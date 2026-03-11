"""
train_rl.py

Tank 자율주행 RL 에이전트 학습 스크립트
- stable-baselines3의 PPO 사용
- v9 스타일 시각화 (주기적으로 Pygame 창에서 관전)
- 체크포인트 저장

[사용법]
    python train_rl.py train --timesteps 500000
    python train_rl.py train --timesteps 500000 --viz --viz-freq 25000

[요구사항]
    pip install stable-baselines3 gymnasium numpy pygame
"""

import os
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

# Gymnasium & SB3
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    CheckpointCallback, 
    EvalCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# 로컬 모듈
from rl_environment import TankNavEnv, SimConfig
from visualization_callback import VisualEvalCallback, ImageSaveCallback


class TensorboardCallback(BaseCallback):
    """Tensorboard 로깅 콜백"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.successes = []
        
    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][idx]
                    
                    if info.get('reached_goal', False):
                        self.successes.append(1)
                    else:
                        self.successes.append(0)
                    
                    if len(self.successes) >= 100:
                        success_rate = np.mean(self.successes[-100:])
                        self.logger.record('custom/success_rate_100', success_rate)
                    
                    final_dist = info.get('distance_to_goal', 0)
                    self.logger.record('custom/final_distance', final_dist)
        
        return True


class ProgressCallback(BaseCallback):
    """학습 진행 상황 출력 콜백"""
    def __init__(self, total_timesteps: int, print_freq: int = 10000, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.episode_count = 0
        self.success_count = 0
        self.recent_rewards = []
        
    def _on_step(self) -> bool:
        if self.locals.get('dones') is not None:
            for idx, done in enumerate(self.locals['dones']):
                if done:
                    self.episode_count += 1
                    info = self.locals['infos'][idx]
                    
                    if info.get('reached_goal', False):
                        self.success_count += 1
                    
                    if 'episode' in info:
                        self.recent_rewards.append(info['episode']['r'])
        
        if self.num_timesteps % self.print_freq == 0:
            progress = self.num_timesteps / self.total_timesteps * 100
            
            success_rate = 0
            if self.episode_count > 0:
                success_rate = self.success_count / self.episode_count * 100
            
            avg_reward = 0
            if self.recent_rewards:
                avg_reward = np.mean(self.recent_rewards[-100:])
            
            print(f"\n{'='*60}")
            print(f"📊 학습 진행: {progress:.1f}% ({self.num_timesteps:,} / {self.total_timesteps:,})")
            print(f"📈 에피소드: {self.episode_count}, 성공률: {success_rate:.1f}%")
            print(f"💰 평균 보상 (최근 100): {avg_reward:.1f}")
            print(f"{'='*60}\n")
        
        return True


def load_obstacles(json_path: str) -> list:
    """장애물 JSON 로드"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('obstacles', [])


def load_terrain_data(height_path: str, slope_path: str):
    """지형 데이터 로드"""
    height_map = None
    slope_map = None
    
    if os.path.exists(height_path):
        height_map = np.load(height_path)
        print(f"✅ Height map 로드: {height_map.shape}")
    
    if os.path.exists(slope_path):
        slope_map = np.load(slope_path)
        print(f"✅ Slope map 로드: {slope_map.shape}")
    
    return height_map, slope_map


def make_env_fn(obstacles, height_map, slope_map, config, rank, seed=0):
    """병렬 환경 생성 함수"""
    def _init():
        env = TankNavEnv(
            obstacles=obstacles,
            height_map=height_map,
            slope_map=slope_map,
            config=config,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def train(
    total_timesteps: int = 500000,
    save_path: str = "./models",
    obstacle_path: str = "env_data/ob_v2.json",
    height_path: str = "env_data/height_map.npy",
    slope_path: str = "env_data/slope_costmap.npy",
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    seed: int = 42,
    eval_freq: int = 10000,
    checkpoint_freq: int = 50000,
    tensorboard_log: str = "./tensorboard_logs",
    enable_viz: bool = False,
    viz_freq: int = 25000,
    n_viz_episodes: int = 3,
):
    """RL 에이전트 학습"""
    print("="*60)
    print("🚀 Tank Navigation RL 학습 시작")
    print("="*60)
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    
    obstacles = []
    if os.path.exists(obstacle_path):
        obstacles = load_obstacles(obstacle_path)
        print(f"✅ 장애물 로드: {len(obstacles)}개")
    else:
        print(f"⚠️ 장애물 파일 없음: {obstacle_path}")
    
    height_map, slope_map = load_terrain_data(height_path, slope_path)
    
    # 환경 설정
    config = SimConfig(
        max_episode_steps=1500,
        goal_threshold=8.0,
        reward_goal=1000.0,
        reward_collision=-500.0,
        reward_approach=10.0,
    )
    
    # 병렬 환경 생성
    print(f"\n🎮 환경 생성 중 (n_envs={n_envs})...")
    
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env_fn(obstacles, height_map, slope_map, config, i, seed)
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env_fn(obstacles, height_map, slope_map, config, 0, seed)
        ])
    
    # 평가용 환경 (별도)
    eval_env = DummyVecEnv([
        make_env_fn(obstacles, height_map, slope_map, config, 0, seed + 1000)
    ])
    
    # 시각화용 환경 (별도, 단일 환경)
    viz_env = TankNavEnv(
        obstacles=obstacles,
        height_map=height_map,
        slope_map=slope_map,
        config=config,
    )
    
    # PPO 모델 생성
    print("\n🧠 PPO 모델 생성 중...")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=seed,
        device="auto",
    )
    
    print(f"✅ 모델 생성 완료")
    print(f"   - Policy: MlpPolicy")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - n_envs: {n_envs}")
    
    # 콜백 설정
    callbacks = [
        ProgressCallback(total_timesteps, print_freq=10000),
        TensorboardCallback(),
        CheckpointCallback(
            save_freq=checkpoint_freq // n_envs,
            save_path=save_path,
            name_prefix="tank_nav",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(save_path, "best"),
            log_path=os.path.join(save_path, "eval_logs"),
            eval_freq=eval_freq // n_envs,
            n_eval_episodes=10,
            deterministic=True,
        ),
    ]
    
    # 시각화 콜백 추가 (v9 스타일)
    if enable_viz:
        visual_callback = VisualEvalCallback(
            eval_env=viz_env,
            eval_freq=viz_freq,
            n_eval_episodes=n_viz_episodes,
            verbose=1,
        )
        callbacks.append(visual_callback)
        print(f"📊 시각화 활성화: {viz_freq} 스텝마다 {n_viz_episodes} 에피소드 관전")
    
    # 학습 시작
    print(f"\n🏋️ 학습 시작 (총 {total_timesteps:,} 스텝)...")
    print(f"   - Tensorboard: tensorboard --logdir {tensorboard_log}")
    print(f"   - 체크포인트: {save_path}")
    
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️ 학습 중단됨 (Ctrl+C)")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # 최종 모델 저장
    final_path = os.path.join(save_path, "tank_nav_final.zip")
    model.save(final_path)
    
    print("\n" + "="*60)
    print("✅ 학습 완료!")
    print(f"   - 소요 시간: {duration}")
    print(f"   - 최종 모델: {final_path}")
    print("="*60)
    
    # 환경 정리
    env.close()
    eval_env.close()
    
    return model, final_path


def evaluate(
    model_path: str,
    obstacle_path: str = "env_data/ob_v2.json",
    height_path: str = "env_data/height_map.npy",
    slope_path: str = "env_data/slope_costmap.npy",
    n_episodes: int = 10,
    render: bool = False,
):
    """학습된 모델 평가"""
    print("="*60)
    print("📊 모델 평가")
    print("="*60)
    
    obstacles = []
    if os.path.exists(obstacle_path):
        obstacles = load_obstacles(obstacle_path)
    
    height_map, slope_map = load_terrain_data(height_path, slope_path)
    
    config = SimConfig()
    
    env = TankNavEnv(
        obstacles=obstacles,
        height_map=height_map,
        slope_map=slope_map,
        config=config,
        render_mode="human" if render else None,
    )
    
    model = PPO.load(model_path)
    print(f"✅ 모델 로드: {model_path}")
    
    successes = 0
    total_rewards = []
    total_steps = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if render:
                env.render()
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        
        if info.get('reached_goal', False):
            successes += 1
            print(f"  Episode {ep+1}: ✅ 성공! (reward={episode_reward:.1f}, steps={steps})")
        else:
            print(f"  Episode {ep+1}: ❌ 실패 (reward={episode_reward:.1f}, steps={steps})")
    
    env.close()
    
    print("\n" + "="*60)
    print(f"📊 평가 결과 ({n_episodes} 에피소드)")
    print(f"   - 성공률: {successes/n_episodes*100:.1f}%")
    print(f"   - 평균 보상: {np.mean(total_rewards):.1f}")
    print(f"   - 평균 스텝: {np.mean(total_steps):.1f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Tank Navigation RL Training")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # 학습 명령
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps")
    train_parser.add_argument("--save-path", type=str, default="./models", help="Model save path")
    train_parser.add_argument("--obstacles", type=str, default="env_data/ob_v2.json", help="Obstacle JSON path")
    train_parser.add_argument("--height", type=str, default="env_data/height_map.npy", help="Height map path")
    train_parser.add_argument("--slope", type=str, default="env_data/slope_costmap.npy", help="Slope map path")
    train_parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--viz", action="store_true", help="Enable visual evaluation")
    train_parser.add_argument("--viz-freq", type=int, default=25000, help="Visual eval frequency (steps)")
    train_parser.add_argument("--viz-episodes", type=int, default=3, help="Episodes per visual eval")
    
    # 평가 명령
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.add_argument("--model", type=str, required=True, help="Model path")
    eval_parser.add_argument("--obstacles", type=str, default="env_data/ob_v2.json", help="Obstacle JSON path")
    eval_parser.add_argument("--height", type=str, default="env_data/height_map.npy", help="Height map path")
    eval_parser.add_argument("--slope", type=str, default="env_data/slope_costmap.npy", help="Slope map path")
    eval_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    eval_parser.add_argument("--render", action="store_true", help="Render visualization")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            obstacle_path=args.obstacles,
            height_path=args.height,
            slope_path=args.slope,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
            enable_viz=args.viz,
            viz_freq=args.viz_freq,
            n_viz_episodes=args.viz_episodes,
        )
    
    elif args.command == "eval":
        evaluate(
            model_path=args.model,
            obstacle_path=args.obstacles,
            height_path=args.height,
            slope_path=args.slope,
            n_episodes=args.episodes,
            render=args.render,
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
