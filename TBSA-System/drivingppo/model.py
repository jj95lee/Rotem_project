from typing import Callable, Literal
import time

from .world import World
from .environment import WorldEnv
from .common import (
    LIDAR_NUM,
    OBSERVATION_IND_SPD,
    OBSERVATION_IND_GOAL_2,
    OBSERVATION_IND_SCALAR_S,
    OBSERVATION_IND_SCALAR_E,
    OBSERVATION_DIM_SCALAR,
    OBSERVATION_IND_LIDAR_DIS_S,
    OBSERVATION_IND_LIDAR_DIS_E,
    OBSERVATION_DIM_LIDAR,
)

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, DummyVecEnv
import torch
import torch.nn as nn
import torch.nn.init as init


import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np


# 훈련 결과 저장
LOG_DIR = "./ppo_tensorboard_logs/"
CHECKPOINT_DIR = './ppo_world_checkpoints/'



class MyFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box):

        speed_and_goal1_dim = OBSERVATION_IND_GOAL_2 - OBSERVATION_IND_SPD
        cnn_channel_num = 2
        feature0_dim = 32
        feature1_dim = 64
        feature2_dim = 128
        total_feature_dim   = speed_and_goal1_dim + feature0_dim + feature1_dim + feature2_dim

        super(MyFeatureExtractor, self).__init__(observation_space, features_dim=total_feature_dim)

        self.layer0 = nn.Sequential(
            nn.Linear(OBSERVATION_DIM_SCALAR, feature0_dim),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            # 노이즈 제거 및 로컬 특징 추출
            nn.Conv1d(in_channels=1, out_channels=cnn_channel_num, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=feature1_dim),
            nn.Flatten(),
            nn.Linear(cnn_channel_num * feature1_dim, feature1_dim)
        )

        self.layer2 = nn.Sequential(
            # 위치 정보 매핑 (Linear)
            nn.Linear(speed_and_goal1_dim + feature1_dim, feature2_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        speed_and_goal1 = observations[:, OBSERVATION_IND_SPD         : OBSERVATION_IND_GOAL_2]
        scalar_data     = observations[:, OBSERVATION_IND_SCALAR_S    : OBSERVATION_IND_SCALAR_E]
        lidar_dis_data  = observations[:, OBSERVATION_IND_LIDAR_DIS_S : OBSERVATION_IND_LIDAR_DIS_E]

        output0 = self.layer0(scalar_data)
        output1 = self.layer1(lidar_dis_data.unsqueeze(1))
        output2 = self.layer2(torch.cat((speed_and_goal1, output1), dim=1))

        return torch.cat((speed_and_goal1, output0, output1, output2), dim=1)



def train_start(
        gen_env:Callable[[], WorldEnv],
        steps:int,
        save_path:str|None=None,
        save_freq:int=0,
        tb_log:bool=False,
        *,
        vec_env:Literal['dummy', 'subp']|VecEnv='dummy',
        lr=3e-4,
        gamma=0.9,
        ent_coef=0.01,
        n_steps=1024,
        batch_size=256,
        seed=42
) -> PPO:
    """
    학습 처음부터
    """

    if vec_env == 'dummy':
        vec_env_cls = DummyVecEnv
    elif vec_env == 'subp':
        vec_env_cls = SubprocVecEnv
    else:
        raise Exception(f'unknown vec_env: {vec_env}')
    vec_env = make_vec_env(gen_env, n_envs=1, vec_env_cls=vec_env_cls, seed=seed)# n_envs: 병렬 환경 수

    policy_kwargs = dict(
        features_extractor_class=MyFeatureExtractor,
        features_extractor_kwargs=dict(),
        net_arch=dict(
            pi=[256, 256], # Actor
            vf=[256, 256, 128]  # Critic
        )
    )

    # PPO 모델
    model = PPO(
        "MlpPolicy", # Multi-Layer Perceptron (피드 포워드 신경망 정책)
        vec_env,
        policy_kwargs=policy_kwargs,

        verbose=1,
        tensorboard_log=LOG_DIR  if tb_log  else None,

        # 학습 하이퍼파라미터
        learning_rate=lr,
        gamma=gamma,           # 미래 보상 할인율
        ent_coef=ent_coef,     # 엔트로피: 장애물 거의 없는 환경 - 약하게
        n_steps=n_steps,       # 데이터 수집 스텝 (버퍼 크기, NUM_ENVS * n_steps = 총 수집 데이터량)
        batch_size=batch_size, # 미니 배치 크기

        device="auto"  # GPU 사용 설정
    )

    # 콜백 - 모델 저장
    checkpoint_callback = None
    if save_freq:
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=CHECKPOINT_DIR,
            name_prefix='check'
        )

    print("=== PPO 학습 시작 ===")

    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback
    )

    # 최종 모델 저장
    if save_path:
        model.save(CHECKPOINT_DIR+save_path)
        print(f"=== 학습 완료: {CHECKPOINT_DIR+save_path} ===")

    vec_env.close()  # 환경 정리

    return model


def train_resume(
        model:PPO|str,
        gen_env: Callable[[], WorldEnv],
        steps: int,
        save_path:str|None=None,
        save_freq:int=0,
        tb_log:bool=False,
        *,
        vec_env:Literal['dummy', 'subp']|VecEnv='dummy',
        log_std=None,
        lr=1e-4,
        gamma=0.99,
        ent_coef=0.05,
        seed=42
) -> PPO:
    """
    기존 모델 추가학습
    """

    if vec_env == 'dummy':
        vec_env_cls = DummyVecEnv
    elif vec_env == 'subp':
        vec_env_cls = SubprocVecEnv
    else:
        raise Exception(f'unknown vec_env: {vec_env}')
    vec_env = make_vec_env(gen_env, n_envs=4, vec_env_cls=vec_env_cls, seed=seed)

    # 모델 로드 (학습된 모델의 환경도 함께 로드)
    if type(model) == str:
        print(f"=== 체크포인트 로드: {CHECKPOINT_DIR+model} ===")
        model = PPO.load(
            path=CHECKPOINT_DIR+model,
            env=vec_env,
            verbose=1,
            tensorboard_log=LOG_DIR  if tb_log  else None,

            learning_rate=lr,
            gamma=gamma,       # 미래 보상 할인율
            ent_coef=ent_coef, # 새로운 환경 -> 새로운 시도 위해 엔트로피 높임.
        )
    assert isinstance(model, PPO)

    # 콜백 - 모델 저장
    checkpoint_callback = None
    if save_freq:
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=CHECKPOINT_DIR,
            name_prefix='check'
        )

    if log_std:
        with torch.no_grad():
            # log_std 값 덮어쓰기 (모델 구조에 따라 접근법이 다를 수 있으나 보통 아래와 같음)
            model.policy.log_std.fill_(log_std)

    total_timesteps = steps + model.num_timesteps

    print(f"=== 학습 재개 (현재 스텝: {model.num_timesteps} / 목표: {total_timesteps} / 남은: {steps}) ===")

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,

        reset_num_timesteps=False # 내부 타임스텝 카운터 초기화 여부
    )

    # 최종 모델 저장
    if save_path:
        model.save(CHECKPOINT_DIR+save_path)
        print(f"=== 학습 완료: {CHECKPOINT_DIR+save_path} ===")

    vec_env.close()  # 환경 정리

    return model


def run(
        world_generator:Callable[[], World],
        model:PPO|str,
        time_spd=2.0,
        time_step=111,
        step_per_control=3,
        auto_close_at_end=True,
    ):
    """
    모델 시각적 확인용 실행
    """

    env = WorldEnv(
        world_generator=world_generator,
        time_step=time_step,
        step_per_control=1,
        render_mode='debug',
        auto_close_at_end=auto_close_at_end
    )

    if type(model) == str:
        model = PPO.load(CHECKPOINT_DIR+model, env=env)
    assert isinstance(model, PPO)

    obs, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0.0

    while not terminated and not truncated:

        action, _ = model.predict(obs, deterministic=True)  # 에이전트가 행동 선택
        for _ in range(step_per_control):
            obs, reward, terminated, truncated, info = env.step(action)  # 행동 실행
            episode_reward += reward
            env.render()  # 시각화 호출
            time.sleep(time_step / 1000.0 / time_spd)# 시각화 프레임을 위해 딜레이 추가
            if terminated or truncated: break

    print(f"에피소드 종료. 총 보상: {episode_reward:.2f}")

    env.close()
