"""
rl_controller.py

학습된 RL 모델을 사용하는 제어기
- hybrid_controller.py의 _pid_control() 대체
- 학습된 모델 로드 및 추론
- 행동을 시뮬레이터 명령으로 변환

[사용법]
    from rl_controller import RLController
    
    controller = RLController(model_path="tank_nav_model.zip")
    command = controller.get_action(curr_x, curr_z, curr_yaw, target, obstacles)
"""

import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
import os


@dataclass
class RLConfig:
    """RL 컨트롤러 설정"""
    # 라이다 설정
    lidar_num_rays: int = 64
    lidar_max_range: float = 50.0 # 시뮬레이터 환경과 동기화
    
    # 정규화 상수
    max_speed: float = 11.3
    
    # 행동 매핑 가중치
    forward_weight: float = 0.5
    turn_weight: float = 0.5
    strong_turn_weight: float = 0.8


class RLController:
    """
    학습된 RL 모델 기반 제어기
    
    PID 제어를 대체하여 상황에 맞는 행동 결정
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[RLConfig] = None,
        obstacle_rects: Optional[List[Tuple]] = None,
    ):
        self.config = config or RLConfig()
        self.model = None
        self.model_loaded = False
        
        # 장애물 정보 (라이다 계산용)
        self.obstacle_rects = obstacle_rects or []
        
        # 상태 추적
        self.prev_action = 5  # 정지
        self.current_speed = 0.0
        
        # 맵 설정
        self.map_size = 300.0
        self.map_margin = 5.0
        
        # 모델 로드
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """학습된 모델 로드"""
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
            self.model_loaded = True
            print(f"✅ RL 모델 로드 완료: {model_path}")
        except ImportError:
            print("⚠️ stable_baselines3가 설치되지 않았습니다.")
            print("   pip install stable-baselines3 로 설치하세요.")
            self.model_loaded = False
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            self.model_loaded = False
    
    def set_obstacles(self, obstacles: List[Dict]):
        """장애물 정보 업데이트"""
        self.obstacle_rects = []
        for obs in obstacles:
            self.obstacle_rects.append((
                obs['x_min'], obs['x_max'],
                obs['z_min'], obs['z_max']
            ))
    
    def get_action(
        self,
        curr_x: float,
        curr_z: float,
        curr_yaw: float,
        target: Tuple[float, float],
        goal: Tuple[float, float],
        current_speed: float = 0.0,
    ) -> Dict[str, Any]:
        """
        현재 상태에서 행동 결정
        
        Args:
            curr_x, curr_z: 현재 위치
            curr_yaw: 현재 방향 (degrees)
            target: 다음 waypoint (x, z)
            goal: 최종 목표 (x, z)
            current_speed: 현재 속도
        
        Returns:
            dict: 시뮬레이터 명령 {"moveWS": {...}, "moveAD": {...}, "fire": False}
        """
        self.current_speed = current_speed
        
        # 관측 벡터 생성
        observation = self._build_observation(curr_x, curr_z, curr_yaw, target, goal)
        
        # 모델이 로드되지 않았으면 간단한 규칙 기반 fallback
        if not self.model_loaded or self.model is None:
            action = self._fallback_action(curr_x, curr_z, curr_yaw, target)
        else:
            # RL 모델 추론
            action, _ = self.model.predict(observation, deterministic=True)
        
        # 행동을 명령으로 변환
        command = self._action_to_command(int(action))
        
        self.prev_action = int(action)
        
        return command
    
    def _build_observation(
        self,
        curr_x: float,
        curr_z: float,
        curr_yaw: float,
        target: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> np.ndarray:
        """관측 벡터 생성"""
        
        # 1. Heading error (타겟 방향과의 오차)
        dx = target[0] - curr_x
        dz = target[1] - curr_z
        target_yaw = math.degrees(math.atan2(dx, dz))
        heading_error = target_yaw - curr_yaw
        while heading_error > 180:
            heading_error -= 360
        while heading_error < -180:
            heading_error += 360
        heading_error_norm = heading_error / 180.0
        
        # 2. Target distance
        target_dist = math.hypot(dx, dz)
        target_dist_norm = min(target_dist / 50.0, 1.0)
        
        # 3. Current speed
        speed_norm = self.current_speed / self.config.max_speed
        
        # 4. Goal distance
        goal_dist = math.hypot(goal[0] - curr_x, goal[1] - curr_z)
        goal_dist_norm = min(goal_dist / 300.0, 1.0)
        
        # 5. Lidar rays
        lidar = self._cast_lidar_rays(curr_x, curr_z, curr_yaw)
        lidar_norm = lidar / self.config.lidar_max_range
        
        obs = np.array(
            [heading_error_norm, target_dist_norm, speed_norm, goal_dist_norm] + 
            lidar_norm.tolist(),
            dtype=np.float32
        )
        
        return obs
    
    def _cast_lidar_rays(self, x: float, z: float, yaw: float) -> np.ndarray:
        """가상 라이다 레이캐스팅"""
        num_rays = self.config.lidar_num_rays
        max_range = self.config.lidar_max_range
        
        rays = np.full(num_rays, max_range)
        
        for i in range(num_rays):
            angle_offset = (i / num_rays) * 360 - 180
            ray_angle = yaw + angle_offset
            ray_angle_rad = math.radians(ray_angle)
            
            ray_dx = math.sin(ray_angle_rad)
            ray_dz = math.cos(ray_angle_rad)
            
            step_size = 1.0
            for d in np.arange(step_size, max_range, step_size):
                check_x = x + ray_dx * d
                check_z = z + ray_dz * d
                
                # 맵 경계
                if check_x < self.map_margin or check_x > self.map_size - self.map_margin:
                    rays[i] = d
                    break
                if check_z < self.map_margin or check_z > self.map_size - self.map_margin:
                    rays[i] = d
                    break
                
                # 장애물
                hit = False
                for x_min, x_max, z_min, z_max in self.obstacle_rects:
                    if x_min <= check_x <= x_max and z_min <= check_z <= z_max:
                        rays[i] = d
                        hit = True
                        break
                if hit:
                    break
        
        return rays
    
    def _fallback_action(
        self,
        curr_x: float,
        curr_z: float,
        curr_yaw: float,
        target: Tuple[float, float],
    ) -> int:
        """
        모델 없을 때 간단한 규칙 기반 행동 결정
        
        Returns:
            int: action (0-5)
        """
        # Heading error 계산
        dx = target[0] - curr_x
        dz = target[1] - curr_z
        target_yaw = math.degrees(math.atan2(dx, dz))
        heading_error = target_yaw - curr_yaw
        while heading_error > 180:
            heading_error -= 360
        while heading_error < -180:
            heading_error += 360
        
        # 전방 장애물 체크 (라이다 중앙 3개 레이)
        lidar = self._cast_lidar_rays(curr_x, curr_z, curr_yaw)
        front_rays = lidar[7:9]  # 중앙 부분
        min_front_dist = np.min(front_rays)
        
        # 행동 결정
        if min_front_dist < 5.0:
            # 전방 장애물 가까움 → 회전
            if heading_error > 0:
                return 4  # 우회전
            else:
                return 3  # 좌회전
        
        if abs(heading_error) > 45:
            # 방향 오차 큼 → 제자리 회전
            if heading_error > 0:
                return 4  # 우회전
            else:
                return 3  # 좌회전
        
        if abs(heading_error) > 15:
            # 방향 오차 중간 → 전진 + 회전
            if heading_error > 0:
                return 2  # 전진 + 우회전
            else:
                return 1  # 전진 + 좌회전
        
        # 방향 맞음 → 직진
        return 0
    
    def _action_to_command(self, action: int) -> Dict[str, Any]:
        """
        이산 행동을 시뮬레이터 명령으로 변환
        
        Action mapping:
            0: 전진 (빠름)
            1: 전진 + 좌회전
            2: 전진 + 우회전
            3: 제자리 좌회전
            4: 제자리 우회전
            5: 정지
        """
        cfg = self.config
        
        if action == 0:  # 전진
            return {
                "moveWS": {"command": "W", "weight": cfg.forward_weight},
                "moveAD": {"command": "", "weight": 0.0},
                "turretQE": {"command": "", "weight": 0.0},
                "turretRF": {"command": "", "weight": 0.0},
                "fire": False
            }
        
        elif action == 1:  # 전진 + 좌회전
            return {
                "moveWS": {"command": "W", "weight": cfg.forward_weight * 0.7},
                "moveAD": {"command": "A", "weight": cfg.turn_weight},
                "turretQE": {"command": "", "weight": 0.0},
                "turretRF": {"command": "", "weight": 0.0},
                "fire": False
            }
        
        elif action == 2:  # 전진 + 우회전
            return {
                "moveWS": {"command": "W", "weight": cfg.forward_weight * 0.7},
                "moveAD": {"command": "D", "weight": cfg.turn_weight},
                "turretQE": {"command": "", "weight": 0.0},
                "turretRF": {"command": "", "weight": 0.0},
                "fire": False
            }
        
        elif action == 3:  # 제자리 좌회전
            return {
                "moveWS": {"command": "STOP", "weight": 1.0},
                "moveAD": {"command": "A", "weight": cfg.strong_turn_weight},
                "turretQE": {"command": "", "weight": 0.0},
                "turretRF": {"command": "", "weight": 0.0},
                "fire": False
            }
        
        elif action == 4:  # 제자리 우회전
            return {
                "moveWS": {"command": "STOP", "weight": 1.0},
                "moveAD": {"command": "D", "weight": cfg.strong_turn_weight},
                "turretQE": {"command": "", "weight": 0.0},
                "turretRF": {"command": "", "weight": 0.0},
                "fire": False
            }
        
        else:  # 정지
            return {
                "moveWS": {"command": "STOP", "weight": 1.0},
                "moveAD": {"command": "", "weight": 0.0},
                "turretQE": {"command": "", "weight": 0.0},
                "turretRF": {"command": "", "weight": 0.0},
                "fire": False
            }
    
    def reset(self):
        """제어기 상태 초기화"""
        self.prev_action = 5
        self.current_speed = 0.0


class RLControllerWithFallback(RLController):
    """
    RL + PID Fallback 컨트롤러
    
    RL 모델이 불확실한 상황에서 PID로 폴백
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        pid_controller=None,
        confidence_threshold: float = 0.7,
        **kwargs
    ):
        super().__init__(model_path=model_path, **kwargs)
        self.pid_controller = pid_controller
        self.confidence_threshold = confidence_threshold
        self.use_rl_count = 0
        self.use_pid_count = 0
    
    def get_action(
        self,
        curr_x: float,
        curr_z: float,
        curr_yaw: float,
        target: Tuple[float, float],
        goal: Tuple[float, float],
        current_speed: float = 0.0,
    ) -> Dict[str, Any]:
        """
        RL 또는 PID로 행동 결정
        
        RL 모델의 확신이 낮으면 PID로 폴백
        """
        self.current_speed = current_speed
        
        if not self.model_loaded or self.model is None:
            # 모델 없으면 폴백
            if self.pid_controller:
                self.use_pid_count += 1
                return self._pid_fallback(curr_x, curr_z, curr_yaw, target)
            else:
                action = self._fallback_action(curr_x, curr_z, curr_yaw, target)
                return self._action_to_command(action)
        
        # 관측 벡터 생성
        observation = self._build_observation(curr_x, curr_z, curr_yaw, target, goal)
        
        # RL 모델 추론 (action_probability 포함)
        action, _ = self.model.predict(observation, deterministic=False)
        
        # TODO: 확신도 기반 폴백 (현재는 항상 RL 사용)
        self.use_rl_count += 1
        command = self._action_to_command(int(action))
        self.prev_action = int(action)
        
        return command
    
    def _pid_fallback(
        self,
        curr_x: float,
        curr_z: float,
        curr_yaw: float,
        target: Tuple[float, float],
    ) -> Dict[str, Any]:
        """PID 제어로 폴백"""
        if self.pid_controller is None:
            action = self._fallback_action(curr_x, curr_z, curr_yaw, target)
            return self._action_to_command(action)
        
        # PID 제어기 호출 (기존 hybrid_controller 스타일)
        dx = target[0] - curr_x
        dz = target[1] - curr_z
        target_angle_deg = math.degrees(math.atan2(dx, dz))
        
        error = target_angle_deg - curr_yaw
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        
        pid_output = self.pid_controller.compute(error)
        
        # 조향
        steer_weight = min(abs(pid_output), 1.0)
        steer_dir = "D" if pid_output > 0 else "A"
        if abs(pid_output) < 0.05:
            steer_dir = ""
            steer_weight = 0.0
        
        # 속도
        speed_weight = max(0.3, 0.5 - steer_weight * 0.5)
        
        return {
            "moveWS": {"command": "W", "weight": round(speed_weight, 2)},
            "moveAD": {"command": steer_dir, "weight": round(steer_weight * 0.7, 2)},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False
        }
    
    def get_stats(self) -> Dict[str, int]:
        """사용 통계 반환"""
        return {
            'rl_count': self.use_rl_count,
            'pid_count': self.use_pid_count,
            'total': self.use_rl_count + self.use_pid_count,
        }
