"""
rl_environment.py

Tank 자율주행을 위한 Gymnasium 환경
- 시뮬레이터 없이 로컬에서 학습 가능
- A* waypoint를 따라가는 국소 제어 학습
- 기존 astar_planner, config와 호환

[Observation Space]
- 목표 방향 오차 (정규화)
- 목표까지 거리 (정규화)
- 현재 속도
- 가상 라이다 (64방향)

[Action Space]
- Discrete(6): 전진, 전진+좌회전, 전진+우회전, 제자리좌회전, 제자리우회전, 정지

[Reward]
- 목표 접근: +
- 충돌: -
- 시간 패널티: -
- 부드러운 조향: +
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class TankState:
    """전차 상태"""
    x: float
    z: float
    yaw: float  # degrees, 0 = +z direction
    speed: float
    

@dataclass 
class SimConfig:
    """시뮬레이션 설정"""
    # 맵 설정
    map_size: float = 300.0
    map_margin: float = 5.0
    
    # 전차 설정
    tank_width: float = 3.667
    tank_length: float = 8.066
    max_speed: float = 11.3  # 시뮬레이터 weight가 0.5일때의 추정 속도
    max_turn_rate: float = 30.0  # deg/s (추정)
    
    # 시뮬레이션 설정
    dt: float = 0.2  # 시간 간격 (시뮬레이터 통신 주기와 유사)
    max_episode_steps: int = 1500  # 최대 스텝 (300초 / 0.2초)
    
    # 목표 설정
    goal_threshold: float = 8.0  # 목표 도달 거리
    
    # 가상 라이다 설정
    lidar_num_rays: int = 64
    lidar_max_range: float = 50.0 # 시뮬레이터 환경과 동기화

    # 안전 마진 설정
    margin_critical: float = 0.1
    margin_warning: float = 0.5
    
    # 보상 설정
    reward_goal: float = 1000.0
    reward_collision: float = -500.0
    reward_approach: float = 10.0  # per meter
    reward_time_penalty: float = -0.1
    reward_smooth_steering: float = 0.5


class TankNavEnv(gym.Env):
    """
    Tank 자율주행 환경
    
    A* waypoint를 따라가는 국소 제어를 학습
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(
        self,
        obstacles: Optional[List[Dict]] = None,
        height_map: Optional[np.ndarray] = None,
        slope_map: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
        config: Optional[SimConfig] = None,
    ):
        super().__init__()
        
        self.config = config or SimConfig()
        self.render_mode = render_mode
        
        # 장애물 로드
        self.obstacles = obstacles or []
        self.obstacle_rects = self._convert_obstacles(self.obstacles)

        self._add_walls()

        if self.obstacle_rects:
            self.obstacles_np = np.array(self.obstacle_rects, dtype=np.float32)
        else:
            self.obstacles_np = np.zeros((0, 4), dtype = np.float32)
        
        # 지형 데이터
        self.height_map = height_map
        self.slope_map = slope_map
        if self.slope_map is not None:
            self.slope_map = np.nan_to_num(self.slope_map, nan=0.0)
        
        # Action Space: 6개 이산 행동
        # 0: 전진 (빠름)
        # 1: 전진 + 좌회전
        # 2: 전진 + 우회전  
        # 3: 제자리 좌회전
        # 4: 제자리 우회전
        # 5: 정지
        self.action_space = spaces.Discrete(6)
        
        # Observation Space
        # [0]: heading_error / 180 (-1 ~ 1)
        # [1]: distance_to_target / 50 (0 ~ 1, clipped)
        # [2]: current_speed / max_speed (0 ~ 1)
        # [3]: goal_distance / 300 (0 ~ 1)
        # [4-19]: lidar rays (16개) / max_range (0 ~ 1)
        obs_dim = 4 + self.config.lidar_num_rays
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # 직사각형 전차 모델링
        ## 라이다 각도 배열 생성
        self.lidar_angles_rel = np.linspace(
            0, 2 * np.pi, self.config.lidar_num_rays, endpoint=False
        )

        ## 각 ray별 중심 ~차체 표면 거리 계산
        half_l = self.config.tank_length / 2.0
        half_w = self.config.tank_width / 2.0
        
        ## 0으로 나누기 방지
        abs_cos = np.maximum(np.abs(np.cos(self.lidar_angles_rel)), 1e-6)
        abs_sin = np.maximum(np.abs(np.sin(self.lidar_angles_rel)), 1e-6)
        
        ## 직사각형의 기하학적 특성을 이용해 거리 계산
        self.tank_boundary_dist = np.minimum(half_l / abs_cos, half_w / abs_sin)

        # 상태 변수
        self.state: Optional[TankState] = None
        self.target: Optional[Tuple[float, float]] = None
        self.goal: Optional[Tuple[float, float]] = None
        self.path: List[Tuple[float, float]] = []
        self.current_path_idx: int = 0
        
        self.step_count: int = 0
        self.prev_distance_to_goal: float = 0.0
        self.prev_action: int = 5  # 정지
        self.collision_count: int = 0
        
        # 시각화용
        self.trajectory: List[Tuple[float, float]] = []
        
    def _convert_obstacles(self, obstacles: List[Dict]) -> List[Tuple[float, float, float, float]]:
        """장애물을 (x_min, x_max, z_min, z_max) 튜플 리스트로 변환"""
        rects = []
        for obs in obstacles:
            rects.append((
                obs['x_min'], obs['x_max'],
                obs['z_min'], obs['z_max']
            ))
        return rects
    
    def _add_walls(self):
        '''맵 경계 벽 추가'''
        size = self.config.map_size
        walls = [
            (-5.0, 0.0, -10.0, size + 10.0),
            (size, size + 5, -10.0, size + 10),
            (-10.0, size + 10, -5.0, 0.0),
            (-10.0, size + 10, size, size + 5)
        ]
        self.obstacle_rects.extend(walls)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # 시작점과 목표점 설정
        if options and 'start' in options:
            start_x, start_z = options['start']
        else:
            # 기본 시작점 또는 랜덤
            start_x, start_z = 49.0, 236.0  # 시나리오 시작점
            
        if options and 'goal' in options:
            self.goal = options['goal']
        else:
            self.goal = (65.0, 30.0)  # 시나리오 목표점
        
        # 경로 설정 (options에서 받거나 직선 경로)
        if options and 'path' in options:
            self.path = options['path']
        else:
            self.path = [self.goal]  # 단순히 목표만
            
        self.current_path_idx = 0
        
        # 초기 yaw: 목표 방향 또는 랜덤
        if options and 'initial_yaw' in options:
            initial_yaw = options['initial_yaw']
        else:
            # 목표 방향으로 약간의 오차를 준 상태로 시작
            dx = self.goal[0] - start_x
            dz = self.goal[1] - start_z
            target_yaw = math.degrees(math.atan2(dx, dz))
            initial_yaw = target_yaw + self.np_random.uniform(-30, 30)
        
        self.state = TankState(
            x=start_x,
            z=start_z,
            yaw=initial_yaw,
            speed=0.0
        )
        
        self.step_count = 0
        self.prev_distance_to_goal = self._distance_to_goal()
        self.prev_action = 5
        self.collision_count = 0
        self.trajectory = [(start_x, start_z)]
        
        # 현재 타겟 설정
        self._update_target()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.state is not None, "Call reset() first"
        
        self.step_count += 1
        
        # 1. 행동 실행 (물리 시뮬레이션)
        self._apply_action(action)
        
        # 2. 타겟 업데이트
        self._update_target()

        # 3. 라이다 raycasting 수행
        lidar_rays = self._cast_lidar_vectorized()
        
        # 4. 충돌 체크
        collision, warning = self._check_collision_and_margin(lidar_rays)
        
        # 5. 목표 도달 체크
        distance_to_goal = self._distance_to_goal()
        reached_goal = distance_to_goal < self.config.goal_threshold
        
        # 6. 보상 계산
        reward = self._calculate_reward(action, collision, warning, reached_goal, distance_to_goal)
        
        # 7. 종료 조건
        terminated = reached_goal or collision
        truncated = self.step_count >= self.config.max_episode_steps
        
        # 8. 상태 업데이트
        self.prev_distance_to_goal = distance_to_goal
        self.prev_action = action
        self.trajectory.append((self.state.x, self.state.z))
        
        observation = self._get_observation()
        info = self._get_info()
        info['collision'] = collision
        info['reached_goal'] = reached_goal
        info['pos'] = (self.state.x, self.state.z)
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action: int):
        """행동을 적용하여 전차 상태 업데이트"""
        dt = self.config.dt
        max_speed = self.config.max_speed
        max_turn = self.config.max_turn_rate
        
        # 경사도에 따른 속도 조절
        slope = self._get_slope_at(self.state.x, self.state.z)
        speed_factor = 1.0
        if slope > 30:
            speed_factor = 0.3
        elif slope > 15:
            speed_factor = 0.6
        
        # 행동 해석
        if action == 0:  # 전진 (빠름)
            target_speed = max_speed * speed_factor
            turn_rate = 0.0
        elif action == 1:  # 전진 + 좌회전
            target_speed = max_speed * 0.7 * speed_factor
            turn_rate = -max_turn * 0.5
        elif action == 2:  # 전진 + 우회전
            target_speed = max_speed * 0.7 * speed_factor
            turn_rate = max_turn * 0.5
        elif action == 3:  # 제자리 좌회전
            target_speed = 0.0
            turn_rate = -max_turn
        elif action == 4:  # 제자리 우회전
            target_speed = 0.0
            turn_rate = max_turn
        else:  # 정지
            target_speed = 0.0
            turn_rate = 0.0
        
        # 속도 변화 (간단한 가속 모델)
        accel = 2.0  # m/s^2
        if target_speed > self.state.speed:
            self.state.speed = min(self.state.speed + accel * dt, target_speed)
        else:
            self.state.speed = max(self.state.speed - accel * dt, target_speed)
        
        # 회전 적용
        self.state.yaw += turn_rate * dt
        # yaw 정규화 (-180 ~ 180)
        while self.state.yaw > 180:
            self.state.yaw -= 360
        while self.state.yaw < -180:
            self.state.yaw += 360
        
        # 위치 업데이트 (Unity 좌표계: yaw=0 → +z 방향)
        yaw_rad = math.radians(self.state.yaw)
        dx = math.sin(yaw_rad) * self.state.speed * dt
        dz = math.cos(yaw_rad) * self.state.speed * dt
        
        new_x = self.state.x + dx
        new_z = self.state.z + dz
        
        # 맵 경계 체크
        margin = self.config.map_margin
        new_x = np.clip(new_x, margin, self.config.map_size - margin)
        new_z = np.clip(new_z, margin, self.config.map_size - margin)
        
        self.state.x = new_x
        self.state.z = new_z
    
    def _update_target(self):
        """현재 타겟 waypoint 업데이트"""
        if not self.path:
            self.target = self.goal
            return
        
        # 현재 위치에서 가장 가까운 경로 노드 찾기
        while self.current_path_idx < len(self.path) - 1:
            wp = self.path[self.current_path_idx]
            dist = math.hypot(wp[0] - self.state.x, wp[1] - self.state.z)
            if dist < 5.0:  # 5m 이내면 다음 waypoint로
                self.current_path_idx += 1
            else:
                break
        
        # Lookahead 거리만큼 앞의 waypoint 선택
        lookahead = 10.0
        cumulative_dist = 0.0
        target_idx = self.current_path_idx
        
        for i in range(self.current_path_idx, len(self.path)):
            if i > self.current_path_idx:
                prev = self.path[i-1]
                curr = self.path[i]
                cumulative_dist += math.hypot(curr[0] - prev[0], curr[1] - prev[1])
            if cumulative_dist >= lookahead:
                target_idx = i
                break
            target_idx = i
        
        self.target = self.path[target_idx]
    
    def _get_slope_at(self, x: float, z: float) -> float:
        """특정 위치의 경사도 반환"""
        if self.slope_map is None:
            return 0.0
        gx, gz = int(x), int(z)
        if 0 <= gx < self.slope_map.shape[1] and 0 <= gz < self.slope_map.shape[0]:
            return self.slope_map[gz, gx]
        return 0.0
    
    def _check_collision_and_margin(self, lidar_rays: np.ndarray) -> Tuple[bool, bool]:
        """충돌 체크"""
        
        # LiDAR 기반 정밀 체크 (직사각형 모양 고려)
        safety_margins = lidar_rays - self.tank_boundary_dist

        is_lidar_collision = np.any(safety_margins < self.config.margin_critical)
        is_warning = np.any((safety_margins >= self.config.margin_critical) & (safety_margins < self.config.margin_warning))

        # 물리적 위치 기반 체크
        x, z = self.state.x, self.state.z
        is_physical_collision = False

        # 맵 경계 충돌
        margin = self.config.map_margin
        if x < margin or x > self.config.map_size - margin:
            is_physical_collision = True
        if z < margin or z > self.config.map_size - margin:
            is_physical_collision = True
        
        # 장애물 AABB 체크
        if not is_physical_collision:
            tank_radius = max(self.config.tank_width, self.config.tank_length) / 2 + 0.5
        
            for x_min, x_max, z_min, z_max in self.obstacle_rects:
                # 장애물과 전차 중심 간 거리
                closest_x = np.clip(x, x_min, x_max)
                closest_z = np.clip(z, z_min, z_max)
                dist = math.hypot(x - closest_x, z - closest_z)
                
                if dist < tank_radius:
                    is_physical_collision = True
                    break
        
        # 결과 통합
        final_collision = is_lidar_collision or is_physical_collision

        if final_collision:
            self.collision_count += 1

        return final_collision, is_warning
    
    def _distance_to_goal(self) -> float:
        """목표까지 거리"""
        return math.hypot(self.goal[0] - self.state.x, self.goal[1] - self.state.z)
    
    def _distance_to_target(self) -> float:
        """현재 타겟까지 거리"""
        if self.target is None:
            return self._distance_to_goal()
        return math.hypot(self.target[0] - self.state.x, self.target[1] - self.state.z)
    
    def _heading_error(self) -> float:
        """타겟 방향과의 각도 오차 (degrees)"""
        if self.target is None:
            return 0.0
        
        dx = self.target[0] - self.state.x
        dz = self.target[1] - self.state.z
        target_yaw = math.degrees(math.atan2(dx, dz))
        
        error = target_yaw - self.state.yaw
        # -180 ~ 180으로 정규화
        while error > 180:
            error -= 360
        while error < -180:
            error += 360
        
        return error
    
    def _cast_lidar_vectorized(self) -> np.ndarray:
        """가상 라이다 레이캐스팅"""
        num_rays = self.config.lidar_num_rays
        max_range = self.config.lidar_max_range

        if len(self.obstacle_rects) == 0:
            return np.full(num_rays, max_range, dtype=np.float32)
        
        yaw_rad = math.radians(self.state.yaw)
        ray_angles = self.lidar_angles_rel + yaw_rad

        ray_dirs = np.stack([np.cos(ray_angles), np.sin(ray_angles)], axis=1)
        ray_dirs_exp = ray_dirs[:, np.newaxis, :]
        obs_exp = self.obstacles_np[np.newaxis, :, :]

        pos_x, pos_z = self.state.x, self.state.z
        
        # 0으로 나누기 방지
        dir_x = np.where(np.abs(ray_dirs_exp[..., 0]) < 1e-6, 1e-6, ray_dirs_exp[..., 0])
        dir_z = np.where(np.abs(ray_dirs_exp[..., 1]) < 1e-6, 1e-6, ray_dirs_exp[..., 1])
        
        # AABB 슬랩 교차 계산
        # obs_exp[..., 0] = x_min, [1] = x_max, [2] = z_min, [3] = z_max
        t1 = (obs_exp[..., 0] - pos_x) / dir_x
        t2 = (obs_exp[..., 1] - pos_x) / dir_x
        t3 = (obs_exp[..., 2] - pos_z) / dir_z
        t4 = (obs_exp[..., 3] - pos_z) / dir_z
        
        t_enter = np.maximum(np.minimum(t1, t2), np.minimum(t3, t4))
        t_exit = np.minimum(np.maximum(t1, t2), np.maximum(t3, t4))
        
        # 유효한 교차 (레이가 박스를 통과)
        mask = (t_enter < t_exit) & (t_exit > 0)
        
        # 교차 거리 (교차 안하면 inf)
        dist = np.where(mask, t_enter, np.inf)
        
        # 레이 시작점이 박스 내부에 있는 경우
        inside = (t_enter < 0) & (t_exit > 0)
        dist = np.where(inside, 0.0, dist)
        
        # 각 레이별 최소 거리
        min_dists = np.min(dist, axis=1)
        
        # 범위 클리핑
        lidar_data = np.clip(min_dists, 0, max_range).astype(np.float32)
        
        return lidar_data
    
    def _get_observation(self, lidar_rays: Optional[np.array]=None) -> np.ndarray:
        """관측 벡터 생성"""

        if lidar_rays is None:
            lidar_rays = self._cast_lidar_rays()
        lidar = lidar_rays / self.config.lidar_max_range

        # Heading error (정규화)
        heading_error = self._heading_error() / 180.0
        
        # Target distance (정규화, 클리핑)
        target_dist = min(self._distance_to_target() / 50.0, 1.0)
        
        # Current speed (정규화)
        speed_norm = self.state.speed / self.config.max_speed
        
        # Goal distance (정규화)
        goal_dist = min(self._distance_to_goal() / 300.0, 1.0)
        
        obs = np.array([heading_error, target_dist, speed_norm, goal_dist] + lidar.tolist(), 
                       dtype=np.float32)
        
        return obs
    
    def _calculate_reward(self, action: int, collision: bool, warning: bool, reached_goal: bool, 
                          distance_to_goal: float) -> float:
        """보상 계산"""
        cfg = self.config
        reward = 0.0
        
        # 1. 목표 도달
        if reached_goal:
            reward += cfg.reward_goal
            return reward
        
        # 2. 충돌 페널티
        if collision:
            reward += cfg.reward_collision
            return reward
        
        # 3. 목표 접근 보상
        distance_delta = self.prev_distance_to_goal - distance_to_goal
        reward += distance_delta * cfg.reward_approach
        
        # 4. 시간 페널티
        reward += cfg.reward_time_penalty
        
        # 5. 부드러운 조향 보상
        if action == self.prev_action:
            reward += cfg.reward_smooth_steering * 0.5
        
        # 6. 방향 정렬 보상
        heading_error = abs(self._heading_error())
        if heading_error < 15:
            reward += 0.5
        elif heading_error < 30:
            reward += 0.2
        
        # 7. 속도 보상 (움직이면 보상)
        if self.state.speed > 1.0:
            reward += 0.3

        if warning:
            reward -= 0.5
        
        return reward
    
    def _get_info(self) -> Dict[str, Any]:
        """정보 딕셔너리"""
        return {
            'step': self.step_count,
            'position': (self.state.x, self.state.z),
            'yaw': self.state.yaw,
            'speed': self.state.speed,
            'distance_to_goal': self._distance_to_goal(),
            'heading_error': self._heading_error(),
            'collision_count': self.collision_count,
        }
    
    def render(self):
        """시각화 (matplotlib)"""
        if self.render_mode != "human":
            return
        
        import matplotlib.pyplot as plt
        
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 장애물
        for x_min, x_max, z_min, z_max in self.obstacle_rects:
            rect = plt.Rectangle((x_min, z_min), x_max - x_min, z_max - z_min,
                                  color='gray', alpha=0.5)
            ax.add_patch(rect)
        
        # 경로
        if self.path:
            path_x = [p[0] for p in self.path]
            path_z = [p[1] for p in self.path]
            ax.plot(path_x, path_z, 'b--', linewidth=1, alpha=0.5, label='Path')
        
        # 궤적
        if self.trajectory:
            traj_x = [p[0] for p in self.trajectory]
            traj_z = [p[1] for p in self.trajectory]
            ax.plot(traj_x, traj_z, 'g-', linewidth=2, label='Trajectory')
        
        # 전차
        ax.plot(self.state.x, self.state.z, 'go', markersize=10, label='Tank')
        
        # 전차 방향 화살표
        arrow_len = 5.0
        dx = math.sin(math.radians(self.state.yaw)) * arrow_len
        dz = math.cos(math.radians(self.state.yaw)) * arrow_len
        ax.arrow(self.state.x, self.state.z, dx, dz, 
                 head_width=2, head_length=1, fc='lime', ec='lime')
        
        # 타겟
        if self.target:
            ax.plot(self.target[0], self.target[1], 'b^', markersize=10, label='Target')
        
        # 목표
        ax.plot(self.goal[0], self.goal[1], 'r*', markersize=15, label='Goal')
        
        ax.set_xlim(0, self.config.map_size)
        ax.set_ylim(0, self.config.map_size)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f'Step: {self.step_count}, Dist: {self._distance_to_goal():.1f}m')
        
        plt.pause(0.01)
        
    def close(self):
        """환경 종료"""
        pass


# 환경 등록 (선택적)
def register_env():
    """Gymnasium에 환경 등록"""
    from gymnasium.envs.registration import register
    register(
        id='TankNav-v0',
        entry_point='rl_environment:TankNavEnv',
        max_episode_steps=1500,
    )
