"""
실제로 동작하는 RL 플래너

문제점:
- 기존 코드는 global_obstacles를 사용하지만 제대로 채워지지 않음
- LiDAR 스캔 생성이 복잡하고 버그 가능성 높음

해결책:
- StateManager의 obstacle_rects를 직접 사용
- 간단하고 확실한 장애물 회피 로직
- 디버그 로그 추가
"""
import numpy as np
import math


class WorkingRLPlanner:
    """
    실제로 동작하는 RL 스타일 플래너

    핵심:
    - obstacle_rects를 직접 사용 (global_obstacles 무시)
    - 간단한 potential field 방식
    - 목표 지향 + 장애물 반발력
    """

    def __init__(self, config, state_manager):
        self.config = config
        self.state = state_manager

        # 🎯 극단적 장애물 회피 (장애물 절대 뚫고 못가게)
        self.w_goal = 1.5          # 목표 끌어당김 (극도로 약화)
        self.w_obstacle = 100.0    # 장애물 밀어냄 (극단적 증가!)
        self.obstacle_range = 30.0 # 장애물 영향 범위 (매우 넓게)

        self.call_count = 0

    def get_action(self, curr_x, curr_z, curr_yaw, goal_x, goal_z):
        """
        Potential Field 방식으로 행동 결정

        Args:
            curr_x, curr_z: 현재 위치
            curr_yaw: 현재 방향 (도)
            goal_x, goal_z: 목표 위치

        Returns:
            dict: {"steering": float, "speed": float}
        """
        self.call_count += 1

        # 🔍 디버그: 첫 호출 시 상태 확인
        if self.call_count == 1:
            print(f"\n🔍 [RL DEBUG] 첫 호출 - 데이터 상태 확인")
            print(f"   StateManager.obstacle_rects 존재: {hasattr(self.state, 'obstacle_rects')}")
            if hasattr(self.state, 'obstacle_rects'):
                print(f"   obstacle_rects 개수: {len(self.state.obstacle_rects) if self.state.obstacle_rects else 0}")
                if self.state.obstacle_rects:
                    print(f"   첫 장애물 예시: {self.state.obstacle_rects[0]}")

        # 1. 목표 방향 계산
        dx_goal = goal_x - curr_x
        dz_goal = goal_z - curr_z
        dist_to_goal = math.hypot(dx_goal, dz_goal)

        if dist_to_goal < 0.1:
            return {"steering": 0.0, "speed": 0.0}

        # 목표 방향 각도
        goal_angle = math.degrees(math.atan2(dx_goal, dz_goal))

        # 현재 yaw와의 차이
        angle_diff = goal_angle - curr_yaw
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360

        # 2. 목표로의 힘 (Attractive Force)
        goal_force_x = dx_goal / dist_to_goal * self.w_goal
        goal_force_z = dz_goal / dist_to_goal * self.w_goal

        # 3. 장애물로부터의 반발력 (Repulsive Force)
        repulsive_force_x = 0.0
        repulsive_force_z = 0.0
        nearest_obstacle_dist = float('inf')
        obstacle_count = 0

        # 🔍 디버그: 장애물 데이터 상태
        has_obstacles = False
        if hasattr(self.state, 'obstacle_rects') and self.state.obstacle_rects:
            has_obstacles = True

        # obstacle_rects 직접 사용
        if has_obstacles:
            for obs in self.state.obstacle_rects:
                # 장애물 중심점
                obs_cx = (obs["x_min"] + obs["x_max"]) / 2
                obs_cz = (obs["z_min"] + obs["z_max"]) / 2

                # 장애물까지의 거리
                dx_obs = curr_x - obs_cx
                dz_obs = curr_z - obs_cz
                dist_to_obs = math.hypot(dx_obs, dz_obs)

                # 최근접 장애물 거리 기록
                if dist_to_obs < nearest_obstacle_dist:
                    nearest_obstacle_dist = dist_to_obs

                # 영향 범위 내 장애물
                if dist_to_obs < self.obstacle_range and dist_to_obs > 0.1:
                    obstacle_count += 1

                    # 거리의 제곱에 반비례 (가까울수록 강한 반발)
                    strength = self.w_obstacle / (dist_to_obs ** 2)

                    # 장애물로부터 멀어지는 방향
                    repulsive_force_x += (dx_obs / dist_to_obs) * strength
                    repulsive_force_z += (dz_obs / dist_to_obs) * strength
        else:
            # 🔍 디버그: 장애물 데이터 없음 경고
            if self.call_count % 20 == 1:
                print(f"\n⚠️ [RL WARNING] 장애물 데이터 없음!")
                print(f"   obstacle_rects가 비어있거나 존재하지 않습니다.")
                print(f"   /update_obstacle API가 호출되고 있는지 확인하세요.")

        # 4. 합력 계산
        total_force_x = goal_force_x + repulsive_force_x
        total_force_z = goal_force_z + repulsive_force_z

        # 5. 합력 방향으로 조향
        if abs(total_force_x) < 0.01 and abs(total_force_z) < 0.01:
            # 힘이 거의 없음 → 목표 방향으로
            desired_angle = goal_angle
        else:
            # 합력 방향
            desired_angle = math.degrees(math.atan2(total_force_x, total_force_z))

        # 목표 방향과의 차이
        steer_angle_diff = desired_angle - curr_yaw
        while steer_angle_diff > 180:
            steer_angle_diff -= 360
        while steer_angle_diff < -180:
            steer_angle_diff += 360

        # 조향 명령 (-1 ~ 1)
        steering = np.clip(steer_angle_diff / 90.0, -1.0, 1.0)

        # 6. 속도 결정 (장애물 거리 기반 극도 보수적)
        base_speed = 0.8

        # 장애물 가까우면 극도로 감속
        if nearest_obstacle_dist < 3.0:
            speed = 0.1  # 거의 정지 (0.3 → 0.1)
        elif nearest_obstacle_dist < 5.0:
            speed = 0.2  # 매우 느리게 (0.3 유지 → 0.2)
        elif nearest_obstacle_dist < 10.0:
            speed = 0.4  # 느리게 (0.5 → 0.4)
        elif nearest_obstacle_dist < 15.0:
            speed = 0.6  # 적당히 (0.7 → 0.6)
        else:
            speed = base_speed

        # 목표 가까우면 감속
        if dist_to_goal < 10.0:
            speed = min(speed, 0.5)

        # 급회전 시 감속
        if abs(steering) > 0.7:
            speed *= 0.6
        elif abs(steering) > 0.5:
            speed *= 0.8

        # 7. 디버그 로그 (매 5번마다 - 더 자주)
        if self.call_count % 5 == 1:
            print(f"\n🤖 [Potential Field] 호출 #{self.call_count}")
            print(f"   위치: ({curr_x:.1f}, {curr_z:.1f}) → 목표: ({goal_x:.1f}, {goal_z:.1f})")
            print(f"   장애물: {obstacle_count}개, 최근접: {nearest_obstacle_dist:.1f}m")
            print(f"   목표힘: ({goal_force_x:.2f}, {goal_force_z:.2f})")
            print(f"   반발력: ({repulsive_force_x:.2f}, {repulsive_force_z:.2f})")
            print(f"   합력:   ({total_force_x:.2f}, {total_force_z:.2f})")
            print(f"   → 조향: {steering:.2f}, 속도: {speed:.2f}")

            # ⚠️ 장애물 매우 가까우면 경고
            if nearest_obstacle_dist < 5.0:
                print(f"   ⚠️⚠️ 장애물 경고! {nearest_obstacle_dist:.1f}m 이내")

        return {
            "steering": float(steering),
            "speed": float(speed)
        }


class WorkingHybridRLPlanner:
    """
    동작하는 하이브리드 플래너

    우선순위:
    1. ONNX 모델 (있으면)
    2. WorkingRLPlanner (Potential Field)
    """

    def __init__(self, model_path, config, state_manager):
        self.config = config
        self.state = state_manager
        self.onnx_planner = None
        self.working_planner = WorkingRLPlanner(config, state_manager)
        self.mode = "working"

        # ONNX 모델 로드 시도
        try:
            from planners.rl_planner import RLPlanner
            self.onnx_planner = RLPlanner(model_path, config)

            if self.onnx_planner.is_available():
                print("✅ ONNX RL 모델 사용")
                self.mode = "onnx"
            else:
                print("⚠️ ONNX 모델 없음")
                print("   Potential Field 기반 RL 플래너 사용")
                self.mode = "working"
        except Exception as e:
            print(f"⚠️ ONNX 로드 실패: {e}")
            print("   Potential Field 기반 RL 플래너 사용")
            self.mode = "working"

    def is_available(self):
        """항상 사용 가능"""
        return True

    def get_action_from_lidar(self, lidar_data, curr_x, curr_z, curr_yaw, goal_x, goal_z, curr_velocity=0.0):
        """
        LiDAR 데이터를 사용하는 인터페이스 (ONNX용)

        하지만 Working 모드에서는 lidar_data 무시하고 obstacle_rects 직접 사용
        """
        # ONNX 모드
        if self.mode == "onnx" and self.onnx_planner is not None:
            action = self.onnx_planner.get_action(
                lidar_data, curr_x, curr_z, curr_yaw, goal_x, goal_z, curr_velocity
            )
            if action is not None:
                return action

        # Working 모드 (LiDAR 데이터 무시, obstacle_rects 직접 사용)
        return self.working_planner.get_action(curr_x, curr_z, curr_yaw, goal_x, goal_z)

    def get_action(self, curr_x, curr_z, curr_yaw, goal_x, goal_z):
        """간단한 인터페이스 (obstacle_rects 직접 사용)"""
        return self.working_planner.get_action(curr_x, curr_z, curr_yaw, goal_x, goal_z)

    def convert_to_command(self, action):
        """행동 → 명령 변환"""
        if action is None:
            return None

        steering = action["steering"]
        speed = action["speed"]

        # 조향
        if abs(steering) < 0.05:
            steer_dir = ""
            steer_weight = 0.0
        else:
            steer_dir = "D" if steering > 0 else "A"
            steer_weight = abs(steering)

        # 속도
        if speed > 0.05:
            ws_cmd = "W"
            ws_weight = speed
        else:
            ws_cmd = "STOP"
            ws_weight = 0.0

        return {
            "moveWS": {"command": ws_cmd, "weight": round(ws_weight, 2)},
            "moveAD": {"command": steer_dir, "weight": round(steer_weight, 2)},
            "fire": False
        }
