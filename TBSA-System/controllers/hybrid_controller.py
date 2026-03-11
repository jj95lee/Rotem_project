"""
하이브리드 제어기 - SEQ별 분리 제어

SEQ 1, 3: A* + PID (전역 경로 추종)
    - A*로 전역 경로 생성
    - PID로 경로 추종
    - SEQ별 다른 obstacle_margin 사용

SEQ 4: 순수 DWA (실시간 장애물 회피)
    - A* 경로 없음
    - 목적지 방향 lookahead 타겟 설정
    - /update_obstacle의 장애물 사각형 기반 가상 라이다
    - DWA로 실시간 장애물 회피
"""
import math
import time
import numpy as np
import os
from controllers.pid_controller import PIDController
from planners.astar_planner import ObstacleRect
from planners.dwa_planner import DWAConfig, calc_dynamic_window, predict_trajectory, calc_to_goal_cost
# from planners.sb3_ppo_planner import HybridSB3PPOPlanner  # 사용 안함 - ppo_planner 사용
from utils.visualization import save_path_image


class HybridController:
    """
    SEQ별 분리 제어기
    
    - SEQ 1, 3: A* + PID
    - SEQ 4: 순수 DWA (가상 라이다 기반)
    """
    
    def __init__(self, config, planner, state_manager):
        self.config = config
        self.planner = planner  # A* 플래너 (SEQ 1, 3용)
        self.state = state_manager

        # DWA 설정 (SEQ 4용)
        self.dwa_config = DWAConfig(config)

        # RL Planner 설정 (SEQ 4 RL 모드용) - PPO Planner
        # 우선순위: withobs_model > ppo.zip > Potential Field
        from planners.ppo_planner import UnifiedHybridPPOPlanner

        # PPO Hybrid Planner (자동으로 최적 모델 선택)
        self.rl_planner = UnifiedHybridPPOPlanner(config, state_manager)

        # RL 모드 활성화 여부
        self.use_rl_for_seq4 = True
        if self.rl_planner.is_available():
            print("✅ SEQ4에서 Unified PPO 강화학습 자율주행 활성화")
            self.state.set_log("✅ SEQ4에서 Unified PPO 강화학습 자율주행 활성화")
        else:
            print("⚠️ Unified PPO 로드 실패, Potential Field 모드로 작동")
            self.state.set_log("⚠️ Unified PPO 로드 실패, Potential Field 모드로 작동")

        # PID 제어기 (SEQ 1, 3용)
        self.steering_pid = PIDController(
            kp=config.PID.KP,
            ki=config.PID.KI,
            kd=config.PID.KD
        )

        # 상태 변수
        self.last_velocity = 0.0
        self.last_yaw_rate = 0.0
        self.stuck_counter = 0
        self.last_position = None

        # Stuck 복구 상태
        self.recovery_mode = False
        self.recovery_start_time = 0
        self.recovery_direction = 1

        # Stop-Steer-Go 상태
        self.ssg_mode = False
        self.ssg_phase = None          # "stop", "steer", "go"
        self.ssg_start_time = 0
        self.ssg_best_direction = None # 최적 조향 방향 ("A" or "D")
        self.ssg_no_valid_count = 0    # DWA 유효경로 없음 연속 카운트

        # 디버그 카운터
        self._compute_count = 0
        
    def reset(self):
        """제어기 상태 초기화"""
        self.steering_pid.reset()
        self.last_velocity = 0.0
        self.last_yaw_rate = 0.0
        self.stuck_counter = 0
        self.last_position = None
        self.recovery_mode = False
        self.ssg_mode = False
        self.ssg_phase = None
        self.ssg_no_valid_count = 0
        
    def compute_action(self, curr_x, curr_z, curr_yaw):
        """메인 제어 루프"""
        
        # 1. 위치 업데이트
        curr_x, curr_z = self.config.clamp_world_xz(curr_x, curr_z)
        self.state.update_robot_pose(curr_x, curr_z)
        
        # SEQ 1, 3에서 obstacle_margin 업데이트
        if self.state.seq in [1, 3]:
            self._update_obstacle_margin()
        
        # 디버깅
        self._compute_count += 1
        if self._compute_count % 50 == 1:
            print(f"🚗 [compute_action] #{self._compute_count} SEQ={self.state.seq} "
                  f"pos=({curr_x:.1f},{curr_z:.1f}) dest={self.state.destination}")
            self.state.set_log(f"🚗 [compute_action] #{self._compute_count} SEQ={self.state.seq} "
                  f"pos=({curr_x:.1f},{curr_z:.1f}) dest={self.state.destination}")
        
        # 2. SEQ 2 사격 처리
        if self.state.seq == 2:
            cmd = self._stop_command()
            cmd["fire"] = True
            self.state.seq = 3
            self.state.status_message = "🔥 사격 완료! 경유지로 출발"
            return cmd

        # 3. 목적지 없으면 정지
        if self.state.destination is None:
            return self._stop_command()
        
        # 4. 도착 확인 및 SEQ 전환
        dist_to_goal = math.hypot(
            self.state.destination[0] - curr_x, 
            self.state.destination[1] - curr_z
        )
        
        if dist_to_goal < self.config.ARRIVAL_THRESHOLD:
            return self._handle_arrival(curr_x, curr_z)

        # 5. Stop-Steer-Go 진행중이면 우선 처리
        if self.ssg_mode:
            return self._stop_steer_go_action(curr_x, curr_z, curr_yaw)

        # 6. Stuck 감지
        self._detect_stuck(curr_x, curr_z)

        # 7. Stuck 복구 모드 처리
        if self.stuck_counter >= self.config.Stuck.STUCK_COUNT_LIMIT:
            return self._recovery_action(curr_x, curr_z, curr_yaw)
        
        # 8. SEQ에 따른 제어 분기
        if self.state.seq == 4:
            # SEQ 4: RL 강화학습 (A* + PPO 하이브리드)
            rl_result = self._seq4_rl_control(curr_x, curr_z, curr_yaw)
            if rl_result is not None:
                return rl_result
            # RL 실패 시 DWA 폴백
            print(f"⚠️ [SEQ4] RL 제어 실패, DWA 폴백")
            return self._seq4_pure_dwa(curr_x, curr_z, curr_yaw)
        else:
            # SEQ 1, 3: A* + PID
            return self._seq13_astar_pid(curr_x, curr_z, curr_yaw)
    
    def _update_obstacle_margin(self):
        """현재 SEQ에 맞는 obstacle_margin 적용"""
        if self.state.seq == 4:
            new_margin = self.config.ASTAR.OBSTACLE_MARGIN_SEQ4
        else:
            new_margin = self.config.ASTAR.get_obstacle_margin(self.state.seq)
        
        if new_margin != self.planner.obstacle_margin:
            self.planner.set_obstacle_margin(new_margin)
            self.state.set_log(f"🔧 SEQ {self.state.seq}: obstacle_margin = {new_margin}")
            print(f"🔧 SEQ {self.state.seq}: obstacle_margin = {new_margin}")
        
    def _handle_arrival(self, curr_x, curr_z):
        """도착 처리 및 SEQ 전환"""
        dist_to_goal = math.hypot(
            self.state.destination[0] - curr_x, 
            self.state.destination[1] - curr_z
        )
        self.state.set_log(f"✅ 도착! 거리={dist_to_goal:.2f}m (임계값={self.config.ARRIVAL_THRESHOLD}m)")
        print(f"✅ 도착! 거리={dist_to_goal:.2f}m (임계값={self.config.ARRIVAL_THRESHOLD}m)")
        
        if self.state.seq == 1:
            self.state.seq = 2
            self.state.status_message = "🎯 정찰지 도착! 사격 시스템 가동 중..."
            self.state.clear_path()
            self.state.destination = None
            print("🔄 SEQ 1→2 전환")
            return self._stop_command()
            
        elif self.state.seq == 3:
            self.state.seq = 4
            self.state.status_message = "🚀 경유지 도착! 자율주행 모드 활성화"
            self.state.clear_path()
            self.state.destination = None
            print("🔄 SEQ 3→4 전환, 자율주행 시작")
            return self._stop_command()
            
        elif self.state.seq == 4:
            self.state.status_message = "🏁 모든 임무 완료!"
            self.state.clear_path()
            self.state.destination = None
            print("🏁 SEQ 4 완료!")
            return self._stop_command()
        
        else:
            self.state.clear_path()
            self.state.destination = None
            return self._stop_command()
    
    # ==================== SEQ 1, 3: A* + PID ====================
    
    def _seq13_astar_pid(self, curr_x, curr_z, curr_yaw):
        """SEQ 1, 3: A* 경로 + PID 제어"""
        
        # 경로가 없으면 생성
        if not self.state.global_path:
            self._generate_astar_path(curr_x, curr_z)
            if not self.state.global_path:
                self.state.set_log("⚠️ A* 경로 생성 실패")
                print("⚠️ A* 경로 생성 실패")
                return self._stop_command()
        
        # 경로 업데이트 (지나간 노드 제거)
        self._update_path(curr_x, curr_z)
        
        # 타겟 포인트 선택
        target_point, _ = self._select_target_point(curr_x, curr_z)
        if not target_point:
            return self._stop_command()
        
        # PID 제어
        return self._pid_control(curr_x, curr_z, curr_yaw, target_point)
    
    def _generate_astar_path(self, curr_x, curr_z):
        """A* 경로 생성"""
        if self.state.destination is None:
            return
        
        dest_x, dest_z = self.state.destination

        mask_zones = []

        if self.state.seq == 1:
            forbidden_zone = ObstacleRect.from_min_max(158.0, 190.0, 115.0, 156.0)
            mask_zones.append(forbidden_zone)
            self.state.set_log(f"🚫 마스킹 영역(No-Go Zone) {len(mask_zones)}개 설정 완료")
            self.planner.update_grid_range(65.0, 200.0, 0.0, 220.0)
            self.state.set_log(f"📏 A* 범위 변경 완료: X(65.0~200.0), Z(0.0~220.0)")

        elif self.state.seq == 3:
            self.planner.update_grid_range(0.0, 200.0, 150.0, 300.0)
            self.state.set_log(f"📏 A* 범위 변경 완료: X(0.0~200.0), Z(150.0~300.0)")
        
        self.planner.set_mask_zones(mask_zones)
        
        path = self.planner.find_path(
            start=(curr_x, curr_z),
            goal=(dest_x, dest_z),
            use_obstacles=True
        )
        
        if path:
            self.state.global_path = path
            self.state.global_path_version += 1
            self.state.set_log(f"✅ A* 경로 생성: {len(path)}개 노드 (SEQ {self.state.seq})")
            print(f"✅ A* 경로 생성: {len(path)}개 노드 (SEQ {self.state.seq})")
            # 경로 이미지 저장
            try:
                obs_count = len(self.planner._obstacles) if self.planner._obstacles else 0
                mode_label = ""
                if self.state.seq == 1:
                    mode_label = "정찰지 (RP1) 이동 (A* + PID)"
                elif self.state.seq == 3:
                    mode_label = "경유지 (RP2) 이동 (A* + PID)"
                
                save_path_image(
                    planner=self.planner,
                    path=path,
                    current_pos=(curr_x, curr_z),
                    current_yaw=self.state.robot_yaw_deg,
                    filename=f"SEQ {self.state.seq}_Global_Path.png",
                    title=f"{mode_label}",
                    state_manager=self.state
                )
                self.state.set_log(f"💾 경로 이미지 저장 완료!")
                print(f"💾 경로 이미지 저장 완료! ({len(path)}개 노드, 장애물 {obs_count}개)")
            except Exception as e:
                self.state.set_log(f"⚠️ 디버그 이미지 저장 실패: {e}")
                print(f"⚠️ 디버그 이미지 저장 실패: {e}")
        else:
            self.state.set_log(f"❌ A* 경로 생성 실패 (SEQ {self.state.seq})")
            print(f"❌ A* 경로 생성 실패 (SEQ {self.state.seq})")
    
    def _update_path(self, curr_x, curr_z):
        """경로 업데이트: 지나간 노드 제거"""
        if not self.state.global_path:
            return
        
        # 현재 위치에서 가장 가까운 경로 노드 찾기
        min_dist = float('inf')
        closest_idx = 0
        
        for i, point in enumerate(self.state.global_path):
            dist = math.hypot(point[0] - curr_x, point[1] - curr_z)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 지나간 노드 제거
        if closest_idx > 0:
            self.state.global_path = self.state.global_path[closest_idx:]
    
    def _select_target_point(self, curr_x, curr_z):
        """Lookahead 거리에 맞는 타겟 포인트 선택"""
        if not self.state.global_path:
            return None, 0
        
        lookahead = self.config.LOOKAHEAD_DIST
        cumulative_dist = 0.0
        prev_point = (curr_x, curr_z)
        
        for i, point in enumerate(self.state.global_path):
            segment_dist = math.hypot(
                point[0] - prev_point[0],
                point[1] - prev_point[1]
            )
            cumulative_dist += segment_dist
            
            if cumulative_dist >= lookahead:
                return point, i
            
            prev_point = point
        
        # 경로 끝에 도달하면 마지막 포인트 반환
        return self.state.global_path[-1], len(self.state.global_path) - 1
    
    def _pid_control(self, curr_x, curr_z, curr_yaw, target_node):
        """PID 조향 제어"""
        # 타겟 방향 계산
        dx = target_node[0] - curr_x
        dz = target_node[1] - curr_z
        target_angle_deg = math.degrees(math.atan2(dx, dz))
        
        # 각도 오차 계산
        error = target_angle_deg - curr_yaw
        while error > 180: 
            error -= 360
        while error < -180: 
            error += 360
        
        # PID 계산
        pid_output = self.steering_pid.compute(error)
        
        # 조향 가중치
        steer_weight = min(abs(pid_output), 1.0)
        steer_dir = "D" if pid_output > 0 else "A"
        if pid_output == 0: 
            steer_dir = ""
        
        # 속도 계산 (조향에 따른 감속)
        max_w = self.config.PID.MAX_SPEED_WEIGHT
        min_w = self.config.PID.MIN_SPEED_WEIGHT
        gain = self.config.PID.SPEED_REDUCT_GAIN
        error_th = self.config.PID.ERROR_THRESHOLD
        error_range = self.config.PID.ERROR_RANGE

        speed_weight = max(min_w, max_w - steer_weight * gain)
        
        if abs(error) > error_th:
            reduction_factor = max(0.0, 1.0 - (abs(error) - error_th) / error_range)
            speed_weight *= reduction_factor
        speed_weight = max(speed_weight, min_w)
        
        if speed_weight <= 0.05:
            cmd_ws = "STOP"
            speed_weight = 1.0
        else:
            cmd_ws = "W"
        
        return {
            "moveWS": {"command": cmd_ws, "weight": round(speed_weight, 2)},
            "moveAD": {"command": steer_dir, "weight": round(steer_weight * self.config.PID.STEER_SENSITIVITY, 2)},
            "fire": False
        }
    
    # ==================== SEQ 4: 순수 DWA ====================
    
    def _seq4_pure_dwa(self, curr_x, curr_z, curr_yaw):
        """SEQ 4: RL 또는 DWA 제어 (A* 없음)"""

        if self.state.destination is None:
            return self._stop_command()

        # 🆕 RL 모드 우선 시도
        if self.use_rl_for_seq4:
            rl_command = self._seq4_rl_control(curr_x, curr_z, curr_yaw)
            if rl_command is not None:
                return rl_command
            # RL 실패 시 DWA로 폴백
            print("⚠️ RL 제어 실패, DWA로 폴백")

        # DWA 제어 (폴백 또는 기본 모드)
        target_point = self._calc_lookahead_target(curr_x, curr_z)

        if self._compute_count % 20 == 1:
            print(f"🎯 [SEQ4 DWA] pos=({curr_x:.1f},{curr_z:.1f}) → "
                  f"target=({target_point[0]:.1f},{target_point[1]:.1f}) → "
                  f"dest=({self.state.destination[0]:.1f},{self.state.destination[1]:.1f}), "
                  f"obstacles={len(self.state.obstacle_rects)}개")

        # DWA 제어 (가상 라이다 비용 사용)
        return self._dwa_control_virtual_lidar(curr_x, curr_z, curr_yaw, target_point)
    
    def _calc_lookahead_target(self, curr_x, curr_z):
        """목적지 방향으로 lookahead 거리만큼의 타겟 계산"""
        dest_x, dest_z = self.state.destination
        
        # 목적지까지의 거리와 방향
        dx = dest_x - curr_x
        dz = dest_z - curr_z
        dist_to_dest = math.hypot(dx, dz)
        
        # SEQ 4 전용 lookahead 사용
        lookahead = self.config.SEQ4.LOOKAHEAD_DIST
        
        # 목적지가 lookahead보다 가까우면 목적지 그대로 사용
        if dist_to_dest <= lookahead:
            return (dest_x, dest_z)
        
        # 목적지 방향으로 lookahead 거리만큼의 타겟
        ratio = lookahead / dist_to_dest
        target_x = curr_x + dx * ratio
        target_z = curr_z + dz * ratio
        
        return (target_x, target_z)
    
    def _dwa_control_virtual_lidar(self, curr_x, curr_z, curr_yaw, target_point):
        """DWA 제어 - 가상 라이다 기반 장애물 비용"""
        
        curr_yaw_rad = math.radians(curr_yaw)
        x = np.array([curr_x, curr_z, curr_yaw_rad, self.last_velocity, self.last_yaw_rate])
        
        # Dynamic Window 계산
        dw = calc_dynamic_window(x, self.dwa_config)
        
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_trajectory = np.array([x])
        valid_trajectories = 0
        total_trajectories = 0
        
        obstacle_margin = self.config.ASTAR.OBSTACLE_MARGIN_SEQ4
        
        # 모든 (v, omega) 조합 탐색
        for v in np.arange(dw[0], dw[1], self.dwa_config.v_resolution):
            for omega in np.arange(dw[2], dw[3], self.dwa_config.yaw_rate_resolution):
                total_trajectories += 1
                trajectory = predict_trajectory(x, v, omega, self.dwa_config)
                
                # 1. 월드 경계 체크
                out_of_bounds = False
                for state in trajectory:
                    tx, tz = state[0], state[1]
                    if (tx < self.config.WORLD_MIN_XZ or tx > self.config.WORLD_MAX_XZ or
                        tz < self.config.WORLD_MIN_XZ or tz > self.config.WORLD_MAX_XZ):
                        out_of_bounds = True
                        break
                
                if out_of_bounds:
                    continue
                
                # 2. 가상 라이다 기반 장애물 비용 계산
                ob_cost = self._calc_virtual_lidar_cost(trajectory, obstacle_margin)
                if ob_cost == float("inf"):
                    continue  # 충돌 경로는 제외
                
                # 3. 목표 비용
                to_goal_cost = self.dwa_config.to_goal_cost_gain * calc_to_goal_cost(
                    trajectory, [target_point[0], target_point[1]]
                )
                
                # 4. 속도 비용 (빠를수록 좋음)
                speed_cost = self.dwa_config.speed_cost_gain * (
                    self.dwa_config.max_speed - trajectory[-1, 3]
                )
                
                # 5. 조향 패널티
                steering_penalty = abs(omega) * self.dwa_config.steering_penalty
                
                # 총 비용
                final_cost = to_goal_cost + speed_cost + ob_cost + steering_penalty
                
                valid_trajectories += 1
                
                if final_cost < min_cost:
                    min_cost = final_cost
                    best_u = [v, omega]
                    best_trajectory = trajectory
        
        # DWA 결과 로깅
        self.state.valid_traj_count = valid_trajectories
        if self._compute_count % 10 == 1:
            print(f"🎯 DWA: 총={total_trajectories}, 유효={valid_trajectories}, "
                  f"비용={min_cost:.2f}, v={best_u[0]:.2f}, ω={best_u[1]:.3f}")
        
        # 유효 경로 없음 → Stop-Steer-Go 또는 후진
        if valid_trajectories == 0:
            ssg_cfg = self.config.StopSteerGo
            if ssg_cfg.ENABLE and not self.ssg_mode:
                self.ssg_no_valid_count += 1
                if self.ssg_no_valid_count >= ssg_cfg.TRIGGER_STUCK_COUNT:
                    # SSG 진입
                    self.ssg_mode = True
                    self.ssg_phase = "stop"
                    self.ssg_start_time = time.time()
                    print(f"🛑 SSG 진입! (연속 {self.ssg_no_valid_count}회 유효경로 없음)")
                    self.state.set_log("🛑 장애물 조우 → 정지-조향-출발 시작")
                    return self._stop_steer_go_action(curr_x, curr_z, curr_yaw)
            # SSG 미진입 시 기존 후진
            print("⚠️ DWA 유효 경로 없음 → 후진 시도")
            return {
                "moveWS": {"command": "S", "weight": 0.3},
                "moveAD": {"command": "", "weight": 0.0},
                "fire": False
            }
        
        # 유효 경로 있음 → SSG 카운터 리셋
        self.ssg_no_valid_count = 0

        # DWA 궤적 저장 (시각화용)
        self.state.last_dwa_traj = best_trajectory
        self.state.last_dwa_target = (float(target_point[0]), float(target_point[1]))
        self.state.local_traj_version += 1
        
        # 속도 업데이트
        desired_v = float(best_u[0])
        desired_omega = float(best_u[1])
        
        # Stuck 방지
        if (abs(desired_v) < self.dwa_config.robot_stuck_flag_cons and 
            abs(x[3]) < self.dwa_config.robot_stuck_flag_cons):
            desired_v = -float(self.config.Recovery.REVERSE_SPEED)
            desired_omega = 0.0
        
        self.last_velocity = desired_v
        self.last_yaw_rate = desired_omega
        
        # 명령어 변환
        steer_command = desired_omega / self.dwa_config.max_yaw_rate
        steer_command = max(min(steer_command, 1.0), -1.0)
        steer_weight = abs(steer_command)
        
        if abs(steer_command) < 0.05:
            steer_dir = ""
            steer_weight = 0.0
        else:
            steer_dir = "D" if steer_command > 0 else "A"
        
        ws_cmd = "W" if desired_v > 0.05 else ("S" if desired_v < -0.05 else "STOP")
        ws_weight = min(max(abs(desired_v) / self.dwa_config.max_speed, 0.0), 1.0)
        
        return {
            "moveWS": {"command": ws_cmd, "weight": round(ws_weight, 2)},
            "moveAD": {"command": steer_dir, "weight": round(steer_weight, 2)},
            "fire": False
        }
    
    def _calc_virtual_lidar_cost(self, trajectory, obstacle_margin):
        """가상 라이다 기반 장애물 비용 계산
        
        - 장애물 사각형(obstacle_rects)과의 거리를 기반으로 비용 계산
        - 충돌(collision_distance 이내) → inf
        - 위험(danger_distance 이내) → 높은 비용
        - 안전(safe_distance 이상) → 낮은 비용
        """
        collision_dist = self.config.DWA.COLLISION_DISTANCE
        danger_dist = self.config.DWA.DANGER_DISTANCE
        safe_dist = self.config.DWA.SAFE_DISTANCE
        
        total_cost = 0.0
        min_dist_overall = float('inf')
        
        # 궤적의 각 포인트에서 장애물 거리 체크
        for i, state in enumerate(trajectory):
            if i < 3:  # 처음 몇 포인트는 스킵 (현재 위치 근처)
                continue
            
            px, pz = state[0], state[1]
            
            # 가장 가까운 장애물까지의 거리
            dist = self.state.get_obstacle_distance(px, pz, obstacle_margin)
            
            if dist < min_dist_overall:
                min_dist_overall = dist
            
            # 충돌 거리 이내 → 무효 경로
            if dist <= collision_dist:
                return float("inf")
        
        # 거리 기반 비용 계산 (🔧 v4: 비용 범위 조정)
        if min_dist_overall <= danger_dist:
            # 위험 구간: 높은 비용 (50.0 → 20.0)
            normalized = (min_dist_overall - collision_dist) / max(danger_dist - collision_dist, 0.1)
            total_cost = 20.0 * (1.0 - normalized)
        elif min_dist_overall <= safe_dist:
            # 주의 구간: 중간 비용 (10.0 → 5.0)
            normalized = (min_dist_overall - danger_dist) / max(safe_dist - danger_dist, 0.1)
            total_cost = 5.0 * (1.0 - normalized)
        else:
            # 안전 구간: 낮은 비용
            total_cost = 0.0
        
        return total_cost * self.dwa_config.obstacle_cost_gain

    def _seq4_rl_control(self, curr_x, curr_z, curr_yaw):
        """🆕 SEQ 4: A* + PPO 하이브리드 제어"""

        # 디버깅: 초기 상태 확인
        if self._compute_count == 1:
            self.state.set_log(f"🔍 [강화학습 사용여부 체크]={self.use_rl_for_seq4}, destination={self.state.destination}")
            print(f"🔍 [강화학습 사용여부 체크]={self.use_rl_for_seq4}, destination={self.state.destination}")

        if not self.use_rl_for_seq4 or self.state.destination is None:
            if self._compute_count % 50 == 1:
                print(f"⚠️ [SEQ4] RL 비활성화 또는 목적지 없음 (use_rl={self.use_rl_for_seq4}, dest={self.state.destination})")
            return None

        # 하이브리드 모드가 비활성화되어 있으면 PPO만 사용
        if not self.config.SEQ4.HYBRID_MODE_ENABLED:
            return self._seq4_ppo_only(curr_x, curr_z, curr_yaw)

        # 하이브리드 모드: A* + PPO 혼합
        dest_x, dest_z = self.state.destination

        # ========================================
        # 1. A* 경로 생성 및 명령 획득
        # ========================================
        astar_command = self._get_astar_command_for_seq4(curr_x, curr_z, curr_yaw)

        if self._compute_count % 20 == 1:
            print(f"🔍 [SEQ4 Debug] A* command: {astar_command is not None}")

        # ========================================
        # 2. PPO 명령 획득
        # ========================================
        ppo_command = self._get_ppo_command(curr_x, curr_z, curr_yaw, dest_x, dest_z)

        if self._compute_count % 20 == 1:
            print(f"🔍 [SEQ4 Debug] PPO command: {ppo_command is not None}")

        # ========================================
        # 3. 명령 혼합
        # ========================================

        # A* 실패 시 PPO만 사용
        if astar_command is None:
            if ppo_command is not None:
                if self._compute_count % 20 == 1:
                    print(f"⚠️ [SEQ4 Hybrid] A* 실패, PPO 100% 사용")
                return ppo_command
            else:
                return None

        # PPO 실패 시 A*만 사용
        if ppo_command is None:
            if self.config.SEQ4.PPO_FALLBACK_TO_ASTAR:
                if self._compute_count % 20 == 1:
                    print(f"⚠️ [SEQ4 Hybrid] PPO 실패, A* 100% 사용")
                return astar_command
            else:
                return None

        # 둘 다 성공: 가중 평균으로 혼합
        if self._compute_count % 20 == 1:
            print(f"✅ [SEQ4 Hybrid] A*와 PPO 둘 다 성공, 명령 혼합 중 (A*: {self.config.SEQ4.ASTAR_WEIGHT:.0%}, PPO: {self.config.SEQ4.PPO_WEIGHT:.0%})")

        blended_command = self._blend_commands(
            astar_command,
            ppo_command,
            astar_weight=self.config.SEQ4.ASTAR_WEIGHT,
            ppo_weight=self.config.SEQ4.PPO_WEIGHT
        )

        # 주기적 로깅
        if self._compute_count % 20 == 1:
            print(f"🎯 [SEQ4 Hybrid] A*({self.config.SEQ4.ASTAR_WEIGHT:.2f}) + "
                  f"PPO({self.config.SEQ4.PPO_WEIGHT:.2f})")
            print(f"   A*: WS={astar_command['moveWS']['command']}({astar_command['moveWS']['weight']:.2f}), "
                  f"AD={astar_command['moveAD']['command']}({astar_command['moveAD']['weight']:.2f})")
            print(f"   PPO: WS={ppo_command['moveWS']['command']}({ppo_command['moveWS']['weight']:.2f}), "
                  f"AD={ppo_command['moveAD']['command']}({ppo_command['moveAD']['weight']:.2f})")
            print(f"   → WS={blended_command['moveWS']['command']}({blended_command['moveWS']['weight']:.2f}), "
                  f"AD={blended_command['moveAD']['command']}({blended_command['moveAD']['weight']:.2f})")

        if self._compute_count % 20 == 1:
            print(f"📤 [SEQ4 Hybrid] 최종 명령 반환: {blended_command is not None}")

        return blended_command

    def _seq4_ppo_only(self, curr_x, curr_z, curr_yaw):
        """SEQ 4: PPO만 사용 (하이브리드 비활성화 시)"""
        dest_x, dest_z = self.state.destination
        ppo_command = self._get_ppo_command(curr_x, curr_z, curr_yaw, dest_x, dest_z)
        return ppo_command

    def _get_astar_command_for_seq4(self, curr_x, curr_z, curr_yaw):
        """SEQ4용 A* 경로 생성 및 명령 획득"""

        # 경로가 없으면 생성
        if not self.state.global_path:
            self._generate_astar_path_for_seq4(curr_x, curr_z)
            if not self.state.global_path:
                return None

        # 경로 업데이트 (지나간 노드 제거)
        self._update_path(curr_x, curr_z)

        # 타겟 포인트 선택
        target_point, _ = self._select_target_point(curr_x, curr_z)
        if not target_point:
            return None

        # PID 제어로 명령 생성
        return self._pid_control(curr_x, curr_z, curr_yaw, target_point)

    def _generate_astar_path_for_seq4(self, curr_x, curr_z):
        """SEQ4용 A* 경로 생성"""
        if self.state.destination is None:
            return

        dest_x, dest_z = self.state.destination

        # SEQ4는 전체 맵 범위 사용
        self.planner.update_grid_range(0.0, 300.0, 0.0, 300.0)
        self.planner.set_mask_zones([])

        path = self.planner.find_path(
            start=(curr_x, curr_z),
            goal=(dest_x, dest_z),
            use_obstacles=True
        )

        if path:
            self.state.global_path = path
            self.state.global_path_version += 1
            if self._compute_count % 20 == 1:
                print(f"✅ [SEQ4] A* 경로 생성: {len(path)}개 노드")
        else:
            if self._compute_count % 20 == 1:
                print(f"❌ [SEQ4] A* 경로 생성 실패")

    def _get_ppo_command(self, curr_x, curr_z, curr_yaw, dest_x, dest_z):
        """PPO 명령 획득"""

        # 1. 가상 LiDAR 스캔 생성 (32개 광선)
        lidar_data = self._generate_virtual_lidar_scan(
            curr_x, curr_z, curr_yaw,
            num_rays=32,
            max_range=50.0
        )

        # 2. 현재 속도
        curr_velocity = self.last_velocity

        # 3. Unified PPO 플래너 호출
        action = self.rl_planner.get_action(
            lidar_data=lidar_data,
            curr_x=curr_x,
            curr_z=curr_z,
            curr_yaw=curr_yaw,
            goal_x=dest_x,
            goal_z=dest_z,
            curr_velocity=curr_velocity
        )

        if action is None:
            return None

        # 4. 명령 변환
        command = self.rl_planner.convert_to_command(action)
        return command

    def _blend_commands(self, astar_cmd, ppo_cmd, astar_weight, ppo_weight):
        """
        두 명령을 가중 평균으로 혼합

        Args:
            astar_cmd: A* 명령 {"moveWS": {...}, "moveAD": {...}, "fire": bool}
            ppo_cmd: PPO 명령 (같은 형식)
            astar_weight: A* 가중치 (0.0 ~ 1.0)
            ppo_weight: PPO 가중치 (0.0 ~ 1.0)

        Returns:
            혼합된 명령 (같은 형식)
        """

        # ========================================
        # 1. moveWS 혼합
        # ========================================

        # A*와 PPO의 속도값 추출 (STOP: 0, W: weight, S: -weight)
        astar_speed_val = 0.0
        if astar_cmd["moveWS"]["command"] == "W":
            astar_speed_val = astar_cmd["moveWS"]["weight"]
        elif astar_cmd["moveWS"]["command"] == "S":
            astar_speed_val = -astar_cmd["moveWS"]["weight"]

        ppo_speed_val = 0.0
        if ppo_cmd["moveWS"]["command"] == "W":
            ppo_speed_val = ppo_cmd["moveWS"]["weight"]
        elif ppo_cmd["moveWS"]["command"] == "S":
            ppo_speed_val = -ppo_cmd["moveWS"]["weight"]

        # 가중 평균
        blended_speed = (
            astar_speed_val * astar_weight +
            ppo_speed_val * ppo_weight
        )
        blended_speed = np.clip(blended_speed, -1.0, 1.0)

        # 명령 및 가중치로 변환
        if abs(blended_speed) < 0.05:
            ws_command = "STOP"
            ws_weight = 0.0
        elif blended_speed > 0:
            ws_command = "W"
            ws_weight = abs(blended_speed)
        else:
            ws_command = "S"
            ws_weight = abs(blended_speed)

        # ========================================
        # 2. moveAD 혼합
        # ========================================

        # A* 조향값 추출 (A: -1, D: +1, "": 0)
        astar_steer_val = 0.0
        if astar_cmd["moveAD"]["command"] == "D":
            astar_steer_val = astar_cmd["moveAD"]["weight"]
        elif astar_cmd["moveAD"]["command"] == "A":
            astar_steer_val = -astar_cmd["moveAD"]["weight"]

        # PPO 조향값 추출
        ppo_steer_val = 0.0
        if ppo_cmd["moveAD"]["command"] == "D":
            ppo_steer_val = ppo_cmd["moveAD"]["weight"]
        elif ppo_cmd["moveAD"]["command"] == "A":
            ppo_steer_val = -ppo_cmd["moveAD"]["weight"]

        # 가중 평균
        blended_steer = (
            astar_steer_val * astar_weight +
            ppo_steer_val * ppo_weight
        )
        blended_steer = np.clip(blended_steer, -1.0, 1.0)

        # 명령 및 가중치로 변환
        if abs(blended_steer) < 0.05:
            ad_command = ""
            ad_weight = 0.0
        elif blended_steer > 0:
            ad_command = "D"
            ad_weight = abs(blended_steer)
        else:
            ad_command = "A"
            ad_weight = abs(blended_steer)

        return {
            "moveWS": {"command": ws_command, "weight": round(ws_weight, 2)},
            "moveAD": {"command": ad_command, "weight": round(ad_weight, 2)},
            "fire": False
        }

    def _generate_virtual_lidar_scan(self, curr_x, curr_z, curr_yaw, num_rays=32, max_range=50.0):
        """
        360도 가상 LiDAR 스캔 생성 (obstacle_rects 기반)

        Args:
            curr_x, curr_z: 현재 위치
            curr_yaw: 현재 방향 (도)
            num_rays: 광선 개수
            max_range: 최대 탐지 거리 (m)

        Returns:
            list: [dist1, dist2, ...] 각 방향의 거리 (0~max_range)
        """
        distances = []
        curr_yaw_rad = math.radians(curr_yaw)

        for i in range(num_rays):
            # 광선 방향 (0도 = 북쪽, 시계방향)
            angle_offset = (2 * math.pi * i) / num_rays
            ray_angle = curr_yaw_rad + angle_offset

            # 광선 방향 벡터
            ray_dx = math.sin(ray_angle)
            ray_dz = math.cos(ray_angle)

            # 광선과 장애물의 최소 거리
            min_dist = max_range

            # obstacle_rects 사용 (실제 장애물 사각형)
            if hasattr(self.state, 'obstacle_rects') and self.state.obstacle_rects:
                for obs in self.state.obstacle_rects:
                    # 장애물 중심점
                    obs_cx = (obs["x_min"] + obs["x_max"]) / 2
                    obs_cz = (obs["z_min"] + obs["z_max"]) / 2

                    # 장애물과의 벡터
                    to_obs_x = obs_cx - curr_x
                    to_obs_z = obs_cz - curr_z

                    # 광선 방향으로의 투영 확인
                    projection = to_obs_x * ray_dx + to_obs_z * ray_dz

                    if projection > 0:  # 앞쪽에 있는 장애물만
                        # 거리 계산
                        dist = math.hypot(to_obs_x, to_obs_z)

                        # 장애물 크기 고려 (반지름 추정)
                        obs_radius = max(
                            (obs["x_max"] - obs["x_min"]) / 2,
                            (obs["z_max"] - obs["z_min"]) / 2
                        )

                        # 실제 충돌 거리 = 중심까지 거리 - 반지름
                        actual_dist = max(0.1, dist - obs_radius)

                        if actual_dist < min_dist:
                            min_dist = actual_dist

            distances.append(min_dist)

        return distances

    # ==================== Stop-Steer-Go 장애물 회피 ====================

    def _ssg_find_best_direction(self, curr_x, curr_z, curr_yaw):
        """가상 LiDAR로 가장 빈 방향 탐색, 목적지 방향도 가중"""
        ssg = self.config.StopSteerGo
        num_rays = ssg.SCAN_RAYS
        max_range = 50.0

        curr_yaw_rad = math.radians(curr_yaw)

        # 목적지 방향 각도
        dest_angle = None
        if self.state.destination:
            dx = self.state.destination[0] - curr_x
            dz = self.state.destination[1] - curr_z
            dest_angle = math.atan2(dx, dz)  # 북쪽 기준

        best_score = -1
        best_angle_offset = 0

        for i in range(num_rays):
            angle_offset = (2 * math.pi * i) / num_rays
            ray_angle = curr_yaw_rad + angle_offset
            ray_dx = math.sin(ray_angle)
            ray_dz = math.cos(ray_angle)

            min_dist = max_range
            if hasattr(self.state, 'obstacle_rects') and self.state.obstacle_rects:
                for obs in self.state.obstacle_rects:
                    obs_cx = (obs["x_min"] + obs["x_max"]) / 2
                    obs_cz = (obs["z_min"] + obs["z_max"]) / 2
                    to_obs_x = obs_cx - curr_x
                    to_obs_z = obs_cz - curr_z
                    projection = to_obs_x * ray_dx + to_obs_z * ray_dz
                    if projection > 0:
                        dist = math.hypot(to_obs_x, to_obs_z)
                        obs_radius = max(
                            (obs["x_max"] - obs["x_min"]) / 2,
                            (obs["z_max"] - obs["z_min"]) / 2
                        )
                        actual_dist = max(0.1, dist - obs_radius)
                        if actual_dist < min_dist:
                            min_dist = actual_dist

            # 점수: 거리 + 목적지 방향 보너스
            score = min_dist
            if dest_angle is not None:
                angle_diff = abs(math.atan2(math.sin(ray_angle - dest_angle),
                                            math.cos(ray_angle - dest_angle)))
                # 목적지 방향에 가까울수록 보너스 (최대 +10)
                score += max(0, 10 * (1 - angle_diff / math.pi))

            if score > best_score:
                best_score = score
                best_angle_offset = angle_offset

        # 최적 방향이 현재 전방 기준 좌/우 어디인지 판별
        # offset을 -pi ~ pi 범위로 정규화
        if best_angle_offset > math.pi:
            best_angle_offset -= 2 * math.pi

        best_clear_dist = best_score
        return best_angle_offset, best_clear_dist

    def _stop_steer_go_action(self, curr_x, curr_z, curr_yaw):
        """Stop-Steer-Go 3단계 장애물 회피

        Phase 1 (STOP): 완전 정지하여 관성 제거
        Phase 2 (STEER): 제자리 회전으로 가장 빈 방향 탐색 후 그 방향으로 조향
        Phase 3 (GO): 클리어된 방향으로 전진
        """
        ssg = self.config.StopSteerGo
        elapsed = time.time() - self.ssg_start_time

        # Phase 1: STOP
        if self.ssg_phase == "stop":
            if elapsed < ssg.STOP_SEC:
                return {
                    "moveWS": {"command": "STOP", "weight": 1.0},
                    "moveAD": {"command": "", "weight": 0.0},
                    "fire": False
                }
            # → Phase 2 전환
            angle_offset, clear_dist = self._ssg_find_best_direction(curr_x, curr_z, curr_yaw)
            if angle_offset >= 0:
                self.ssg_best_direction = "D"  # 우회전
            else:
                self.ssg_best_direction = "A"  # 좌회전

            self.ssg_phase = "steer"
            self.ssg_start_time = time.time()
            print(f"🔄 SSG Phase2(STEER): 방향={self.ssg_best_direction}, "
                  f"클리어거리={clear_dist:.1f}m, 각도차={math.degrees(angle_offset):.0f}°")
            self.state.set_log(f"🔄 SSG 조향 탐색: {self.ssg_best_direction} 방향")

        # Phase 2: STEER (제자리 회전)
        if self.ssg_phase == "steer":
            steer_elapsed = time.time() - self.ssg_start_time
            if steer_elapsed < ssg.STEER_SEC:
                return {
                    "moveWS": {"command": "STOP", "weight": 1.0},
                    "moveAD": {"command": self.ssg_best_direction,
                               "weight": ssg.STEER_WEIGHT},
                    "fire": False
                }
            # → Phase 3 전환
            self.ssg_phase = "go"
            self.ssg_start_time = time.time()
            print(f"🚗 SSG Phase3(GO): 전진 재개 ({self.ssg_best_direction} 방향)")
            self.state.set_log(f"🚗 SSG 전진 재개")

        # Phase 3: GO (전진)
        if self.ssg_phase == "go":
            go_elapsed = time.time() - self.ssg_start_time
            if go_elapsed < ssg.GO_SEC:
                return {
                    "moveWS": {"command": "W", "weight": ssg.GO_WS_WEIGHT},
                    "moveAD": {"command": self.ssg_best_direction,
                               "weight": ssg.GO_AD_WEIGHT},
                    "fire": False
                }
            # SSG 완료 → 정상 복귀
            print("✅ SSG 완료! 정상 제어 복귀")
            self.state.set_log("✅ 장애물 회피 완료, 정상 주행 복귀")
            self.ssg_mode = False
            self.ssg_phase = None
            self.ssg_no_valid_count = 0
            self.stuck_counter = 0
            self.last_position = None
            self.state.clear_path()  # 경로 재생성 유도
            return self._stop_command()

        # fallback
        self.ssg_mode = False
        return self._stop_command()

    # ==================== Stuck 감지/복구 ====================
    
    def _detect_stuck(self, curr_x, curr_z):
        """Stuck 감지"""
        if self.last_position is None:
            self.last_position = (curr_x, curr_z)
            return
        
        dist = math.hypot(
            curr_x - self.last_position[0],
            curr_z - self.last_position[1]
        )
        
        if dist < self.config.Stuck.STUCK_THRESHOLD:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        
        self.last_position = (curr_x, curr_z)
    
    def _recovery_action(self, curr_x, curr_z, curr_yaw):
        """Stuck 복구 동작"""
        rc = self.config.Recovery
        
        if not self.recovery_mode:
            self.recovery_mode = True
            self.recovery_start_time = time.time()
            self.recovery_direction = 1 if (self.stuck_counter % 2 == 0) else -1
            print(f"🔧 복구 시작: {'좌회전' if self.recovery_direction > 0 else '우회전'} 후진")
        
        elapsed = time.time() - self.recovery_start_time
        
        if elapsed < rc.PHASE1_SEC:
            # Phase 1: 후진 + 회전
            return {
                "moveWS": {"command": "S", "weight": rc.PHASE1_WS_WEIGHT},
                "moveAD": {"command": "D" if self.recovery_direction > 0 else "A", 
                          "weight": rc.PHASE1_AD_WEIGHT},
                "fire": False
            }
        
        elif elapsed < rc.PHASE1_SEC + rc.PHASE2_SEC:
            # Phase 2: 제자리 회전
            return {
                "moveWS": {"command": "STOP", "weight": 1.0},
                "moveAD": {"command": "D" if self.recovery_direction > 0 else "A", 
                          "weight": rc.PHASE2_AD_WEIGHT},
                "fire": False
            }
        
        else:
            # 복구 완료
            print("✅ 복구 완료!")
            self.recovery_mode = False
            self.stuck_counter = 0
            self.last_position = None
            self.state.clear_path()  # 경로 재생성 유도
            return self._stop_command()
    
    @staticmethod
    def _stop_command():
        """정지 명령"""
        return {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0}, 
            "fire": False
        }
