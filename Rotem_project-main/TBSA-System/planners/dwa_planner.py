"""
DWA (Dynamic Window Approach) 로컬 플래너 - 진단 버전

=== 디버그 모드 ===
DEBUG_DWA = True로 설정하면 상세 로그 출력

=== 좌표계 확인 필수 ===
게임에서 curr_yaw가 어떤 방향을 0으로 사용하는지 확인하세요:
- 0도 = 북쪽(+z)인 경우: 현재 코드 그대로 사용
- 0도 = 동쪽(+x)인 경우: motion_model 수정 필요
"""
import math
import numpy as np

# 🔧 디버그 플래그
DEBUG_DWA = False
DEBUG_INTERVAL = 20  # N번에 1번 출력


class DWAConfig:
    """DWA 알고리즘 파라미터"""
    
    def __init__(self, config):
        self.max_speed = config.DWA.MAX_SPEED
        self.min_speed = config.DWA.MIN_SPEED
        self.max_yaw_rate = config.DWA.MAX_YAW_RATE
        self.max_accel = config.DWA.MAX_ACCEL
        self.max_delta_yaw_rate = config.DWA.MAX_DELTA_YAW_RATE
        self.v_resolution = config.DWA.V_RESOLUTION
        self.yaw_rate_resolution = config.DWA.YAW_RATE_RESOLUTION
        self.dt = config.DWA.DT
        self.predict_time = config.DWA.PREDICT_TIME
        self.to_goal_cost_gain = config.DWA.TO_GOAL_COST_GAIN
        self.speed_cost_gain = config.DWA.SPEED_COST_GAIN
        self.obstacle_cost_gain = config.DWA.OBSTACLE_COST_GAIN
        self.robot_radius = config.DWA.ROBOT_RADIUS
        self.robot_stuck_flag_cons = config.DWA.ROBOT_STUCK_FLAG_CONS
        self.steering_penalty = config.DWA.STEERING_PENALTY


_call_count = 0  # 전역 호출 카운터


def motion_model(x, u, dt):
    """
    운동 모델
    
    가정 (게임 좌표계):
    - x[0]: x좌표 (동쪽이 +)
    - x[1]: z좌표 (북쪽이 +)  
    - x[2]: yaw (라디안), 0=북쪽, 양수=시계방향(오른쪽)
    
    yaw=0일 때 전진 → z 증가 (북쪽)
    yaw=90°일 때 전진 → x 증가 (동쪽)
    """
    x[2] += u[1] * dt
    x[0] += u[0] * math.sin(x[2]) * dt
    x[1] += u[0] * math.cos(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x


def calc_dynamic_window(x, config):
    """Dynamic Window 계산"""
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]
    
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]
    
    if x[3] < 0 and Vd[1] < 0:
        Vd[1] = 0.02
    elif x[3] > 0 and Vd[0] > 0:
        Vd[0] = -0.02
    
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    return dw


def predict_trajectory(x_init, v, y, config):
    """궤적 예측"""
    x = np.array(x_init)
    trajectory = np.array(x)
    time_val = 0
    while time_val <= config.predict_time:
        x = motion_model(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time_val += config.dt
    return trajectory


def calc_costmap_cost(traj, costmap, origin, resolution, state_manager=None):
    """Costmap 기반 장애물 비용 (근접 비용 포함)"""
    if costmap is None and state_manager is None:
        return 0.0

    from config import Config

    max_cell_cost = 0.0
    proximity_cost = 0.0  # 장애물 근접 비용
    
    for i, state in enumerate(traj):
        if i < 2: continue

        x = state[0]
        z = state[1]

        # 맵 경계 체크
        if (x <= Config.WORLD_MIN_XZ + 0.2 or x >= Config.WORLD_MAX_XZ - 0.2 or 
            z <= Config.WORLD_MIN_XZ + 0.2 or z >= Config.WORLD_MAX_XZ - 0.2):
            return float("inf")
        
        # 전역 장애물 충돌 체크
        if state_manager is not None:
            if state_manager.is_global_obstacle(x, z):
                return float("inf")
            
            # 🆕 장애물 근접 비용 계산 (가까울수록 높은 비용)
            min_dist = state_manager.get_min_obstacle_distance(x, z)
            if min_dist < 8.0:  # 8m 이내 장애물
                # 거리에 반비례하는 비용 (가까울수록 급격히 증가)
                prox = (8.0 - min_dist) / 8.0  # 0~1 정규화
                proximity_cost = max(proximity_cost, prox * prox * 2.0)  # 제곱해서 급격히 증가
        
        if costmap is None or origin is None:
            continue

        ix = int((x - origin[0]) / resolution)
        iz = int((z - origin[1]) / resolution)

        if ix < 0 or iz < 0 or iz >= costmap.shape[0] or ix >= costmap.shape[1]:
            continue

        cell_cost = float(costmap[iz, ix])

        if cell_cost >= 0.88:
            return float("inf")

        if cell_cost > max_cell_cost:
            max_cell_cost = cell_cost
    
    return max_cell_cost + proximity_cost


def calc_to_goal_cost(trajectory, goal):
    """
    목표 도달 비용
    
    목표 방향과 현재 heading의 차이를 비용으로 반환
    
    중요: atan2(dx, dz)를 사용 (표준 atan2(y,x)가 아님!)
    이유: 게임에서 yaw=0이 +z방향(북쪽)을 가리키기 때문
    """
    global _call_count
    _call_count += 1
    
    dx = goal[0] - trajectory[-1, 0]
    dz = goal[1] - trajectory[-1, 1]
    
    # 목표 방향 계산
    # atan2(dx, dz): dz>0 & dx=0 → 0 (북쪽)
    #               dz=0 & dx>0 → π/2 (동쪽)  
    goal_angle = math.atan2(dx, dz)
    
    current_yaw = trajectory[-1, 2]
    angle_diff = goal_angle - current_yaw
    
    while angle_diff > math.pi:
        angle_diff -= 2.0 * math.pi
    while angle_diff < -math.pi:
        angle_diff += 2.0 * math.pi

    cost = abs(angle_diff)
    
    # 디버그 출력
    if DEBUG_DWA and _call_count % (DEBUG_INTERVAL * 50) == 1:  # 매우 드물게 출력
        dist = math.hypot(dx, dz)
        print(f"🧭 [Goal Cost Debug]")
        print(f"   현재 위치: ({trajectory[-1,0]:.1f}, {trajectory[-1,1]:.1f})")
        print(f"   목표 위치: ({goal[0]:.1f}, {goal[1]:.1f})")
        print(f"   차이 벡터: dx={dx:.1f}, dz={dz:.1f}, 거리={dist:.1f}m")
        print(f"   목표 방향: {math.degrees(goal_angle):.1f}°")
        print(f"   현재 yaw: {math.degrees(current_yaw):.1f}°")
        print(f"   각도 차이: {math.degrees(angle_diff):.1f}° → 비용: {cost:.3f}")
        
        # 좌표계 검증
        if dz < -100:  # 남쪽으로 100m 이상
            expected_dir = "남쪽 (약 180° 또는 -180°)"
        elif dz > 100:
            expected_dir = "북쪽 (약 0°)"
        elif dx > 100:
            expected_dir = "동쪽 (약 90°)"
        elif dx < -100:
            expected_dir = "서쪽 (약 -90°)"
        else:
            expected_dir = "근거리"
        print(f"   예상 방향: {expected_dir}")
        print(f"   계산된 방향: {math.degrees(goal_angle):.1f}° - 일치 확인 필요!")
    
    return cost


def verify_coordinate_system(curr_x, curr_z, curr_yaw_deg, goal_x, goal_z):
    """
    좌표계 검증 함수 - hybrid_controller에서 호출하여 좌표계 확인
    
    사용법:
    from planners.dwa_planner import verify_coordinate_system
    verify_coordinate_system(curr_x, curr_z, curr_yaw, goal[0], goal[1])
    """
    dx = goal_x - curr_x
    dz = goal_z - curr_z
    dist = math.hypot(dx, dz)
    
    # 목표 방향 (우리 가정: atan2(dx, dz))
    goal_angle_our = math.degrees(math.atan2(dx, dz))
    
    # 목표 방향 (표준 수학: atan2(dz, dx))
    goal_angle_std = math.degrees(math.atan2(dz, dx))
    
    print("\n" + "="*60)
    print("🔍 좌표계 검증")
    print("="*60)
    print(f"현재 위치: ({curr_x:.1f}, {curr_z:.1f})")
    print(f"현재 yaw: {curr_yaw_deg:.1f}°")
    print(f"목표 위치: ({goal_x:.1f}, {goal_z:.1f})")
    print(f"거리: {dist:.1f}m")
    print(f"\n벡터: dx={dx:.1f}, dz={dz:.1f}")
    print(f"\n목표 방향 계산:")
    print(f"  - atan2(dx, dz) = {goal_angle_our:.1f}° (현재 코드)")
    print(f"  - atan2(dz, dx) = {goal_angle_std:.1f}° (표준 수학)")
    
    # 방향 판단
    if dz < -50:
        print(f"\n⚠️ 목표가 남쪽(z 감소)에 있음")
        print(f"   올바른 방향은 약 ±180° 근처여야 함")
        if abs(goal_angle_our) > 150:
            print(f"   ✅ atan2(dx,dz)={goal_angle_our:.1f}° - 올바름")
        else:
            print(f"   ❌ atan2(dx,dz)={goal_angle_our:.1f}° - 잘못됨!")
    
    # yaw와 비교
    yaw_diff = goal_angle_our - curr_yaw_deg
    while yaw_diff > 180: yaw_diff -= 360
    while yaw_diff < -180: yaw_diff += 360
    
    print(f"\n현재 yaw와의 차이: {yaw_diff:.1f}°")
    if abs(yaw_diff) < 30:
        print("   ✅ 거의 목표를 향하고 있음 - 직진해야 함")
    elif abs(yaw_diff) > 150:
        print("   ⚠️ 목표와 반대 방향 - 큰 회전 필요")
    else:
        print(f"   → {'우회전' if yaw_diff > 0 else '좌회전'} 필요")
    
    print("="*60 + "\n")