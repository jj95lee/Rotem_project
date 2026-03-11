"""
server_rl.py

RL 기반 자율주행 Flask 서버
- A* 경로 계획 + RL 국소 제어
- 기존 hybrid_controller 구조 활용
- 학습된 모델 로드 및 추론

[사용법]
    python server_rl.py --model models/tank_nav_final.zip --port 5000

[엔드포인트]
    POST /info       - 시뮬레이터 상태 수신
    POST /get_action - 이동 명령 반환 (RL 기반)
    POST /collision  - 충돌 정보 수신
    GET  /init       - 초기 설정
    GET  /start      - 시작 명령
    POST /set_destination - 목적지 설정
"""

import os
import sys
import json
import math
import time
import argparse
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from flask import Flask, request, jsonify
import numpy as np

# 로컬 모듈
from rl_controller import RLController, RLConfig


# ==============================================================================
# 상태 관리자
# ==============================================================================

@dataclass
class TankState:
    """전차 상태"""
    x: float = 0.0
    z: float = 0.0
    yaw: float = 0.0  # playerBodyX (degrees)
    speed: float = 0.0
    health: float = 100.0
    
    
@dataclass
class ServerState:
    """서버 전역 상태"""
    # 전차 상태
    tank: TankState = field(default_factory=TankState)
    
    # 목적지
    destination: Optional[Tuple[float, float]] = None
    
    # A* 경로
    global_path: List[Tuple[float, float]] = field(default_factory=list)
    current_path_idx: int = 0
    
    # 장애물
    obstacle_rects: List[Dict] = field(default_factory=list)
    
    # 에피소드 정보
    episode_start_time: float = 0.0
    step_count: int = 0
    collision_count: int = 0
    
    # 플래그
    is_running: bool = False
    reached_goal: bool = False
    
    def reset(self):
        """상태 초기화"""
        self.destination = None
        self.global_path = []
        self.current_path_idx = 0
        self.episode_start_time = time.time()
        self.step_count = 0
        self.collision_count = 0
        self.is_running = False
        self.reached_goal = False


# ==============================================================================
# 서버 설정
# ==============================================================================

@dataclass  
class ServerConfig:
    """서버 설정"""
    # 경로 추종
    lookahead_dist: float = 10.0
    goal_threshold: float = 8.0
    
    # 맵 설정
    map_size: float = 300.0
    map_margin: float = 5.0
    
    # A* 설정
    obstacle_margin: float = 3.0
    
    # 시간 제한
    max_episode_time: float = 300.0  # 5분
    
    # RL 모델 경로
    model_path: str = "models/tank_nav_final.zip"
    
    # 시작점/목표점 (시나리오 기본값)
    default_start: Tuple[float, float] = (49.0, 236.0)
    default_goal: Tuple[float, float] = (65.0, 30.0)


# ==============================================================================
# Flask 앱
# ==============================================================================

app = Flask(__name__)

# 전역 상태
state = ServerState()
config = ServerConfig()
rl_controller: Optional[RLController] = None
planner = None  # A* 플래너 (나중에 로드)


def init_controller(model_path: str):
    """RL 컨트롤러 초기화"""
    global rl_controller
    
    rl_config = RLConfig(
        forward_weight=0.5,
        turn_weight=0.5,
        strong_turn_weight=0.8,
    )
    
    rl_controller = RLController(
        model_path=model_path,
        config=rl_config,
    )
    
    print(f"✅ RL 컨트롤러 초기화 완료")
    if rl_controller.model_loaded:
        print(f"   - 모델: {model_path}")
    else:
        print(f"   - 모델 없음, 규칙 기반 폴백 사용")


def init_planner(obstacles: List[Dict]):
    """A* 플래너 초기화"""
    global planner
    
    try:
        from astar_planner import AStarPlanner, ObstacleRect
        
        planner = AStarPlanner(
            grid_min_x=0.0,
            grid_max_x=config.map_size,
            grid_min_z=0.0,
            grid_max_z=config.map_size,
            cell_size=1.0,
            obstacle_margin=config.obstacle_margin,
            allow_diagonal=True,
            safety_weight=1.5,
            proximity_radius=8.0,
        )
        
        # 장애물 설정
        obs_list = []
        for obs in obstacles:
            obs_list.append(ObstacleRect.from_min_max(
                x_min=obs['x_min'],
                x_max=obs['x_max'],
                z_min=obs['z_min'],
                z_max=obs['z_max'],
            ))
        planner.set_obstacles(obs_list)
        
        print(f"✅ A* 플래너 초기화 완료 ({len(obstacles)}개 장애물)")
        
    except ImportError as e:
        print(f"⚠️ A* 플래너 로드 실패: {e}")
        print("   - 직선 경로 사용")
        planner = None


def load_obstacles(json_path: str) -> List[Dict]:
    """장애물 JSON 로드"""
    if not os.path.exists(json_path):
        print(f"⚠️ 장애물 파일 없음: {json_path}")
        return []
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    obstacles = data.get('obstacles', [])
    print(f"✅ 장애물 로드: {len(obstacles)}개")
    return obstacles


def generate_path(start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
    """A* 경로 생성"""
    if planner is None:
        # 직선 경로 (플래너 없을 때)
        return [start, goal]
    
    path = planner.find_path(start, goal)
    
    if path:
        print(f"✅ 경로 생성: {len(path)}개 노드")
        return path
    else:
        print(f"⚠️ 경로 생성 실패, 직선 경로 사용")
        return [start, goal]


def get_target_point(curr_x: float, curr_z: float) -> Optional[Tuple[float, float]]:
    """현재 타겟 waypoint 반환"""
    if not state.global_path:
        return state.destination
    
    # 지나간 노드 제거
    while state.current_path_idx < len(state.global_path) - 1:
        wp = state.global_path[state.current_path_idx]
        dist = math.hypot(wp[0] - curr_x, wp[1] - curr_z)
        if dist < 5.0:
            state.current_path_idx += 1
        else:
            break
    
    # Lookahead 거리만큼 앞의 waypoint 선택
    cumulative_dist = 0.0
    target_idx = state.current_path_idx
    
    for i in range(state.current_path_idx, len(state.global_path)):
        if i > state.current_path_idx:
            prev = state.global_path[i-1]
            curr = state.global_path[i]
            cumulative_dist += math.hypot(curr[0] - prev[0], curr[1] - prev[1])
        if cumulative_dist >= config.lookahead_dist:
            target_idx = i
            break
        target_idx = i
    
    return state.global_path[target_idx]


def check_goal_reached(curr_x: float, curr_z: float) -> bool:
    """목표 도달 체크"""
    if state.destination is None:
        return False
    
    dist = math.hypot(state.destination[0] - curr_x, state.destination[1] - curr_z)
    return dist < config.goal_threshold


def stop_command() -> Dict:
    """정지 명령"""
    return {
        "moveWS": {"command": "STOP", "weight": 1.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False
    }


# ==============================================================================
# 엔드포인트
# ==============================================================================

@app.route('/init', methods=['GET'])
def init():
    """초기화 설정"""
    state.reset()
    state.episode_start_time = time.time()
    
    # 기본 목적지 설정
    state.destination = config.default_goal
    
    # 경로 생성
    if planner is not None:
        state.global_path = generate_path(config.default_start, config.default_goal)
    else:
        state.global_path = [config.default_start, config.default_goal]
    
    init_config = {
        "startMode": "start",
        "blStartX": config.default_start[0],
        "blStartY": 10,
        "blStartZ": config.default_start[1],
        "rdStartX": 59,
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": True,
        "detectMode": False,
        "logMode": True,  # playerBodyX 받으려면 True
        "stereoCameraMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000,
        "destoryObstaclesOnHit": True
    }
    
    print(f"🛠️ 초기화 완료")
    print(f"   - 시작점: {config.default_start}")
    print(f"   - 목적지: {config.default_goal}")
    print(f"   - 경로: {len(state.global_path)}개 노드")
    
    return jsonify(init_config)


@app.route('/start', methods=['GET'])
def start():
    """시작 명령"""
    state.is_running = True
    print("🚀 시작!")
    return jsonify({"control": "start"})


@app.route('/info', methods=['POST'])
def info():
    """시뮬레이터 상태 수신"""
    data = request.get_json(force=True)
    
    if not data:
        return jsonify({"status": "error", "message": "No data"}), 400
    
    # 상태 업데이트
    if 'playerPos' in data:
        state.tank.x = data['playerPos'].get('x', 0)
        state.tank.z = data['playerPos'].get('z', 0)
    
    if 'playerBodyX' in data:
        state.tank.yaw = data['playerBodyX']
    
    if 'playerSpeed' in data:
        state.tank.speed = data['playerSpeed']
    
    if 'playerHealth' in data:
        state.tank.health = data['playerHealth']
    
    # 시간 체크
    elapsed = time.time() - state.episode_start_time
    if elapsed > config.max_episode_time:
        print(f"⏰ 시간 초과 ({elapsed:.1f}s)")
        return jsonify({"status": "success", "control": "pause"})
    
    return jsonify({"status": "success", "control": ""})


@app.route('/get_action', methods=['POST'])
def get_action():
    """이동 명령 반환 (RL 기반)"""
    global rl_controller
    
    data = request.get_json(force=True)
    
    # 위치 정보 추출
    position = data.get("position", {})
    curr_x = position.get("x", state.tank.x)
    curr_z = position.get("z", state.tank.z)
    curr_yaw = state.tank.yaw  # /info에서 업데이트됨
    
    state.step_count += 1
    
    # 디버그 출력 (50스텝마다)
    if state.step_count % 50 == 1:
        dist_to_goal = math.hypot(state.destination[0] - curr_x, state.destination[1] - curr_z) if state.destination else 0
        print(f"📍 Step {state.step_count}: pos=({curr_x:.1f}, {curr_z:.1f}), yaw={curr_yaw:.1f}°, "
              f"dist={dist_to_goal:.1f}m")
    
    # 목적지 없으면 정지
    if state.destination is None:
        return jsonify(stop_command())
    
    # 목표 도달 체크
    if check_goal_reached(curr_x, curr_z):
        state.reached_goal = True
        elapsed = time.time() - state.episode_start_time
        print(f"🎉 목표 도달! (시간: {elapsed:.1f}s, 스텝: {state.step_count})")
        return jsonify(stop_command())
    
    # 타겟 포인트 선택
    target = get_target_point(curr_x, curr_z)
    if target is None:
        target = state.destination
    
    # RL 컨트롤러로 행동 결정
    if rl_controller is None:
        init_controller(config.model_path)
    
    # 장애물 정보 업데이트
    rl_controller.obstacle_rects = [
        (obs['x_min'], obs['x_max'], obs['z_min'], obs['z_max'])
        for obs in state.obstacle_rects
    ]
    
    command = rl_controller.get_action(
        curr_x=curr_x,
        curr_z=curr_z,
        curr_yaw=curr_yaw,
        target=target,
        goal=state.destination,
        current_speed=state.tank.speed,
    )
    
    return jsonify(command)


@app.route('/set_destination', methods=['POST'])
def set_destination():
    """목적지 설정"""
    data = request.get_json()
    
    if not data or "destination" not in data:
        return jsonify({"status": "error", "message": "Missing destination"}), 400
    
    try:
        x, y, z = map(float, data["destination"].split(","))
        state.destination = (x, z)
        
        # 경로 재생성
        start = (state.tank.x, state.tank.z)
        if start[0] == 0 and start[1] == 0:
            start = config.default_start
        
        state.global_path = generate_path(start, state.destination)
        state.current_path_idx = 0
        
        print(f"🎯 목적지 설정: ({x}, {z})")
        print(f"   - 경로: {len(state.global_path)}개 노드")
        
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/collision', methods=['POST'])
def collision():
    """충돌 정보 수신"""
    data = request.get_json()
    
    if not data:
        return jsonify({"status": "error"}), 400
    
    state.collision_count += 1
    
    obj_name = data.get('objectName', 'unknown')
    pos = data.get('position', {})
    
    print(f"💥 충돌 #{state.collision_count}: {obj_name} at ({pos.get('x', 0):.1f}, {pos.get('z', 0):.1f})")
    
    return jsonify({"status": "OK", "message": "Collision received"})


@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    """장애물 정보 업데이트"""
    data = request.get_json()
    
    if not data:
        return jsonify({"status": "error"}), 400
    
    obstacles = data.get('obstacles', [])
    state.obstacle_rects = obstacles
    
    # RL 컨트롤러에도 업데이트
    if rl_controller:
        rl_controller.set_obstacles(obstacles)
    
    # A* 플래너에도 업데이트
    if planner:
        try:
            from astar_planner import ObstacleRect
            obs_list = [
                ObstacleRect.from_min_max(
                    x_min=obs['x_min'], x_max=obs['x_max'],
                    z_min=obs['z_min'], z_max=obs['z_max']
                )
                for obs in obstacles
            ]
            planner.set_obstacles(obs_list)
        except:
            pass
    
    print(f"🪨 장애물 업데이트: {len(obstacles)}개")
    
    return jsonify({"status": "OK"})


@app.route('/detect', methods=['POST'])
def detect():
    """객체 탐지 (더미)"""
    return jsonify([])


@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    """포탄 충돌 정보"""
    data = request.get_json()
    print(f"💥 포탄 충돌: {data}")
    return jsonify({"status": "OK"})


@app.route('/stereo_image', methods=['POST'])
def stereo_image():
    """스테레오 이미지 (더미)"""
    return jsonify({"result": "success"})


# ==============================================================================
# 메인
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="RL-based Tank Navigation Server")
    parser.add_argument("--model", type=str, default="models/tank_nav_final.zip",
                        help="RL model path")
    parser.add_argument("--obstacles", type=str, default="ob_v2.json",
                        help="Obstacle JSON path")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Tank Navigation RL Server")
    print("="*60)
    
    # 설정 업데이트
    config.model_path = args.model
    
    # 장애물 로드
    obstacles = load_obstacles(args.obstacles)
    state.obstacle_rects = obstacles
    
    # A* 플래너 초기화
    init_planner(obstacles)
    
    # RL 컨트롤러 초기화
    init_controller(args.model)
    
    # 장애물 정보를 RL 컨트롤러에 전달
    if rl_controller:
        rl_controller.set_obstacles(obstacles)
    
    print(f"\n🌐 서버 시작: http://{args.host}:{args.port}")
    print("="*60)
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
