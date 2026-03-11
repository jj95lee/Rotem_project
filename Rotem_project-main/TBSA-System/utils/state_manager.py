"""
전역 상태 관리 - 주행 시스템(SEQ 1,3,4) + 전투 시스템(SEQ 2) 통합

[클래스 구조]
├── LidarTargetLockState: LiDAR 기반 타겟 잠금 상태 (SEQ 2)
└── StateManager: 시스템 전역 상태 관리
    ├── 주행 시스템 상태 (SEQ 1, 3, 4)
    │   ├── 로봇 위치/자세
    │   ├── 경로 정보
    │   ├── Costmap
    │   ├── 전역 장애물 맵
    │   └── DWA 상태
    │
    └── 전투 시스템 상태 (SEQ 2)
        ├── 타이밍 정보
        ├── 타겟 추적
        ├── 이벤트 처리
        ├── Hit 감지
        ├── State Machine (SCAN/STANDBY/FIRE)
        ├── LiDAR 잠금
        └── 카메라 정보
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import math
from config import precision_cfg 



@dataclass
class LidarTargetLockState:
    """
    LiDAR 기반 타겟 잠금 상태 (SEQ 2)
    
    Attributes:
        locked: 잠금 활성화 여부
        lock_time: 잠금 시작 시각
        locked_angle: 잠금 시 수평 각도
        locked_vertical_angle: 잠금 시 수직 각도
        locked_distance: 잠금 시 거리
        locked_position: 잠금 시 3D 위치
        current_*: 현재 추적 중인 타겟 정보
        lock_count: 총 잠금 횟수
        successful_fires: 성공적인 발사 횟수
        last_update_time: 마지막 업데이트 시각
    """
    locked: bool = False
    lock_time: float = 0.0
    
    # 잠금된 타겟 정보
    locked_angle: float = 0.0
    locked_vertical_angle: float = 0.0
    locked_distance: float = 0.0
    locked_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # 현재 추적 정보
    current_angle: float = 0.0
    current_vertical_angle: float = 0.0
    current_distance: float = 0.0
    current_lidar_points: int = 0
    
    # 통계
    lock_count: int = 0
    successful_fires: int = 0
    last_update_time: float = 0.0


class StateManager:
    """
    시스템 전역 상태 관리 클래스
    
    주행 시스템(SEQ 1, 3, 4)과 전투 시스템(SEQ 2)의 
    모든 상태를 통합 관리
    """
    
    def __init__(self, config):
        """
        StateManager 초기화
        
        Args:
            config: Config 클래스 인스턴스
        """
        self.config = config
        
        # ==================================================================
        # 주행 시스템 상태 (SEQ 1, 3, 4)
        # ==================================================================
        
        # 시퀀스 및 상태 메시지
        self.seq = 1
        self.status_message = "목적지를 설정해주세요 (SEQ 1)"
        self.last_log = "시스템 대기 중"
        
        # 로봇 상태
        self.robot_pose = None  # (x, y, z) 튜플
        self.robot_yaw_deg = None
        
        # 경로 관련
        self.destination = None  # (x, z) 튜플
        self.global_path = []  # A* 경로
        
        # Costmap (LiDAR 기반 장애물 지도)
        self.costmap = None
        self.costmap_origin = None
        self.costmap_stats = None
        
        # 전역 장애물 맵 (메모리에 누적)
        self.global_obstacles = []  # [(x, z), ...] 리스트 (FIFO)
        self.global_obstacle_grid_size = 4.0
        self.global_obstacles_updated = False
        self.MAX_GLOBAL_OBSTACLES = 300
        
        # DWA 상태
        self.last_dwa_traj = None
        self.last_dwa_target = None
        self.valid_traj_count = 0
        
        # 버전 관리 (시각화용)
        self.costmap_version = 0
        self.global_path_version = 0
        self.local_traj_version = 0
        
        # 실시간 경로 추적 스냅샷 (SEQ 1, 3용) - 번갈아가며 표시
        self.realtime_snapshot_1_bytes = None  # Realtime 1 창용
        self.realtime_snapshot_2_bytes = None  # Realtime 2 창용
        self.realtime_snapshot_index = 0  # 다음에 업데이트할 창 (0 또는 1)
        self.realtime_snapshot_ts = 0.0  # 마지막 스냅샷 시각
        
        # Unity에서 받아온 추가 정보 (DataFrame 병합용)
        self.turret_camera_pos = {'x': 0, 'y': 0, 'z': 0}
        self.stereo_left_rot = {'x': 0, 'y': 0, 'z': 0}
        self.stereo_left_pos = {'x': 0, 'y': 0, 'z': 0}
        self.stereo_right_pos = {'x': 0, 'y': 0, 'z': 0}
        self.stereo_right_rot = {'x': 0, 'y': 0, 'z': 0}
        self.lidar_rotation = {'x': 0, 'y': 0, 'z': 0}
        
        # ==================================================================
        # 전투 시스템 상태 (SEQ 2)
        # ==================================================================
        
        # 타이밍 정보
        self.last_pitch_cmd_ts = 0.0
        self.last_pose_ts = 0.0
        self.last_target_ts = 0.0
        self.last_bullet_ts = 0.0
        self.last_obstacle_ts = 0.0
        self.last_shot_ts = 0.0
        self.mode_ts = 0.0
        self.hit_ts = 0.0
        self.locked_ts = 0.0
        self.locked_update_ts = 0.0
        self.locked_start_ts = None
        self.overlay_left_ts = 0.0
        self.aim_stable_start_ts = 0.0
        self.target_lost_start_ts = 0.0
        self.output_hold_ts = 0.0
        
        # 스무싱 (조준 부드럽게)
        self.smooth_turret_yaw = 0.0
        self.smooth_turret_pitch = 0.0
        
        # 타겟 추적
        self.last_target = None  # 현재 타겟 정보 (dict)
        self.detected_targets = []  # 탐지된 모든 타겟 리스트
        self.last_detected_distance = None
        self.locked_bbox = None  # 잠긴 타겟의 BBox
        self.last_scan_targets = []  # SCAN 모드에서 탐지된 타겟 (모드 전환 후에도 유지)
        
        # 이벤트
        self.last_bullet_event = None
        self.last_obstacle_event = None
        self.last_hit_xyz = None
        
        # Hit 감지
        self.hit_flag = False
        self.hit_count = 0
        
        # State Machine (SCAN → STANDBY → FIRE)
        self.combat_mode = "SCAN"
        self.last_action = None
        
        # SCAN 모드 상태
        self.scan_start_ts = 0.0
        self.scan_direction = None  # None: 입력 대기, "Q": 좌측, "E": 우측
        self.scan_init_msg_sent = False
        self.enemy_msg_sent = False
        self.is_lowering_barrel = False
        
        # STANDBY 모드 상태
        self.standby_target = None
        self.locked_tid = None  # Tracking ID
        self.standby_start_ts = 0.0
        self.is_aim_aligned = False  # 포신 정렬 완료 여부
        self.fire_ready = False  # 발사 준비 완료 (버튼 활성화)
        
        # RETREAT 모드 상태
        self.retreat_aligned = False

        # FIRE 모드 상태
        self.fire_requested = False
        self.fire_executed_ts = 0.0
        
        # UI 버튼 액션 ("FIRE", "RESCAN", "RETREAT")
        self.user_action = None

        # SEQ 자동 전환 요청 변수 추가
        self.seq_change_request = None
        
        # Output hold (연속 출력 방지)
        self.last_sent_boxes = None
        
        # 카메라
        self.camera_img_bytes = None
        self.overlay_left_bytes = None
        
        # 카메라 포즈 (센서 퓨전용)
        self.cam_pos = None
        self.cam_rot = None
        self.cam_C = None
        self.cam_R_wc = None
        self.cam_axes = None
        
        # 카운터
        self.fallback_count = 0
        
        # LiDAR 타겟 잠금
        self.lidar_lock = LidarTargetLockState()
        
        # 현재 터렛/차체 위치
        self.current_player_turret_x = 0.0
        self.current_turret_pitch = 0.0

        self.player_turret_x = 0.0
        self.player_body_x = 0.0
        
        # [추가] 타겟 락 검증용 상태 변수
        self.pending_tid = None        # 검증 중인 타겟의 Track ID
        self.pending_start_ts = 0.0    # 검증 시작 시간
    
    # ==================================================================
    # 주행 시스템 메서드
    # ==================================================================
    
    def update_robot_pose(self, x: float, z: float, y: float = None):
        """
        로봇 위치 업데이트
        
        Args:
            x: X 좌표
            z: Z 좌표
            y: Y 좌표 (높이) - 제공되지 않으면 기존 y값 유지
        """
        x, z = self.config.clamp_world_xz(x, z)
        
        # y가 제공되지 않으면 기존 y 값 유지
        if y is None:
            if self.robot_pose is not None:
                y = self.robot_pose[1]  # 기존 y 값 유지
            else:
                y = 0.0  # 초기값
        
        self.robot_pose = (x, y, z)
    
    def set_destination(self, x: float, z: float):
        """
        목적지 설정 및 경로 초기화
        
        Args:
            x: 목적지 X 좌표
            z: 목적지 Z 좌표
        """
        x, z = self.config.clamp_world_xz(x, z)
        self.clear_path()
        self.destination = (x, z)
        
        # 상태 메시지 업데이트
        if self.seq == 2:
            self.status_message = f"⚔️ 전투 모드 (SEQ {self.seq}) - 목적지: ({x:.1f}, {z:.1f})"
        else:
            self.status_message = f"🚗 주행 중 (SEQ {self.seq}) → 목적지: ({x:.1f}, {z:.1f})"
    
    def clear_path(self):
        """경로 정보 초기화"""
        self.global_path = []
        self.valid_traj_count = 0
    
    def update_costmap(self, costmap, origin):
        """
        Costmap 업데이트 및 통계 계산
        
        Args:
            costmap: Numpy 배열 (H x W)
            origin: Costmap 원점 좌표 (x, z)
        """
        self.costmap = costmap
        self.costmap_origin = origin
        self.costmap_version += 1
        
        # 통계 계산
        if costmap is not None:
            import numpy as np
            total_cells = int(costmap.size)
            obstacle_cells = int(np.sum(costmap >= 1.0))
            self.costmap_stats = {
                "total_cells": total_cells,
                "obstacle_cells": obstacle_cells,
                "obstacle_ratio": float(obstacle_cells / max(total_cells, 1)),
                "shape": [int(costmap.shape[0]), int(costmap.shape[1])],
                "origin": [float(origin[0]), float(origin[1])] if origin is not None else None,
                "version": int(self.costmap_version),
            }
    
    # ==================================================================
    # Unity 정보 업데이트 메서드
    # ==================================================================
    def set_log(self, msg: str):
        self.last_log = msg
    
    def update_camera_turret_info(self, data: Dict[str, Any]):
        """
        Unity에서 받은 카메라/터렛 정보 업데이트
        
        Args:
            data: Unity에서 전송한 JSON 데이터
        """
        # JSON 키와 내부 변수 이름 매핑
        mapping = {
            'turretCameraPos': 'turret_camera_pos',
            'stereoCameraLeftPos': 'stereo_left_pos',
            'stereoCameraLeftRot': 'stereo_left_rot',
            'stereoCameraRightPos': 'stereo_right_pos',
            'stereoCameraRightRot': 'stereo_right_rot',
            'lidarRotation': 'lidar_rotation'
        }
        
        for json_key, attr_name in mapping.items():
            val = data.get(json_key)
            if val is not None and isinstance(val, dict):
                getattr(self, attr_name).update(val)
        
        if 'playerTurretX' in data:
            self.player_turret_x = data['playerTurretX']
    
    def get_camera_turret_dict(self) -> Dict[str, Any]:
        """
        카메라/터렛 정보를 딕셔너리로 반환
        
        Returns:
            카메라/터렛 정보가 담긴 딕셔너리
        """
        res = {}
        
        # 딕셔너리 형태의 데이터들을 x, y, z로 풀어서 저장
        target_groups = {
            'turretCam': self.turret_camera_pos,
            'camLeftPos': self.stereo_left_pos,
            'camLeftRot': self.stereo_left_rot,
            'camRightPos': self.stereo_right_pos,
            'camRightRot': self.stereo_right_rot,
            'lidarRot': self.lidar_rotation
        }
        
        for prefix, d in target_groups.items():
            res[f"{prefix}_x"] = d.get('x', 0.0)
            res[f"{prefix}_y"] = d.get('y', 0.0)
            res[f"{prefix}_z"] = d.get('z', 0.0)
        
        res['playerTurretX'] = self.player_turret_x
        return res
    
    # ==================================================================
    # 전역 장애물 맵 관리 메서드
    # ==================================================================
    
    def add_global_obstacles(self, x_or_point, z=None):
        """
        전역 장애물 맵에 장애물 추가
        
        메모리에 누적 저장하며, 최대 개수 제한 적용
        
        Args:
            x_or_point: X 좌표 또는 (x, z) 튜플/리스트
            z: Z 좌표 (x_or_point가 좌표인 경우)
        """
        try:
            if z is not None:
                points = [(x_or_point, z)]
            # 인자 처리
            elif isinstance(x_or_point, (list, tuple, set)):
                if not x_or_point: return
                first_elem = next(iter(x_or_point))
                if isinstance(first_elem, (int, float)):
                    points = [x_or_point]
                else:
                    points = x_or_point
            else:
                return
            
            for p in points:
                px, pz = p[0], p[-1]
                new_obs = (round(px, 1), round(pz, 1))
                # 중복 확인 후 추가
                if new_obs not in self.global_obstacles:
                    self.global_obstacles.append(new_obs)
                    self.global_obstacles_updated = True
                    
                    # 최대 개수 제한 (FIFO)
                    if len(self.global_obstacles) > self.MAX_GLOBAL_OBSTACLES:
                        self.global_obstacles.pop(0)
        except Exception as e:
            pass  # 조용히 실패 (로그 스팸 방지)
    
    def is_global_obstacle(self, x: float, z: float) -> bool:
        """
        특정 위치가 장애물 회피 반경 내에 있는지 확인
        
        Args:
            x: 확인할 X 좌표
            z: 확인할 Z 좌표
        
        Returns:
            bool: 장애물 회피 반경 내에 있으면 True
        """
        AVOID_RADIUS = 1.5  # 회피 반경 (m)
        
        for obs_x, obs_z in self.global_obstacles:
            dist = math.hypot(obs_x - x, obs_z - z)
            if dist < AVOID_RADIUS:
                return True
        return False
    
    def clear_global_obstacles(self):
        """전역 장애물 맵 초기화"""
        count = len(self.global_obstacles)
        self.global_obstacles = []
        print(f"🧹 전역 장애물 초기화: {count}개 삭제됨")
    
    def get_virtual_lidar_dist(self, curr_x: float, curr_z: float, 
                               max_range: float = 30.0) -> float:
        """
        현재 위치에서 가장 가까운 전역 장애물과의 거리 반환 (가상 LiDAR)
        
        Args:
            curr_x: 현재 X 좌표
            curr_z: 현재 Z 좌표
            max_range: 최대 탐지 거리 (m)
        
        Returns:
            float: 가장 가까운 장애물과의 거리 (m)
        """
        min_dist = max_range
        
        for obs_x, obs_z in self.global_obstacles:
            dist = math.hypot(obs_x - curr_x, obs_z - curr_z)
            if dist < min_dist:
                min_dist = dist
        
        return min_dist
    
    def get_min_obstacle_distance(self, x: float, z: float, max_range: float = 15.0) -> float:
        """
        특정 위치에서 가장 가까운 장애물까지의 거리 (DWA 근접 비용용)
        get_virtual_lidar_dist의 별칭
        """
        return self.get_virtual_lidar_dist(x, z, max_range)
    
    # ==================================================================
    # SCAN 타겟 관리 메서드 (SEQ 2)
    # ==================================================================
    
    def save_scan_targets(self, targets: List[Dict]):
        """
        SCAN 모드에서 탐지된 타겟 목록 저장
        
        Args:
            targets: 탐지된 타겟 리스트
        """
        self.last_scan_targets = targets.copy() if targets else []
        print(f"📋 SCAN 타겟 저장: {len(self.last_scan_targets)}개")
    
    def get_scan_targets_for_display(self) -> List[Dict]:
        """
        UI 표시용 SCAN 타겟 목록 반환
        
        locked 타겟에는 is_locked=True 플래그 추가
        잠금된 타겟을 리스트 최상단으로 정렬
        
        Returns:
            List[Dict]: 타겟 리스트 (각 타겟에 is_locked 필드 포함, 잠금된 타겟이 최상단)
        """
        result = []
        for target in self.last_scan_targets:
            t = target.copy()
            t['is_locked'] = self._is_target_locked(target)
            result.append(t)
        
        # 잠금된 타겟을 최상단으로 정렬
        # is_locked=True인 항목이 먼저 오도록 정렬 (True=1, False=0이므로 역순 정렬)
        result.sort(key=lambda x: x.get('is_locked', False), reverse=True)
        
        return result
    
    def _is_target_locked(self, target: Dict) -> bool:
        """
        타겟이 현재 locked 상태인지 확인 (엄격한 매칭)
        
        Args:
            target: 확인할 타겟 딕셔너리
        
        Returns:
            bool: True if locked, False otherwise
        """
        if not self.last_target:
            return False
        
        # 1. track_id로 매칭 (가장 정확 - 최우선)
        if (target.get('track_id') is not None and 
            self.last_target.get('track_id') is not None):
            return target['track_id'] == self.last_target['track_id']
        
        # 2. category 체크 (기본 필터)
        if target.get('category') != self.last_target.get('category'):
            return False
        
        # 3. confidence + bbox 조합 매칭 (AND 조건)
        conf_target = target.get('confidence')
        conf_last = self.last_target.get('confidence')
        
        if conf_target is None or conf_last is None:
            return False
        
        # Confidence 체크 (5% 이내)
        conf_diff = abs(conf_target - conf_last)
        if conf_diff >= 0.05:
            return False
        
        # Bbox 위치 체크 (50px 이내)
        t_bbox = target.get('bbox')
        l_bbox = self.last_target.get('bbox')
        
        if not (t_bbox and l_bbox):
            return False
        
        t_cx = (t_bbox[0] + t_bbox[2]) / 2
        t_cy = (t_bbox[1] + t_bbox[3]) / 2
        l_cx = (l_bbox[0] + l_bbox[2]) / 2
        l_cy = (l_bbox[1] + l_bbox[3]) / 2
        
        dist = math.hypot(t_cx - l_cx, t_cy - l_cy)
        
        # 둘 다 만족해야 True
        return dist < 50
    
    def clear_scan_targets(self):
        """SCAN 타겟 목록 초기화"""
        self.last_scan_targets = []
        print("🧹 SCAN 타겟 초기화")

    # ==================================================================
    # SEQ 4용 장애물 사각형 관리
    # ==================================================================
    
    def update_obstacle_rects(self, obstacles_data: list):
        """장애물 사각형 리스트 업데이트 (/update_obstacle에서 호출)
        
        Args:
            obstacles_data: [{'x_min', 'x_max', 'z_min', 'z_max'}, ...] 형태의 리스트
        """
        self.obstacle_rects = obstacles_data
        if len(obstacles_data) > 0:
            print(f"🗺️ 장애물 사각형 업데이트: {len(obstacles_data)}개")
    
    def get_obstacle_distance(self, x: float, z: float, obstacle_margin: float = 2.5) -> float:
        """특정 좌표에서 가장 가까운 장애물까지의 거리
        
        Args:
            x, z: 확인할 좌표
            obstacle_margin: 장애물 마진
            
        Returns:
            가장 가까운 장애물까지의 거리 (장애물이 없으면 float('inf'))
        """
        min_dist = float('inf')
        
        for obs in self.obstacle_rects:
            # 마진 적용된 장애물 경계
            x_min = obs['x_min'] - obstacle_margin
            x_max = obs['x_max'] + obstacle_margin
            z_min = obs['z_min'] - obstacle_margin
            z_max = obs['z_max'] + obstacle_margin
            
            # 점과 사각형 사이의 최단 거리
            dx = max(x_min - x, 0, x - x_max)
            dz = max(z_min - z, 0, z - z_max)
            dist = math.hypot(dx, dz)
            
            if dist < min_dist:
                min_dist = dist
        
        return min_dist
    
    def is_point_in_obstacle(self, x: float, z: float, obstacle_margin: float = 2.5) -> bool:
        """특정 좌표가 장애물(마진 포함) 내부인지 확인"""
        for obs in self.obstacle_rects:
            if (obs['x_min'] - obstacle_margin <= x <= obs['x_max'] + obstacle_margin and
                obs['z_min'] - obstacle_margin <= z <= obs['z_max'] + obstacle_margin):
                return True
        return False
    

    ##  0127 함수 추가 
    def parse_unity_combat_data(self, data: dict):
        """
        Unity 데이터에서 Turret 및 Body 각도를 안전하게 추출 (4중 안전장치)
        """
        turret = data.get("turret", {})
        pos = data.get("position", {}) 

        # 1. Turret 데이터 (X: Q/E 좌우, Y: R/F 상하)
        curr_tx = float(turret.get("x", 0))
        curr_ty = float(turret.get("y", 0))

        # 2. Body 데이터 안전하게 읽기 (4중 안전장치)
        body_yaw = 0.0
        body_pitch = 0.0
        
        if "rotationY" in pos: 
            body_yaw = float(pos["rotationY"])
        if "rotationX" in pos: 
            body_pitch = float(pos["rotationX"])
            
        if body_yaw == 0 and "rotationY" in data:
            body_yaw = float(data["rotationY"])
        if body_pitch == 0 and "rotationX" in data:
            body_pitch = float(data["rotationX"])

        if body_yaw == 0:
            rot = data.get("rotation", {})
            if rot:
                body_yaw = float(rot.get("y", 0))
                body_pitch = float(rot.get("x", 0))

        if body_yaw == 0:
            ppos = data.get("playerPos", {})
            if "rotationY" in ppos:
                body_yaw = float(ppos["rotationY"])
                body_pitch = float(ppos.get("rotationX", 0))

        return curr_tx, curr_ty, body_yaw, body_pitch

    def compute_precision_attack(self, curr_tx, curr_ty, curr_bx, curr_by):
            """
            정밀 조준 및 사격 명령 생성
            로직의 변경 없이 config 값만 precision_cfg에서 가져옵니다.
            """

            # 설정값 로드
            rel_target_x = self.target_yaw if hasattr(self, 'target_yaw') else precision_cfg.TARGET_YAW
            rel_target_y = self.target_pitch if hasattr(self, 'target_pitch') else precision_cfg.TARGET_PITCH
            tolerance = precision_cfg.TOLERANCE
            weight = precision_cfg.TURRET_WEIGHT
            
            abs_target_x = curr_bx + rel_target_x
            abs_target_y = curr_by + rel_target_y
            
            # 오차 및 조준 로직 (기존과 동일)
            err_x = (abs_target_x - curr_tx + 180) % 360 - 180
            err_y = (abs_target_y - curr_ty + 180) % 360 - 180

            command = {"turretQE": {"command": "", "weight": 0.0}, "turretRF": {"command": "", "weight": 0.0}, "fire": False}

            if abs(err_x) > tolerance:
                command["turretQE"] = {"command": "E" if err_x > 0 else "Q", "weight": weight}
                self.status_message = f"STEP 1: X축 조정 중"
                return command

            if abs(err_y) > tolerance:
                command["turretRF"] = {"command": "R" if err_y > 0 else "F", "weight": weight}
                self.status_message = f"STEP 2: Y축 조정 중"
                return command
            
            # 모든 조건 만족 시 발사
            command["fire"] = True
            self.auto_attack_active = False
            
            # [추가] 포격 후 자동 정렬 및 후퇴 시퀀스 트리거
            self.user_action = "RETREAT"
            self.combat_mode = "SCAN"
            
            # 이전 후퇴 기록 초기화
            if hasattr(self, 'retreat_aligned'):
                delattr(self, 'retreat_aligned')
            
            self.status_message = "💥 정밀 사격 완료!"
            return command
 
def handle_user_combat_action(self, action_name: str):
        """
        [통합 전투 액션 처리]
        기존의 FIRE, RESCAN, RETREAT 로직과 새로운 AUTO_ATTACK 로직을 모두 수용합니다.
        """
        action = action_name.upper()

        # 1. 새로운 기능: AUTO_ATTACK (정밀 조준 시퀀스 가동)
        if action == 'AUTO_ATTACK':
            self.auto_attack_active = True
            # 목표 각도 설정 (하드 코딩 대신 config의 설정값 사용)
            self.target_yaw = precision_cfg.TARGET_YAW   
            self.target_pitch = precision_cfg.TARGET_PITCH
            self.status_message = "🚀 지정 좌표 조준 시퀀스 가동"
            return True, "OK"

        # 2. 기존 기능: FIRE, RESCAN, RETREAT 처리
        self.user_action = action
        self.status_message = f"User Action Set: {action}"
        return True, "OK"