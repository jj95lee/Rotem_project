"""
LiDAR 파일 모니터링 및 Costmap 생성 + 카메라/터렛 정보 병합
+ 센서 퓨전 (3D→2D 투영, 거리 계산, 오버레이)
"""
import json, time, threading
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw

from models.lidar_frame import LidarFrame, gridify, fit_local_planes
from models.lidar_frame import compute_cell_features, build_costmap
from config import Config, fusion_cfg  

# ============================================================
# 센서 퓨전 유틸 함수들
# ============================================================

def build_intrinsic_from_fov(width: int, height: int, hfov_deg: float, vfov_deg: float) -> np.ndarray:
    """FOV로부터 카메라 내부 파라미터 행렬 생성"""
    hfov = np.deg2rad(hfov_deg)
    vfov = np.deg2rad(vfov_deg)

    fx = (width * 0.5) / np.tan(hfov * 0.5)
    fy = (height * 0.5) / np.tan(vfov * 0.5)
    cx = width * 0.5
    cy = height * 0.5

    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)


def euler_deg_to_R(rot_xyz_deg: Dict[str, float]) -> np.ndarray:
    """오일러 각도(도) → 회전 행렬 변환"""
    rx = np.deg2rad(float(rot_xyz_deg["x"]))
    ry = np.deg2rad(float(rot_xyz_deg["y"]))
    rz = np.deg2rad(float(rot_xyz_deg["z"]))

    cx, sx = np.cos(rx), np.sin(rx)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)

    cy, sy = np.cos(ry), np.sin(ry)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)

    cz, sz = np.cos(rz), np.sin(rz)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)

    return Rz @ Ry @ Rx


def project_world_to_image(
    Pw: np.ndarray,
    cam_pos: Dict[str, float],
    cam_rot: Dict[str, float],
    K: np.ndarray,
    width: int,
    height: int,
    original_distances: Optional[np.ndarray] = None,
    show_details: bool = True  # ← 디버깅 출력 여부
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """월드 좌표 3D 포인트를 카메라 이미지 2D로 투영 (디버깅 버전)"""
    
    C = np.array([cam_pos["x"], cam_pos["y"], cam_pos["z"]], dtype=np.float32)
    R_wc = euler_deg_to_R(cam_rot).astype(np.float32)
    
    d = (Pw - C[None, :]).astype(np.float32)
    right, up, forward = R_wc[:, 0], R_wc[:, 1], R_wc[:, 2]
    
    x_cam = d @ right
    y_cam = d @ up
    z_cam = d @ forward
    
    mask = z_cam > 0.1
    if not np.any(mask):
        return np.zeros((0, 2), dtype=np.int32), mask, []
    
    x = x_cam[mask] / z_cam[mask]
    y = -y_cam[mask] / z_cam[mask]
    
    u = (K[0, 0] * x + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * y + K[1, 2]).astype(np.int32)
    
    in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    uv = np.stack([u[in_img], v[in_img]], axis=1)
    
    final_mask = np.zeros(len(mask), dtype=bool)
    true_indices = np.where(mask)[0]
    final_mask[true_indices[in_img]] = True
    
    mapping_info = []
    
    # 투영된 포인트들의 원본 XYZ 추출
    projected_indices = np.where(final_mask)[0]
    
    for idx, proj_idx in enumerate(projected_indices):
        # 원본 3D 좌표
        original_xyz = Pw[proj_idx]
        
        # 투영된 2D 좌표
        pixel_uv = uv[idx]
        
        # 카메라 좌표계에서의 거리
        cam_distance = z_cam[mask][true_indices == proj_idx][0]

        # LiDAR 원본 거리 추출
        if original_distances is not None:
            lidar_dist = original_distances[proj_idx]
        else:
            lidar_dist = cam_distance
        
        mapping_info.append({
            'original_index': int(proj_idx),
            'world_x': float(original_xyz[0]),
            'world_y': float(original_xyz[1]),
            'world_z': float(original_xyz[2]),
            'pixel_u': int(pixel_uv[0]),
            'pixel_v': int(pixel_uv[1]),
            'camera_distance': float(cam_distance),
            'lidar_distance': float(lidar_dist)
        })
    
    return uv, final_mask, mapping_info


def get_distance_for_bboxes(
    detections: List[Dict],
    uv_points: Optional[np.ndarray],
    distances: Optional[np.ndarray],
    mapping_info: List[Dict],  # ← 추가
    w_img: int,
    h_img: int,
    tank_pos = (100.0, 10.0, 100.0),
    margin_px: int = 0
) -> List[Dict]:
    """bbox 내부 LiDAR 포인트로 거리 추정 (XYZ 정보 포함)"""
    
    if uv_points is None or len(uv_points) == 0:
        for det in detections:
            det["point_count"] = 0
            det["distance_m"] = None
            det["aim_uv"] = None
            det["matched_xyz"] = []  # ← XYZ 리스트
        return detections
    
    # mapping_info를 UV 좌표로 인덱싱할 수 있도록 딕셔너리 생성
    uv_to_xyz = {}
    for info in mapping_info:
        key = (info['pixel_u'], info['pixel_v'])
        uv_to_xyz[key] = (info['world_x'], info['world_y'], info['world_z'])
    
    # 전차 위치 언패킹
    tank_x, tank_y, tank_z = tank_pos
    
    for det in detections:
        xmin, ymin, xmax, ymax = det["bbox"]
        
        x1 = max(0, int(xmin) - int(margin_px))
        y1 = max(0, int(ymin) - int(margin_px))
        x2 = min(int(w_img) - 1, int(xmax) + int(margin_px))
        y2 = min(int(h_img) - 1, int(ymax) + int(margin_px))
        
        mask = (
            (uv_points[:, 0] >= x1) & (uv_points[:, 0] <= x2) &
            (uv_points[:, 1] >= y1) & (uv_points[:, 1] <= y2)
        )
        
        in_uv = uv_points[mask]
        in_box_distances = distances[mask]
        det["point_count"] = int(len(in_box_distances))
        
        # 매칭된 포인트들의 XYZ 정보 추출
        matched_xyz_list = []
        
        if det["point_count"] > 0:
            print(f"\n📦 [객체 내 센서 퓨전 포인트 - XYZ 포함 (총 {det['point_count']}개)]")
            print("-" * 80)
            
            for i, (u, v) in enumerate(in_uv):
                dist = in_box_distances[i]
                
                # UV로 원본 XYZ 찾기
                xyz = uv_to_xyz.get((int(u), int(v)), (None, None, None))
                matched_xyz_list.append({
                    'uv': [int(u), int(v)],
                    'xyz': list(xyz),
                    'distance': float(dist)
                })

                # 실제 전차 위치
                rel_dist = ((xyz[0] - tank_x)**2 + 
                        (xyz[1] - tank_y)**2 + 
                        (xyz[2] - tank_z)**2)**0.5
                
                print(f"  rel_XYZ=({abs(xyz[0] - tank_x):.2f}, "
                    f"{abs(xyz[1] - tank_y):.2f}, "
                    f"{abs(xyz[2] - tank_z):.2f}), "
                    f"Dist={rel_dist:.2f}m")
                
                print(f"  Point {i:2d}: UV=[{int(u):4d}, {int(v):4d}], XYZ=({xyz[0]:7.2f}, {xyz[1]:6.2f}, {xyz[2]:7.2f}), Dist={dist:.2f}m")
            
            # ============================================================
            # 높이 기반 필터링 추가 (지면 및 낮은 장애물 제거)
            # ============================================================
            MIN_HEIGHT_THRESHOLD = fusion_cfg.min_height_threshold # ✅ Config 값 사용
            height_mask = []
            
            for i, (u, v) in enumerate(in_uv):
                xyz = uv_to_xyz.get((int(u), int(v)), (None, None, None))
                # Y 좌표(xyz[1])가 설정값 이상인 포인트만 인덱스 저장
                if xyz[1] is not None and xyz[1] >= MIN_HEIGHT_THRESHOLD:
                    height_mask.append(i)
            
            if len(height_mask) > 0:
                # 높이 조건을 만족하는 포인트들 중에서 가장 가까운 거리 선택
                filtered_distances = in_box_distances[height_mask]
                filtered_uv = in_uv[height_mask]
                
                median_val = np.median(filtered_distances)
                min_i = int(np.argmin(np.abs(filtered_distances - median_val)))
                target_uv = filtered_uv[min_i]
                target_dist = filtered_distances[min_i]
                print(f"✅ 높이 필터 + 중앙값 로직 적용: {len(height_mask)}개 중 중앙값({median_val:.2f}m) 근접 포인트 선택")
            else:
                # 높이 조건을 만족하는 점이 없으면 전체 점 중에서 중앙값 적용
                median_val = np.median(in_box_distances)
                min_i = int(np.argmin(np.abs(in_box_distances - median_val)))
                
                target_uv = in_uv[min_i]
                target_dist = in_box_distances[min_i]
                print(f"⚠️ 높이 조건 미달로 전체 포인트 중 중앙값({median_val:.2f}m) 기반 선택")
            
            target_xyz = uv_to_xyz.get((int(target_uv[0]), int(target_uv[1])), (None, None, None))
            
            print(f"🎯 [포격 타겟 확정] "
                  f"UV={target_uv}, "
                  f"XYZ=({target_xyz[0]:.2f}, {target_xyz[1]:.2f}, {target_xyz[2]:.2f}), "
                  f"거리={target_dist:.2f}m\n")
            
            if det["point_count"] != 0:
                dist_est = float(rel_dist)
            else:
                dist_est = float(np.median(in_box_distances))
            
            det["distance_m"] = round(dist_est, 2)
            det["aim_uv"] = [int(target_uv[0]), int(target_uv[1])]
            det["matched_xyz"] = matched_xyz_list  # ← XYZ 정보 저장
            # 포격 타겟 확정 로그의 좌표를 담기 위해서 추가 
            det["position"] = {
            "x": round(float(target_xyz[0]), 2),
            "y": round(float(target_xyz[1]), 2),
            "z": round(float(target_xyz[2]), 2)
            }
        else:
            det["distance_m"] = None
            det["aim_uv"] = None
            det["matched_xyz"] = []
    
    return detections


def get_rainbow_color_smooth(distance: float, max_dist: float = 120.0) -> Tuple[int, int, int]:
    """
    거리에 따른 그레이스케일 색상 반환 (180m 기준)
    
    색상 범위:
    - 0m (가까움): 검은색 (0, 0, 0)
    - 150m (멀리): 흰색 (255, 255, 255)
    
    가까울수록 어둡고, 멀수록 밝아집니다.
    
    Args:
        distance: 거리 (미터)
        max_dist: 최대 거리 (기본값: 180m)
    
    Returns:
        (R, G, B) 튜플 (0-255 범위)
    """
    # 거리를 0-1 범위로 정규화
    ratio = min(distance / max_dist, 1.0)
    
    # 거리에 비례하여 밝기 증가 (0=검은색, 1=흰색)
    brightness = int(ratio * 255)
    
    return (brightness, brightness, brightness)


def draw_points_on_rgb(
    rgb_img: np.ndarray,
    uv: np.ndarray,
    distances: Optional[np.ndarray] = None,
    radius: int = 6,
    highlight_mask: Optional[np.ndarray] = None,
    highlight_radius: int = 12,
    highlight_color: Tuple[int, int, int] = (255, 255, 255)  # 흰색으로 변경
) -> np.ndarray:
    """
    거리별 색상:
    RGB 이미지에 LiDAR 포인트 그리기 (150m 그레이스케일)
    
    거리별 색상:
    - 0m (매우 가까움): ⚫ 검은색
    - 120m (매우 멀리): ⚪ 흰색
    - 중간 거리: 그레이 톤 (거리에 비례하여 밝아짐)
    
    Args:
        rgb_img: 원본 RGB 이미지
        uv: LiDAR 포인트 2D 좌표 [N, 2]
        distances: 각 포인트의 거리 [N] (미터)
        radius: 포인트 반경 (픽셀)
        highlight_mask: 강조 포인트 마스크
        highlight_radius: 강조 포인트 반경
        highlight_color: 강조 포인트 색상 (기본값: 흰색)
    
    Returns:
        포인트가 그려진 이미지
    """
    img = rgb_img.copy()
    if len(uv) == 0:
        return img

    # 🌈 거리에 따른 무지개 색상 생성 (180m 기준)
    if distances is not None and len(distances) == len(uv):
        max_dist = 120.0  # 50m → 180m로 변경
        colors = [get_rainbow_color_smooth(d, max_dist) for d in distances]
    else:
        colors = [(0, 255, 0)] * len(uv)

    # 일반 포인트 그리기
    for (u, v), color in zip(uv, colors):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = v + dy, u + dx
                if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]:
                    img[ny, nx] = color

    # 강조 포인트 그리기 (흰색)
    if highlight_mask is not None and len(highlight_mask) == len(uv):
        hv = uv[highlight_mask]
        for (u, v) in hv:
            for dy in range(-highlight_radius, highlight_radius + 1):
                for dx in range(-highlight_radius, highlight_radius + 1):
                    ny, nx = int(v) + dy, int(u) + dx
                    if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]:
                        img[ny, nx] = highlight_color

    return img


# YOLO로 감지된 적 탱크의 바운딩 박스 그리기 + LiDAR 정보 추가
def draw_lidar_association_boxes(
    rgb: np.ndarray,
    detections: list,
    box_color=(0, 255, 0),
    width: int = 4,
    fill_alpha: int = 50,
    show_label: bool = True
) -> np.ndarray:
    
    """LiDAR 매칭 객체 박스 시각화 (반투명 지원)"""
    base = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for det in detections:
        if int(det.get("point_count", 0)) <= 0:
            continue

        x1, y1, x2, y2 = map(int, det["bbox"])

        if fill_alpha > 0:
            draw.rectangle(
                [x1, y1, x2, y2],
                fill=(box_color[0], box_color[1], box_color[2], fill_alpha)
            )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(box_color[0], box_color[1], box_color[2], 255),
            width=width
        )

        if show_label:
            d = det.get("distance_m", None)
            pc = int(det.get("point_count", 0))
            text = f"LiDAR: {float(d):.1f}m  pts={pc}" if d else f"LiDAR: CAL  pts={pc}"
            ty = max(0, y1 - 16)
            draw.text((x1 + 4, ty), text, fill=(255, 255, 255, 255))

    out = Image.alpha_composite(base, overlay).convert("RGB")
    return np.array(out)


# ============================================================
# LidarLogger 클래스
# ============================================================

class LidarLogger:
    """LiDAR 파일 모니터링 및 처리 + 카메라/터렛 정보 병합"""
    
    def __init__(self, lidar_folder, file_pattern, state_manager, save_csv=False,
                 auto_cleanup_mode="after_process", max_files=10, max_age_sec=30.0,
                 costmap_inflation=Config.Terrain.COSTMAP_INFLATION):
        """
        Args:
            lidar_folder: LiDAR 파일이 저장되는 폴더 경로
            file_pattern: 파일 패턴 (예: "*.json")
            state_manager: StateManager 인스턴스
            save_csv: CSV 디버그 저장 여부
            auto_cleanup_mode: 자동 정리 모드
                - "none": 자동 정리 비활성화
                - "after_process": costmap 처리 완료 후 즉시 삭제 (기본값)
                - "keep_recent": 최신 N개 파일만 유지 (max_files 사용)
                - "max_age": 일정 시간이 지난 파일 삭제 (max_age_sec 사용)
            max_files: keep_recent 모드에서 유지할 최대 파일 개수
            max_age_sec: max_age 모드에서 파일 최대 수명 (초)
            costmap_inflation: Costmap 장애물 팽창 반경 (기본값: 5)
        """
        self.lidar_folder = lidar_folder
        self.file_pattern = file_pattern
        self.state = state_manager
        
        self.last_lidar_file = None
        self.last_lidar_mtime = 0
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = state_manager.config.Lidar.MONITOR_INTERVAL
        
        # 로컬 범위
        self.local_radius = state_manager.config.Lidar.LOCAL_RADIUS
        self.grid_size = state_manager.config.Lidar.GRID_SIZE
        
        # CSV 저장 옵션
        self.save_csv = save_csv
        self.csv_counter = 0
        
        # 센서 퓨전용: 최신 통합 DataFrame
        self.latest_merged_df = None
        
        # 🧹 자동 정리 옵션
        self.auto_cleanup_mode = auto_cleanup_mode
        self.max_files = max_files
        self.max_age_sec = max_age_sec
        self.processed_files = set()  # 처리된 파일 추적
        
        # 🗺️ Costmap 생성 제어 (SEQ 4에서만 필요)
        self.build_costmap_enabled = False
        self.costmap_inflation = costmap_inflation  # ⭐ inflation 설정
        
        Path(self.lidar_folder).mkdir(parents=True, exist_ok=True)
        
        # CSV 저장 폴더 생성
        if self.save_csv:
            self.csv_folder = Path(self.lidar_folder) / "csv_debug"
            self.csv_folder.mkdir(parents=True, exist_ok=True)
            print(f"📊 CSV 저장 모드 활성화: {self.csv_folder}")

        # 자동 정리 모드 출력
        if self.auto_cleanup_mode != "none":
            mode_desc = {
                "after_process": "처리 완료 후 즉시 삭제",
                "keep_recent": f"최신 {self.max_files}개 파일만 유지",
                "max_age": f"{self.max_age_sec}초 경과 파일 삭제"
            }
            print(f"🧹 자동 정리 모드: {mode_desc.get(self.auto_cleanup_mode, self.auto_cleanup_mode)}")

        # 시작 시 기존 LiDAR JSON 파일 삭제
        self._cleanup_old_data()
    
    def start(self):
        """모니터링 시작"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, 
                daemon=True
            )
            self.monitor_thread.start()
            print(f"📡 LiDAR 파일 모니터링 시작... 폴더: {self.lidar_folder}")
    
    def stop(self):
        """모니터링 중지"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def enable_costmap(self):
        """Costmap 생성 활성화 (SEQ 4에서 사용)"""
        self.build_costmap_enabled = True
        print("🗺️ Costmap 생성 활성화")
    
    def disable_costmap(self):
        """Costmap 생성 비활성화 (SEQ 2에서 센서퓨전만 사용)"""
        self.build_costmap_enabled = False
        print("🗺️ Costmap 생성 비활성화 (센서퓨전만 사용)")
    
    def _monitor_loop(self):
        """모니터링 루프 - Windows 파일 잠금 대응"""
        print("⏳ LiDAR 파일 대기 중...")
        
        consecutive_errors = 0
        max_errors = 10
        update_count = 0
        
        while self.monitoring_active:
            try:
                # Windows 파일 접근 오류 대응
                try:
                    lidar_files = list(Path(self.lidar_folder).glob(self.file_pattern))
                except (OSError, PermissionError) as e:
                    time.sleep(self.monitor_interval)
                    continue
                
                if not lidar_files:
                    if consecutive_errors == 0:
                        print("⏳ LiDAR 파일 대기 중...")
                    consecutive_errors += 1
                    time.sleep(self.monitor_interval)
                    continue
                
                if consecutive_errors > 0:
                    print(f"✅ LiDAR 파일 감지됨! ({len(lidar_files)}개)")
                    consecutive_errors = 0
                
                # 최신 파일 선택 (파일 접근 오류 대응)
                try:
                    latest_file = max(lidar_files, key=lambda p: p.stat().st_mtime)
                    file_mtime = latest_file.stat().st_mtime
                except (OSError, PermissionError):
                    # 파일 잠금 - 건너뛰고 다음 루프에서 재시도
                    time.sleep(self.monitor_interval)
                    continue
                
                # 새 파일인지 확인
                if latest_file == self.last_lidar_file and file_mtime == self.last_lidar_mtime:
                    time.sleep(self.monitor_interval)
                    continue
                
                # LiDAR 데이터 로드 (포인트만)
                lidar_points, timestamp = self._load_lidar(latest_file)
                
                if lidar_points is None or len(lidar_points) == 0:
                    time.sleep(self.monitor_interval)
                    continue
                
                self.last_lidar_file = latest_file
                self.last_lidar_mtime = file_mtime
                
                # LiDAR 데이터 처리 (robot_pose가 있을 때만)
                if self.state.robot_pose is not None:
                    self._process_lidar(lidar_points, timestamp, latest_file.name)
                    update_count += 1
                    
                    if update_count % 5 == 0:
                        if self.build_costmap_enabled and self.state.costmap is not None:
                            print(f"🗺️ Costmap #{update_count}: {latest_file.name} → {self.state.costmap.shape}")
                        else:
                            print(f"📡 LiDAR #{update_count}: {latest_file.name} (센서퓨전 모드)")
                
            except (OSError, PermissionError) as e:
                # Windows WinError 32 등 파일 접근 오류
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    print(f"⚠️ 파일 접근 대기 중... ({consecutive_errors}/3)")
                time.sleep(self.monitor_interval * 2)  # 더 오래 대기
                
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= max_errors:
                    print(f"❌ 모니터링 오류 ({consecutive_errors}/{max_errors}): {e}")
            
            time.sleep(self.monitor_interval)
    
    def _load_lidar(self, filepath):
        """LiDAR JSON 파일 로드 (포인트 데이터만 추출) - Windows 파일 잠금 대응"""
        max_retries = self.state.config.Lidar.LIDAR_MAX_RETRIES
        
        for attempt in range(max_retries):
            try:
                # 파일 존재 확인
                if not filepath.exists():
                    return None, None
                
                # 파일 크기 확인 (쓰기 중인지 체크)
                try:
                    size1 = filepath.stat().st_size
                    time.sleep(0.08)  # 약간 더 길게 대기
                    size2 = filepath.stat().st_size
                except (OSError, PermissionError):
                    # Windows 파일 잠금 - 다음 시도
                    if attempt < max_retries - 1:
                        time.sleep(0.2)
                        continue
                    return None, None
                
                if size1 != size2 or size1 == 0:
                    if attempt < max_retries - 1:
                        time.sleep(0.15)
                        continue
                    else:
                        return None, None
                
                # JSON 로드 (Windows 파일 잠금 대응)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    data = json.loads(content)
                except (PermissionError, OSError) as e:
                    # WinError 32: 다른 프로세스가 파일 사용 중
                    if attempt < max_retries - 1:
                        time.sleep(0.2)
                        continue
                    return None, None
                except json.JSONDecodeError:
                    # 불완전한 JSON (쓰기 중)
                    if attempt < max_retries - 1:
                        time.sleep(0.15)
                        continue
                    return None, None
                
                # 포인트 데이터 추출
                if isinstance(data, dict):
                    lidar_points = data.get('data')
                    timestamp = time.time()
                elif isinstance(data, list):
                    lidar_points = data
                    timestamp = time.time()
                else:
                    return None, None
                
                if not lidar_points:
                    return None, None
                
                # 포인트 정규화
                normalized_points = []
                for pt in lidar_points:
                    if 'position' in pt:
                        normalized_points.append(pt)
                    elif 'x' in pt and 'y' in pt and 'z' in pt:
                        normalized_points.append({
                            'angle': pt.get('angle', 0),
                            'verticalAngle': pt.get('verticalAngle', 0),
                            'distance': pt.get('distance', 0),
                            'position': {
                                'x': pt['x'],
                                'y': pt['y'],
                                'z': pt['z']
                            },
                            'channelIndex': pt.get('ringID', pt.get('channelIndex', 0)),
                            'isDetected': pt.get('isDetected', True)
                        })
                
                if len(normalized_points) == 0:
                    return None, None
                
                return normalized_points, timestamp
                
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(0.15)
                    continue
                else:
                    return None, None
        
        return None, None
    
    def _process_lidar(self, lidar_points, timestamp, filename):
        """LiDAR 데이터와 StateManager의 최신 정보(카메라/터렛) 통합 및 Costmap 생성"""
        # 1. 현재 전차 위치 가져오기
        if self.state.robot_pose is None:
            print(f"⚠️ [LIDAR] robot_pose가 None입니다! /info가 호출되지 않음")
            return
        
        # robot_pose = (x,y,z)
        cx, cy, cz = self.state.robot_pose[0], self.state.robot_pose[1], self.state.robot_pose[2]
        
        # 2. LiDAR 데이터프레임 생성
        lf = LidarFrame(lidar_points, timestamp)
        lidar_df = lf.to_dataframe()
        
        # 3. StateManager에서 통합 데이터(카메라/터렛/회전 등) 가져오기
        # /info 엔드포인트로 들어온 최신 정보가 여기에 포함됩니다.
        integrated_info = self.state.get_camera_turret_dict()

        # 4. 데이터프레임에 각 필드 주입
        for key, value in integrated_info.items():
            lidar_df[key] = value

        # 전차의 현재 위치 DataFrame에 추가
        lidar_df['tank_x'] = cx
        lidar_df['tank_y'] = cy
        lidar_df['tank_z'] = cz
        
        # 센서 퓨전용: 통합된 DataFrame 저장 (로컬 필터링 전)
        self.latest_merged_df = lidar_df.copy()
        
        # 🔍 CSV 저장 및 디버깅
        if self.save_csv:
            csv_path = self.csv_folder / f"step1_merged_{self.csv_counter:04d}.csv"
            lidar_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # 10번마다 한 번씩 병합 상태 확인
            if self.csv_counter % 10 == 0:
                print(f"✅ [파일 #{self.csv_counter}] 데이터 병합 확인:")
                print(f"   - turretCam: ({lidar_df['turretCam_x'].iloc[0]:.2f}, {lidar_df['turretCam_y'].iloc[0]:.2f}, {lidar_df['turretCam_z'].iloc[0]:.2f})")
                print(f"   - playerTurretX: {lidar_df['playerTurretX'].iloc[0]:.2f}")
            
            self.csv_counter += 1
            
        # 5. 로컬 영역 필터링
        lidar_df = lidar_df[
            (lidar_df['x'] - cx)**2 + (lidar_df['z'] - cz)**2 < self.local_radius**2
        ].copy()
        
        if len(lidar_df) < 50:
            return
        
        # 6. 지형 분석 및 Costmap 업데이트 (build_costmap_enabled일 때만)
        if self.build_costmap_enabled:
            lidar_df = gridify(lidar_df, grid_size=self.grid_size)
            lidar_df = fit_local_planes(lidar_df)
            cell_df = compute_cell_features(lidar_df)
            
            if len(cell_df) > 0:
                costmap, origin = build_costmap(cell_df, inflation=self.costmap_inflation)
                
                # 디버깅: 전차 위치와 costmap 범위 비교
                cm_min_x, cm_min_z = origin
                cm_max_x = cm_min_x + costmap.shape[1] * self.grid_size
                cm_max_z = cm_min_z + costmap.shape[0] * self.grid_size
                in_range = (cm_min_x <= cx <= cm_max_x) and (cm_min_z <= cz <= cm_max_z)
                
                if not in_range:
                    print(f"⚠️ [COSTMAP] 전차가 범위 밖! 전차=({cx:.1f}, {cz:.1f}), "
                          f"Costmap=({cm_min_x:.0f}~{cm_max_x:.0f}, {cm_min_z:.0f}~{cm_max_z:.0f})")
                
                self.state.update_costmap(costmap, origin)
                
                # 전역 장애물 맵에 누적
                self._accumulate_global_obstacles(costmap, origin)
        
        # 7. 🧹 처리 완료 후 자동 정리
        self._auto_cleanup_after_process(filename)
    
    def _accumulate_global_obstacles(self, costmap, origin):
        """Costmap에서 장애물을 추출하여 전역 장애물 맵에 누적
        
        Args:
            costmap: 2D numpy array (cost values)
            origin: (min_x, min_z) costmap 원점
        """
        import numpy as np
        
        # 장애물 임계값
        obstacle_threshold = 1.0
        
        # 전역 장애물 그리드 크기
        global_grid_size = self.state.global_obstacle_grid_size
        
        # 장애물 셀 찾기
        obstacle_indices = np.where(costmap >= obstacle_threshold)
        
        new_obstacles = set()
        for iz, ix in zip(obstacle_indices[0], obstacle_indices[1]):
            # Costmap 인덱스 → 월드 좌표
            world_x = origin[0] + ix * self.grid_size
            world_z = origin[1] + iz * self.grid_size
            
            # 전역 그리드로 스냅 (중복 방지)
            grid_x = int(world_x / global_grid_size) * global_grid_size
            grid_z = int(world_z / global_grid_size) * global_grid_size
            
            new_obstacles.add((grid_x, grid_z))
        
        # 전역 장애물 맵에 추가
        if new_obstacles:
            self.state.add_global_obstacles(new_obstacles)
    
    def _auto_cleanup_after_process(self, processed_filename):
        """처리 완료 후 자동 정리 (auto_cleanup_mode에 따라 동작)"""
        if self.auto_cleanup_mode == "none":
            return
        
        p = Path(self.lidar_folder)
        
        if self.auto_cleanup_mode == "after_process":
            # 처리 완료된 파일 즉시 삭제
            self._delete_processed_file(processed_filename)
            
        elif self.auto_cleanup_mode == "keep_recent":
            # 최신 N개 파일만 유지
            self._keep_recent_files()
            
        elif self.auto_cleanup_mode == "max_age":
            # 일정 시간 경과 파일 삭제
            self._delete_old_files()
    
    def _delete_processed_file(self, filename):
        """처리 완료된 특정 파일 삭제"""
        try:
            filepath = Path(self.lidar_folder) / filename
            if filepath.exists():
                filepath.unlink()
                self.processed_files.add(filename)
        except Exception as e:
            pass  # 삭제 실패는 무시 (다음 정리 시 재시도됨)
    
    def _keep_recent_files(self):
        """최신 N개 파일만 유지하고 나머지 삭제 - Windows 파일 잠금 대응"""
        try:
            p = Path(self.lidar_folder)
            
            # 파일 목록 가져오기 (접근 오류 대응)
            try:
                lidar_files = list(p.glob(self.file_pattern))
            except (OSError, PermissionError):
                return
            
            if len(lidar_files) <= self.max_files:
                return
            
            # 수정 시간 기준 정렬 (오래된 것 먼저) - 접근 오류 대응
            try:
                sorted_files = sorted(lidar_files, key=lambda f: f.stat().st_mtime)
            except (OSError, PermissionError):
                return
            
            # 오래된 파일 삭제 (최신 max_files개 유지)
            files_to_delete = sorted_files[:-self.max_files]
            deleted_count = 0
            
            for f in files_to_delete:
                try:
                    f.unlink()
                    deleted_count += 1
                except (OSError, PermissionError):
                    # Windows 파일 잠금 - 다음에 재시도
                    pass
                except:
                    pass
            
            if deleted_count > 0 and deleted_count % 10 == 0:
                print(f"🧹 정리: {deleted_count}개 파일 삭제 (최신 {self.max_files}개 유지)")
                
        except Exception as e:
            pass
    
    def _delete_old_files(self):
        """지정된 시간(max_age_sec)보다 오래된 파일 삭제"""
        try:
            p = Path(self.lidar_folder)
            lidar_files = list(p.glob(self.file_pattern))
            
            now = time.time()
            deleted_count = 0
            
            for f in lidar_files:
                try:
                    file_age = now - f.stat().st_mtime
                    if file_age > self.max_age_sec:
                        f.unlink()
                        deleted_count += 1
                except:
                    pass
            
            if deleted_count > 0 and deleted_count % 10 == 0:
                print(f"🧹 정리: {deleted_count}개 오래된 파일 삭제 (>{self.max_age_sec}초)")
                
        except Exception as e:
            pass
    
    def set_cleanup_mode(self, mode, max_files=None, max_age_sec=None):
        """런타임에서 자동 정리 모드 변경
        
        Args:
            mode: "none", "after_process", "keep_recent", "max_age"
            max_files: keep_recent 모드에서 유지할 파일 수
            max_age_sec: max_age 모드에서 최대 파일 수명
        """
        valid_modes = ["none", "after_process", "keep_recent", "max_age"]
        if mode not in valid_modes:
            print(f"⚠️ 유효하지 않은 모드: {mode}. 사용 가능: {valid_modes}")
            return
        
        self.auto_cleanup_mode = mode
        
        if max_files is not None:
            self.max_files = max_files
        if max_age_sec is not None:
            self.max_age_sec = max_age_sec
        
        mode_desc = {
            "none": "자동 정리 비활성화",
            "after_process": "처리 완료 후 즉시 삭제",
            "keep_recent": f"최신 {self.max_files}개 파일만 유지",
            "max_age": f"{self.max_age_sec}초 경과 파일 삭제"
        }
        print(f"🧹 정리 모드 변경: {mode_desc[mode]}")
    
    def force_cleanup(self):
        """강제로 모든 LiDAR 파일 정리"""
        self._cleanup_old_data()
    
    def _cleanup_old_data(self):
        """실행 시 lidar_data 폴더 내의 이전 세션 JSON 파일 및 글로벌 경로 이미지를 정리합니다."""
        p = Path(self.lidar_folder)
        
        # 모든 파일 삭제
        try:
            # 폴더 내의 모든 파일 및 디렉토리 항목을 가져옵니다.
            all_items = list(p.glob("*")) 
            
            if all_items:
                print(f"🧹 LiDAR 폴더 전체 정리 중... (대상: {len(all_items)}개)")
                for item in all_items:
                    if item.is_file(): # 파일인 경우만 삭제
                        item.unlink()
                    elif item.is_dir(): # 혹시 하위 폴더가 있다면 (필요 시 삭제)
                        import shutil
                        shutil.rmtree(item)
                print(f"✅ 모든 데이터가 완전히 삭제되었습니다.")
            else:
                print("✨ LiDAR 폴더가 이미 비어 있습니다.")
        except Exception as e:
            print(f"⚠️ 폴더 정리 중 오류 발생: {e}")
        
        try:
            # 현재 작업 디렉토리 기준
            root_path = Path('.')
            target_images = ["SEQ 1_Global_Path.png", "SEQ 3_Global_Path.png"]

            for image in target_images:
                image_file = root_path / image
                if image_file.exists():
                    image_file.unlink()
                    print(f"🗑️ 기존 경로 이미지 삭제됨: {image}")
        except Exception as e:
            print(f"⚠️ 이미지 삭제 중 오류 발생: {e}")
    
    def get_latest_dataframe(self):
        """메모리에서 최신 통합 LiDAR 데이터프레임 가져오기 (센서 퓨전용)"""
        if self.latest_merged_df is None:
            return None
        
        df = self.latest_merged_df.copy()
        
        # 필수 컬럼 확인
        required_cols = ['x', 'y', 'z', 'distance', 'isDetected']
        if not all(col in df.columns for col in required_cols):
            print(f"[LIDAR_DF] ❌ 필수 컬럼 누락: {required_cols}")
            return None
        
        # isDetected == True인 포인트만
        df = df[df['isDetected'] == True].copy()
        
        return df