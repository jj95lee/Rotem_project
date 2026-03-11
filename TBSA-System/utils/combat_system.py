"""
combat_system.py - SEQ 2 전투 시스템 핵심 로직

[모듈 구조]
├── 1. 기본 유틸리티 (Basic Utilities)
│   └── ensure_attr, _normalize_lidar_points, detect_all_objects_dual
│
├── 2. BBox 유틸리티 (Bounding Box Utilities)  
│   └── _iou, _clip_bbox, _expand_bbox, _shift_bbox, hex_to_rgb, make_det_overlay_bytes
│
├── 3. 좌표/각도 변환 (Coordinate & Angle Conversion)
│   └── lidar_to_cartesian, calculate_angle_from_bbox, _calc_pitch_offset_deg
│
├── 4. LiDAR 타겟 잠금 (LiDAR Target Lock)
│   └── lock_lidar_target, unlock_lidar_target, unlock_all_combat_locks
│   └── update_lidar_locked_target, get_lidar_target_info
│   └── find_lidar_points_in_angle_range
│
├── 5. YOLO 타겟 선택 (YOLO Target Selection)
│   └── select_best_target, calculate_aim_errors
│
├── 6. ROI 기반 타겟 추적 (ROI-based Target Tracking)
│   └── update_locked_bbox_by_roi_yolo, predict_bbox_by_cam_delta
│
└── 7. 전투 액션 계산 (Combat Action Computation)
    └── compute_combat_action

[사용처]
- app.py의 /get_action 엔드포인트에서 SEQ 2 전투 모드 시 호출
- app.py의 /detect 엔드포인트에서 타겟 선택/조준 오차 계산에 사용
"""

import time
import math
import io
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from PIL import Image, ImageDraw, ImageFont

from config import (
    pitch_cfg, smooth_cfg, fusion_cfg, aim_cfg, combat_config,
    turret_cfg, offset_cfg, lock_cfg, camera_cfg ,precision_cfg
)


# ==============================================================================
# 전역 변수 (Thread-safe)
# ==============================================================================
yolo_track_lock = threading.Lock()
lidar_lock = threading.Lock()  # LiDAR 관련 상태 접근 시 사용
pose_buffer = deque(maxlen=fusion_cfg.pose_buffer_maxlen)  # 카메라 포즈 버퍼


# ==============================================================================
# 1. 기본 유틸리티 (Basic Utilities)
# ==============================================================================

def ensure_attr(obj, name: str, default):
    """
    객체에 속성이 없으면 기본값으로 생성
    
    Args:
        obj: 대상 객체 (보통 state_manager)
        name: 속성 이름
        default: 기본값
    
    Example:
        ensure_attr(state, "hit_count", 0)
    """
    if not hasattr(obj, name):
        setattr(obj, name, default)


def _normalize_lidar_points(lidar_pts_raw):
    """
    LiDAR 포인트 데이터 정규화
    
    - 유효한 포인트만 필터링
    - angle/verticalAngle/distance 형식 또는 position 형식 지원
    
    Args:
        lidar_pts_raw: 원시 LiDAR 포인트 리스트
    
    Returns:
        list: 정규화된 LiDAR 포인트 리스트
    """
    if not lidar_pts_raw:
        return []
    
    normalized = []
    for pt in lidar_pts_raw:
        if isinstance(pt, dict):
            # 각도 기반 형식
            if 'angle' in pt and 'verticalAngle' in pt and 'distance' in pt:
                normalized.append(pt)
            # 좌표 기반 형식
            elif 'position' in pt:
                pos = pt['position']
                if isinstance(pos, dict) and all(k in pos for k in ('x', 'y', 'z')):
                    normalized.append(pt)
    
    return normalized

def detect_all_objects_dual(
    image_input,
    model_cannon,
    model_integrated,
    combat_config,
    fusion_cfg,
    nms_iou_th: float = 0.5,
    use_onnx: bool = False,
):
    '''
    Args:
        img_input: 파일 경로 (str) 또는 PIL Image 객체
        model_cannon: Cannon 모델 (YOLO 또는 OnnxYoloDetector)
        model_integrated: 통합 모델 (YOLO 또는 OnnxYoloDetector)
        combat_config: CombatSystemConfig 인스턴스
        fusion_cfg: FusionConfig 인스턴스
        nms_iou_th: NMS IoU 임계값
        use_onnx: True면 ONNX 모드, False면 PyTorch 모드
    '''

    if isinstance(image_input, str):
        # 파일 경로인 경우
        img_pil = Image.open(image_input).convert("RGB")
        image_path = image_input
    elif isinstance(image_input, Image.Image):
        # PIL Image인 경우
        img_pil = image_input.convert("RGB") if image_input.mode != "RGB" else image_input
        image_path = None
    else:
        raise ValueError(f"img_input must be str or PIL.Image, got {type(image_input)}")
    
    temp_detections = []

    if use_onnx:
        # ═══════════════════════════════════════════════════════════════
        # ONNX 모드: OnnxYoloDetector 사용
        # ═══════════════════════════════════════════════════════════════
        model_configs = [
            {"detector": model_cannon, "mapping": combat_config.map_cannon, "color": combat_config.color_cannon},
            {"detector": model_integrated, "mapping": combat_config.map_integrated, "color": combat_config.color_integrated},
        ]

        for cfg in model_configs:
            # ONNX 추론
            detections = cfg["detector"].detect(
                img_pil,
                conf_threshold=fusion_cfg.min_det_conf,
                iou_threshold=0.45
            )

            for det in detections:
                class_id = det["class_id"]
                
                # 매핑 확인
                if class_id not in cfg["mapping"]:
                    continue

                class_name = cfg["mapping"][class_id]
                bbox = det["bbox"]
                xmin, ymin, xmax, ymax = bbox
                
                # bbox 크기 필터
                if (xmax - xmin) < fusion_cfg.min_box_w or (ymax - ymin) < fusion_cfg.min_box_h:
                    continue

                temp_detections.append({
                    "bbox": bbox,
                    "confidence": det["confidence"],
                    "class_name": class_name,
                    "color": cfg["color"],
                })
    else:
        # ═══════════════════════════════════════════════════════════════
        # PyTorch 모드: ultralytics YOLO 사용
        # ═══════════════════════════════════════════════════════════════
        model_configs = [
            {"model": model_cannon, "mapping": combat_config.map_cannon, "color": combat_config.color_cannon},
            {"model": model_integrated, "mapping": combat_config.map_integrated, "color": combat_config.color_integrated},
        ]

        for cfg in model_configs:
            # YOLO 추론
            results = cfg["model"](image_input, conf=fusion_cfg.min_det_conf, verbose=False)
            detections = results[0].boxes.data.cpu().numpy()

            for box in detections:
                # Box 길이에 따라 Tracking 모드 판단
                box_len = len(box)
                
                if box_len == 7:  # Tracking 활성화 상태
                    xmin, ymin, xmax, ymax = [float(x) for x in box[:4]]
                    track_id = int(box[4])
                    confidence = float(box[5])
                    class_id = int(box[6])
                elif box_len == 6:  # 일반 탐지
                    xmin, ymin, xmax, ymax = [float(x) for x in box[:4]]
                    confidence = float(box[4])
                    class_id = int(box[5])
                else:
                    continue
                
                # 매핑 확인
                if class_id not in cfg["mapping"]:
                    continue

                class_name = cfg["mapping"][class_id]
                
                # bbox 크기 필터
                if (xmax - xmin) < fusion_cfg.min_box_w or (ymax - ymin) < fusion_cfg.min_box_h:
                    continue

                # 탐지 추가
                temp_detections.append({
                    "bbox": [xmin, ymin, xmax, ymax],
                    "confidence": confidence,
                    "class_name": class_name,
                    "color": cfg["color"],
                })
                
    # NMS (confidence 높은 순으로 IoU overlap 제거)
    temp_detections.sort(key=lambda x: x["confidence"], reverse=True)
    final_detections = []
    for cur in temp_detections:
        overlapped = False
        for kept in final_detections:
            if _iou(cur["bbox"], kept["bbox"]) > nms_iou_th:
                overlapped = True
                break
        if not overlapped:
            final_detections.append(cur)

    # 결과 가공 (UI 스키마)
    filtered_results = []
    tank_count = 0
    red_count = 0
    last_cannon_bbox = None
    
    # ═══════════════════════════════════════════════════════════════
    # 🎨 bbox 오버레이 커스터마이징 설정
    # ═══════════════════════════════════════════════════════════════
    bbox_styles = {
        "Tank": {
            "color": "#FF0000",      # 빨간색
            "filled": True,          # 반투명 채우기
            "show_confidence": True, # 신뢰도 표시
        },
        "Red": {
            "color": "#FF4444",      # 밝은 빨간색
            "filled": True,
            "show_confidence": True,
        },
        "Tree": {
            "color": "#AAAAAA",      # 회색
            "filled": True,         # 테두리만
            "show_confidence": False,
        },
        "Rock": {
            "color": "#AAAAAA",      # 회색
            "filled": True,
            "show_confidence": False,
        },
        "default": {
            "color": "#FFFFFF",      # 흰색 (기본값)
            "filled": True,
            "show_confidence": False,
        }
    }

    for det in final_detections:
        name = det["class_name"]
        conf = det["confidence"]

        if name == "Tank":
            tank_count += 1
        elif name == "Red":
            red_count += 1
        elif name == "Cannon":
            last_cannon_bbox = det["bbox"]
            continue  # Cannon은 그리지 않음


        # 스타일 가져오기
        style = bbox_styles.get(name, bbox_styles["default"])

        filtered_results.append({
            "className": name,              # 이름만 표시
            "category": name.lower(),
            "bbox": det["bbox"],
            "confidence": conf,
            "color": style["color"],
            "filled": style["filled"],
            "updateBoxWhileMoving": False,
        })

    meta = {
        "tank_count": tank_count,
        "red_count": red_count,
        "final_detections": final_detections,
        "last_cannon_bbox": last_cannon_bbox,
    }
    
    return filtered_results, meta

# ==============================================================================
# 2. BBox 유틸리티 (Bounding Box Utilities)
# ==============================================================================

def _iou(bbox_a: List[float], bbox_b: List[float]) -> float:
    """
    두 bounding box의 IoU (Intersection over Union) 계산
    
    Args:
        bbox_a: [xmin, ymin, xmax, ymax]
        bbox_b: [xmin, ymin, xmax, ymax]
    
    Returns:
        float: IoU 값 (0.0 ~ 1.0)
    """
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    
    # 교집합 영역
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    
    # 합집합 영역
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    
    return (inter / union) if union > 0 else 0.0


def _clip_bbox(bbox, w, h):
    """
    bbox를 이미지 범위 내로 클리핑
    
    Args:
        bbox: [x1, y1, x2, y2]
        w: 이미지 너비
        h: 이미지 높이
    
    Returns:
        list: 클리핑된 [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(x1), w - 1))
    y1 = max(0.0, min(float(y1), h - 1))
    x2 = max(0.0, min(float(x2), w - 1))
    y2 = max(0.0, min(float(y2), h - 1))
    
    # 최소 크기 보장
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    
    return [x1, y1, x2, y2]


def _expand_bbox(bbox, w, h, pad_px):
    """
    bbox를 지정된 픽셀만큼 확장 (ROI 영역 생성용)
    
    Args:
        bbox: [x1, y1, x2, y2]
        w: 이미지 너비
        h: 이미지 높이
        pad_px: 확장할 픽셀 수
    
    Returns:
        list: 확장된 [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    return _clip_bbox([x1 - pad_px, y1 - pad_px, x2 + pad_px, y2 + pad_px], w, h)


def _shift_bbox(bbox, dx, dy, w, h):
    """
    bbox를 지정된 픽셀만큼 이동
    
    Args:
        bbox: [x1, y1, x2, y2]
        dx: X축 이동량 (픽셀)
        dy: Y축 이동량 (픽셀)
        w: 이미지 너비
        h: 이미지 높이
    
    Returns:
        list: 이동된 [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, x1 + dx))
    x2 = max(0, min(w - 1, x2 + dx))
    y1 = max(0, min(h - 1, y1 + dy))
    y2 = max(0, min(h - 1, y2 + dy))
    
    # 뒤집힘 방지
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    
    return [float(x1), float(y1), float(x2), float(y2)]

def hex_to_rgb(hex_color: str, default=(0, 255, 0)):
    '''
    HEX 색상을 RGB로 변환
    
    Args:
        hex_color: 색상 값
        default: 변환할 색상
    '''
    try:
        if isinstance(hex_color, str) and hex_color.startswith("#") and len(hex_color) == 7:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            return (r, g, b)
    except Exception:
        pass
    return default

def make_det_overlay_bytes(img_pil: Image.Image, dets: list, target_bbox=None, target_iou_th=0.5):
    """
    dets: [{'bbox':[xmin,ymin,xmax,ymax], 'color':'#RRGGBB', ...}, ...]
    """
    img = img_pil.copy()
    draw = ImageDraw.Draw(img, "RGBA")

    # 폰트 로드 (설정된 크기 사용)
    try:
        # 윈도우/리눅스 환경에 따라 폰트 경로가 다를 수 있음. 기본 폰트나 시스템 폰트 활용
        font = ImageFont.truetype(combat_config.overlay_font_path, combat_config.overlay_font_size)
    except IOError:
        # 폰트 파일이 없으면 기본 폰트 사용
        font = ImageFont.load_default()

    for d in dets:
        xmin, ymin, xmax, ymax = d["bbox"]
        rgb = hex_to_rgb(d.get("color", "#FFFFFF"), default=(255, 255, 255))

        # 타겟 박스는 더 두껍게/채우기
        is_target = (target_bbox is not None and _iou(d["bbox"], target_bbox) > target_iou_th)
        width = 8 if is_target else 2
        fill = (rgb[0], rgb[1], rgb[2], 70) if is_target else None

        if fill is not None:
            draw.rectangle([xmin, ymin, xmax, ymax], outline=rgb, width=width, fill=fill)
        else:
            draw.rectangle([xmin, ymin, xmax, ymax], outline=rgb, width=width)
        label = d.get("className", d.get("class_name", "Unknown"))
        # 텍스트 배경 (가독성 확보)
        text_bbox = draw.textbbox((xmin, ymin), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        # 텍스트가 박스 위에 위치하도록 조정
        text_origin = (xmin, max(0, ymin - text_h - 4))
        
        # 텍스트 배경 사각형 (색상과 동일하게, 반투명)
        draw.rectangle(
            [text_origin[0], text_origin[1], text_origin[0] + text_w + 4, text_origin[1] + text_h + 4],
            fill=(rgb[0], rgb[1], rgb[2], 180) 
        )
        
        # 텍스트 쓰기 (흰색)
        draw.text((text_origin[0] + 2, text_origin[1] + 2), label, fill=(255, 255, 255), font=font)
        
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# ==============================================================================
# 3. 좌표/각도 변환 (Coordinate & Angle Conversion)
# ==============================================================================

def lidar_to_cartesian(angle_deg: float, vertical_angle_deg: float, distance: float) -> Tuple[float, float, float]:
    """
    LiDAR 구면 좌표 → 직교 좌표(XYZ) 변환
    
    Args:
        angle_deg: 수평 각도 (도)
        vertical_angle_deg: 수직 각도 (도)
        distance: 거리 (m)
    
    Returns:
        Tuple[x, y, z]: 직교 좌표
    """
    theta = np.deg2rad(angle_deg)
    phi = np.deg2rad(vertical_angle_deg)
    
    x = distance * np.cos(phi) * np.sin(theta)
    y = distance * np.sin(phi)
    z = distance * np.cos(phi) * np.cos(theta)
    
    return float(x), float(y), float(z)


def calculate_angle_from_bbox(bbox: List[float], w_img: int, h_img: int) -> Tuple[float, float]:
    """
    bbox 중심 좌표로부터 yaw/pitch 각도 계산
    
    Args:
        bbox: [xmin, ymin, xmax, ymax]
        w_img: 이미지 너비
        h_img: 이미지 높이
    
    Returns:
        Tuple[yaw_angle, pitch_angle]: 각도 (도)
    """
    xmin, ymin, xmax, ymax = bbox
    bbox_center_x = (xmin + xmax) / 2
    bbox_center_y = (ymin + ymax) / 2
    
    # 이미지 중심 기준 정규화 (-1 ~ 1)
    cx, cy = w_img / 2, h_img / 2
    x_norm = (bbox_center_x - cx) / cx
    y_norm = (cy - bbox_center_y) / cy
    
    # FOV 기반 각도 변환
    yaw_angle = x_norm * (camera_cfg.h_fov_deg / 2)
    pitch_angle = y_norm * (camera_cfg.v_fov_deg / 2)
    
    return yaw_angle, pitch_angle


def _calc_pitch_offset_deg(dist_m: Optional[float]) -> float:
    """
    거리 기반 pitch 오프셋 계산 (탄도 보정용)
    
    - 가까운 거리: 오프셋 없음
    - 먼 거리: 최대 오프셋 적용
    - 중간 거리: 선형 보간
    
    Args:
        dist_m: 타겟까지 거리 (m)
    
    Returns:
        float: pitch 오프셋 (도)
    """
    if dist_m is None:
        return 0.0
    
    d = float(dist_m)
    
    if d <= offset_cfg.pitch_offset_min_dist:
        return 0.0
    if d >= offset_cfg.pitch_offset_full_dist:
        return offset_cfg.pitch_offset_deg
    
    # 선형 보간
    t = (d - offset_cfg.pitch_offset_min_dist) / (
        offset_cfg.pitch_offset_full_dist - offset_cfg.pitch_offset_min_dist
    )
    return offset_cfg.pitch_offset_deg * t


# ==============================================================================
# 4. LiDAR 타겟 잠금 (LiDAR Target Lock)
# ==============================================================================

def find_lidar_points_in_angle_range(lidar_points, target_angle, target_vertical_angle, angle_tolerance=3.0):
    """
    지정된 각도 범위 내의 LiDAR 포인트 검색
    
    Args:
        lidar_points: LiDAR 포인트 리스트
        target_angle: 목표 수평 각도 (도)
        target_vertical_angle: 목표 수직 각도 (도)
        angle_tolerance: 허용 각도 오차 (도)
    
    Returns:
        list: 매칭된 LiDAR 포인트 리스트
    """
    matched_points = []
    
    for point in lidar_points:
        if not isinstance(point, dict) or not point.get('isDetected', False):
            continue
        
        angle = point.get('angle', 0.0)
        v_angle = point.get('verticalAngle', 0.0)
        
        # 순환 각도 차이 계산 (최단 거리 방식)
        angle_diff = abs((angle - target_angle + 180) % 360 - 180)
        v_angle_diff = abs(v_angle - target_vertical_angle)
        
        if angle_diff <= angle_tolerance and v_angle_diff <= angle_tolerance:
            matched_points.append(point)
    
    return matched_points


def lock_lidar_target(state, angle: float, vertical_angle: float, distance: float):
    """
    LiDAR 기반 타겟 잠금 시작
    
    Args:
        state: StateManager 인스턴스
        angle: 수평 각도 (도)
        vertical_angle: 수직 각도 (도)
        distance: 거리 (m)
    """
    now = time.time()
    
    with lidar_lock:
        state.lidar_lock.locked = True
        state.lidar_lock.lock_time = now
        state.lidar_lock.locked_angle = angle
        state.lidar_lock.locked_vertical_angle = vertical_angle
        state.lidar_lock.locked_distance = distance
        
        # 3D 좌표 계산 및 저장
        x, y, z = lidar_to_cartesian(angle, vertical_angle, distance)
        state.lidar_lock.locked_position = (x, y, z)
        
        state.lidar_lock.lock_count += 1
        state.lidar_lock.last_update_time = now
    
    print(f"🎯 [LIDAR LOCK] angle={angle:.2f}°, vAngle={vertical_angle:.2f}°, dist={distance:.2f}m")


def update_lidar_locked_target(state, lidar_points: List[Dict]) -> bool:
    """
    잠금된 타겟의 위치를 LiDAR 데이터로 업데이트
    
    Args:
        state: StateManager 인스턴스
        lidar_points: 최신 LiDAR 포인트 리스트
    
    Returns:
        bool: 업데이트 성공 여부 (실패 시 잠금 해제됨)
    """
    if not state.lidar_lock.locked:
        return False
    
    now = time.time()
    
    # 잠금 시간 초과 체크
    if (now - state.lidar_lock.lock_time) > lock_cfg.lock_duration:
        unlock_lidar_target(state)
        return False
    
    # 업데이트 주기 체크
    if (now - state.lidar_lock.last_update_time) < lock_cfg.update_interval:
        return True
    
    if not lidar_points:
        return True
    
    # 잠금 각도 근처 포인트 검색
    matched_points = find_lidar_points_in_angle_range(
        lidar_points,
        state.lidar_lock.locked_angle,
        state.lidar_lock.locked_vertical_angle,
        angle_tolerance=lock_cfg.angle_tolerance
    )
    
    # 포인트 부족 시 잠금 해제
    if len(matched_points) < lock_cfg.min_lidar_points:
        print(f"⚠️ [LIDAR LOCK] Lost target - points: {len(matched_points)} < {lock_cfg.min_lidar_points}")
        unlock_lidar_target(state)
        return False
    
    # 매칭 포인트 평균으로 현재 위치 업데이트
    angles = [p['angle'] for p in matched_points]
    v_angles = [p['verticalAngle'] for p in matched_points]
    distances = [p['distance'] for p in matched_points]
    
    with lidar_lock:
        state.lidar_lock.current_angle = float(np.mean(angles))
        state.lidar_lock.current_vertical_angle = float(np.mean(v_angles))
        state.lidar_lock.current_distance = float(np.mean(distances))
        state.lidar_lock.current_lidar_points = len(matched_points)
        state.lidar_lock.last_update_time = now
    
    return True


def unlock_lidar_target(state):
    """
    LiDAR 타겟 잠금 해제
    
    Args:
        state: StateManager 인스턴스
    """
    with lidar_lock:
        if state.lidar_lock.locked:
            duration = time.time() - state.lidar_lock.lock_time
            print(f"🔓 [LIDAR UNLOCK] Duration: {duration:.2f}s, "
                  f"Locks: {state.lidar_lock.lock_count}, "
                  f"Fires: {state.lidar_lock.successful_fires}")
        
        state.lidar_lock.locked = False
        state.lidar_lock.lock_time = 0.0
        state.lidar_lock.current_lidar_points = 0


def unlock_all_combat_locks(state, reason: str = ""):
    """
    모든 전투 관련 잠금 상태 일괄 해제
    
    - LiDAR 잠금
    - YOLO bbox 잠금
    - 타겟 정보 초기화
    
    Args:
        state: StateManager 인스턴스
        reason: 해제 사유 (로그/UI 표시용)
    """
    # LiDAR 잠금 해제
    try:
        unlock_lidar_target(state)
    except Exception:
        pass
    
    # YOLO bbox 잠금 해제
    if hasattr(state, "locked_bbox"):
        state.locked_bbox = None
    if hasattr(state, "locked_ts"):
        state.locked_ts = 0.0
    
    # 타겟 정보 초기화
    if hasattr(state, "last_target"):
        state.last_target = None
    if hasattr(state, "last_target_ts"):
        state.last_target_ts = 0.0
    if hasattr(state, "last_detected_distance"):
        state.last_detected_distance = None
    
    # UI 표시용 데이터 초기화
    if hasattr(state, "detected_targets"):
        state.detected_targets = []
    if reason and hasattr(state, "status_message"):
        state.status_message = f"🔓 UNLOCK ({reason})"


def get_lidar_target_info(state) -> Optional[Dict]:
    """
    현재 LiDAR 잠금 타겟 정보 조회
    
    Args:
        state: StateManager 인스턴스
    
    Returns:
        dict: 타겟 정보 (잠금 없으면 None)
    """
    if not state.lidar_lock.locked:
        return None
    
    return {
        'locked': True,
        'angle': state.lidar_lock.current_angle,
        'verticalAngle': state.lidar_lock.current_vertical_angle,
        'distance': state.lidar_lock.current_distance,
        'position': state.lidar_lock.locked_position,
        'lidar_points': state.lidar_lock.current_lidar_points,
        'lock_age': time.time() - state.lidar_lock.lock_time
    }


# ==============================================================================
# 5. YOLO 타겟 선택 (YOLO Target Selection)
# ==============================================================================

def select_best_target(
    tank_candidates: List[Dict],
    locked_bbox: Optional[List[float]],
    locked_ts: float,
    now: float,
    min_fire_conf: float = 0.40,
    fire_min_dist: float = 2.0
) -> Optional[Dict]:
    """
    YOLO 탐지 결과에서 최적 타겟 선택
    
    선택 우선순위:
    1. 기존 잠금 타겟과 IoU가 높은 후보 (락 유지)
    2. 거리가 가까운 후보
    3. confidence가 높은 후보
    
    Args:
        tank_candidates: Tank 클래스 탐지 결과 리스트
        locked_bbox: 현재 잠금된 bbox (없으면 None)
        locked_ts: 잠금 시작 타임스탬프
        now: 현재 시간
        min_fire_conf: 최소 confidence 임계값
        fire_min_dist: 최소 사격 거리 (m)
    
    Returns:
        dict: 선택된 타겟 (없으면 None)
    """
    if not tank_candidates:
        return None
    
    # 유효한 후보 필터링
    valid_candidates = [
        d for d in tank_candidates
        if float(d.get("confidence", 0.0)) >= min_fire_conf
        and (d.get("distance_m") is None or float(d.get("distance_m")) >= fire_min_dist)
    ]
    
    if not valid_candidates:
        return None
    
    best = None
    
    # (1) 기존 잠금 타겟 유지 시도 (IoU 기반)
    if locked_bbox is not None:
        cand_with_iou = [
            (d, _iou(d["bbox"], locked_bbox))
            for d in valid_candidates
        ]
        cand_with_iou = [
            (d, iou) for d, iou in cand_with_iou
            if iou >= lock_cfg.iou_thresh
        ]
        
        if cand_with_iou:
            best = max(cand_with_iou, key=lambda x: x[1])[0]
    
# (2) 새 타겟 선택
    if best is None:
        with_dist = [d for d in valid_candidates if d.get("distance_m") is not None]
        
        if with_dist:
            best = min(
                with_dist,
                key=lambda d: (
                    d.get("distance_m", 9999),
                    -d.get("point_count", 0),
                    -d.get("confidence", 0.0)
                )
            )
        else:
            # 센서 퓨전 실패 시: IoU로 이전 타겟 유지 시도 (0116)
            if locked_bbox is not None:
                cand_with_iou = [
                    (d, _iou(d["bbox"], locked_bbox))
                    for d in valid_candidates
                ]
                # 낮은 임계값으로 매칭 (거리 정보 없어도 OK)
                cand_with_iou = [(d, iou) for d, iou in cand_with_iou if iou >= 0.15]
                
                if cand_with_iou:
                    best = max(cand_with_iou, key=lambda x: x[1])[0]

    return best

def check_target_lost(
    tank_candidates: List[Dict],
    last_detected_distance: Optional[float],
    locked_bbox: Optional[List[float]],
    distance_tolerance=None
) -> bool:
    """타겟 소실 여부 감지"""
    
    if distance_tolerance is None:
        distance_tolerance = lock_cfg.target_lost_distance_tolerance
    
    # 탱크가 하나라도 탐지되면 소실 아님
    if tank_candidates:
        # IoU 기반 확인
        if locked_bbox is not None:
            for t in tank_candidates:
                if _iou(t["bbox"], locked_bbox) >= 0.10:  # 매우 낮은 임계값
                    return False  # 타겟 유지
        
        # 거리 기반 확인
        if last_detected_distance is not None:
            for t in tank_candidates:
                dm = t.get("distance_m")
                if dm is not None:
                    if abs(float(dm) - float(last_detected_distance)) < distance_tolerance:
                        return False  # 타겟 유지
    
    # 아무 탱크도 탐지 안 되면 소실
    return len(tank_candidates) == 0

def calculate_aim_errors(
    bbox: List[float],
    aim_uv: Optional[List[int]],
    distance_m: Optional[float],
    w_img: int,
    h_img: int,
    yaw_offset_threshold: float = 1.0
) -> Dict[str, float]:
    """
    조준 오차 계산 (Yaw/Pitch + 오프셋 보정)
    
    Args:
        bbox: 타겟 bbox [xmin, ymin, xmax, ymax]
        aim_uv: 조준점 픽셀 좌표 [u, v] (없으면 bbox 중심 사용)
        distance_m: 타겟 거리 (m)
        w_img: 이미지 너비
        h_img: 이미지 높이
        yaw_offset_threshold: yaw 오프셋 적용 임계값 (도)
    
    Returns:
        dict: {
            'yaw_base': 기본 yaw 오차,
            'pitch_base': 기본 pitch 오차,
            'yaw_error_deg': 보정된 yaw 오차,
            'pitch_error_deg': 보정된 pitch 오차,
            'pitch_offset_deg': 적용된 pitch 오프셋
        }
    """
    xmin, ymin, xmax, ymax = bbox
    cx, cy = w_img / 2, h_img / 2
    
    # 조준점 결정 (aim_uv 없으면 bbox 중심)
    if aim_uv:
        u, v = aim_uv
    else:
        u = (xmin + xmax) / 2
        v = (ymin + ymax) / 2
    
    # FOV 기반 초점 거리 계산
    fx = (w_img * 0.5) / np.tan(np.deg2rad(camera_cfg.h_fov_deg / 2))
    fy = (h_img * 0.5) / np.tan(np.deg2rad(camera_cfg.v_fov_deg / 2))
    
    # 기본 각도 오차 계산
    yaw_base = float(np.degrees(np.arctan((u - cx) / fx)))
    pitch_base = float(np.degrees(np.arctan((cy - v) / fy)))
    
    # Yaw 오프셋 적용
    if abs(yaw_base) > yaw_offset_threshold:
        yaw_offset = offset_cfg.yaw_offset_deg if yaw_base > 0 else -offset_cfg.yaw_offset_deg
        yaw_error_deg = yaw_base + yaw_offset
    else:
        yaw_error_deg = yaw_base
    
    # Pitch 오프셋 적용 (거리 기반 탄도 보정)
    pitch_offset_deg = _calc_pitch_offset_deg(distance_m)
    pitch_error_deg = pitch_base + pitch_offset_deg
    
    return {
        'yaw_base': yaw_base,
        'pitch_base': pitch_base,
        'yaw_error_deg': yaw_error_deg,
        'pitch_error_deg': pitch_error_deg,
        'pitch_offset_deg': pitch_offset_deg
    }


# ==============================================================================
# 6. ROI 기반 타겟 추적 (ROI-based Target Tracking)
# ==============================================================================

def update_locked_bbox_by_roi_yolo(
    img_np: np.ndarray,
    prev_bbox: Optional[list],
    w_img: int,
    h_img: int,
    yolo_model,
    class_mapping: Dict[int, str],
    roi_pad_px: int,
    roi_conf: float,
    roi_iou_th: float
) -> Optional[list]:
    """
    ROI crop + YOLO로 잠금된 bbox 업데이트
    
    - 이전 bbox 주변을 crop하여 YOLO 재탐지
    - IoU가 높은 Tank 탐지 결과로 bbox 갱신
    
    Args:
        img_np: 전체 이미지 (numpy array)
        prev_bbox: 이전 프레임 bbox
        w_img: 이미지 너비
        h_img: 이미지 높이
        yolo_model: YOLO 모델 (model_integrated)
        class_mapping: 클래스 ID → 이름 매핑
        roi_pad_px: ROI 확장 픽셀
        roi_conf: YOLO confidence 임계값
        roi_iou_th: IoU 임계값
    
    Returns:
        list: 업데이트된 bbox (실패 시 prev_bbox 반환)
    """
    if prev_bbox is None:
        return None
    
    # ROI 영역 계산
    rx1, ry1, rx2, ry2 = _expand_bbox(prev_bbox, w_img, h_img, roi_pad_px)
    x1i, y1i, x2i, y2i = map(int, [rx1, ry1, rx2, ry2])
    
    # 이미지 crop
    crop = img_np[y1i:y2i, x1i:x2i]
    if crop.size == 0:
        return prev_bbox
    
    # YOLO 탐지
    results = yolo_model(crop, conf=roi_conf, verbose=False)
    if results is None or len(results) == 0:
        return prev_bbox
    
    r0 = results[0]
    if r0.boxes is None or r0.boxes.data is None or len(r0.boxes.data) == 0:
        return prev_bbox
    
    dets = r0.boxes.data.cpu().numpy()
    
    # 최적 매칭 탐지 결과 찾기
    best_bbox = None
    best_iou = 0.0
    
    for box in dets:
        class_id = int(box[5])
        if class_id not in class_mapping:
            continue
        
        name = class_mapping[class_id].lower()
        if "tank" not in name:
            continue
        
        # crop 좌표 → 전체 이미지 좌표 변환
        cx1, cy1, cx2, cy2 = [float(v) for v in box[:4]]
        full_bbox = _clip_bbox(
            [cx1 + x1i, cy1 + y1i, cx2 + x1i, cy2 + y1i],
            w_img, h_img
        )
        
        iou = _iou(full_bbox, prev_bbox)
        if iou > best_iou:
            best_iou = iou
            best_bbox = full_bbox
    
    # IoU 임계값 이상이면 업데이트
    if best_bbox is not None and best_iou >= roi_iou_th:
        return best_bbox
    
    return prev_bbox


def predict_bbox_by_cam_delta(
    prev_bbox,
    prev_rot,
    curr_rot,
    w_img: int,
    h_img: int,
    h_fov_deg: float,
    v_fov_deg: float
):
    """
    카메라(포신) 회전량으로 bbox 위치 예측
    
    - 포신이 회전하면 화면 상의 타겟 위치도 이동
    - 회전량을 픽셀 이동량으로 변환하여 bbox 예측
    
    Args:
        prev_bbox: 이전 프레임 bbox
        prev_rot: 이전 카메라 회전 {"x": pitch, "y": yaw, "z": roll}
        curr_rot: 현재 카메라 회전
        w_img: 이미지 너비
        h_img: 이미지 높이
        h_fov_deg: 수평 FOV (도)
        v_fov_deg: 수직 FOV (도)
    
    Returns:
        Tuple[predicted_bbox, abs_dx, abs_dy]: 예측 bbox와 이동량
    """
    if prev_rot is None or curr_rot is None:
        return prev_bbox, 0.0, 0.0
    
    # 회전 변화량 계산
    prev_yaw = float(prev_rot.get("y", 0.0))
    curr_yaw = float(curr_rot.get("y", 0.0))
    prev_pitch = float(prev_rot.get("x", 0.0))
    curr_pitch = float(curr_rot.get("x", 0.0))
    
    dyaw = curr_yaw - prev_yaw
    dpitch = curr_pitch - prev_pitch
    
    # FOV 기반 초점 거리
    fx = w_img / (2.0 * math.tan(math.radians(h_fov_deg) / 2.0))
    fy = h_img / (2.0 * math.tan(math.radians(v_fov_deg) / 2.0))
    
    # 회전량 → 픽셀 이동량 변환
    # (우회전 → 화면이 좌로 이동)
    dx = -fx * math.tan(math.radians(dyaw))
    dy = -fy * math.tan(math.radians(dpitch))
    
    pred = _shift_bbox(prev_bbox, dx, dy, w_img, h_img)
    return pred, abs(dx), abs(dy)


# ==============================================================================
# 7. 전투 액션 계산 (Combat Action Computation)
# ==============================================================================

def compute_combat_action(state, lidar_points, sm_cfg):
    """
    SEQ 2 전투 시스템 - 4단계 모드 State Machine
    
    모드 흐름:
    1. SCAN: 터렛 회전하며 주변 탐색
       - 일시 정지 후, 사용자의 방향 전환 값을 받으면 시작함
       - 적 감지 시 → STANDBY로 전환
       - RESCAN 버튼 -> SCAN 모드로 돌아가서 기존 방향대로 재탐색
       - RETREAT 버튼 -> SEQ=3으로 전환
       
    2. STANDBY: 가장 가까운 적에 타겟 고정 + 포신 정렬
       - 정렬 완료 시 fire_ready = True (버튼 활성화)
       - 사용자 버튼 선택 대기:
         * FIRE 버튼 → FIRE 모드
         * RE-SCAN 버튼 → SCAN 모드
         * RETREAT 버튼 → SEQ=3으로 전환
    
    3. FIRE: 발사 실행
       - 발사 완료 후 → SEQ=3으로 전환
    """
    
    # 기본 명령 (정지)
    command = {
        "moveWS": {"command": "STOP", "weight": 1.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False,
    }
    
    now = time.time()
    lidar_points = _normalize_lidar_points(lidar_points)
    
    mode = state.combat_mode
    user_action = state.user_action

    if user_action == "RETREAT":
        # 포탑 정렬 후 이동하도록 각도 계산
        turret_x = getattr(state, 'player_turret_x', 0.0)
        body_x = getattr(state, 'player_body_x', 0.0)

        # 유효성 검증
        if turret_x is None or body_x is None:
            print("⚠️ [RETREAT] 터렛/본체 각도 정보가 없습니다. SEQ 3으로 즉시 전환합니다.")
            state.seq_change_request = 3
            state.retreat_aligned = False
            state.user_action = None
            return command

        angle_diff = abs(turret_x - body_x)

        if angle_diff > 180: angle_diff = 360 - angle_diff
        threshold = sm_cfg.turret_alignment_threshold

        # 정렬 시작 알림
        if not hasattr(state, 'retreat_aligned'):
            state.retreat_aligned = False
            state.retreat_start_ts = now
            state.status_message = "[RETREAT] 후퇴 명령 - 터렛-바디 정렬 시작..."
            print(f"   터렛: {turret_x:.2f}°, 바디: {body_x:.2f}°, 차이: {angle_diff:.2f}°")

        # 정렬 완료 후, 후퇴 시작
        if angle_diff <= threshold:
            state.seq_change_request = 3
            state.user_action = None
            state.retreat_aligned = False
            state.status_message = "[RETREAT] 터렛-바디 정렬 완료! 후퇴 시작"
            print(f"   Turret X: {turret_x:.2f}° ≈ Body X: {body_x:.2f}°")
            return command 
        
        # 정렬 실행 - 회전 명령 생성 -> 터렛을 바디 방향으로 회전
        raw_diff = turret_x - body_x

        while raw_diff > 180: raw_diff -= 360
        while raw_diff < -180: raw_diff += 360

        # 회전 방향 결정
        # raw_diff > 0: 터렛이 바디보다 우측 → Q(좌회전)
        # raw_diff < 0: 터렛이 바디보다 좌측 → E(우회전)
        turn_direction = "Q" if raw_diff > 0 else "E"

        # 회전 속도 계산
        turn_weight = min(0.2 + abs(raw_diff) * 0.008, 0.4)
        
        command["turretQE"] = {
            "command": turn_direction,
            "weight": turn_weight
        }
        state.status_message = f"🏃 후퇴 준비 중 - 터렛 정렬 중... (차이: {angle_diff:.2f}°)"

        return command
    
    if user_action == "RESCAN":
        state.combat_mode = "SCAN"
        state.mode_ts = now
        state.scan_completed = False  # 플래그 초기화
        state.fire_ready = False
        state.standby_target = None
        state.detected_targets = []
        state.last_scan_targets = []  
        state.last_target = None      
        state.locked_bbox = None       
        state.locked_ts = 0.0       
        # 완전한 상태 초기화
        if hasattr(state, 'locked_tid'):
            state.locked_tid = None
        if hasattr(state, 'locked_update_ts'):
            state.locked_update_ts = 0.0
        if hasattr(state, 'locked_start_ts'):
            state.locked_start_ts = None
        if hasattr(state, 'last_target_ts'):
            state.last_target_ts = 0.0
        if hasattr(state, 'last_detected_distance'):
            state.last_detected_distance = None
        state.user_action = None
        state.is_lowering_barrel = False
        print(f"🔄 [RESCAN] 탐색 모드 전환")
        return command
    
    
    # ─────────────────────────────────────────────────────────────
    # [SCAN 모드] 터렛 회전하며 탐색
    # ─────────────────────────────────────────────────────────────

    if mode == "SCAN":
        # 1. 방향이 설정되지 않았을 때 정지 유지 (최우선)
        if state.scan_direction is None:
            state.status_message = "SCAN 대기 중 - 방향(Q/E)을 선택하세요."
            return command

        # 2. 포신 하향 먼저 처리 (방향이 정해지면 무조건 실행)
        scan_elapsed = now - state.scan_start_ts
        
        if state.is_lowering_barrel:
            if scan_elapsed >= sm_cfg.lowering_sec:
                # 포신 하향 완료
                state.is_lowering_barrel = False
                state.mode_ts = now
                state.status_message = "포신 하향 완료. 회전 탐색 실행"
            else:
                # 포신 하향 중 (적 감지 여부와 무관하게 계속 진행)
                state.status_message = f"탐색 준비 중... 포신 하향 ({(sm_cfg.lowering_sec - scan_elapsed):.1f}s)"
                command["turretRF"] = {"command": "F", "weight": 0.8}
                return command

        # 3. 포신 하향 완료 후에만 적 감지 체크
        if not hasattr(state, 'scan_completed'):
            state.scan_completed = False
        
        if not state.scan_completed:
            tanks = [t for t in state.detected_targets if t.get('category') == 'tank']
            reds = [t for t in state.detected_targets if t.get('category') == 'red']
            
            enemy_detected = (
                len(tanks) >= sm_cfg.min_tanks_to_detect or 
                len(reds) >= sm_cfg.min_reds_to_detect
            )
            
            scan_elapsed_mode = now - state.mode_ts
            if enemy_detected and scan_elapsed_mode > sm_cfg.scan_hold_sec:
                state.scan_completed = True
                state.status_message = "적 감지! - 다음 행동을 선택하세요."

        # 4. 적 감지되면 정지
        if state.scan_completed:
            state.status_message = "적 감지! - 다음 행동을 선택하세요."
            return command

        # 5. 터렛 회전 (적 미감지 + 포신 하향 완료 상태)
        command["turretQE"] = {
            "command": state.scan_direction,
            "weight": sm_cfg.scan_turret_speed
        }
        command["turretRF"] = {"command": "", "weight": 0.0}
        
        return command
    
    # ─────────────────────────────────────────────────────────────
    # [STANDBY 모드] 타겟 고정 + 포신 정렬 + 버튼 대기
    # ─────────────────────────────────────────────────────────────
    elif mode == "STANDBY":
            # 1) 센서 퓨전 조준 오차 판별 (미세 조종 제외)
            # app.py의 perform_sensor_fusion -> calculate_aim_errors에 의해 갱신된 정보 사용
            target = getattr(state, 'last_target', None)
            threshold = precision_cfg.TOLERANCE  # 조준 완료 허용 오차 (도) 적용
            
            if target and target.get("bbox") is not None:
                # yaw 및 pitch 오차의 절대값 확인
                yaw_err = abs(target.get("yaw_error_deg", 999))
                pitch_err = abs(target.get("pitch_error_deg", 999))
                
                # 임계값 이내일 때만 fire_ready 활성화
                if yaw_err <= threshold and pitch_err <= threshold:
                    state.fire_ready = True
                else:
                    state.fire_ready = False
                    state.status_message = "포격 버튼을 눌러주세요."
            else:
                state.fire_ready = False
                state.status_message = "🔒 STANDBY 모드 - 타겟 대기 중..."

            # 2) 사용자 버튼 입력 및 AUTO_ATTACK 로직 통합
            user_action = state.user_action
            
            # 사용자가 UI에서 FIRE(포격) 버튼을 눌렀을 때
            if user_action == "FIRE":
                if state.fire_ready:
                    state.combat_mode = "FIRE"
                    state.user_action = None
                    state.auto_attack_active = False # 자동 포격 플래그 초기화
                    print("🔥 [STANDBY→FIRE] 즉시 발사 실행")
                else:
                    state.auto_attack_active = True
                    state.user_action = None
                    print("⚔️ [AUTO_ATTACK] 활성화 - 조준 정렬 대기 중")

            # 3) AUTO_ATTACK 강제 실행 체크
            if getattr(state, 'auto_attack_active', False) and state.fire_ready:
                state.combat_mode = "FIRE"
                state.auto_attack_active = False
                print("🚀 [AUTO_ATTACK] 조준 일치 - 자동 발사!")
                
            elif user_action == "RESCAN":
                # RE-SCAN 버튼 클릭 → SCAN 모드로 복귀, 방향 전환
                state.combat_mode = "SCAN"
                state.scan_completed = False  
                state.fire_ready = False
                state.standby_target = None
                state.last_scan_targets = []   
                state.last_target = None       
                state.locked_bbox = None  
                state.locked_ts = 0.0   
                if hasattr(state, 'locked_tid'):
                    state.locked_tid = None
                if hasattr(state, 'locked_update_ts'):
                    state.locked_update_ts = 0.0
                if hasattr(state, 'locked_start_ts'):
                    state.locked_start_ts = None
                if hasattr(state, 'last_target_ts'):
                    state.last_target_ts = 0.0
                if hasattr(state, 'last_detected_distance'):
                    state.last_detected_distance = None

                state.user_action = None
                print(f"🔄 [STANDBY→SCAN] Re-Scan 시작 (방향: {state.scan_direction})")
                return command

    # ─────────────────────────────────────────────────────────────
    # [FIRE 모드] 발사 실행
    # ─────────────────────────────────────────────────────────────
    elif mode == "FIRE":
        # 발사!
        command["fire"] = True
        state.fire_executed_ts = now
        state.lidar_lock.successful_fires += 1
        
        # [추가] 사격 후 자동 후퇴 로직
        state.user_action = "RETREAT"
        state.combat_mode = "SCAN"
        state.fire_ready = False
        state.auto_attack_active = False
        
        # 이전 후퇴 기록 초기화 (정렬을 새로 시작하기 위함)
        if hasattr(state, 'retreat_aligned'):
            delattr(state, 'retreat_aligned')
        
        state.status_message = "💥 포격 완료! 즉시 후퇴(RETREAT) 모드로 자동 전환합니다."
        print(f"💥 [FIRE] 발사 성공 -> 유저 액션을 'RETREAT'으로 강제 변경하여 후퇴를 시작합니다.")
        
        return command

# ==============================================================================
# 8. 전차 (Tank)만 Track 모드로 탐지
# - 대기 모드에서 전차에게만 특정 id를 부여하여 추적 및 관리하는 함수
# ==============================================================================

def detect_tank_only_track(
    image_input,
    yolo_model,
    class_map: dict,
    color_hex: str,
    min_det_conf: float,
    min_box_w: float,
    min_box_h: float,
    track_lock=None,
    use_onnx: bool = False,
    prev_detections: list = None, # ONNX 트래킹용 이전 탐지 결과
):
    if isinstance(image_input, str):
        img_pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img_pil = image_input.convert("RGB") if image_input.mode != "RGB" else image_input
    else:
        raise ValueError(f"img_input must be str or PIL.Image, got {type(image_input)}")
    
    # Tank class id 자동 추출
    tank_cls_ids = [cid for cid, name in class_map.items() if name == "Tank"]
    if not tank_cls_ids:
        tank_cls_ids = None  # (비권장) 맵핑에 Tank 없으면 필터 없이 수행

    if use_onnx:
        # ═══════════════════════════════════════════════════════════════
        # ONNX 모드: OnnxYoloDetector + IoU 기반 트래킹
        # ═══════════════════════════════════════════════════════════════
        all_detections = yolo_model.detect(
            img_pil,
            conf_threshold=min_det_conf,
            iou_threshold=0.45
        )
        
        # Tank 클래스만 필터링
        tank_detections = []
        for det in all_detections:
            if tank_cls_ids is None or det["class_id"] in tank_cls_ids:
                bbox = det["bbox"]
                xmin, ymin, xmax, ymax = bbox
                
                if (xmax - xmin) < min_box_w or (ymax - ymin) < min_box_h:
                    continue
                
                tank_detections.append({
                    "bbox": bbox,
                    "confidence": det["confidence"],
                    "class_id": det["class_id"]
                })
        
        # IoU 기반 트래킹 (track_id 할당)
        out = []
        iou_threshold = 0.3
        
        if prev_detections:
            used_prev_ids = set()
            
            for det in tank_detections:
                best_iou = 0
                best_prev_id = None
                
                for prev in prev_detections:
                    prev_tid = prev.get("track_id")
                    if prev_tid in used_prev_ids:
                        continue
                    
                    iou = _iou(det["bbox"], prev["bbox"])
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_prev_id = prev_tid
                
                if best_prev_id is not None:
                    track_id = best_prev_id
                    used_prev_ids.add(track_id)
                else:
                    # 새로운 track_id 할당
                    max_existing = max([p.get("track_id", 0) for p in prev_detections] + [0])
                    track_id = max_existing + 1
                
                det["track_id"] = track_id
        else:
            # 이전 탐지 없으면 순차적으로 ID 할당
            for i, det in enumerate(tank_detections):
                det["track_id"] = i + 1
        
        # 결과 포맷팅
        for det in tank_detections:
            track_id = det.get("track_id")
            conf = det["confidence"]
            
            display = f"Tank"
            if track_id is not None:
                display = f"[ID:{track_id}] Tank ({conf:.2f})"
            
            out.append({
                "className": display,
                "category": "tank",
                "bbox": det["bbox"],
                "confidence": conf,
                "color": color_hex,
                "filled": False,
                "updateBoxWhileMoving": False,
                "track_id": track_id,
            })
        
        return out
    else:
        # ═══════════════════════════════════════════════════════════════
        # PyTorch 모드: ultralytics YOLO tracking 사용 (PIL Image 지원)
        # ═══════════════════════════════════════════════════════════════
        if track_lock is not None:
            with track_lock:
                results = yolo_model.track(
                    img_pil,
                    conf=min_det_conf,
                    classes=tank_cls_ids,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
        else:
            results = yolo_model.track(
                img_pil,
                conf=min_det_conf,
                classes=tank_cls_ids,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )

        r = results[0]
        boxes = r.boxes

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else []
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        tids  = boxes.id.int().cpu().numpy() if boxes.id is not None else [None] * len(xyxy)

        out = []
        for bb, conf, tid in zip(xyxy, confs, tids):
            xmin, ymin, xmax, ymax = map(float, bb.tolist())
            conf = float(conf)

            if (xmax - xmin) < min_box_w or (ymax - ymin) < min_box_h:
                continue

            track_id = int(tid) if tid is not None else None

            display = f"Tank"
            if track_id is not None:
                display = f"[ID:{track_id}] Tank ({conf:.2f})"

            out.append({
                "className": display,
                "category": "tank",
                "bbox": [xmin, ymin, xmax, ymax],
                "confidence": conf,
                "color": color_hex,
                "filled": False,
                "updateBoxWhileMoving": False,
                "track_id": track_id,
            })

        return out

def verify_target_stability(state, best_target, now, delay_sec):
    """
    [검증 로직] 타겟 락을 확정하기 전 일정 시간 동안 후보를 검증
    
    Args:
        state: StateManager 인스턴스
        best_target: 현재 프레임의 최적 타겟 후보 (select_best_target 결과)
        now: 현재 시간
        delay_sec: 검증 대기 시간 (config.lock_cfg.lock_delay)
        
    Returns:
        bool: True면 '검증 통과(락 걸어도 됨)', False면 '아직 검증 중'
    """
    # 1. 후보가 없으면 검증 상태 초기화
    if best_target is None:
        if state.pending_tid is not None:
            print(f"👋 [검증 취소] 타겟 소실")
        state.pending_tid = None
        state.pending_start_ts = 0.0
        return False

    curr_tid = best_target.get("track_id")
    
    # 2. Track ID가 없는 경우 (추적 불가) -> 검증 리셋
    if curr_tid is None:
        state.pending_tid = None
        state.pending_start_ts = 0.0
        return False

    # 3. 새로운 타겟 후보가 나타난 경우 (ID 변경)
    if state.pending_tid != curr_tid:
        state.pending_tid = curr_tid
        state.pending_start_ts = now
        # 로그는 필요시 주석 해제
        # print(f"🕵️ [검증 시작] 새로운 후보 ID:{curr_tid} (거리: {best_target.get('distance_m')}m)")
        return False

    # 4. 동일 타겟 유지 중 -> 시간 체크
    elapsed = now - state.pending_start_ts
    if elapsed >= delay_sec:
        # 검증 시간 초과 -> 락 확정!
        # 확정되었으므로 펜딩 상태는 초기화하지 않고, 외부(app.py)에서 locked_bbox 설정 시 자연스럽게 처리됨
        return True
    
    # 아직 시간 부족
    return False

# =========================================================
# Lock-on 대상 선택
# - 이전 locked_bbox가 있으면 IoU 기반으로 추적 느낌 유지
# =========================================================
def pick_lock_target_yolo_only(tank_candidates: list, prev_locked_bbox, iou_gate=0.15):
    if not tank_candidates:
        return None

    if prev_locked_bbox is not None:
        tank_candidates.sort(key=lambda d: _iou(d["bbox"], prev_locked_bbox), reverse=True)
        best = tank_candidates[0]
        if _iou(best["bbox"], prev_locked_bbox) >= iou_gate:
            return best

    tank_candidates.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
    return tank_candidates[0]