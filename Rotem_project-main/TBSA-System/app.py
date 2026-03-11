"""
Flask 서버 메인 - SEQ 기반 통합 시스템

[시스템 구조]
    SEQ 1: A* + PID 주행 (정적 장애물 회피)
    SEQ 2: YOLO + LiDAR 센서퓨전 전투 시스템
    SEQ 3: A* + PID 주행 (SEQ 1과 동일, 파라미터 다름)
    SEQ 4: PPO + A* 하이브리드 주행 (동적 재계획)

[주요 기능]
    - 듀얼 YOLO 모델 (Cannon 전용 + 통합 객체 인식)
    - LiDAR + 이미지 센서 퓨전
    - 실시간 타겟 추적 및 조준
    - 자동 경로 재계획 (SEQ 4)
"""

# ==============================================================================
# 라이브러리 임포트
# ==============================================================================
import os
import math
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'
import io
import time
import threading
import numpy as np
from flask import Flask, request, jsonify, send_file, Response, render_template
from ultralytics import YOLO
from PIL import Image, ImageDraw

# ==============================================================================
# 프로젝트 모듈 임포트
# ==============================================================================
# 설정
from config import (
    Config, combat_config, sm_cfg,lock_cfg,
    camera_cfg, fusion_cfg, aim_cfg, turret_cfg
)

# 제어기
from controllers.hybrid_controller import HybridController

# 경로 계획
from planners.astar_planner import AStarPlanner

# 유틸리티
from utils.state_manager import StateManager
from utils.lidar_logger import (
    LidarLogger,
    build_intrinsic_from_fov, 
    project_world_to_image,
    get_distance_for_bboxes, 
    draw_points_on_rgb, 
    draw_lidar_association_boxes
)
from utils.visualization import VisualizationManager
from utils.combat_system import (
    compute_combat_action, 
    lock_lidar_target, 
    unlock_lidar_target, 
    make_det_overlay_bytes,
    get_lidar_target_info,
    detect_tank_only_track, 
    detect_all_objects_dual,
    select_best_target, 
    check_target_lost, 
    calculate_aim_errors,
    verify_target_stability
)

# ==============================================================================
# Flask 앱 초기화
# ==============================================================================
app = Flask(__name__)

# ==============================================================================
# 전역 객체 초기화
# ==============================================================================

# 설정 관리
config = Config()
state_manager = StateManager(config)

# 모델 로드 (ONNX 또는 PyTorch)
USE_ONNX = combat_config.use_onnx

if USE_ONNX:
    # ONNX 모델 로드
    from utils.onnx_detector import OnnxYoloDetector
    
    print("=" * 60)
    print("🚀 ONNX 모드 활성화")
    print("=" * 60)
    
    try:
        model_cannon = OnnxYoloDetector(
            combat_config.onnx_cannon_path,
            input_size=(combat_config.onnx_input_size, combat_config.onnx_input_size),
            use_gpu=combat_config.onnx_use_gpu,
            fp16=combat_config.onnx_fp16
        )
        model_integrated = OnnxYoloDetector(
            combat_config.onnx_integrated_path,
            input_size=(combat_config.onnx_input_size, combat_config.onnx_input_size),
            use_gpu=combat_config.onnx_use_gpu,
            fp16=combat_config.onnx_fp16
        )
        # Legacy 모델도 ONNX로 (best.onnx가 있는 경우)
        onnx_best_path = combat_config.model_path.replace(".pt", ".onnx")
        if os.path.exists(onnx_best_path):
            model = OnnxYoloDetector(
                onnx_best_path,
                input_size=(combat_config.onnx_input_size, combat_config.onnx_input_size),
                use_gpu=combat_config.onnx_use_gpu,
                fp16=combat_config.onnx_fp16
            )
        else:
            model = model_integrated  # ONNX best 없으면 integrated 사용
            
    except Exception as e:
        print(f"[ONNX] ⚠️ ONNX 모델 로드 실패: {e}")
        print("[ONNX] PyTorch 모드로 폴백합니다.")
        USE_ONNX = False
        # 폴백: PyTorch 모델 로드
        model_cannon = YOLO(combat_config.model_cannon_path)
        model_integrated = YOLO(combat_config.model_integrated_path)
        model = YOLO(combat_config.model_path) if os.path.exists(combat_config.model_path) else None

else:
    # PyTorch YOLO 모델 로드 (듀얼 모델)
    print("=" * 60)
    print("🔷 PyTorch 모드 활성화")
    print("=" * 60)
    model_cannon = YOLO(combat_config.model_cannon_path)
    model_integrated = YOLO(combat_config.model_integrated_path)
    
    # Legacy 모델 (기존 호환성 유지, SEQ 2 전용)
    model = YOLO(combat_config.model_path) if os.path.exists(combat_config.model_path) else None

# 트래킹용 이전 탐지 결과 저장 (ONNX 모드)
prev_tank_detections = []

# 주행 모듈
planner = AStarPlanner(
    grid_min_x=0.0, 
    grid_max_x=300.0,
    grid_min_z=0.0, 
    grid_max_z=300.0,
    cell_size=config.ASTAR.CELL_SIZE, 
    obstacle_margin=config.ASTAR.OBSTACLE_MARGIN, 
    allow_diagonal=True,
    safety_weight=config.ASTAR.SAFETY_WEIGHT,
    proximity_radius=config.ASTAR.PROXIMITY_RADIUS
)

controller = HybridController(config, planner, state_manager)

# LiDAR 로거 (자동 파일 정리 기능 포함)
lidar_logger = LidarLogger(
    config.Lidar.LIDAR_FOLDER, 
    config.Lidar.LIDAR_FILE_PATTERN, 
    state_manager, 
    save_csv=False,
    auto_cleanup_mode="keep_recent",
    max_files=20,
    costmap_inflation=config.Terrain.COSTMAP_INFLATION
)
lidar_logger.start()

# 시각화 관리자
viz_manager = VisualizationManager(state_manager, config.Lidar.GRID_SIZE)


# ==============================================================================
# 헬퍼 함수 - 센서 퓨전
# ==============================================================================

def perform_sensor_fusion(img_pil, tank_candidates, state_manager, now):
    """
    LiDAR와 이미지 센서 퓨전 수행
    
    Args:
        img_pil: PIL 이미지 객체
        tank_candidates: 탱크 객체 탐지 결과 리스트
        state_manager: 상태 관리자
        now: 현재 시각
    
    Returns:
        tuple: (fusion_ok, merged_results, uv_valid, dist_valid)
            - fusion_ok: 센서 퓨전 성공 여부
            - merged_results: 퓨전된 탐지 결과
            - uv_valid: 유효한 LiDAR 포인트 픽셀 좌표
            - dist_valid: 유효한 LiDAR 포인트 거리
    """
    w_img, h_img = img_pil.size
    merged_results = tank_candidates.copy()
    uv_valid = None
    dist_valid = None
    
    # LiDAR 데이터 가져오기
    lidar_df = lidar_logger.get_latest_dataframe()

    state_manager.cam_pos = {"x": float(lidar_df["turretCam_x"].iloc[0]),
                                "y": float(lidar_df["turretCam_y"].iloc[0]),
                                "z": float(lidar_df["turretCam_z"].iloc[0])}
    state_manager.cam_rot = {"x": float(lidar_df["camLeftRot_x"].iloc[0]),
                                "y": float(lidar_df["camLeftRot_y"].iloc[0]),
                                "z": float(lidar_df["camLeftRot_z"].iloc[0])}
    
    # 포즈 데이터 유효성 확인
    pose_ok = (
        state_manager.cam_pos is not None and
        state_manager.cam_rot is not None and
        (now - state_manager.last_pose_ts) < fusion_cfg.pose_timeout_sec
    )
    
    fusion_ok = (pose_ok and lidar_df is not None and len(lidar_df) > 0)
    
    if not fusion_ok or not tank_candidates:
        return fusion_ok, merged_results, uv_valid, dist_valid
    
    try:
        # 카메라 내적 행렬 생성
        K = build_intrinsic_from_fov(
            w_img, h_img,
            camera_cfg.h_fov_deg,
            camera_cfg.v_fov_deg
        )
        
        # LiDAR 포인트 월드 좌표 추출
        Pw = lidar_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        distances = lidar_df["distance"].to_numpy(dtype=np.float32)
        
        # 월드 좌표를 이미지 좌표로 투영
        uv, mask, mapping_info = project_world_to_image(
            Pw, state_manager.cam_pos, state_manager.cam_rot,
            K, w_img, h_img,
            original_distances=distances,
            show_details=False  # ← True로 설정하면 콘솔에 출력됨
        )
        
        # 이미지 범위 내 포인트만 필터링
        if uv is not None and len(uv) > 0:
            valid_indices = []
            margin = margin = fusion_cfg.screen_margin  # 픽셀 확장시 config screen_margin에서 설정 

            for i, (u, v) in enumerate(uv):
                if -margin <= u < w_img + margin and -margin <= v < h_img + margin:
                    valid_indices.append(i)
            print(f"📐 필터링 범위: [{-margin}, {w_img + margin}] x [{-margin}, {h_img + margin}]")

            for i, (u, v) in enumerate(uv):
                if 0 <= u < w_img and 0 <= v < h_img:
                    valid_indices.append(i)
            
            if valid_indices:
                uv_valid = uv[valid_indices]
                dist_valid = np.array([mapping_info[i]['lidar_distance'] for i in valid_indices])
            else:
                uv_valid = np.array([])
                dist_valid = np.array([])
        else:
            uv_valid = np.array([])
            dist_valid = np.array([])
        
        # BBox와 LiDAR 포인트 매칭하여 거리 정보 추가
        if len(uv_valid) > 0:
            # 전차 위치 가져오기
            tank_pos = state_manager.robot_pose if state_manager.robot_pose else (100.0, 10.0, 100.0)

            merged_results = get_distance_for_bboxes(
                merged_results, 
                uv_valid, 
                dist_valid, 
                mapping_info, 
                w_img, h_img, 
                tank_pos = tank_pos, 
                margin_px=20
            )
    
    except Exception as e:
        print(f"[SENSOR_FUSION] ❌ 오류 발생: {e}")
        fusion_ok = False
    
    return fusion_ok, merged_results, uv_valid, dist_valid


def create_fusion_overlay(img_pil, merged_results, uv_valid, dist_valid):
    """
    센서 퓨전 결과 오버레이 이미지 생성
    
    Args:
        img_pil: PIL 이미지 객체
        merged_results: 퓨전된 탐지 결과
        uv_valid: 유효한 LiDAR 포인트 픽셀 좌표
        dist_valid: 유효한 LiDAR 포인트 거리
    
    Returns:
        bytes: PNG 형식의 오버레이 이미지
    """
    rgb = np.array(img_pil)
    
    # 타겟과 일반 객체 분리
    target_dets = [d for d in merged_results if d.get("filled") is True]
    other_dets = [d for d in merged_results if not d.get("filled")]
    
    # LiDAR 포인트 그리기
    overlay = draw_points_on_rgb(rgb, uv_valid, dist_valid, highlight_mask=None)
    
    # 일반 객체 박스 (흰색)
    overlay = draw_lidar_association_boxes(
        overlay, other_dets,
        box_color=(255, 255, 255), width=4, fill_alpha=70, show_label=False
    )
    
    # 타겟 박스 (빨간색) - point_count 최소값 보장
    for td in target_dets:
        if td.get("point_count", 0) == 0:
            td["point_count"] = 1
    
    overlay = draw_lidar_association_boxes(
        overlay, target_dets,
        box_color=(255, 0, 0), width=4, fill_alpha=90, show_label=False
    )
    
    # 이미지를 바이트로 변환
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# ==============================================================================
# 헬퍼 함수 - SEQ 2 모드별 처리
# ==============================================================================

def handle_scan_mode(image_path, img_pil, state_manager):
    """
    SCAN 모드: 전체 객체 탐지만 수행
    
    Args:
        image_path: 이미지 파일 경로
        img_pil: PIL 이미지 객체
        state_manager: 상태 관리자
    
    Returns:
        dict: JSON 응답 데이터
    """
    # 듀얼 모델로 전체 객체 탐지
    filtered_results, meta = detect_all_objects_dual(
        img_pil,
        model_cannon=model_cannon,
        model_integrated=model_integrated,
        combat_config=combat_config,
        fusion_cfg=fusion_cfg,
        use_onnx=USE_ONNX,
    )
    
    # 탐지 결과 저장
    state_manager.detected_targets = filtered_results.copy()
    state_manager.save_scan_targets(filtered_results)
    
    # 상태 메시지 업데이트
    if state_manager.scan_direction is not None:
        state_manager.status_message = "SCAN 모드 - 객체 식별 중"
    else:
        state_manager.status_message = "SCAN 대기 중 - 방향(Q/E)을 선택하세요."
    
    # 오버레이 생성
    state_manager.overlay_left_bytes = make_det_overlay_bytes(
        img_pil, filtered_results, target_bbox=None
    )
    
    return jsonify(filtered_results)


def handle_standby_mode(image_path, img_pil, state_manager, now):
    """
    STANDBY/FIRE 모드: 탱크 추적 + 센서 퓨전 + 타겟 선택
    
    Args:
        image_path: 이미지 파일 경로
        img_pil: PIL 이미지 객체
        state_manager: 상태 관리자
        now: 현재 시각
    
    Returns:
        dict: JSON 응답 데이터
    """
    global prev_tank_detections

    w_img, h_img = img_pil.size
    
    # 1. 탱크 객체만 추적 탐지
    tracked_tanks = detect_tank_only_track(
        img_pil,
        model_integrated,
        combat_config.map_integrated,
        combat_config.color_integrated,
        fusion_cfg.min_det_conf,
        fusion_cfg.min_box_w,
        fusion_cfg.min_box_h,
        use_onnx=USE_ONNX,
        prev_detections=prev_tank_detections if USE_ONNX else None
    )
    
    if USE_ONNX:
        prev_tank_detections = tracked_tanks.copy()

    # 2. 센서 퓨전 수행
    fusion_ok, merged_results, uv_valid, dist_valid = perform_sensor_fusion(
        img_pil, tracked_tanks, state_manager, now
    )
    
    # 3. 탱크 후보 필터링
    tank_candidates = [d for d in merged_results if d.get("category") == "tank"]
    
    # 4. 최적 타겟 선택
    best = select_best_target(
        tank_candidates,
        state_manager.locked_bbox,
        state_manager.locked_ts,
        now,
        min_fire_conf=fusion_cfg.min_fire_conf,
        fire_min_dist=aim_cfg.fire_min_dist
    )
    
    # 디버그 로그 출력
    try:
        mask_pts = 0 if uv_valid is None else len(uv_valid)
        lidar_df = lidar_logger.get_latest_dataframe()
        pose_ok = (
            state_manager.cam_pos is not None and
            state_manager.cam_rot is not None and
            (now - state_manager.last_pose_ts) < fusion_cfg.pose_timeout_sec
        )
        
        print(
            f"[LOCKSRC] pose_ok={pose_ok} "
            f"lidar_ok={lidar_df is not None and len(lidar_df) > 0} "
            f"mask_pts={mask_pts} "
            f"best_dist={None if best is None else best.get('distance_m')} "
            f"best_pts={None if best is None else best.get('point_count')} "
            f"best_tid={None if best is None else best.get('track_id')}"
        )
        print(
            f"[LOCKDBG] t={now:.3f} seq={state_manager.seq} "
            f"mode={state_manager.combat_mode} "
            f"tanks={len(tank_candidates)} "
            f"locked={'Y' if state_manager.locked_bbox is not None else 'N'} "
            f"locked_age={0.0 if state_manager.locked_ts==0 else (now - state_manager.locked_ts):.2f}s "
            f"best={'Y' if best is not None else 'N'} "
            f"best_conf={None if best is None else round(best.get('confidence',-1),3)} "
            f"best_dist={None if best is None else best.get('distance_m')} "
            f"best_tid={None if best is None else best.get('track_id')}"
        )
    except Exception:
        pass
    
    # 5. 타겟 잠금 및 상태 업데이트
    # if best is not None:
    #     # 잠금 시작 시각 기록
    #     if state_manager.locked_bbox is None:
    #         state_manager.locked_ts = now
    # [수정 후] 검증 로직 적용
    should_lock = False
    
    # 1. 이미 락이 걸려있는 상태라면? -> 계속 유지 (select_best_target이 이미 필터링함)
    if state_manager.locked_bbox is not None:
        should_lock = (best is not None)
    
    # 2. 락이 없는 상태라면? -> 검증(Verify) 시도
    else:
        # 검증 함수 호출
        is_verified = verify_target_stability(
            state_manager, 
            best, 
            now, 
            lock_cfg.lock_delay  # config.py에서 설정한 시간
        )
        
        if is_verified:
            print(f"🎯 [검증 완료] 타겟 락 확정! ID:{best.get('track_id')}")
            should_lock = True
        elif best is not None:
            # 아직 검증 중일 때 메시지 표시 (선택 사항)
            elapsed = now - state_manager.pending_start_ts
            state_manager.status_message = f"타겟 검증 중... {elapsed:.1f}s / {lock_cfg.lock_delay}s"

    # 3. 락 실행 및 정보 업데이트
    if should_lock and best is not None:
        # 잠금 시작 시각 기록 (최초 1회)
        if state_manager.locked_bbox is None:
            state_manager.locked_ts = now
            state_manager.locked_start_ts = now
            state_manager.locked_tid = best.get("track_id")
            # 락 걸리면 펜딩 상태 초기화
            state_manager.pending_tid = None 
            state_manager.pending_start_ts = 0.0



            state_manager.locked_start_ts = now
            state_manager.locked_tid = best.get("track_id")
        
        state_manager.locked_update_ts = now
        state_manager.locked_bbox = best["bbox"]
        if best.get("track_id") is not None:
            state_manager.locked_tid = best.get("track_id")
        
        # 조준 오차 계산
        aim_errors = calculate_aim_errors(
            best["bbox"],
            best.get("aim_uv"),
            best.get("distance_m"),
            w_img, h_img,
            yaw_offset_threshold=turret_cfg.yaw_offset_threshold
        )
        
        # 타겟 정보 업데이트
        state_manager.last_target = {
            "category": best.get("category", "tank"),
            "displayName": best.get("className", "Tank [TARGET]"),
            "distance_m": best.get("distance_m"),
            "position": best.get("position"), #<- 함수 내부에서 값을 가져옴
            "point_count": best.get("point_count", 0),
            "yaw_error_deg": aim_errors.get("yaw_error_deg"),
            "yaw_base": aim_errors.get("yaw_base"),
            "pitch_base": aim_errors.get("pitch_base"),
            "pitch_offset_deg": aim_errors.get("pitch_offset_deg"),
            "pitch_error_deg": aim_errors.get("pitch_error_deg"),
            "bbox": best["bbox"],
            "confidence": best.get("confidence", 0.0),
            "lidar_yaw": aim_errors.get("yaw_base"),
            "lidar_pitch": aim_errors.get("pitch_base"),
            "track_id": best.get("track_id")
        }
        state_manager.last_target_ts = now
        state_manager.last_detected_distance = best.get("distance_m")
        
        # UI 표시 업데이트
        best["filled"] = True
        dist_str = f"{best['distance_m']:.1f}m" if best.get("distance_m") is not None else ""
        best["className"] = f"Tank [TARGET] {dist_str}".strip()
        
        state_manager.status_message = "대기 모드(STANDBY) - 타겟 선정 완료"
    else:
        # 타겟 없음 - 잠금 해제 판단
        state_manager.last_target = None
        state_manager.last_target_ts = now
        state_manager.status_message = "대기 모드(STANDBY) - 타겟 미선정"
        
        if check_target_lost(
            tank_candidates,
            getattr(state_manager, "last_detected_distance", None),
            getattr(state_manager, "locked_bbox", None)
        ):
            state_manager.locked_bbox = None
            state_manager.locked_ts = 0.0
            state_manager.locked_tid = None
            state_manager.locked_update_ts = 0.0
            state_manager.locked_start_ts = None
    
    # 6. 오버레이 생성
    if fusion_ok and uv_valid is not None and dist_valid is not None:
        state_manager.overlay_left_bytes = create_fusion_overlay(
            img_pil, merged_results, uv_valid, dist_valid)
    else:
        # 센서 퓨전 실패 시 기본 오버레이
        target_bbox = state_manager.locked_bbox if state_manager.locked_bbox is not None else None
        state_manager.overlay_left_bytes = make_det_overlay_bytes(
            img_pil, merged_results, target_bbox=target_bbox
        )
    
    return jsonify(tank_candidates)


# ==============================================================================
# 공통 엔드포인트
# ==============================================================================

@app.route('/detect', methods=['POST'])
def detect():
    """
    객체 탐지 엔드포인트 (SEQ 2 전용)
    
    모드별 동작:
        - SCAN: 듀얼 모델로 전체 객체 탐지만 수행
        - STANDBY/FIRE: 탱크 추적 + 센서 퓨전 + 타겟 선택
    """
    # 이미지 수신 확인
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400
    
    # SEQ 2가 아니면 에러 반환
    if state_manager.seq != 2:
        return jsonify({"error": "Detection only available in SEQ 2"}), 400
    
    now = time.time()
    
    # 이미지 저장 및 로드
    image_path = 'temp_image.jpg'
    image.save(image_path)
    img_pil = Image.open(image_path).convert("RGB")
    
    # 현재 모드 확인
    seq2_mode = (state_manager.combat_mode or "SCAN").upper()
    
    # 모드별 처리
    if seq2_mode == "SCAN":
        return handle_scan_mode(image_path, img_pil, state_manager)
    elif seq2_mode in ["STANDBY", "FIRE"]:
        return handle_standby_mode(image_path, img_pil, state_manager, now)
    else:
        return jsonify({"error": f"Unknown mode: {seq2_mode}"}), 400


@app.route('/set_seq2_mode', methods=['POST'])
def set_seq2_mode():
    """
    SEQ 2 전투 모드 변경 (SCAN ↔ STANDBY)
    """
    if state_manager.seq != 2:
        return jsonify({'status': 'error', 'msg': 'Not in SEQ 2'}), 400
    
    data = request.get_json(force=True) or {}
    mode = (data.get('mode') or '').upper().strip()
    
    if mode not in ["SCAN", "STANDBY"]:
        return jsonify({'status': 'error', 'msg': 'mode must be SCAN or STANDBY'}), 400
    
    state_manager.combat_mode = mode
    state_manager.mode_ts = time.time()
    
    # SCAN 모드로 전환 시 잠금 상태 초기화
    if mode == "SCAN":
        state_manager.locked_tid = None
        state_manager.locked_bbox = None
        state_manager.locked_ts = 0.0
        state_manager.last_target = None
        state_manager.last_target_ts = time.time()
        state_manager.last_detected_distance = None
        
        # LiDAR 잠금도 해제
        try:
            unlock_lidar_target(state_manager)
        except Exception:
            pass
    
    return jsonify({'status': 'OK', 'mode': state_manager.combat_mode})


@app.route('/info', methods=['POST'])
def info():
    """
    로봇 위치 및 카메라/터렛 정보 수신
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400
    
    # 기본 위치 정보 업데이트
    position = data.get("playerPos", {})
    rx = position.get('x', 0)
    ry = position.get('y', 0)
    rz = position.get('z', 0)
    state_manager.update_robot_pose(rx, rz, ry)

    if 'playerTurretX' in data:
        state_manager.player_turret_x = float(data['playerTurretX'])
    if 'playerBodyX' in data:
        state_manager.player_body_x = float(data['playerBodyX'])
    
    # 카메라/터렛 정보 업데이트 (DataFrame 병합용)
    state_manager.update_camera_turret_info(data)
    
    # SEQ 2 전투 시스템용 카메라 포즈 업데이트
    if state_manager.seq == 2:
        stereo_left_pos = data.get("stereoCameraLeftPos")
        stereo_left_rot = data.get("stereoCameraLeftRot")
        
        if stereo_left_pos and stereo_left_rot:
            state_manager.cam_pos = stereo_left_pos
            state_manager.cam_rot = stereo_left_rot
            state_manager.last_pose_ts = time.time()
    
    return jsonify({"status": "success"})

@app.route('/get_action', methods=['POST'])
def get_action():
    """
    제어 명령 생성 - SEQ에 따라 분기
    
    SEQ 2: 전투 시스템 (compute_combat_action)
    SEQ 1, 3, 4: 주행 시스템 (HybridController)
    """
    data = request.get_json(force=True)
    
    # 디버깅 카운트 로직 유지
    if not hasattr(get_action, '_call_count'):
        get_action._call_count = 0
    get_action._call_count += 1
    if get_action._call_count % 20 == 1:
        print(f"📡 [get_action] 호출 #{get_action._call_count}, "
              f"SEQ={state_manager.seq}, dest={state_manager.destination}")

    # SEQ 2: 전투 시스템
    if state_manager.seq == 2:
        curr_tx, curr_ty, curr_bx, curr_by = state_manager.parse_unity_combat_data(data)
        # 정밀 자동 공격 모드일 경우 우선 실행
        if getattr(state_manager, 'auto_attack_active', False):
            return jsonify(state_manager.compute_precision_attack(curr_tx, curr_ty, curr_bx, curr_by))

        # 일반 전투 로직 유지
        mode = (state_manager.combat_mode or "SCAN").upper()
        lidar_points = data.get('lidarPoints', [])
        
        # STANDBY가 아니면 LiDAR 포인트 무시
        if mode != "STANDBY":
            lidar_points = []
            try:
                if state_manager.lidar_lock.locked:
                    unlock_lidar_target(state_manager)
            except Exception: pass
        
        command = compute_combat_action(state_manager, lidar_points, sm_cfg)
        
        # SEQ 전환 로직 유지
        if getattr(state_manager, 'seq_change_request', None):
            new_seq = state_manager.seq_change_request
            state_manager.seq_change_request = None
            state_manager.seq = new_seq
            state_manager.combat_mode = "SCAN"
            state_manager.fire_ready = False

            if new_seq == 3:
                state_manager.destination = (49, 236)
                state_manager.clear_path()
                state_manager.set_log("[AUTO] 사전 저장된 경유지 좌표로 이동")

        return jsonify(command)
    
    # SEQ 1, 3, 4: 주행 시스템
    pos = data.get("position", {})
    curr_x = pos.get("x", 0)
    curr_z = pos.get("z", 0)
    
    turret = data.get("turret", {})
    curr_yaw = turret.get("x", 0)

    command = controller.compute_action(curr_x, curr_z, curr_yaw)

    return jsonify(command)


@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    """
    탄환 이벤트 수신 (SEQ 2)
    
    탄착 확인 및 Hit 카운트 업데이트
    """
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR"}), 400
    
    # SEQ 2에서만 처리
    if state_manager.seq != 2:
        return jsonify({"status": "OK"})
    
    ts = time.time()
    state_manager.last_bullet_event = data
    state_manager.last_bullet_ts = ts
    
    # Hit 감지 로직
    candidate_hit = False
    hit_position = None
    
    if isinstance(data, dict):
        # Hit 플래그 확인
        if data.get("hit") is True or data.get("isHit") is True:
            candidate_hit = True
        
        # 데미지 확인
        if "damage" in data and float(data.get("damage", 0) or 0) > 0:
            candidate_hit = True
        
        # Hit 위치 확인
        hit_pos = data.get("hitPosition") or data.get("position")
        if isinstance(hit_pos, dict) and all(k in hit_pos for k in ("x", "y", "z")):
            hit_position = hit_pos
            candidate_hit = True
        elif all(k in data for k in ("x", "y", "z")):
            hit_position = {
                "x": float(data["x"]), 
                "y": float(data["y"]), 
                "z": float(data["z"])
            }
            candidate_hit = True
    
    if candidate_hit:
        # 중복 Hit 방지 (0.3초 이내 중복 무시)
        if state_manager.hit_flag and (ts - state_manager.hit_ts) < 0.3:
            print("[BULLET] ⚠️ 중복 사격 무시")
            return jsonify({"status": "ok", "duplicate": True})
        
        # Hit 상태 업데이트
        state_manager.hit_flag = True
        state_manager.hit_ts = ts
        state_manager.hit_count += 1
        if hit_position:
            state_manager.last_hit_xyz = hit_position
        
        print(f"[BULLET] 🎯 HIT CONFIRMED! count={state_manager.hit_count}")
    
    return jsonify({
        "status": "ok", 
        "hit_detected": candidate_hit, 
        "hit_count": state_manager.hit_count
    })


@app.route('/change_seq', methods=['POST'])
def change_seq():
    """
    SEQ 변경 전용 엔드포인트
    
    SEQ별 LiDAR 및 Costmap 제어:
        - SEQ 1, 3: LiDAR OFF, Costmap OFF
        - SEQ 2: LiDAR ON, Costmap OFF (센서퓨전 전용)
        - SEQ 4: LiDAR ON, Costmap ON (장애물 회피)
    """
    data = request.get_json()
    
    if "seq" not in data:
        return jsonify({"status": "ERROR", "msg": "seq parameter required"}), 400
    
    new_seq = int(data["seq"])
    if new_seq not in [1, 2, 3, 4]:
        return jsonify({"status": "ERROR", "msg": "seq must be 1, 2, 3, or 4"}), 400
    
    old_seq = state_manager.seq
    state_manager.seq = new_seq
    print(f"🔄 SEQ 변경: {old_seq} → {new_seq}")
    
    # SEQ별 A* 안전성 파라미터 동적 조정
    planner.set_obstacle_margin(config.ASTAR.get_obstacle_margin(new_seq))
    planner.set_safety_weight(config.ASTAR.get_safety_weight(new_seq))
    planner.set_proximity_radius(config.ASTAR.get_proximity_radius(new_seq))
    
    # SEQ별 LiDAR 및 Costmap 제어
    if new_seq == 2:
        lidar_logger.enable_costmap()
    else:
        lidar_logger.disable_costmap()
    
    # SEQ 2: 전투 시스템 초기화
    if new_seq == 2:
        state_manager.combat_mode = "SCAN"
        state_manager.scan_start_ts = time.time()
        state_manager.scan_direction = None  # 사용자 입력 대기
        state_manager.is_lowering_barrel = True # 포신 하향 시퀀스 활성화
        state_manager.fire_ready = False
        state_manager.is_aim_aligned = False
        state_manager.standby_target = None
        state_manager.user_action = None
        state_manager.mode_ts = time.time()
        
        # 경로 정보 클리어
        state_manager.destination = None
        state_manager.clear_path()
        print("⚔️ 전투 시스템 활성화 (SEQ 2) - SCAN 모드 시작")
    
    # SEQ 4: 자율주행 시스템 초기화
    elif new_seq == 4:
        planner.set_obstacles([])  # 기존 A* 장애물 클리어
        state_manager.destination = None
        state_manager.clear_path()
        state_manager.costmap = None
        state_manager.costmap_origin = None
        print("🤖 자율주행 시스템 활성화 (SEQ 4) - LiDAR Costmap 주행 모드")
    
    # SEQ 1, 3: 일반 주행 시스템
    else:
        if state_manager.destination is not None:
            state_manager.destination = None
            state_manager.clear_path()
        print(f"🚗 주행 시스템 활성화 (SEQ {new_seq})")
    
    return jsonify({
        "status": "OK",
        "old_seq": old_seq,
        "new_seq": new_seq,
        "msg": f"SEQ changed from {old_seq} to {new_seq}"
    })


@app.route('/set_destination', methods=['POST'])
def set_destination_route():
    """
    목적지 설정 엔드포인트
    
    SEQ 변경 및 목적지 좌표 설정 동시 지원
    """
    data = request.get_json()
    
    # SEQ 변경 처리
    if "seq" in data:
        new_seq = int(data["seq"])
        state_manager.seq = new_seq
        print(f"🔄 SEQ 변경: {new_seq}")
        
        # SEQ별 LiDAR 모니터링 및 Costmap 제어
        if new_seq in [2, 4]:
            lidar_logger.start()
            if new_seq == 4:
                lidar_logger.enable_costmap()
            else:
                lidar_logger.disable_costmap()
        else:
            lidar_logger.stop()
            lidar_logger.disable_costmap()
        
        # SEQ 2 전투 모드 초기화 (Legacy 호환성)
        if new_seq == 2:
            state_manager.combat_mode = "ENGAGE"
            state_manager.mode_ts = time.time()
            print("⚔️ 전투 시스템 활성화 (SEQ 2)")
    
    # 목적지 설정 처리
    if "destination" in data:
        try:
            x, y, z = map(float, data["destination"].split(","))
            state_manager.set_destination(x, z)
            controller.reset()
            state_manager.set_log(f"🚩 목적지 설정: ({x:.1f}, {z:.1f}) - SEQ {state_manager.seq}")
            # print(f"🚩 목적지 설정: ({x:.1f}, {z:.1f}) - SEQ {state_manager.seq}")
            
            return jsonify({
                "status": "OK",
                "destination": {"x": x, "y": y, "z": z},
                "seq": state_manager.seq
            })
        except Exception as e:
            state_manager.set_log(f"❌ 목적지 설정 오류: {e}")
            print(f"❌ 목적지 설정 오류: {e}")
            return jsonify({"status": "ERROR", "msg": str(e)})
    
    return jsonify({"status": "ERROR"})


@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    """
    장애물 정보 업데이트
    
    SEQ 2: 이벤트만 저장 (처리 안함)
    SEQ 1, 3, 4: A* 플래너에 즉시 반영
    """
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error'}), 400
    
    # SEQ 2에서는 장애물 이벤트만 저장
    if state_manager.seq == 2:
        ts = time.time()
        state_manager.last_obstacle_event = data
        state_manager.last_obstacle_ts = ts
        print(f"[OBS] ✅ Received t={ts:.3f}")
        return jsonify({'status': 'success'})
    
    # SEQ 1, 3, 4에서는 A* 플래너에 즉시 업데이트
    planner.update_obstacles_from_payload(data)
    if state_manager.seq in [1, 3, 4]:
        obstacles = data.get("obstacles", [])
        for obs in obstacles:
            # 사각형 장애물의 중심점 계산
            cx = (obs["x_min"] + obs["x_max"]) / 2
            cz = (obs["z_min"] + obs["z_max"]) / 2
            
            # 전역 장애물 리스트에 추가 (DWA가 가상 라이다처럼 참조함)
            state_manager.add_global_obstacles(cx, cz)
            
    return jsonify({'status': 'success'})


@app.route('/collision', methods=['POST'])
def collision():
    """
    충돌 감지 처리
    
    stuck_counter 강제 증가로 빠른 복구 유도
    """
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error'}), 400
    
    # Stuck 카운터 강제 증가
    controller.stuck_counter = max(
        controller.stuck_counter, 
        config.Stuck.STUCK_COUNT_LIMIT - 1
    )
    
    # UI 로그 메시지 업데이트
    msg = f"⚠️ 충돌 감지! stuck_counter={controller.stuck_counter}"
    state_manager.set_log(msg=msg)
    
    return jsonify({'status': 'success'})


# ==============================================================================
# LiDAR 파일 관리 엔드포인트
# ==============================================================================

@app.route('/set_lidar_cleanup', methods=['POST'])
def set_lidar_cleanup():
    """
    LiDAR 파일 자동 정리 모드 설정
    
    지원 모드:
        - none: 정리 안함
        - after_process: 처리 후 삭제
        - keep_recent: 최신 N개 파일만 유지
        - max_age: 일정 시간 경과 파일 삭제
    
    Request JSON:
        mode: 정리 모드
        max_files: (선택) keep_recent 모드에서 유지할 파일 수
        max_age_sec: (선택) max_age 모드에서 최대 파일 수명
    """
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'msg': 'No JSON received'}), 400
    
    mode = data.get('mode')
    if not mode:
        return jsonify({'status': 'error', 'msg': 'mode parameter required'}), 400
    
    valid_modes = ["none", "after_process", "keep_recent", "max_age"]
    if mode not in valid_modes:
        return jsonify({
            'status': 'error',
            'msg': f'Invalid mode. Valid modes: {valid_modes}'
        }), 400
    
    max_files = data.get('max_files')
    max_age_sec = data.get('max_age_sec')
    
    lidar_logger.set_cleanup_mode(mode, max_files, max_age_sec)
    
    return jsonify({
        'status': 'success',
        'mode': mode,
        'max_files': lidar_logger.max_files,
        'max_age_sec': lidar_logger.max_age_sec
    })


@app.route('/force_lidar_cleanup', methods=['POST'])
def force_lidar_cleanup():
    """LiDAR 폴더 강제 정리"""
    lidar_logger.force_cleanup()
    return jsonify({'status': 'success', 'msg': 'LiDAR folder cleaned'})


@app.route('/lidar_cleanup_status', methods=['GET'])
def lidar_cleanup_status():
    """현재 LiDAR 정리 모드 상태 확인"""
    return jsonify({
        'mode': lidar_logger.auto_cleanup_mode,
        'max_files': lidar_logger.max_files,
        'max_age_sec': lidar_logger.max_age_sec,
        'folder': lidar_logger.lidar_folder
    })


# ==============================================================================
# Unity 초기화 엔드포인트
# ==============================================================================

@app.route('/init', methods=['GET'])
def init():
    """Unity 초기화 설정 반환"""
    config_data = {
        "startMode": "start",
        "blStartX": 130, "blStartY": 15, "blStartZ": 30,
        "rdStartX": 300, "rdStartY": 10, "rdStartZ": 300,
        "trackingMode": True,
        "detectMode": True,
        "logMode": True,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": True,
        "saveLidarData": True,
        "lux": 30000,
        "destoryObstaclesOnHit": True
    }
    return jsonify(config_data)


@app.route('/start', methods=['GET'])
def start():
    """Unity 시작 명령"""
    return jsonify({"control": ""})


# ==============================================================================
# SEQ 2 전투 시스템 전용 엔드포인트
# ==============================================================================

@app.route('/lock_target', methods=['POST'])
def lock_target_endpoint():
    """
    수동 타겟 잠금 (SEQ 2)
    
    Request JSON:
        angle: 수평 각도
        verticalAngle: 수직 각도
        distance: 거리
    """
    try:
        data = request.get_json()
        angle = data.get('angle', 0.0)
        v_angle = data.get('verticalAngle', 0.0)
        distance = data.get('distance', 0.0)
        
        lock_lidar_target(state_manager, angle, v_angle, distance)
        
        return jsonify({
            "status": "success",
            "message": "Target locked",
            "target": {
                "angle": angle,
                "verticalAngle": v_angle,
                "distance": distance
            }
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/unlock_target', methods=['POST'])
def unlock_target_endpoint():
    """타겟 잠금 해제 (SEQ 2)"""
    try:
        unlock_lidar_target(state_manager)
        return jsonify({"status": "success", "message": "Target unlocked"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/target_status', methods=['GET'])
def target_status():
    """
    타겟 상태 확인 (SEQ 2)
    
    Returns:
        잠금 상태 및 타겟 정보
    """
    try:
        target_info = get_lidar_target_info(state_manager)
        
        if target_info:
            return jsonify({
                "status": "locked",
                "target": target_info
            }), 200
        else:
            return jsonify({
                "status": "unlocked"
            }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/overlay/left", methods=["GET"])
def get_overlay():
    """
    오버레이 이미지 반환 (SEQ 2)
    
    센서 퓨전 결과 시각화 이미지 제공
    """
    if state_manager.overlay_left_bytes is None:
        # 기본 이미지 생성
        img = Image.new("RGB", (640, 480), color=(50, 50, 50))
        draw = ImageDraw.Draw(img)
        draw.text((200, 230), "No data yet", fill=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    
    return send_file(
        io.BytesIO(state_manager.overlay_left_bytes), 
        mimetype="image/png"
    )


@app.route('/set_scan_direction', methods=['POST'])
def set_scan_direction():
    """
    SCAN 모드 방향 설정 (Q: 좌측, E: 우측)
    
    Request JSON:
        direction: "Q" 또는 "E"
    """
    data = request.get_json()
    if not data or 'direction' not in data:
        return jsonify({'status': 'error', 'msg': 'direction required'}), 400
    
    direction = data.get('direction', '').upper()
    if direction not in ["Q", "E"]:
        return jsonify({'status': 'error', 'msg': 'Invalid direction'}), 400
    
    state_manager.scan_direction = direction
    state_manager.is_lowering_barrel = True
    state_manager.scan_start_ts = time.time()
    state_manager.mode_ts = time.time()
    state_manager.scan_init_msg_sent = False  # 새 방향 로그 출력 리셋
    state_manager.enemy_msg_sent = False      # 적 감지 상태 리셋
    
    return jsonify({'status': 'OK', 'direction': direction})


@app.route('/combat_action', methods=['POST'])
def combat_action():
    """
    SEQ 2 사용자 액션 처리
    
    지원 액션:
        - FIRE: 자동 조준 및 포격 (STANDBY/SCAN 모드에서)
        - RESCAN: 재탐색 (SCAN/STANDBY 모드에서)
        - RETREAT: 후퇴 (SCAN/STANDBY 모드에서)
    
    FIRE 동작:
        1. auto_attack_active 플래그 활성화
        2. 시스템이 자동으로 정밀 조준
        3. 조준 완료 시 자동 발사
    """
    data = request.get_json()
    if not data or 'action' not in data:
        return jsonify({'status': 'error', 'msg': 'action required'}), 400
    
    action = data['action'].upper()

    # SEQ 2 확인
    if state_manager.seq != 2:
        return jsonify({'status': 'error', 'msg': 'Not in SEQ 2'}), 400

    # 1. FIRE 액션 처리 (자동 포격)
    if action == "FIRE":
        # 자동 포격 플래그 활성화
        state_manager.auto_attack_active = True
        state_manager.user_action = "FIRE" 
        return jsonify({
            "status": "OK", 
            "action": action,
            "msg": "Fire command received. Auto-aiming and firing."
        })

    # 2. 기존 액션 (RESCAN, RETREAT) 처리
    if action not in ["RESCAN", "RETREAT"]:
        return jsonify({'status': 'error', 'msg': 'Invalid action'}), 400
    
    # 허용 가능한 모드 확인
    allowed_modes = ["STANDBY", "SCAN"]
    if state_manager.combat_mode not in allowed_modes:
        return jsonify({
            'status': 'error',
            'msg': f'{action} 액션은 현재 모드에서 불가능합니다.'
        }), 400
    
    # StateManager에 액션 전달
    state_manager.user_action = action
    
    return jsonify({
        'status': 'OK',
        'action': action,
        'current_mode': state_manager.combat_mode
    })

# ==============================================================================
# 디버그/시각화 엔드포인트
# ==============================================================================

@app.route('/debug_status')
def debug_status():
    """
    통합 디버그 상태 조회
    
    SEQ별 상태 정보:
        - 공통: 메시지, 로그, SEQ 번호
        - SEQ 2: 전투 시스템 상태 (모드, 타겟, Hit 정보 등)
        - SEQ 4: 자율주행 시스템 상태 (장애물, 경로 등)
    """
    status = viz_manager.get_status_json()
    status["msg"] = state_manager.status_message
    status["log"] = state_manager.last_log
    status["seq"] = state_manager.seq
    
    # 좌표 범위 정보 추가 (SEQ별로 동적 제공)
    seq = state_manager.seq
    if seq == 1:
        # SEQ 1: A* 플래너 범위
        status["path_bounds"] = {
            "x_min": 65.0,
            "x_max": 200.0,
            "z_min": 0.0,
            "z_max": 220.0
        }
    elif seq == 3:
        # SEQ 3: 다른 맵 범위
        status["path_bounds"] = {
            "x_min": 0.0,
            "x_max": 200.0,
            "z_min": 150.0,
            "z_max": 300.0
        }
    
    # SEQ 2 전투 시스템 정보
    if state_manager.seq == 2:
        now = time.time()
        status["combat_mode"] = state_manager.combat_mode
        status["scan_direction"] = state_manager.scan_direction
        status["fire_ready"] = state_manager.fire_ready
        status["is_aim_aligned"] = state_manager.is_aim_aligned
        status["hit_count"] = state_manager.hit_count
        status["lidar_lock_active"] = state_manager.lidar_lock.locked
        status["lidar_lock_count"] = state_manager.lidar_lock.lock_count
        status["lidar_lock_fires"] = state_manager.lidar_lock.successful_fires
        status["last_shot_age_sec"] = (
            round(now - state_manager.last_shot_ts, 2) 
            if state_manager.last_shot_ts else None
        )
        status["detected_targets"] = state_manager.get_scan_targets_for_display()
        status["locked_target"] = state_manager.last_target
    
    # SEQ 4 자율주행 시스템 정보
    if state_manager.seq == 4:
        nearby_obstacles = []
        lidar_range = 50.0
        pose = status.get('tank_pose', [60.0, 0.0, 200.0])
        tank_x = pose[0]
        tank_z = pose[-1]

        # obstacle_rects에서 직접 장애물 추출 (가장 신뢰성 높은 소스)
        if hasattr(state_manager, 'obstacle_rects') and state_manager.obstacle_rects:
            for obs in state_manager.obstacle_rects:
                cx = (obs['x_min'] + obs['x_max']) / 2
                cz = (obs['z_min'] + obs['z_max']) / 2
                size_x = obs['x_max'] - obs['x_min']
                size_z = obs['z_max'] - obs['z_min']
                dist = math.hypot(cx - tank_x, cz - tank_z)

                if dist <= lidar_range:
                    nearby_obstacles.append({
                        'x': float(cx),
                        'z': float(cz),
                        'size': float(max(size_x, size_z)),
                        'type': 'rect',
                        'distance': float(dist)
                    })

        # global_obstacles 보조 소스 (costmap에서 누적된 점)
        if hasattr(state_manager, 'global_obstacles') and state_manager.global_obstacles:
            existing_keys = set(f"{o['x']:.0f}_{o['z']:.0f}" for o in nearby_obstacles)
            grid_size = state_manager.global_obstacle_grid_size
            for ox, oz in state_manager.global_obstacles:
                key = f"{ox:.0f}_{oz:.0f}"
                if key not in existing_keys:
                    dist = math.hypot(ox - tank_x, oz - tank_z)
                    if dist <= lidar_range:
                        nearby_obstacles.append({
                            'x': float(ox),
                            'z': float(oz),
                            'size': float(grid_size),
                            'type': 'global',
                            'distance': float(dist)
                        })

        # SEQ4 전용 데이터
        heading = 0.0
        if hasattr(state_manager, 'robot_yaw_deg') and state_manager.robot_yaw_deg is not None:
            heading = state_manager.robot_yaw_deg

        seq4_data = {
            "lidar_range": lidar_range,
            "nearby": nearby_obstacles,
            "heading": heading,
            "path": state_manager.global_path[:20] if state_manager.global_path else [],
        }
        status["seq4"] = seq4_data

        # 기존 호환성 유지
        status["lidar_obstacles"] = nearby_obstacles
        status["obstacle_count"] = len(nearby_obstacles)
    
    return jsonify(status)


@app.route('/view_costmap')
def view_costmap():
    """Costmap 시각화 이미지 반환"""
    buf = viz_manager.render_scene("costmap")
    return send_file(buf, mimetype="image/png")


@app.route('/view_global')
def view_global():
    """전역 경로 시각화 이미지 반환"""
    buf = viz_manager.render_scene("global")
    return send_file(buf, mimetype="image/png")


@app.route('/view_local')
def view_local():
    """로컬 경로 시각화 이미지 반환"""
    buf = viz_manager.render_scene("local")
    return send_file(buf, mimetype="image/png")


@app.route('/view_path')
def view_path():
    """실시간 경로 이미지 (현재 위치 포함)"""
    buf = viz_manager.render_path(planner)
    return send_file(buf, mimetype="image/png")


@app.route('/realtime_path_image')
def realtime_path_image():
    """
    SEQ 1, 3용 실시간 경로 추적 이미지
    
    - A* 경로와 장애물을 심플하게 표시
    - 지나온 경로는 실선, 남은 경로는 점선
    - 현재 전차 위치를 빨간색 갈매기 화살표로 표시
    - 목적지를 검정색 깃발로 표시
    
    Query Parameters:
        width: 이미지 너비 (기본 640)
        height: 이미지 높이 (기본 640)
    """
    # 이미지 크기 파라미터
    width = request.args.get('width', 640, type=int)
    height = request.args.get('height', 640, type=int)
    
    # 크기 제한 (100~1200px)
    width = max(100, min(1200, width))
    height = max(100, min(1200, height  ))
    
    buf = viz_manager.render_realtime_path_image(planner, image_size=(width, height))
    return send_file(buf, mimetype="image/png")


@app.route('/view_autonomous')
def view_autonomous():
    """
    SEQ 4 자율주행용 통합 뷰
    
    Costmap + 경로 + LiDAR 장애물 통합 시각화
    """
    buf = viz_manager.render_autonomous(planner, lidar_logger)
    return send_file(buf, mimetype="image/png")


@app.route('/get_static_path/<int:seq>')
def get_static_path(seq):
    """
    SEQ별 정적 전역 경로 이미지 반환
    
    Args:
        seq: SEQ 번호 (1 또는 3)
    """
    filename = f"SEQ {seq}_Global_Path.png"
    # 파일이 루트에 있다면 직접 전송, 혹은 static 폴더에 있다면 경로 수정 필요
    try:
        return send_file(filename, mimetype='image/png')
    except Exception:
        # 파일이 없을 경우 빈 이미지 반환
        img = Image.new("RGB", (640, 480), color=(50, 50, 50))
        draw = ImageDraw.Draw(img)
        draw.text((200, 220), f"SEQ {seq} 경로 이미지 없음", fill=(150, 150, 150))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")


@app.route('/monitor')
def monitor():
    """SEQ 기반 자동 전환 모니터링 UI"""
    return render_template('monitor.html')


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 TBSA System (Navigation + Combat)")
    print("=" * 60)
    print(f"📂 LiDAR: {config.Lidar.LIDAR_FOLDER}")
    print(f"🚗 SEQ 1, 3: A* + PID 주행")
    print(f"⚔️ SEQ 2: LiDAR + YOLO 전장 상황 인식")
    print(f"🤖 SEQ 4: PPO 강화학습 + A* 하이브리드 자율주행")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)