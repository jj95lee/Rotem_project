"""
onnx_detector.py - ONNX 기반 YOLO 객체 탐지 모듈

[주요 기능]
- ONNX Runtime을 사용한 YOLO 모델 추론
- FP16 + 640x640 해상도 지원
- 수동 NMS (Non-Maximum Suppression) 구현
- PIL Image 직접 입력 지원 (파일 경로 불필요)

[사용법]
    from utils.onnx_detector import OnnxYoloDetector
    
    # 단일 모델
    detector = OnnxYoloDetector("models/cannon.onnx")
    results = detector.detect(pil_image, conf_threshold=0.25)
    
    # 듀얼 모델 탐지
    results, meta = detect_all_objects_dual_onnx(
        img_pil, detector_cannon, detector_integrated, ...
    )
"""

import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import time

# ONNX Runtime import (GPU 우선, CPU 폴백)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[ONNX_DETECTOR] ⚠️ onnxruntime not installed. pip install onnxruntime-gpu")


class OnnxYoloDetector:
    """
    ONNX 기반 YOLO 객체 탐지기
    
    Args:
        model_path: ONNX 모델 파일 경로
        input_size: 모델 입력 크기 (width, height), 기본값 (640, 640)
        use_gpu: GPU 사용 여부, 기본값 True (사용 가능 시)
        fp16: FP16 모드 여부, 기본값 False
    
    Example:
        detector = OnnxYoloDetector("models/best.onnx")
        detections = detector.detect(pil_image, conf_threshold=0.25)
    """
    
    def __init__(
        self, 
        model_path: str, 
        input_size: Tuple[int, int] = (640, 640),
        use_gpu: bool = True,
        fp16: bool = False
    ):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed")
        
        self.model_path = model_path
        self.input_size = input_size  # (width, height)
        self.fp16 = fp16
        
        # ONNX Runtime 세션 생성
        self.session = self._create_session(model_path, use_gpu)
        
        # 입력/출력 정보 추출
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # 입력 shape 확인 (NCHW 형식)
        input_shape = self.session.get_inputs()[0].shape
        if input_shape[2] is not None and input_shape[3] is not None:
            self.input_size = (input_shape[3], input_shape[2])  # (W, H)
        
        print(f"[ONNX_DETECTOR] ✅ 모델 로드 완료: {model_path}")
        print(f"  - 입력 크기: {self.input_size}")
        print(f"  - FP16 모드: {self.fp16}")
        print(f"  - 출력 레이어: {self.output_names}")
    
    def _create_session(self, model_path: str, use_gpu: bool) -> 'ort.InferenceSession':
        """ONNX Runtime 세션 생성"""
        providers = []
        
        if use_gpu:
            # CUDA 사용 가능 시 GPU 우선
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
                print("[ONNX_DETECTOR] 🚀 CUDA GPU 가속 활성화")
            elif 'DmlExecutionProvider' in ort.get_available_providers():
                providers.append('DmlExecutionProvider')
                print("[ONNX_DETECTOR] 🚀 DirectML GPU 가속 활성화")
        
        # CPU 폴백
        providers.append('CPUExecutionProvider')
        
        # GPU 가속이 없으면 CPU 사용 메시지 출력
        if len(providers) == 1:
            print("[ONNX_DETECTOR] 🐢 CPU 모드로 실행 (GPU 가속 없음)")
            
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        return ort.InferenceSession(model_path, sess_options, providers=providers)
    
    def preprocess(self, img_pil: Image.Image) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """
        이미지 전처리 (PIL Image -> ONNX 입력 텐서)
        
        Args:
            img_pil: PIL Image 객체
        
        Returns:
            Tuple: (input_tensor, scale_factors, original_size)
                - input_tensor: NCHW 형식의 float32 텐서
                - scale_factors: (scale_x, scale_y) 스케일 비율
                - original_size: (원본_width, 원본_height)
        """
        original_size = img_pil.size  # (width, height)
        target_w, target_h = self.input_size
        
        # 리사이즈 (비율 유지하며 letterbox)
        img_resized, scale, pad = self._letterbox(img_pil, (target_w, target_h))
        
        # PIL -> numpy (RGB)
        img_np = np.array(img_resized, dtype=np.float32)
        
        # 정규화 [0, 255] -> [0, 1]
        img_np = img_np / 255.0
        
        # HWC -> CHW
        img_np = img_np.transpose(2, 0, 1)
        
        # 배치 차원 추가 (NCHW)
        img_np = np.expand_dims(img_np, axis=0)
        img_np = img_np.astype(np.float32)
        
        # 스케일 팩터 계산 (복원용)
        scale_factors = (scale, pad)
        
        return img_np, scale_factors, original_size
    
    def _letterbox(
        self, 
        img_pil: Image.Image, 
        target_size: Tuple[int, int],
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[Image.Image, float, Tuple[int, int]]:
        """
        Letterbox 리사이징 (비율 유지 + 패딩)
        
        Args:
            img_pil: 원본 PIL Image
            target_size: 목표 크기 (width, height)
            color: 패딩 색상 (R, G, B)
        
        Returns:
            Tuple: (리사이즈된 이미지, 스케일, 패딩)
        """
        orig_w, orig_h = img_pil.size
        target_w, target_h = target_size
        
        # 스케일 계산 (비율 유지)
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # 리사이즈
        img_resized = img_pil.resize((new_w, new_h), Image.BILINEAR)
        
        # 패딩 계산
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        # 새 캔버스에 배치
        new_img = Image.new("RGB", target_size, color)
        new_img.paste(img_resized, (pad_w, pad_h))
        
        return new_img, scale, (pad_w, pad_h)
    
    def postprocess(
        self, 
        outputs: List[np.ndarray], 
        scale_factors: Tuple[float, Tuple[int, int]],
        original_size: Tuple[int, int],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict]:
        """
        모델 출력 후처리 (NMS 포함)
        
        Args:
            outputs: ONNX 모델 출력
            scale_factors: (scale, (pad_w, pad_h))
            original_size: (원본_width, 원본_height)
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
        
        Returns:
            List[Dict]: 탐지 결과 리스트
                각 딕셔너리: {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class_id": int
                }
        """
        # YOLO 출력 형식: (1, num_classes + 4, num_boxes) 또는 (1, num_boxes, num_classes + 4)
        output = outputs[0]
        
        # 출력 shape 확인 및 변환
        if len(output.shape) == 3:
            # (1, 84, 8400) -> (8400, 84) 형식으로 변환
            if output.shape[1] < output.shape[2]:
                output = output[0].T  # (8400, 84)
            else:
                output = output[0]    # (8400, 84)
        elif len(output.shape) == 2:
            pass  # 이미 (num_boxes, features) 형식
        
        # bbox (cx, cy, w, h) + class scores 분리
        # YOLOv8 형식: [cx, cy, w, h, class0_score, class1_score, ...]
        boxes = output[:, :4]  # (N, 4)
        scores = output[:, 4:]  # (N, num_classes)
        
        # 각 박스의 최대 클래스 점수 및 클래스 ID
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # 신뢰도 필터링
        mask = confidences >= conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
        
        # cx, cy, w, h -> x1, y1, x2, y2 변환
        boxes_xyxy = self._cxcywh_to_xyxy(boxes)
        
        # 좌표 복원 (letterbox 역변환)
        scale, (pad_w, pad_h) = scale_factors
        orig_w, orig_h = original_size
        
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale
        
        # 이미지 범위로 클리핑
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)
        
        # NMS 적용
        indices = self._nms(boxes_xyxy, confidences, iou_threshold)
        
        # 결과 정리
        results = []
        for idx in indices:
            results.append({
                "bbox": boxes_xyxy[idx].tolist(),
                "confidence": float(confidences[idx]),
                "class_id": int(class_ids[idx])
            })
        
        return results
    
    def _cxcywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """
        중심 좌표 형식을 코너 좌표 형식으로 변환
        (cx, cy, w, h) -> (x1, y1, x2, y2)
        """
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return boxes_xyxy
    
    def _nms(
        self, 
        boxes: np.ndarray, 
        scores: np.ndarray, 
        iou_threshold: float
    ) -> List[int]:
        """
        Non-Maximum Suppression (NMS) 구현
        
        Args:
            boxes: (N, 4) 형식의 박스 좌표 [x1, y1, x2, y2]
            scores: (N,) 형식의 신뢰도 점수
            iou_threshold: IoU 임계값
        
        Returns:
            List[int]: 유지할 박스의 인덱스 리스트
        """
        if len(boxes) == 0:
            return []
        
        # 좌표 추출
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # 면적 계산
        areas = (x2 - x1) * (y2 - y1)
        
        # 점수 기준 내림차순 정렬
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            # 가장 높은 점수의 박스 선택
            i = order[0]
            keep.append(i)
            
            if len(order) == 1:
                break
            
            # 나머지 박스들과의 IoU 계산
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            inter_area = inter_w * inter_h
            
            union_area = areas[i] + areas[order[1:]] - inter_area
            iou = inter_area / (union_area + 1e-6)
            
            # IoU가 임계값 이하인 박스만 유지
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect(
        self, 
        img_pil: Image.Image,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict]:
        """
        객체 탐지 수행
        
        Args:
            img_pil: PIL Image 객체
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
        
        Returns:
            List[Dict]: 탐지 결과 리스트
        """
        # 전처리
        input_tensor, scale_factors, original_size = self.preprocess(img_pil)
        
        # 추론
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 후처리 (NMS 포함)
        results = self.postprocess(
            outputs, scale_factors, original_size,
            conf_threshold, iou_threshold
        )
        
        return results
    
    def detect_with_tracking_format(
        self,
        img_pil: Image.Image,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> np.ndarray:
        """
        YOLO tracking 형식과 호환되는 출력 반환
        
        Returns:
            np.ndarray: (N, 6) 형식 [x1, y1, x2, y2, conf, class_id]
        """
        results = self.detect(img_pil, conf_threshold, iou_threshold)
        
        if not results:
            return np.array([])
        
        output = []
        for det in results:
            bbox = det["bbox"]
            output.append([
                bbox[0], bbox[1], bbox[2], bbox[3],
                det["confidence"], det["class_id"]
            ])
        
        return np.array(output)


# ==============================================================================
# 듀얼 모델 탐지 함수 (기존 combat_system.py의 함수 대체)
# ==============================================================================

def detect_all_objects_dual_onnx(
    img_pil: Image.Image,
    detector_cannon: OnnxYoloDetector,
    detector_integrated: OnnxYoloDetector,
    combat_config,
    fusion_cfg,
    nms_iou_th: float = 0.5,
) -> Tuple[List[Dict], Dict]:
    """
    ONNX 듀얼 모델로 객체를 탐지하고 NMS 중첩 처리
    
    Args:
        img_pil: PIL Image 객체 (파일 경로 대신)
        detector_cannon: Cannon 전용 ONNX 탐지기
        detector_integrated: 통합 객체 ONNX 탐지기
        combat_config: CombatSystemConfig 인스턴스
        fusion_cfg: FusionConfig 인스턴스
        nms_iou_th: NMS IoU 임계값
    
    Returns:
        Tuple[List[Dict], Dict]: (탐지 결과 리스트, 메타데이터)
    """
    temp_detections = []
    
    model_configs = [
        {
            "detector": detector_cannon,
            "mapping": combat_config.map_cannon,
            "color": combat_config.color_cannon
        },
        {
            "detector": detector_integrated,
            "mapping": combat_config.map_integrated,
            "color": combat_config.color_integrated
        },
    ]
    
    for cfg in model_configs:
        # ONNX 추론
        detections = cfg["detector"].detect(
            img_pil,
            conf_threshold=fusion_cfg.min_det_conf,
            iou_threshold=0.45  # 내부 NMS
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
    
    # 크로스 모델 NMS (confidence 높은 순으로 IoU overlap 제거)
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
    
    # bbox 오버레이 커스터마이징 설정
    bbox_styles = {
        "Tank": {
            "color": "#FF0000",
            "filled": True,
            "show_confidence": True,
        },
        "Red": {
            "color": "#FF4444",
            "filled": True,
            "show_confidence": True,
        },
        "Tree": {
            "color": "#AAAAAA",
            "filled": True,
            "show_confidence": False,
        },
        "Rock": {
            "color": "#AAAAAA",
            "filled": True,
            "show_confidence": False,
        },
        "default": {
            "color": "#FFFFFF",
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
        
        style = bbox_styles.get(name, bbox_styles["default"])
        
        filtered_results.append({
            "className": name,
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
        "cannon_bbox": last_cannon_bbox
    }
    
    return filtered_results, meta


def detect_tank_only_track_onnx(
    img_pil: Image.Image,
    detector: OnnxYoloDetector,
    class_map: dict,
    color_hex: str,
    min_det_conf: float,
    min_box_w: float,
    min_box_h: float,
    prev_detections: List[Dict] = None,
    iou_threshold: float = 0.3
) -> List[Dict]:
    """
    ONNX 기반 Tank 전용 탐지 (간단한 IoU 기반 트래킹)
    
    Args:
        img_pil: PIL Image 객체
        detector: ONNX 탐지기
        class_map: 클래스 ID -> 이름 매핑
        color_hex: 박스 색상 (hex)
        min_det_conf: 최소 신뢰도
        min_box_w: 최소 박스 너비
        min_box_h: 최소 박스 높이
        prev_detections: 이전 프레임 탐지 결과 (트래킹용)
        iou_threshold: 트래킹 IoU 임계값
    
    Returns:
        List[Dict]: Tank 탐지 결과 리스트
    """
    # Tank class id 추출
    tank_cls_ids = [cid for cid, name in class_map.items() if name == "Tank"]
    
    # ONNX 추론
    all_detections = detector.detect(
        img_pil,
        conf_threshold=min_det_conf,
        iou_threshold=0.45
    )
    
    # Tank 클래스만 필터링
    tank_detections = []
    for det in all_detections:
        if det["class_id"] in tank_cls_ids:
            bbox = det["bbox"]
            xmin, ymin, xmax, ymax = bbox
            
            if (xmax - xmin) < min_box_w or (ymax - ymin) < min_box_h:
                continue
            
            tank_detections.append({
                "bbox": bbox,
                "confidence": det["confidence"],
                "class_id": det["class_id"]
            })
    
    # 간단한 IoU 기반 트래킹 (track_id 할당)
    results = []
    next_track_id = 1
    
    # 이전 탐지 결과가 있으면 IoU 기반으로 track_id 연결
    if prev_detections:
        used_prev_ids = set()
        
        for det in tank_detections:
            best_iou = 0
            best_prev_id = None
            
            for prev in prev_detections:
                if prev.get("track_id") in used_prev_ids:
                    continue
                
                iou = _iou(det["bbox"], prev["bbox"])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_prev_id = prev.get("track_id")
            
            if best_prev_id is not None:
                track_id = best_prev_id
                used_prev_ids.add(track_id)
            else:
                # 새로운 track_id 할당
                track_id = max([p.get("track_id", 0) for p in prev_detections] + [0]) + 1
            
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
        
        results.append({
            "className": display,
            "category": "tank",
            "bbox": det["bbox"],
            "confidence": conf,
            "color": color_hex,
            "filled": False,
            "updateBoxWhileMoving": False,
            "track_id": track_id,
        })
    
    return results


def _iou(box1: List[float], box2: List[float]) -> float:
    """
    두 박스 간의 IoU (Intersection over Union) 계산
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        float: IoU 값 (0.0 ~ 1.0)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


# ==============================================================================
# 유틸리티 함수
# ==============================================================================

def convert_pt_to_onnx(
    pt_path: str,
    onnx_path: str,
    input_size: int = 640,
    fp16: bool = True,
    simplify: bool = True
):
    """
    PyTorch YOLO 모델(.pt)을 ONNX로 변환
    
    Args:
        pt_path: 입력 .pt 모델 경로
        onnx_path: 출력 .onnx 모델 경로
        input_size: 입력 해상도 (정사각형)
        fp16: FP16 변환 여부
        simplify: ONNX simplify 적용 여부
    
    Note:
        ultralytics 패키지가 필요합니다.
        변환 명령어: yolo export model=best.pt format=onnx imgsz=640 half=True simplify=True
    """
    try:
        from ultralytics import YOLO
        
        model = YOLO(pt_path)
        model.export(
            format="onnx",
            imgsz=input_size,
            half=fp16,
            simplify=simplify,
            opset=12
        )
        
        print(f"[CONVERT] ✅ ONNX 변환 완료: {onnx_path}")
        
    except ImportError:
        print("[CONVERT] ❌ ultralytics 패키지가 필요합니다.")
        print("  pip install ultralytics")
    except Exception as e:
        print(f"[CONVERT] ❌ 변환 실패: {e}")


def benchmark_detector(detector: OnnxYoloDetector, img_pil: Image.Image, iterations: int = 100):
    """
    탐지기 성능 벤치마크
    
    Args:
        detector: OnnxYoloDetector 인스턴스
        img_pil: 테스트 이미지
        iterations: 반복 횟수
    """
    # 워밍업
    for _ in range(10):
        detector.detect(img_pil)
    
    # 벤치마크
    start = time.time()
    for _ in range(iterations):
        detector.detect(img_pil)
    elapsed = time.time() - start
    
    avg_ms = (elapsed / iterations) * 1000
    fps = iterations / elapsed
    
    print(f"[BENCHMARK] 평균 추론 시간: {avg_ms:.2f}ms")
    print(f"[BENCHMARK] FPS: {fps:.1f}")


# ==============================================================================
# 테스트 코드
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ONNX YOLO Detector 테스트")
    print("=" * 60)
    
    # 테스트 이미지 생성
    test_img = Image.new("RGB", (1280, 720), color=(100, 100, 100))
    
    print("\n[테스트] PIL Image 전처리")
    if ONNX_AVAILABLE:
        # 실제 모델이 없으므로 전처리만 테스트
        print("  - onnxruntime 사용 가능")
        print("  - 실제 테스트를 위해서는 .onnx 모델 파일이 필요합니다.")
        
        # 변환 명령어 안내
        print("\n[모델 변환 방법]")
        print("  yolo export model=models/best.pt format=onnx imgsz=640 half=True simplify=True")
    else:
        print("  - onnxruntime이 설치되지 않았습니다.")
        print("  - pip install onnxruntime-gpu  (GPU 사용 시)")
        print("  - pip install onnxruntime      (CPU 전용)")
