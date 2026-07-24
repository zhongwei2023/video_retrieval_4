# YOLO-World 检测器
from typing import List
import torch
from PIL import Image

from .base import BaseDetector, DetectedObject


class YOLOWorldDetector(BaseDetector):
    """YOLO-World 开放词汇目标检测器"""

    AVAILABLE = {
        "s": "yolov8s-worldv2.pt",
        "m": "yolov8m-worldv2.pt",
        "l": "yolov8l-worldv2.pt",
        "x": "yolov8x-worldv2.pt",
    }

    def __init__(self, model_name: str = "l", device: str = "cuda"):
        if model_name not in self.AVAILABLE:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.AVAILABLE)}")
        super().__init__(model_name, device)

    def _load_model(self):
        from ultralytics import YOLOWorld
        print(f"[YOLOWorld] Loading {self.AVAILABLE[self.model_name]} ...")
        self._model = YOLOWorld(self.AVAILABLE[self.model_name])
        if self.device == "cuda" and torch.cuda.is_available():
            self._model.to(self.device)
        self._processor = None  # YOLO-World 不需要单独的 processor

    @torch.no_grad()
    def _detect_impl(self, image: Image.Image, texts: List[str],
                     box_threshold: float, text_threshold: float = 0.0) -> List[DetectedObject]:
        # YOLO-World: set_classes 设置开放词汇
        self.model.set_classes(texts)
        results = self.model.predict(image, conf=box_threshold, verbose=False)

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            cls_ids = result.boxes.cls.cpu().tolist()
            for b, s, cid in zip(boxes, scores, cls_ids):
                label = texts[int(cid)] if int(cid) < len(texts) else str(int(cid))
                detections.append(DetectedObject(bbox=tuple(b), confidence=s, label=label))
        return detections

    @staticmethod
    def available_models() -> List[str]:
        return list(YOLOWorldDetector.AVAILABLE.keys())
