# OWLv2 检测器
from typing import List
import torch
from PIL import Image

from .base import BaseDetector, DetectedObject


class OWLv2Detector(BaseDetector):
    """OWLv2 开放词汇目标检测器"""

    AVAILABLE = {
        "base": "google/owlv2-base-patch16-ensemble",
        "large": "google/owlv2-large-patch14-ensemble",
    }

    def __init__(self, model_name: str = "base", device: str = "cuda"):
        if model_name not in self.AVAILABLE:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.AVAILABLE)}")
        super().__init__(model_name, device)
        self._hf_id = self.AVAILABLE[model_name]

    def _load_model(self):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection
        print(f"[OWLv2] Loading {self._hf_id} ...")
        self._processor = Owlv2Processor.from_pretrained(self._hf_id, local_files_only=True)
        self._model = Owlv2ForObjectDetection.from_pretrained(
            self._hf_id, local_files_only=True
        ).to(self.device)
        self._model.eval()

    @torch.no_grad()
    def _detect_impl(self, image: Image.Image, texts: List[str],
                     box_threshold: float, text_threshold: float = 0.0) -> List[DetectedObject]:
        query_texts = [texts]
        inputs = self.processor(text=query_texts, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=box_threshold
        )
        result = results[0]
        boxes = result["boxes"].cpu().tolist()
        scores = result["scores"].cpu().tolist()
        label_indices = result["labels"].cpu().tolist()
        class_names = [texts[idx] for idx in label_indices]

        return [
            DetectedObject(bbox=tuple(b), confidence=s, label=l)
            for b, s, l in zip(boxes, scores, class_names)
        ]

    @staticmethod
    def available_models() -> List[str]:
        return list(OWLv2Detector.AVAILABLE.keys())
