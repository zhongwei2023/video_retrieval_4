# Grounding DINO 检测器
import os
from typing import List
import torch
from PIL import Image

from .base import BaseDetector, DetectedObject

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class GroundingDINODetector(BaseDetector):
    """Grounding DINO 开放词汇目标检测器"""

    AVAILABLE = {
        "tiny": "IDEA-Research/grounding-dino-tiny",
        "base": "IDEA-Research/grounding-dino-base",
    }

    def __init__(self, model_name: str = "base", device: str = "cuda"):
        if model_name not in self.AVAILABLE:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.AVAILABLE)}")
        super().__init__(model_name, device)
        self._hf_id = self.AVAILABLE[model_name]

    def _load_model(self):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        print(f"[GroundingDINO] Loading {self._hf_id} ...")
        self._processor = AutoProcessor.from_pretrained(self._hf_id, local_files_only=True)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._hf_id, local_files_only=True
        ).to(self.device)
        self._model.eval()

    @torch.no_grad()
    def _detect_impl(self, image: Image.Image, texts: List[str],
                     box_threshold: float, text_threshold: float) -> List[DetectedObject]:
        text_query = ". ".join([t.lower() for t in texts]) + "."
        inputs = self.processor(images=image, text=text_query, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )
        result = results[0]
        boxes = result["boxes"].cpu().tolist()
        scores = result["scores"].cpu().tolist()
        labels = result["labels"]

        return [
            DetectedObject(bbox=tuple(b), confidence=s, label=l)
            for b, s, l in zip(boxes, scores, labels)
        ]

    @staticmethod
    def available_models() -> List[str]:
        return list(GroundingDINODetector.AVAILABLE.keys())
