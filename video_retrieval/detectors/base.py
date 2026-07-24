# 统一检测器基类
from dataclasses import dataclass
from typing import List
from PIL import Image


@dataclass
class DetectedObject:
    """检测结果"""
    bbox: tuple          # (x1, y1, x2, y2) in pixel coordinates
    confidence: float    # 0~1
    label: str           # 类别名/查询短语


class BaseDetector:
    """
    所有检测器的统一接口。
    子类需要实现 _load_model() 和 _detect_impl()。
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._processor = None

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._load_model()
        return self._processor

    def _load_model(self):
        raise NotImplementedError

    def detect(self, image: Image.Image, texts: List[str],
               box_threshold: float = 0.3,
               text_threshold: float = 0.25) -> List[DetectedObject]:
        return self._detect_impl(image, texts, box_threshold, text_threshold)

    def _detect_impl(self, image: Image.Image, texts: List[str],
                     box_threshold: float, text_threshold: float) -> List[DetectedObject]:
        raise NotImplementedError

    @staticmethod
    def available_models() -> List[str]:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device})"
