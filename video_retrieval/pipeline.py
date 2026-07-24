# 视频目标检索主 Pipeline
import os
import time
from typing import List, Optional, Dict
from dataclasses import dataclass, field

import cv2
import torch
from PIL import Image

from .detectors import BaseDetector, DetectedObject
from .scorer import score_detection, select_best_across_frames


@dataclass
class RetrievalResult:
    """检索结果"""
    video_path: str
    query_text: str
    detector_name: str
    # 最优帧信息
    best_frame_idx: int = -1
    best_timestamp_sec: float = -1.0
    best_detection: Optional[DetectedObject] = None
    best_score: float = 0.0
    # 统计
    total_frames: int = 0
    frames_with_detections: int = 0
    total_detections: int = 0
    elapsed_seconds: float = 0.0
    # 所有检测结果（用于分析）
    all_results: List = field(default_factory=list)


class VideoRetrievalPipeline:
    """
    视频目标检索 Pipeline。

    用法:
        detector = GroundingDINODetector("base")
        pipeline = VideoRetrievalPipeline(detector)
        result = pipeline.retrieve("video.mp4", "a person in red shirt", fps=1)
        print(f"Best frame: {result.best_frame_idx}, score: {result.best_score:.4f}")
    """

    def __init__(self, detector: BaseDetector, device: str = "cuda"):
        self.detector = detector
        self.device = device

    def retrieve(
        self,
        video_path: str,
        query_texts: List[str],
        fps: float = 1.0,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        max_frames: Optional[int] = None,
        progress_callback=None,
    ) -> RetrievalResult:
        """
        从视频中检索目标的最优帧。

        Args:
            video_path: 视频文件路径
            query_texts: 查询短语列表，如 ["a person in red shirt"]
            fps: 采样帧率（默认 1fps）
            box_threshold: bbox 置信度阈值
            text_threshold: 文本匹配阈值
            max_frames: 最大处理帧数（None=全部）
            progress_callback: 进度回调 fn(current, total)

        Returns:
            RetrievalResult
        """
        t_start = time.time()

        # 1. 打开视频获取元信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_video_frames / video_fps if video_fps > 0 else 0

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 计算采样间隔
        if fps <= 0 or video_fps <= 0:
            sample_interval = 1
        else:
            sample_interval = max(1, int(video_fps / fps))

        # 计算要处理的帧数
        total_to_process = total_video_frames // sample_interval
        if max_frames:
            total_to_process = min(total_to_process, max_frames)

        print(f"[Pipeline] Video: {video_path}")
        print(f"  Resolution: {frame_width}x{frame_height}, Duration: {duration_sec:.1f}s")
        print(f"  Source FPS: {video_fps:.1f}, Sample FPS: {fps:.1f}")
        print(f"  Frames to process: {total_to_process} (every {sample_interval} frames)")

        # 2. 逐帧处理
        all_results = []
        frame_idx = 0
        processed = 0

        while processed < total_to_process:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, bgr = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            timestamp = frame_idx / video_fps if video_fps > 0 else frame_idx

            detections = self.detector.detect(
                pil_image, query_texts,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            all_results.append((frame_idx, timestamp, detections))
            processed += 1
            frame_idx += sample_interval

            if progress_callback:
                progress_callback(processed, total_to_process)

        cap.release()

        # 3. 选出全局最优
        best = select_best_across_frames(all_results, frame_width, frame_height)

        # 统计
        frames_with_det = sum(1 for _, _, d in all_results if len(d) > 0)
        total_dets = sum(len(d) for _, _, d in all_results)

        elapsed = time.time() - t_start

        result = RetrievalResult(
            video_path=video_path,
            query_text=", ".join(query_texts),
            detector_name=repr(self.detector),
            total_frames=processed,
            frames_with_detections=frames_with_det,
            total_detections=total_dets,
            elapsed_seconds=elapsed,
            all_results=all_results,
        )

        if best:
            result.best_frame_idx = best["frame_idx"]
            result.best_timestamp_sec = best["timestamp_sec"]
            result.best_detection = best["detection"]
            result.best_score = best["score"]
            print(f"[Pipeline] Best frame: #{best['frame_idx']} "
                  f"({best['timestamp_sec']:.1f}s), score={best['score']:.4f}")
        else:
            print("[Pipeline] No detections found in any frame.")

        print(f"[Pipeline] Elapsed: {elapsed:.1f}s, "
              f"Avg: {elapsed/max(1,processed):.2f}s/frame")
        return result

    def crop_best(self, result: RetrievalResult, output_path: str, margin: float = 0.0):
        """
        将最优帧的目标区域裁切并保存。

        Args:
            result: RetrievalResult
            output_path: 输出图片路径
            margin: bbox 扩展比例 (0=不扩展, 0.1=四边各扩展 10%)
        """
        if result.best_detection is None:
            print("[Pipeline] No detection to crop.")
            return

        cap = cv2.VideoCapture(result.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, result.best_frame_idx)
        ret, bgr = cap.read()
        cap.release()
        if not ret:
            print("[Pipeline] Failed to read best frame.")
            return

        img_h, img_w = bgr.shape[:2]
        x1, y1, x2, y2 = result.best_detection.bbox

        # Apply margin
        w, h = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - w * margin))
        y1 = max(0, int(y1 - h * margin))
        x2 = min(img_w, int(x2 + w * margin))
        y2 = min(img_h, int(y2 + h * margin))

        crop = bgr[y1:y2, x1:x2]
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        Image.fromarray(rgb_crop).save(output_path)
        print(f"[Pipeline] Crop saved to {output_path} ({x2-x1}x{y2-y1})")
