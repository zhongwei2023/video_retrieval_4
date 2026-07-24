# 消融实验：对比不同开放词汇检测器在视频目标检索上的表现
import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from video_retrieval import (
    VideoRetrievalPipeline,
    GroundingDINODetector,
    OWLv2Detector,
    YOLOWorldDetector,
)


def build_detector(detector_type: str, model_size: str, device: str = "cuda"):
    """工厂函数：根据类型和大小构建检测器"""
    d_type = detector_type.lower()
    if d_type == "dino":
        return GroundingDINODetector(model_size, device)
    elif d_type == "owlv2":
        return OWLv2Detector(model_size, device)
    elif d_type == "yolo-world":
        return YOLOWorldDetector(model_size, device)
    else:
        raise ValueError(f"Unknown detector: {detector_type}")


def run_experiment(
    detector_type: str,
    model_size: str,
    video_path: str,
    query_texts: list,
    fps: float,
    box_threshold: float,
    text_threshold: float,
    output_dir: str,
    device: str,
):
    """运行单次实验"""
    print(f"\n{'='*60}")
    print(f"  Experiment: {detector_type} ({model_size})")
    print(f"  Video: {video_path}")
    print(f"  Query: {query_texts}")
    print(f"  FPS: {fps}, Box threshold: {box_threshold}")
    print(f"{'='*60}")

    detector = build_detector(detector_type, model_size, device)
    pipeline = VideoRetrievalPipeline(detector, device)

    result = pipeline.retrieve(
        video_path=video_path,
        query_texts=query_texts,
        fps=fps,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # 裁切最优帧
    exp_name = f"{detector_type}_{model_size}"
    if result.best_detection:
        crop_path = os.path.join(output_dir, f"{exp_name}_best_crop.jpg")
        pipeline.crop_best(result, crop_path, margin=0.1)

    # 保存统计信息
    stats_path = os.path.join(output_dir, f"{exp_name}_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"Detector: {detector_type} ({model_size})\n")
        f.write(f"Video: {result.video_path}\n")
        f.write(f"Query: {result.query_text}\n")
        f.write(f"Total frames processed: {result.total_frames}\n")
        f.write(f"Frames with detections: {result.frames_with_detections}\n")
        f.write(f"Total detections: {result.total_detections}\n")
        f.write(f"Elapsed: {result.elapsed_seconds:.1f}s\n")
        f.write(f"Avg per frame: {result.elapsed_seconds/max(1,result.total_frames):.3f}s\n")
        f.write(f"Best frame: #{result.best_frame_idx} ({result.best_timestamp_sec:.1f}s)\n")
        f.write(f"Best score: {result.best_score:.6f}\n")
        if result.best_detection:
            f.write(f"Best bbox: {result.best_detection.bbox}\n")
            f.write(f"Best confidence: {result.best_detection.confidence:.4f}\n")

    print(f"  Stats saved to {stats_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Video Object Retrieval Ablation")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--query", required=True, nargs="+", help="Query text phrases")
    parser.add_argument("--detectors", nargs="+",
                        default=["dino:base", "dino:tiny", "owlv2:base", "yolo-world:l"],
                        help="Detectors to test (format: type:size)")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling FPS")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--output", default="./ablation_output", help="Output directory")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    results = {}
    for det_spec in args.detectors:
        parts = det_spec.split(":")
        det_type = parts[0]
        det_size = parts[1] if len(parts) > 1 else "base"

        try:
            result = run_experiment(
                detector_type=det_type,
                model_size=det_size,
                video_path=args.video,
                query_texts=args.query,
                fps=args.fps,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                output_dir=args.output,
                device=args.device,
            )
            results[det_spec] = result
        except Exception as e:
            print(f"  [ERROR] {det_spec} failed: {e}")
            import traceback
            traceback.print_exc()

    # 打印汇总
    print(f"\n{'='*60}")
    print("  ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Detector':<25} {'Frames':>7} {'Hits':>7} {'Time':>8} {'BestScore':>10}")
    print("-" * 60)
    for det_spec, r in results.items():
        print(f"{det_spec:<25} {r.total_frames:>7} {r.frames_with_detections:>7} "
              f"{r.elapsed_seconds:>7.1f}s {r.best_score:>10.6f}")


if __name__ == "__main__":
    main()
