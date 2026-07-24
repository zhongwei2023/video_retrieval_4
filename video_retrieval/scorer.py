# 最优帧评分模块

def score_detection(obj, frame_width: int, frame_height: int) -> float:
    """
    综合评分：confidence × 归一化 bbox 面积

    Args:
        obj: DetectedObject
        frame_width: 帧宽度
        frame_height: 帧高度

    Returns:
        综合得分 (0~1)，越高越好
    """
    x1, y1, x2, y2 = obj.bbox
    bbox_area = max(0, (x2 - x1) * (y2 - y1))
    frame_area = frame_width * frame_height
    area_ratio = bbox_area / frame_area if frame_area > 0 else 0

    # confidence × area_ratio: 目标既要置信度高，也要在画面中有足够尺寸
    return obj.confidence * area_ratio


def select_best_detection(detections, frame_width: int, frame_height: int):
    """
    从一帧的检测结果中选出最佳目标。

    Returns:
        (best_obj, best_score) 或 (None, 0.0)
    """
    if not detections:
        return None, 0.0

    best = None
    best_score = -1.0
    for obj in detections:
        s = score_detection(obj, frame_width, frame_height)
        if s > best_score:
            best_score = s
            best = obj
    return best, best_score


def select_best_across_frames(frame_results, frame_width: int, frame_height: int):
    """
    跨所有帧选出全局最优检测。

    Args:
        frame_results: list of (frame_idx, timestamp_sec, list_of_detections)

    Returns:
        {
            "frame_idx": int,
            "timestamp_sec": float,
            "detection": DetectedObject,
            "score": float,
        }
        或 None
    """
    global_best = None
    for frame_idx, ts, detections in frame_results:
        best_obj, score = select_best_detection(detections, frame_width, frame_height)
        if best_obj is not None and (global_best is None or score > global_best["score"]):
            global_best = {
                "frame_idx": frame_idx,
                "timestamp_sec": ts,
                "detection": best_obj,
                "score": score,
            }
    return global_best
