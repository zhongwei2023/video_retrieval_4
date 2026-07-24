import os
import time
from collections import Counter
import torch
from PIL import Image, ImageDraw

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def detect_grounding_dino(
    image_path: str,
    texts: list,
    box_threshold: float = 0.4,
    text_threshold: float = 0.3,
):
    """
    使用 Grounding DINO 进行零样本目标检测，仅返回结果。
    返回：boxes (list), scores (list), class_names (list)
    """
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    print("加载 Grounding DINO 模型...")
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    image = Image.open(image_path).convert("RGB")

    # Grounding DINO 需要小写句点分隔的查询文本
    text_query = ". ".join([t.lower() for t in texts]) + "."

    inputs = processor(images=image, text=text_query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )
    result = results[0]
    boxes = result["boxes"].cpu().tolist()    # [[x1,y1,x2,y2], ...]
    scores = result["scores"].cpu().tolist()
    labels = result["labels"]                 # 已经是具体的类别名称，如 "cushion"

    print(f"Grounding DINO 检测到 {len(boxes)} 个目标")
    return boxes, scores, labels


def detect_owlv2(
    image_path: str,
    texts: list,
    threshold: float = 0.1,
):
    """
    使用 OWLv2 进行零样本目标检测，仅返回结果。
    返回：boxes (list), scores (list), class_names (list)
    """
    from transformers import Owlv2Processor, Owlv2ForObjectDetection

    print("加载 OWLv2 模型...")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

    image = Image.open(image_path).convert("RGB")
    query_texts = [texts]   # OWLv2 需要双层列表

    inputs = processor(text=query_texts, images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.Tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=threshold
    )

    i = 0  # 单图结果
    text_queries = query_texts[i]
    boxes = results[i]["boxes"].cpu().tolist()
    scores = results[i]["scores"].cpu().tolist()
    label_indices = results[i]["labels"].cpu().tolist()
    class_names = [text_queries[idx] for idx in label_indices]

    print(f"OWLv2 检测到 {len(boxes)} 个目标")
    return boxes, scores, class_names


def visualize_and_save(
    image_path: str,
    boxes: list,
    scores: list,
    class_names: list,
    output_dir: str,
):
    """
    统一的检测结果可视化与保存函数。
    会在 output_dir 下生成：
      - cropped/   : 裁剪出的小图
      - annotated.jpg : 带标注的全图
    """
    # 创建输出目录
    crop_dir = os.path.join(output_dir, "cropped")
    os.makedirs(crop_dir, exist_ok=True)

    # 重新打开原始图像（避免修改原图）
    image = Image.open(image_path).convert("RGB")

    # ---- 1. 裁剪并保存每一张目标小图 ----
    label_counter = Counter()
    for box, cls_name in zip(boxes, class_names):
        x1, y1, x2, y2 = map(int, box)
        crop = image.crop((x1, y1, x2, y2))
        # 生成安全的文件名
        safe_name = cls_name.replace(" ", "_")
        label_counter[cls_name] += 1
        count = label_counter[cls_name]
        filename = f"{safe_name}.jpg" if count == 1 else f"{safe_name}_{count}.jpg"
        crop.save(os.path.join(crop_dir, filename))

    # ---- 2. 在原图上绘制框、标签和置信度 ----
    draw = ImageDraw.Draw(image)
    for box, score, cls_name in zip(boxes, scores, class_names):
        x1, y1, x2, y2 = map(int, box)
        # 红色矩形框
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        # 文本：类别名 + 置信度
        text = f"{cls_name} {score:.2f}"
        text_y = y1 - 10 if y1 - 10 > 0 else y1 + 5
        draw.text((x1, text_y), text, fill="white", stroke_width=1, stroke_fill="black")

    # 保存最终结果图
    output_image = os.path.join(output_dir, "annotated.jpg")
    image.save(output_image)
    print(f"可视化结果已保存至 {output_dir}")


if __name__ == "__main__":
    # ===== 用户配置区域 =====
    IMAGE_PATH = r"F:\project_of_codex\dino\test.jpg"
    QUERY_TEXTS = ["cat","a remote control"]                     # 可写多个，如 ["cushion", "cat"]
    MODEL_TYPE = "dino"                           # 选择 "dino" 或 "owlv2"

    start_time = time.time()

    # 1. 执行检测（只返回数据）
    if MODEL_TYPE.lower() == "dino":
        boxes, scores, class_names = detect_grounding_dino(IMAGE_PATH, QUERY_TEXTS)
        output_folder = "output_dino"
    elif MODEL_TYPE.lower() == "owlv2":
        boxes, scores, class_names = detect_owlv2(IMAGE_PATH, QUERY_TEXTS)
        output_folder = "output_owlv2"
    else:
        raise ValueError("MODEL_TYPE 必须是 'dino' 或 'owlv2'")

    # 2. 可视化并保存到对应文件夹
    visualize_and_save(IMAGE_PATH, boxes, scores, class_names, output_folder)

    elapsed = time.time() - start_time
    print(f"\n✅ 总执行时间: {elapsed:.2f} 秒")