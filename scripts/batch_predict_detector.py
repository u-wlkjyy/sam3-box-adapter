# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sam3_frozen_detector import (  # noqa: E402
    Sam3FrozenBackboneDetector,
    cxcywh_to_xyxy,
    decode_predictions,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(images_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in images_dir.rglob("*")
        if p.suffix.lower() in IMG_EXTS and not any(part.startswith(".") for part in p.parts)
    )


def read_yolo_labels(label_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = []
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = parts
            labels.append(int(cls))
            boxes.append([float(cx), float(cy), float(w), float(h)])
    if not boxes:
        return torch.empty(0, dtype=torch.long), torch.empty(0, 4)
    return (
        torch.tensor(labels, dtype=torch.long),
        cxcywh_to_xyxy(torch.tensor(boxes, dtype=torch.float32)).clamp(0, 1),
    )


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def greedy_match(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, iou_thresh: float) -> int:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return 0
    ious = box_iou(pred_boxes.cpu(), gt_boxes.cpu())
    matched_pred = set()
    matched_gt = set()
    while True:
        best = torch.argmax(ious)
        best_iou = float(ious.flatten()[best].item())
        if best_iou < iou_thresh:
            break
        pred_idx = int(best // ious.shape[1])
        gt_idx = int(best % ious.shape[1])
        if pred_idx in matched_pred or gt_idx in matched_gt:
            ious[pred_idx, gt_idx] = -1
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        ious[pred_idx, :] = -1
        ious[:, gt_idx] = -1
    return len(matched_gt)


def draw_boxes(image: Image.Image, boxes: torch.Tensor, scores: torch.Tensor, hide_text: bool) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    w, h = out.size
    boxes_px = boxes.detach().cpu().clone()
    if boxes_px.numel() > 0:
        boxes_px[:, [0, 2]] *= w
        boxes_px[:, [1, 3]] *= h
    for box, score in zip(boxes_px.tolist(), scores.detach().cpu().tolist()):
        draw.rectangle(box, outline=(0, 220, 70), width=3)
        if hide_text:
            continue
        text = f"{score:.2f}"
        tx, ty = box[0], max(0, box[1] - 13)
        bbox = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle(bbox, fill=(0, 90, 35))
        draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
    return out


def make_contact_sheet(paths: List[Path], output_path: Path, cols: int, thumb_width: int):
    if not paths:
        return
    thumbs = []
    font = ImageFont.load_default()
    for path in paths:
        img = Image.open(path).convert("RGB")
        scale = thumb_width / img.size[0]
        thumb = img.resize((thumb_width, max(1, int(img.size[1] * scale))))
        draw = ImageDraw.Draw(thumb)
        label = path.name
        bbox = draw.textbbox((4, 4), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((4, 4), label, fill=(255, 255, 255), font=font)
        thumbs.append(thumb)

    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(t.size[1] for t in thumbs)
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h), (30, 30, 30))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * thumb_width
        y = (idx // cols) * cell_h
        sheet.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--sam3-checkpoint", default="checkpoints/sam3.pt")
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--eval-iou", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hide-text", action="store_true")
    parser.add_argument("--contact-cols", type=int, default=5)
    parser.add_argument("--thumb-width", type=int, default=320)
    args = parser.parse_args()

    device = torch.device(args.device)
    images_dir = Path(args.images)
    labels_dir = Path(args.labels) if args.labels else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.detector_checkpoint, map_location="cpu")
    adapter_state = ckpt.get("adapter_state", {})
    model = Sam3FrozenBackboneDetector(
        checkpoint_path=args.sam3_checkpoint,
        num_classes=ckpt["num_classes"],
        device=args.device,
        train_backbone=False,
        feature_levels=ckpt.get("feature_levels", -1),
        adapter_layers=ckpt.get("adapter_layers", len(adapter_state)),
        adapter_rank=ckpt.get("adapter_rank", 32),
        adapter_scale=ckpt.get("adapter_scale", 1.0),
    ).to(device)
    if adapter_state:
        model.load_adapter_state_dict(adapter_state)
    model.head.load_state_dict(ckpt["head"])
    model.eval()

    resolution = ckpt.get("resolution", args.resolution)
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    image_paths = collect_images(images_dir)
    total_preds = 0
    total_gt = 0
    total_matched = 0
    saved_paths = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        inp = transform(image).unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
            (cls_outputs, box_outputs), features = model(inp)
        boxes, scores, _labels = decode_predictions(
            [x[0:1] for x in cls_outputs],
            [x[0:1] for x in box_outputs],
            [x[0:1] for x in features],
            args.score_thresh,
            args.nms_thresh,
            args.topk,
        )
        total_preds += int(boxes.shape[0])

        if labels_dir is not None:
            rel = image_path.relative_to(images_dir).with_suffix(".txt")
            _gt_labels, gt_boxes = read_yolo_labels(labels_dir / rel)
            total_gt += int(gt_boxes.shape[0])
            total_matched += greedy_match(boxes, gt_boxes, args.eval_iou)

        rel_out = image_path.relative_to(images_dir)
        out_path = output_dir / rel_out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        draw_boxes(image, boxes, scores, args.hide_text).save(out_path)
        saved_paths.append(out_path)
        print(f"{idx + 1}/{len(image_paths)} {image_path.name} preds={boxes.shape[0]}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    contact_path = output_dir / "contact_sheet.jpg"
    make_contact_sheet(saved_paths, contact_path, args.contact_cols, args.thumb_width)

    precision = total_matched / total_preds if total_preds else 0.0
    recall = total_matched / total_gt if total_gt else 0.0
    ms_per_image = elapsed * 1000 / max(1, len(image_paths))
    summary = [
        f"images={len(image_paths)}",
        f"checkpoint={args.detector_checkpoint}",
        f"score_thresh={args.score_thresh}",
        f"nms_thresh={args.nms_thresh}",
        f"preds={total_preds}",
        f"avg_preds_per_image={total_preds / max(1, len(image_paths)):.3f}",
        f"elapsed_sec={elapsed:.3f}",
        f"ms_per_image={ms_per_image:.2f}",
        f"contact_sheet={contact_path}",
    ]
    if labels_dir is not None:
        summary.extend(
            [
                f"gt={total_gt}",
                f"matched_iou_{args.eval_iou:g}={total_matched}",
                f"precision_iou_{args.eval_iou:g}={precision:.4f}",
                f"recall_iou_{args.eval_iou:g}={recall:.4f}",
            ]
        )
    text = "\n".join(summary)
    (output_dir / "summary.txt").write_text(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
