# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sam3_backbone_distill import FeatureSpec, TinyStudentBackbone, make_transform  # noqa: E402
from scripts.sam3_frozen_detector import (  # noqa: E402
    FCOSHead,
    YoloBoxDataset,
    collate_fn,
    cxcywh_to_xyxy,
    decode_predictions,
    detector_loss,
)


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class StudentDetector(nn.Module):
    def __init__(self, student: TinyStudentBackbone, num_classes: int):
        super().__init__()
        self.student = student
        if not student.target_specs:
            raise ValueError("student backbone has no target feature specs")
        in_channels = int(student.target_specs[0].channels)
        self.head = FCOSHead(in_channels=in_channels, num_classes=num_classes)

    def extract_features(self, images):
        return self.student(images)

    def forward(self, images):
        features = self.extract_features(images)
        return self.head(features), features


def count_params(module: nn.Module, trainable_only=False) -> int:
    params = module.parameters()
    if trainable_only:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


def load_student(student_checkpoint, device, load_weights: bool = True) -> Tuple[TinyStudentBackbone, dict]:
    ckpt = torch.load(student_checkpoint, map_location="cpu")
    config = ckpt.get("student_config")
    if config is None:
        raise KeyError(f"{student_checkpoint} does not contain student_config")
    specs = [FeatureSpec(**spec) for spec in config["target_specs"]]
    student = TinyStudentBackbone(
        target_specs=specs,
        resolution=config["resolution"],
        widths=config["widths"],
        depths=config["depths"],
        expansion=config.get("expansion", 2.0),
    )
    if load_weights:
        student.load_state_dict(ckpt["student"])
    return student.to(device), ckpt


def save_detector(model: StudentDetector, student_ckpt: dict, args, epoch: int, metrics: dict):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "sam3_student_detector_v1",
            "student": model.student.state_dict(),
            "student_config": model.student.config(),
            "head": model.head.state_dict(),
            "num_classes": args.num_classes,
            "resolution": int(student_ckpt.get("resolution", model.student.resolution)),
            "feature_levels": int(student_ckpt.get("feature_levels", len(model.student.target_specs))),
            "freeze_student": bool(args.freeze_student),
            "random_student": bool(getattr(args, "random_student", False)),
            "max_images": int(getattr(args, "max_images", 0)),
            "cache_images": bool(getattr(args, "cache_images", False)),
            "epoch": int(epoch),
            "metrics": metrics,
            "source_student_checkpoint": args.student_checkpoint,
        },
        output,
    )


def load_detector(detector_checkpoint, device) -> Tuple[StudentDetector, dict]:
    ckpt = torch.load(detector_checkpoint, map_location="cpu")
    config = ckpt.get("student_config")
    if config is None:
        raise KeyError(f"{detector_checkpoint} does not contain student_config")
    specs = [FeatureSpec(**spec) for spec in config["target_specs"]]
    student = TinyStudentBackbone(
        target_specs=specs,
        resolution=config["resolution"],
        widths=config["widths"],
        depths=config["depths"],
        expansion=config.get("expansion", 2.0),
    )
    model = StudentDetector(student=student, num_classes=ckpt["num_classes"])
    model.student.load_state_dict(ckpt["student"])
    model.head.load_state_dict(ckpt["head"])
    model.to(device)
    model.eval()
    return model, ckpt


def collect_images(images_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.suffix.lower() in IMG_EXTS and not any(part.startswith(".") for part in path.parts)
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
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
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


def draw_boxes(image: Image.Image, boxes: torch.Tensor, scores: torch.Tensor, hide_text=False):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    width, height = out.size
    boxes_px = boxes.detach().cpu().clone()
    if boxes_px.numel() > 0:
        boxes_px[:, [0, 2]] *= width
        boxes_px[:, [1, 3]] *= height
    for box, score in zip(boxes_px.tolist(), scores.detach().cpu().tolist()):
        draw.rectangle(box, outline=(0, 220, 70), width=3)
        if hide_text:
            continue
        text = f"{score:.2f}"
        pos = (box[0], max(0, box[1] - 13))
        bbox = draw.textbbox(pos, text, font=font)
        draw.rectangle(bbox, fill=(0, 90, 35))
        draw.text(pos, text, fill=(255, 255, 255), font=font)
    return out


def make_contact_sheet(paths: List[Path], output_path: Path, cols: int, thumb_width: int):
    if not paths:
        return
    thumbs = []
    font = ImageFont.load_default()
    for path in paths:
        image = Image.open(path).convert("RGB")
        scale = thumb_width / image.size[0]
        thumb = image.resize((thumb_width, max(1, int(image.size[1] * scale))))
        draw = ImageDraw.Draw(thumb)
        bbox = draw.textbbox((4, 4), path.name, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((4, 4), path.name, fill=(255, 255, 255), font=font)
        thumbs.append(thumb)
    rows = (len(thumbs) + cols - 1) // cols
    cell_h = max(thumb.size[1] for thumb in thumbs)
    sheet = Image.new("RGB", (cols * thumb_width, rows * cell_h), (30, 30, 30))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * thumb_width, (idx // cols) * cell_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def train(args):
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    student, student_ckpt = load_student(
        args.student_checkpoint,
        device,
        load_weights=not args.random_student,
    )
    model = StudentDetector(student, args.num_classes).to(device)

    if args.freeze_student:
        for param in model.student.parameters():
            param.requires_grad_(False)
        model.student.eval()

    resolution = int(student_ckpt.get("resolution", student.resolution))
    dataset = YoloBoxDataset(args.images, args.labels, resolution=resolution)
    if args.max_images > 0:
        dataset.image_paths = dataset.image_paths[: args.max_images]
    if args.cache_images:
        print(f"cache_images loading={len(dataset)}", flush=True)
        dataset = [dataset[idx] for idx in range(len(dataset))]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if args.cache_images else args.workers,
        collate_fn=collate_fn,
    )
    optim = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(
        f"images={len(dataset)} resolution={resolution} freeze_student={args.freeze_student} "
        f"random_student={args.random_student} student_params={count_params(model.student)} "
        f"trainable_params={count_params(model, True)} cache_images={args.cache_images}",
        flush=True,
    )

    best_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    use_early_stop = args.early_stop_patience > 0

    for epoch in range(args.epochs):
        model.train()
        if args.freeze_student:
            model.student.eval()
        loss_sum = 0.0
        cls_sum = 0.0
        l1_sum = 0.0
        giou_sum = 0.0
        pos_sum = 0
        sample_sum = 0
        for step, batch in enumerate(loader):
            images = batch["images"].to(device)
            with torch.set_grad_enabled(not args.freeze_student):
                pass
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                (cls_outputs, box_outputs), features = model(images)
                losses = detector_loss(
                    cls_outputs,
                    box_outputs,
                    features,
                    {"boxes": batch["boxes"], "labels": batch["labels"]},
                    args.num_classes,
                )
            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                args.grad_clip,
            )
            optim.step()

            batch_size = int(images.shape[0])
            loss_sum += float(losses["loss"].item()) * batch_size
            cls_sum += float(losses["loss_cls"].item()) * batch_size
            l1_sum += float(losses["loss_l1"].item()) * batch_size
            giou_sum += float(losses["loss_giou"].item()) * batch_size
            pos_sum += int(losses["num_pos"])
            sample_sum += batch_size

            if step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step} loss={losses['loss'].item():.4f} "
                    f"cls={losses['loss_cls'].item():.4f} "
                    f"l1={losses['loss_l1'].item():.4f} "
                    f"giou={losses['loss_giou'].item():.4f} "
                    f"pos={losses['num_pos']}",
                    flush=True,
                )

        normalizer = max(sample_sum, 1)
        metrics = {
            "avg_loss": loss_sum / normalizer,
            "avg_cls": cls_sum / normalizer,
            "avg_l1": l1_sum / normalizer,
            "avg_giou": giou_sum / normalizer,
            "pos": pos_sum,
        }
        print(
            f"epoch={epoch} avg_loss={metrics['avg_loss']:.4f} "
            f"avg_cls={metrics['avg_cls']:.4f} avg_l1={metrics['avg_l1']:.4f} "
            f"avg_giou={metrics['avg_giou']:.4f} pos={metrics['pos']}",
            flush=True,
        )
        improved = metrics["avg_loss"] < best_loss - args.early_stop_min_delta
        if improved:
            best_loss = metrics["avg_loss"]
            best_epoch = epoch
            stale_epochs = 0
            save_detector(model, student_ckpt, args, epoch, metrics)
            print(f"saved best {args.output}", flush=True)
        else:
            stale_epochs += 1

        if (
            use_early_stop
            and epoch + 1 >= args.min_epochs
            and stale_epochs >= args.early_stop_patience
        ):
            print(
                f"early_stop epoch={epoch} best_epoch={best_epoch} "
                f"best_loss={best_loss:.4f} stale_epochs={stale_epochs}",
                flush=True,
            )
            break


@torch.inference_mode()
def predict_one(model, image: Image.Image, transform, device, args):
    inp = transform(image).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        (cls_outputs, box_outputs), features = model(inp)
    return decode_predictions(
        [output[0:1] for output in cls_outputs],
        [output[0:1] for output in box_outputs],
        [feature[0:1] for feature in features],
        args.score_thresh,
        args.nms_thresh,
        args.topk,
    )


@torch.inference_mode()
def predict(args):
    device = torch.device(args.device)
    model, ckpt = load_detector(args.detector_checkpoint, device)
    transform = make_transform(int(ckpt.get("resolution", args.resolution)))
    image = Image.open(args.image).convert("RGB")
    boxes, scores, labels = predict_one(model, image, transform, device, args)
    width, height = image.size
    boxes_px = boxes.detach().cpu().clone()
    if boxes_px.numel() > 0:
        boxes_px[:, [0, 2]] *= width
        boxes_px[:, [1, 3]] *= height
    for score, label, box in zip(scores.cpu().tolist(), labels.cpu().tolist(), boxes_px.tolist()):
        print(f"{score:.4f}\t{label}\t{box[0]:.1f}\t{box[1]:.1f}\t{box[2]:.1f}\t{box[3]:.1f}")
    if args.output_image:
        draw_boxes(image, boxes, scores, args.hide_text).save(args.output_image)


@torch.inference_mode()
def batch_predict(args):
    device = torch.device(args.device)
    model, ckpt = load_detector(args.detector_checkpoint, device)
    transform = make_transform(int(ckpt.get("resolution", args.resolution)))
    images_dir = Path(args.images)
    labels_dir = Path(args.labels) if args.labels else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(images_dir)
    saved_paths = []
    total_preds = 0
    total_gt = 0
    total_matched = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        boxes, scores, _labels = predict_one(model, image, transform, device, args)
        total_preds += int(boxes.shape[0])

        if labels_dir is not None:
            rel_label = image_path.relative_to(images_dir).with_suffix(".txt")
            _gt_labels, gt_boxes = read_yolo_labels(labels_dir / rel_label)
            total_gt += int(gt_boxes.shape[0])
            total_matched += greedy_match(boxes, gt_boxes, args.eval_iou)

        out_path = output_dir / image_path.relative_to(images_dir)
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
    summary = [
        f"images={len(image_paths)}",
        f"checkpoint={args.detector_checkpoint}",
        f"score_thresh={args.score_thresh}",
        f"nms_thresh={args.nms_thresh}",
        f"preds={total_preds}",
        f"avg_preds_per_image={total_preds / max(1, len(image_paths)):.3f}",
        f"elapsed_sec={elapsed:.3f}",
        f"ms_per_image={elapsed * 1000 / max(1, len(image_paths)):.2f}",
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


def add_predict_args(parser):
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--nms-thresh", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hide-text", action="store_true")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    train_p = subparsers.add_parser("train")
    train_p.add_argument("--images", required=True)
    train_p.add_argument("--labels", required=True)
    train_p.add_argument("--student-checkpoint", required=True)
    train_p.add_argument("--output", default="checkpoints/student_detector.pt")
    train_p.add_argument("--num-classes", type=int, default=1)
    train_p.add_argument(
        "--random-student",
        action="store_true",
        help="Use the checkpoint only as an architecture template and randomly initialize the student.",
    )
    train_p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Use only the first N training images after sorting. 0 uses all images.",
    )
    train_p.add_argument(
        "--cache-images",
        action="store_true",
        help="Preload transformed training samples into RAM before training.",
    )
    train_p.add_argument("--seed", type=int, default=0)
    train_p.add_argument("--batch-size", type=int, default=8)
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--grad-clip", type=float, default=1.0)
    train_p.add_argument("--min-epochs", type=int, default=0)
    train_p.add_argument("--early-stop-patience", type=int, default=0)
    train_p.add_argument("--early-stop-min-delta", type=float, default=0.0)
    train_p.add_argument("--workers", type=int, default=2)
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--freeze-student", action="store_true", default=True)
    train_p.add_argument("--train-student", dest="freeze_student", action="store_false")
    train_p.add_argument("--log-every", type=int, default=10)
    train_p.set_defaults(func=train)

    pred_p = subparsers.add_parser("predict")
    pred_p.add_argument("--image", required=True)
    pred_p.add_argument("--output-image")
    add_predict_args(pred_p)
    pred_p.set_defaults(func=predict)

    batch_p = subparsers.add_parser("batch-predict")
    batch_p.add_argument("--images", required=True)
    batch_p.add_argument("--labels")
    batch_p.add_argument("--output-dir", required=True)
    batch_p.add_argument("--eval-iou", type=float, default=0.3)
    batch_p.add_argument("--contact-cols", type=int, default=5)
    batch_p.add_argument("--thumb-width", type=int, default=320)
    add_predict_args(batch_p)
    batch_p.set_defaults(func=batch_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
