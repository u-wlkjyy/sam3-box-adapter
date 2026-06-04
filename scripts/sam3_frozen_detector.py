# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import generalized_box_iou, nms
from torchvision.transforms import v2


def _unwrap_feature(x):
    return x.tensors if hasattr(x, "tensors") else x


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1
    )


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


class YoloBoxDataset(Dataset):
    """YOLO txt labels: class cx cy w h, all normalized to [0, 1]."""

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, images_dir, labels_dir, resolution=1008):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.resolution = resolution
        self.image_paths = sorted(
            p for p in self.images_dir.rglob("*") if p.suffix.lower() in self.IMG_EXTS
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def _label_path(self, image_path: Path) -> Path:
        rel = image_path.relative_to(self.images_dir).with_suffix(".txt")
        return self.labels_dir / rel

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)

        labels = []
        boxes = []
        label_path = self._label_path(image_path)
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = parts
                labels.append(int(cls))
                boxes.append([float(cx), float(cy), float(w), float(h)])

        if boxes:
            labels_t = torch.tensor(labels, dtype=torch.long)
            boxes_t = cxcywh_to_xyxy(torch.tensor(boxes, dtype=torch.float32))
            boxes_t = boxes_t.clamp(0, 1)
        else:
            labels_t = torch.empty(0, dtype=torch.long)
            boxes_t = torch.empty(0, 4, dtype=torch.float32)

        return {
            "image": tensor,
            "boxes": boxes_t,
            "labels": labels_t,
            "path": str(image_path),
        }


def collate_fn(batch):
    return {
        "images": torch.stack([x["image"] for x in batch], dim=0),
        "boxes": [x["boxes"] for x in batch],
        "labels": [x["labels"] for x in batch],
        "paths": [x["path"] for x in batch],
    }


class FCOSHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=1, num_convs=4, hidden=256):
        super().__init__()
        cls_layers = []
        box_layers = []
        for i in range(num_convs):
            c_in = in_channels if i == 0 else hidden
            cls_layers += [nn.Conv2d(c_in, hidden, 3, padding=1), nn.GroupNorm(32, hidden), nn.SiLU()]
            box_layers += [nn.Conv2d(c_in, hidden, 3, padding=1), nn.GroupNorm(32, hidden), nn.SiLU()]
        self.cls_tower = nn.Sequential(*cls_layers)
        self.box_tower = nn.Sequential(*box_layers)
        self.cls_logits = nn.Conv2d(hidden, num_classes, 3, padding=1)
        self.box_reg = nn.Conv2d(hidden, 4, 3, padding=1)
        prior_prob = 0.01
        bias = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias)

    def forward(self, features: List[torch.Tensor]):
        cls_outputs = []
        box_outputs = []
        for feat in features:
            cls_outputs.append(self.cls_logits(self.cls_tower(feat)))
            box_outputs.append(F.softplus(self.box_reg(self.box_tower(feat))))
        return cls_outputs, box_outputs


class ResidualAdapter(nn.Module):
    """Low-rank residual adapter over the ViT token/channel dimension."""

    def __init__(self, dim: int, rank: int = 32, scale: float = 1.0):
        super().__init__()
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        self.act = nn.SiLU()
        self.gate = nn.Parameter(torch.tensor(float(scale)))
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gate * self.up(self.act(self.down(x)))


class BlockWithAdapter(nn.Module):
    def __init__(self, block: nn.Module, adapter: ResidualAdapter):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.block(x))


class FrozenBlockNoGrad(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.block(x)


class DifferentiableMlp(nn.Module):
    """Drop-in MLP wrapper that avoids SAM3's inference-only fused MLP kernel."""

    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return self.mlp(x)
        x = self.mlp.fc1(x)
        x = self.mlp.act(x)
        x = self.mlp.drop1(x)
        x = self.mlp.norm(x)
        x = self.mlp.fc2(x)
        x = self.mlp.drop2(x)
        return x


def unwrap_frozen_block(block: nn.Module) -> nn.Module:
    return block.block if isinstance(block, FrozenBlockNoGrad) else block


def make_block_mlp_differentiable(block: nn.Module):
    base = unwrap_frozen_block(block)
    if isinstance(base, BlockWithAdapter):
        base = base.block
    mlp = getattr(base, "mlp", None)
    if mlp is None or isinstance(mlp, DifferentiableMlp):
        return
    base.mlp = DifferentiableMlp(mlp)


def get_sam3_vision_trunk(sam3_model: nn.Module) -> nn.Module:
    backbone = getattr(sam3_model, "backbone", None)
    vision_backbone = getattr(backbone, "vision_backbone", None)
    trunk = getattr(vision_backbone, "trunk", None)
    if trunk is None or not hasattr(trunk, "blocks"):
        raise RuntimeError("Could not locate SAM3 vision trunk blocks")
    return trunk


def get_block_dim(block: nn.Module, fallback_dim: int) -> int:
    norm1 = getattr(block, "norm1", None)
    shape = getattr(norm1, "normalized_shape", None)
    if shape:
        return int(shape[0])
    return int(fallback_dim)


class Sam3FrozenBackboneDetector(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        num_classes,
        device="cuda",
        train_backbone=False,
        feature_levels=-1,
        adapter_layers=0,
        adapter_rank=32,
        adapter_scale=1.0,
    ):
        super().__init__()
        from sam3.model_builder import build_sam3_image_model

        self.sam3 = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            enable_segmentation=False,
            enable_inst_interactivity=False,
            eval_mode=True,
            device=device,
        )
        self.feature_levels = feature_levels
        self.adapter_block_indices: List[int] = []
        self.adapter_rank = adapter_rank
        self.adapter_scale = adapter_scale
        if train_backbone:
            raise ValueError("train_backbone is not supported by SAM3's inference-only fused MLP")
        for p in self.sam3.parameters():
            p.requires_grad_(False)
        if adapter_layers > 0:
            self.install_adapters(adapter_layers, adapter_rank, adapter_scale)
        self.head = FCOSHead(in_channels=256, num_classes=num_classes)

    def install_adapters(self, adapter_layers: int, adapter_rank: int, adapter_scale: float):
        trunk = get_sam3_vision_trunk(self.sam3)
        num_blocks = len(trunk.blocks)
        if adapter_layers < 0 or adapter_layers > num_blocks:
            raise ValueError(f"adapter_layers must be in [0, {num_blocks}], got {adapter_layers}")
        start = num_blocks - adapter_layers
        fallback_dim = getattr(trunk, "channel_list", [1024])[-1]
        self.adapter_block_indices = list(range(start, num_blocks))
        for idx in range(start):
            block = trunk.blocks[idx]
            if not isinstance(block, FrozenBlockNoGrad):
                trunk.blocks[idx] = FrozenBlockNoGrad(block)
        for idx in self.adapter_block_indices:
            block = trunk.blocks[idx]
            if isinstance(block, BlockWithAdapter):
                continue
            make_block_mlp_differentiable(block)
            dim = get_block_dim(block, fallback_dim)
            adapter = ResidualAdapter(dim=dim, rank=adapter_rank, scale=adapter_scale)
            trunk.blocks[idx] = BlockWithAdapter(block, adapter)

    def adapter_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        trunk = get_sam3_vision_trunk(self.sam3)
        state = {}
        for idx in self.adapter_block_indices:
            block = trunk.blocks[idx]
            if not isinstance(block, BlockWithAdapter):
                continue
            state[str(idx)] = block.adapter.state_dict()
        return state

    def load_adapter_state_dict(self, adapter_state: Dict[str, Dict[str, torch.Tensor]]):
        trunk = get_sam3_vision_trunk(self.sam3)
        for idx_s, state in adapter_state.items():
            idx = int(idx_s)
            block = trunk.blocks[idx]
            if not isinstance(block, BlockWithAdapter):
                raise RuntimeError(f"SAM3 block {idx} does not have an adapter")
            block.adapter.load_state_dict(state)

    def adapter_parameters(self):
        trunk = get_sam3_vision_trunk(self.sam3)
        seen = set()
        for idx in self.adapter_block_indices:
            block = trunk.blocks[idx]
            if not isinstance(block, BlockWithAdapter):
                continue
            for p in block.adapter.parameters():
                if id(p) in seen:
                    continue
                seen.add(id(p))
                yield p

    def extract_features(self, images):
        enable_grad = self.training and any(p.requires_grad for p in self.sam3.backbone.parameters())
        with torch.set_grad_enabled(enable_grad):
            out = self.sam3.backbone.forward_image(images)
        feats = [_unwrap_feature(x) for x in out["backbone_fpn"]]
        if self.feature_levels > 0:
            feats = feats[-self.feature_levels :]
        return feats

    def forward(self, images):
        feats = self.extract_features(images)
        return self.head(feats), feats


def make_locations(features: List[torch.Tensor], device) -> List[torch.Tensor]:
    locations = []
    for feat in features:
        _, _, h, w = feat.shape
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        locations.append(torch.stack([xx, yy], dim=-1).reshape(-1, 2))
    return locations


def assign_targets(locations, gt_boxes, gt_labels, num_classes):
    all_locations = torch.cat(locations, dim=0)
    n = all_locations.shape[0]
    cls_targets = torch.zeros(n, num_classes, device=all_locations.device)
    box_targets = torch.zeros(n, 4, device=all_locations.device)
    pos_mask = torch.zeros(n, dtype=torch.bool, device=all_locations.device)
    if gt_boxes.numel() == 0:
        return cls_targets, box_targets, pos_mask

    x, y = all_locations[:, 0], all_locations[:, 1]
    l = x[:, None] - gt_boxes[None, :, 0]
    t = y[:, None] - gt_boxes[None, :, 1]
    r = gt_boxes[None, :, 2] - x[:, None]
    b = gt_boxes[None, :, 3] - y[:, None]
    reg = torch.stack([l, t, r, b], dim=-1)
    inside = reg.min(dim=-1).values > 0
    areas = box_area(gt_boxes)[None].repeat(n, 1)
    areas[~inside] = float("inf")
    min_area, matched = areas.min(dim=1)
    pos_mask = torch.isfinite(min_area)
    if pos_mask.any():
        cls_targets[pos_mask, gt_labels[matched[pos_mask]]] = 1.0
        box_targets[pos_mask] = reg[pos_mask, matched[pos_mask]]
    return cls_targets, box_targets, pos_mask


def detector_loss(cls_outputs, box_outputs, features, targets, num_classes):
    device = cls_outputs[0].device
    locations = make_locations(features, device)
    cls_flat = []
    box_flat = []
    for cls, box in zip(cls_outputs, box_outputs):
        cls_flat.append(cls.permute(0, 2, 3, 1).reshape(cls.shape[0], -1, num_classes))
        box_flat.append(box.permute(0, 2, 3, 1).reshape(box.shape[0], -1, 4))
    cls_pred = torch.cat(cls_flat, dim=1)
    box_pred = torch.cat(box_flat, dim=1)

    total_cls = cls_pred.new_tensor(0.0)
    total_l1 = cls_pred.new_tensor(0.0)
    total_giou = cls_pred.new_tensor(0.0)
    total_pos = 0
    for b_idx, (gt_boxes, gt_labels) in enumerate(zip(targets["boxes"], targets["labels"])):
        gt_boxes = gt_boxes.to(device)
        gt_labels = gt_labels.to(device)
        cls_t, box_t, pos = assign_targets(locations, gt_boxes, gt_labels, num_classes)
        total_cls = total_cls + sigmoid_focal_loss(cls_pred[b_idx], cls_t)
        if pos.any():
            points = torch.cat(locations, dim=0)[pos]
            pred_ltrb = box_pred[b_idx][pos]
            target_ltrb = box_t[pos]
            pred_xyxy = torch.stack(
                [
                    points[:, 0] - pred_ltrb[:, 0],
                    points[:, 1] - pred_ltrb[:, 1],
                    points[:, 0] + pred_ltrb[:, 2],
                    points[:, 1] + pred_ltrb[:, 3],
                ],
                dim=-1,
            ).clamp(0, 1)
            target_xyxy = torch.stack(
                [
                    points[:, 0] - target_ltrb[:, 0],
                    points[:, 1] - target_ltrb[:, 1],
                    points[:, 0] + target_ltrb[:, 2],
                    points[:, 1] + target_ltrb[:, 3],
                ],
                dim=-1,
            ).clamp(0, 1)
            total_l1 = total_l1 + F.l1_loss(pred_ltrb, target_ltrb, reduction="sum")
            giou = generalized_box_iou(pred_xyxy, target_xyxy).diag()
            total_giou = total_giou + (1 - giou).sum()
            total_pos += int(pos.sum().item())

    normalizer = max(total_pos, 1)
    return {
        "loss": (total_cls + total_l1 + 2.0 * total_giou) / normalizer,
        "loss_cls": total_cls.detach() / normalizer,
        "loss_l1": total_l1.detach() / normalizer,
        "loss_giou": total_giou.detach() / normalizer,
        "num_pos": total_pos,
    }


@torch.inference_mode()
def decode_predictions(cls_outputs, box_outputs, features, score_thresh, nms_thresh, topk):
    device = cls_outputs[0].device
    locations = make_locations(features, device)
    boxes_all = []
    scores_all = []
    labels_all = []
    for cls, box, points in zip(cls_outputs, box_outputs, locations):
        num_classes = cls.shape[1]
        scores = cls.sigmoid().permute(0, 2, 3, 1).reshape(-1, num_classes)
        regs = box.permute(0, 2, 3, 1).reshape(-1, 4)
        scores_flat, labels = scores.max(dim=1)
        keep = scores_flat > score_thresh
        if not keep.any():
            continue
        points = points[keep]
        regs = regs[keep]
        boxes = torch.stack(
            [
                points[:, 0] - regs[:, 0],
                points[:, 1] - regs[:, 1],
                points[:, 0] + regs[:, 2],
                points[:, 1] + regs[:, 3],
            ],
            dim=-1,
        ).clamp(0, 1)
        boxes_all.append(boxes)
        scores_all.append(scores_flat[keep])
        labels_all.append(labels[keep])
    if not boxes_all:
        return (
            torch.empty(0, 4, device=device),
            torch.empty(0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )
    boxes = torch.cat(boxes_all)
    scores = torch.cat(scores_all)
    labels = torch.cat(labels_all)
    keep = nms(boxes, scores, nms_thresh)
    keep = keep[:topk]
    return boxes[keep], scores[keep], labels[keep]


def train(args):
    device = torch.device(args.device)
    dataset = YoloBoxDataset(args.images, args.labels, resolution=args.resolution)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )
    model = Sam3FrozenBackboneDetector(
        checkpoint_path=args.sam3_checkpoint,
        num_classes=args.num_classes,
        device=args.device,
        train_backbone=args.train_backbone,
        feature_levels=args.feature_levels,
        adapter_layers=args.adapter_layers,
        adapter_rank=args.adapter_rank,
        adapter_scale=args.adapter_scale,
    ).to(device)
    model.train()
    model.sam3.eval()
    if args.adapter_lr is not None and args.adapter_layers > 0:
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        adapter_params = [p for p in model.adapter_parameters() if p.requires_grad]
        head_param_ids = {id(p) for p in head_params}
        adapter_param_ids = {id(p) for p in adapter_params}
        other_params = [
            p
            for p in model.parameters()
            if p.requires_grad
            and id(p) not in adapter_param_ids
            and id(p) not in head_param_ids
        ]
        param_groups = [
            {"params": head_params, "lr": args.lr},
            {"params": adapter_params, "lr": args.adapter_lr},
        ]
        if other_params:
            param_groups.append({"params": other_params, "lr": args.lr})
        optim = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"trainable_params={trainable} adapter_layers={args.adapter_layers} "
        f"adapter_rank={args.adapter_rank} feature_levels={args.feature_levels}",
        flush=True,
    )
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            images = batch["images"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
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
            optim.step()
            if step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step} loss={losses['loss'].item():.4f} "
                    f"cls={losses['loss_cls'].item():.4f} "
                    f"l1={losses['loss_l1'].item():.4f} "
                    f"giou={losses['loss_giou'].item():.4f} "
                    f"pos={losses['num_pos']}",
                    flush=True,
                )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "head": model.head.state_dict(),
                "adapter_state": model.adapter_state_dict(),
                "adapter_layers": args.adapter_layers,
                "adapter_rank": args.adapter_rank,
                "adapter_scale": args.adapter_scale,
                "adapter_block_indices": model.adapter_block_indices,
                "num_classes": args.num_classes,
                "feature_levels": args.feature_levels,
                "resolution": args.resolution,
            },
            args.output,
        )
        print(f"saved {args.output}", flush=True)


@torch.inference_mode()
def predict(args):
    device = torch.device(args.device)
    ckpt = torch.load(args.detector_checkpoint, map_location="cpu")
    adapter_state = ckpt.get("adapter_state", {})
    adapter_layers = ckpt.get("adapter_layers", len(adapter_state))
    model = Sam3FrozenBackboneDetector(
        checkpoint_path=args.sam3_checkpoint,
        num_classes=ckpt["num_classes"],
        device=args.device,
        train_backbone=False,
        feature_levels=ckpt.get("feature_levels", -1),
        adapter_layers=adapter_layers,
        adapter_rank=ckpt.get("adapter_rank", 32),
        adapter_scale=ckpt.get("adapter_scale", 1.0),
    ).to(device)
    if adapter_state:
        model.load_adapter_state_dict(adapter_state)
    model.head.load_state_dict(ckpt["head"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    orig_w, orig_h = image.size
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(ckpt.get("resolution", args.resolution), ckpt.get("resolution", args.resolution))),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    inp = transform(image).unsqueeze(0).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
        (cls_outputs, box_outputs), features = model(inp)
    boxes, scores, labels = decode_predictions(
        [x[0:1] for x in cls_outputs],
        [x[0:1] for x in box_outputs],
        [x[0:1] for x in features],
        args.score_thresh,
        args.nms_thresh,
        args.topk,
    )
    boxes_px = boxes.detach().cpu()
    boxes_px[:, [0, 2]] *= orig_w
    boxes_px[:, [1, 3]] *= orig_h
    for score, label, box in zip(scores.cpu().tolist(), labels.cpu().tolist(), boxes_px.tolist()):
        print(f"{score:.4f}\t{label}\t{box[0]:.1f}\t{box[1]:.1f}\t{box[2]:.1f}\t{box[3]:.1f}")

    if args.output_image:
        draw = ImageDraw.Draw(image)
        for score, label, box in zip(scores.cpu().tolist(), labels.cpu().tolist(), boxes_px.tolist()):
            draw.rectangle(box, outline=(0, 220, 70), width=3)
            draw.text((box[0], box[1]), f"{label}:{score:.2f}", fill=(0, 220, 70))
        image.save(args.output_image)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    train_p = subparsers.add_parser("train")
    train_p.add_argument("--images", required=True)
    train_p.add_argument("--labels", required=True)
    train_p.add_argument("--sam3-checkpoint", default="checkpoints/sam3.pt")
    train_p.add_argument("--output", default="checkpoints/sam3_frozen_detector_head.pt")
    train_p.add_argument("--num-classes", type=int, default=1)
    train_p.add_argument("--resolution", type=int, default=1008)
    train_p.add_argument("--feature-levels", type=int, default=-1)
    train_p.add_argument("--batch-size", type=int, default=1)
    train_p.add_argument("--epochs", type=int, default=20)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--workers", type=int, default=2)
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--train-backbone", action="store_true")
    train_p.add_argument("--adapter-layers", type=int, default=0)
    train_p.add_argument("--adapter-rank", type=int, default=32)
    train_p.add_argument("--adapter-scale", type=float, default=1.0)
    train_p.add_argument("--adapter-lr", type=float)
    train_p.add_argument("--log-every", type=int, default=10)
    train_p.set_defaults(func=train)

    pred_p = subparsers.add_parser("predict")
    pred_p.add_argument("--image", required=True)
    pred_p.add_argument("--sam3-checkpoint", default="checkpoints/sam3.pt")
    pred_p.add_argument("--detector-checkpoint", required=True)
    pred_p.add_argument("--resolution", type=int, default=1008)
    pred_p.add_argument("--score-thresh", type=float, default=0.3)
    pred_p.add_argument("--nms-thresh", type=float, default=0.5)
    pred_p.add_argument("--topk", type=int, default=100)
    pred_p.add_argument("--device", default="cuda")
    pred_p.add_argument("--output-image")
    pred_p.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
