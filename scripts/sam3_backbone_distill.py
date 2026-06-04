# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import argparse
import hashlib
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sam3_frozen_detector import Sam3FrozenBackboneDetector, _unwrap_feature  # noqa: E402


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class FeatureSpec:
    channels: int
    height: int
    width: int


def parse_int_list(value: str) -> List[int]:
    parsed = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not parsed:
        raise ValueError(f"empty integer list: {value}")
    return parsed


def collect_images(images_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in images_dir.rglob("*")
        if p.suffix.lower() in IMG_EXTS and not any(part.startswith(".") for part in p.parts)
    )


def make_transform(resolution: int):
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class UnlabeledImageDataset(Dataset):
    def __init__(self, images_dir, resolution=1008):
        self.images_dir = Path(images_dir)
        self.image_paths = collect_images(self.images_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")
        self.transform = make_transform(resolution)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return {
            "image": self.transform(image),
            "path": str(image_path),
        }


def collate_unlabeled(batch):
    return {
        "images": torch.stack([x["image"] for x in batch], dim=0),
        "paths": [x["path"] for x in batch],
    }


def group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = nn.GroupNorm(group_count(out_channels), out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class InvertedResidualBlock(nn.Module):
    """MobileNet/RepViT-style block using depthwise spatial mixing."""

    def __init__(self, in_channels, out_channels, stride=1, expansion=2.0):
        super().__init__()
        hidden = max(out_channels, int(round(in_channels * expansion)))
        self.use_residual = stride == 1 and in_channels == out_channels
        self.expand = ConvGnAct(in_channels, hidden, kernel_size=1, stride=1)
        self.depthwise = ConvGnAct(hidden, hidden, kernel_size=3, stride=stride, groups=hidden)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(group_count(out_channels), out_channels),
        )

    def forward(self, x):
        out = self.project(self.depthwise(self.expand(x)))
        if self.use_residual:
            out = out + x
        return out


class ProjectionNeck(nn.Module):
    """Maps the small student channel width to the teacher feature channel width."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        hidden = max(in_channels, out_channels)
        self.net = nn.Sequential(
            ConvGnAct(in_channels, hidden, kernel_size=1),
            nn.Conv2d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class TinyStudentBackbone(nn.Module):
    """Small high-resolution CNN that exports SAM3-compatible feature tensors."""

    def __init__(
        self,
        target_specs: Sequence[FeatureSpec],
        resolution: int,
        widths: Sequence[int] = (32, 64, 128, 192),
        depths: Sequence[int] = (1, 2, 4, 2),
        expansion: float = 2.0,
    ):
        super().__init__()
        if len(widths) != len(depths):
            raise ValueError("widths and depths must have the same length")
        self.target_specs = list(target_specs)
        self.resolution = int(resolution)
        self.widths = [int(x) for x in widths]
        self.depths = [int(x) for x in depths]
        self.expansion = float(expansion)

        self.stem = ConvGnAct(3, self.widths[0], kernel_size=3, stride=2)
        self.stages = nn.ModuleList()
        self.stage_strides: List[int] = []

        in_channels = self.widths[0]
        current_stride = 2
        for idx, (width, depth) in enumerate(zip(self.widths, self.depths)):
            blocks = []
            if idx > 0:
                blocks.append(InvertedResidualBlock(in_channels, width, stride=2, expansion=expansion))
                current_stride *= 2
                in_channels = width
            for _ in range(max(0, depth - 1)):
                blocks.append(InvertedResidualBlock(in_channels, width, stride=1, expansion=expansion))
            if not blocks:
                blocks.append(InvertedResidualBlock(in_channels, width, stride=1, expansion=expansion))
            self.stages.append(nn.Sequential(*blocks))
            self.stage_strides.append(current_stride)

        self.level_stage_indices = [self._choose_stage_for_spec(spec) for spec in self.target_specs]
        self.projections = nn.ModuleList(
            [
                ProjectionNeck(self.widths[stage_idx], spec.channels)
                for stage_idx, spec in zip(self.level_stage_indices, self.target_specs)
            ]
        )

    def _choose_stage_for_spec(self, spec: FeatureSpec) -> int:
        target_size = max(spec.height, spec.width)
        target_stride = self.resolution / max(1, target_size)
        scores = [
            abs(math.log(max(stride, 1) / max(target_stride, 1e-6)))
            for stride in self.stage_strides
        ]
        return int(min(range(len(scores)), key=scores.__getitem__))

    def extract_stage_features(self, x) -> List[torch.Tensor]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

    def forward(self, x) -> List[torch.Tensor]:
        stage_features = self.extract_stage_features(x)
        outputs = []
        for spec, stage_idx, projection in zip(
            self.target_specs,
            self.level_stage_indices,
            self.projections,
        ):
            feat = projection(stage_features[stage_idx])
            if feat.shape[-2:] != (spec.height, spec.width):
                feat = F.interpolate(
                    feat,
                    size=(spec.height, spec.width),
                    mode="bilinear",
                    align_corners=False,
                )
            outputs.append(feat)
        return outputs

    def config(self) -> Dict:
        return {
            "name": "tiny_depthwise_cnn",
            "target_specs": [asdict(spec) for spec in self.target_specs],
            "resolution": self.resolution,
            "widths": self.widths,
            "depths": self.depths,
            "expansion": self.expansion,
            "stage_strides": self.stage_strides,
            "level_stage_indices": self.level_stage_indices,
        }


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def build_teacher(args, device):
    teacher_ckpt = None
    if args.teacher_detector_checkpoint:
        teacher_ckpt = torch.load(args.teacher_detector_checkpoint, map_location="cpu")
        adapter_state = teacher_ckpt.get("adapter_state", {})
        adapter_layers = teacher_ckpt.get("adapter_layers", len(adapter_state))
        adapter_rank = teacher_ckpt.get("adapter_rank", 32)
        adapter_scale = teacher_ckpt.get("adapter_scale", 1.0)
        feature_levels = teacher_ckpt.get("feature_levels", args.feature_levels)
        resolution = teacher_ckpt.get("resolution", args.resolution)
    else:
        adapter_state = {}
        adapter_layers = 0
        adapter_rank = 32
        adapter_scale = 1.0
        feature_levels = args.feature_levels
        resolution = args.resolution

    teacher = Sam3FrozenBackboneDetector(
        checkpoint_path=args.sam3_checkpoint,
        num_classes=1,
        device=str(device),
        train_backbone=False,
        feature_levels=feature_levels,
        adapter_layers=adapter_layers,
        adapter_rank=adapter_rank,
        adapter_scale=adapter_scale,
    ).to(device)
    if adapter_state:
        teacher.load_adapter_state_dict(adapter_state)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher, int(resolution), int(feature_levels)


@torch.inference_mode()
def infer_teacher_specs(teacher: Sam3FrozenBackboneDetector, sample: torch.Tensor, device) -> List[FeatureSpec]:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        features = teacher.extract_features(sample.unsqueeze(0).to(device))
    specs = []
    for feature in features:
        feature = _unwrap_feature(feature)
        _, channels, height, width = feature.shape
        specs.append(FeatureSpec(int(channels), int(height), int(width)))
    return specs


def feature_cache_path(cache_dir: Path, image_path: str) -> Path:
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.pt"


def stack_cached_features(cached_items: Sequence[Sequence[torch.Tensor]], device) -> List[torch.Tensor]:
    num_levels = len(cached_items[0])
    stacked = []
    for level_idx in range(num_levels):
        level_features = [item[level_idx].to(device) for item in cached_items]
        stacked.append(torch.cat(level_features, dim=0))
    return stacked


@torch.inference_mode()
def get_teacher_features(
    teacher: Sam3FrozenBackboneDetector,
    images: torch.Tensor,
    paths: Sequence[str],
    cache_dir: Optional[Path],
    device,
) -> List[torch.Tensor]:
    if cache_dir is None:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return [feature.detach() for feature in teacher.extract_features(images)]

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = [feature_cache_path(cache_dir, path) for path in paths]
    if all(path.exists() for path in cache_paths):
        cached_items = [
            torch.load(path, map_location="cpu")["features"]
            for path in cache_paths
        ]
        return stack_cached_features(cached_items, device)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        features = [feature.detach() for feature in teacher.extract_features(images)]
    for image_idx, cache_path in enumerate(cache_paths):
        if cache_path.exists():
            continue
        torch.save(
            {
                "features": [
                    feature[image_idx : image_idx + 1].detach().cpu().to(torch.float16)
                    for feature in features
                ]
            },
            cache_path,
        )
    return features


def distill_feature_loss(
    student_features: Sequence[torch.Tensor],
    teacher_features: Sequence[torch.Tensor],
    cosine_weight: float,
    l1_weight: float,
) -> Dict[str, torch.Tensor]:
    if len(student_features) != len(teacher_features):
        raise ValueError(
            f"student features={len(student_features)} teacher features={len(teacher_features)}"
        )

    total_cos = student_features[0].new_tensor(0.0)
    total_l1 = student_features[0].new_tensor(0.0)
    for student, teacher in zip(student_features, teacher_features):
        teacher = teacher.detach()
        if student.shape != teacher.shape:
            raise ValueError(f"feature shape mismatch: student={student.shape} teacher={teacher.shape}")

        student_n = F.normalize(student.float(), dim=1)
        teacher_n = F.normalize(teacher.float(), dim=1)
        total_cos = total_cos + (1.0 - (student_n * teacher_n).sum(dim=1)).mean()
        total_l1 = total_l1 + F.smooth_l1_loss(student_n, teacher_n)

    return {
        "loss": cosine_weight * total_cos + l1_weight * total_l1,
        "loss_cos": total_cos.detach(),
        "loss_l1": total_l1.detach(),
    }


def save_checkpoint(
    student: TinyStudentBackbone,
    args,
    resolution: int,
    feature_levels: int,
    epoch: int,
    metrics: Dict[str, float],
):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "sam3_backbone_distill_v1",
            "student": student.state_dict(),
            "student_config": student.config(),
            "resolution": int(resolution),
            "feature_levels": int(feature_levels),
            "epoch": int(epoch),
            "metrics": metrics,
            "teacher": {
                "sam3_checkpoint": args.sam3_checkpoint,
                "teacher_detector_checkpoint": args.teacher_detector_checkpoint,
            },
            "distill": {
                "cosine_weight": args.cosine_weight,
                "l1_weight": args.l1_weight,
                "student_widths": args.student_widths,
                "student_depths": args.student_depths,
                "student_expansion": args.student_expansion,
            },
        },
        output,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Distill SAM3 image-backbone features into a small student backbone."
    )
    parser.add_argument("--images", required=True, help="Unlabeled image directory.")
    parser.add_argument("--sam3-checkpoint", default="checkpoints/sam3.pt")
    parser.add_argument(
        "--teacher-detector-checkpoint",
        help="Optional SAM3-BoxAdapter detector checkpoint. If set, its adapters are used in the teacher.",
    )
    parser.add_argument("--output", default="checkpoints/sam3_student_backbone.pt")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--feature-levels", type=int, default=1)
    parser.add_argument("--student-widths", default="32,64,128,192")
    parser.add_argument("--student-depths", default="1,2,4,2")
    parser.add_argument("--student-expansion", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--l1-weight", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-epochs", type=int, default=0)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving epochs. 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Required average-loss improvement to reset early stopping patience.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--feature-cache-dir",
        help="Optional directory for per-image fp16 teacher feature cache.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    teacher, resolution, feature_levels = build_teacher(args, device)

    dataset = UnlabeledImageDataset(args.images, resolution=resolution)
    target_specs = infer_teacher_specs(teacher, dataset[0]["image"], device)
    widths = parse_int_list(args.student_widths)
    depths = parse_int_list(args.student_depths)
    student = TinyStudentBackbone(
        target_specs=target_specs,
        resolution=resolution,
        widths=widths,
        depths=depths,
        expansion=args.student_expansion,
    ).to(device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_unlabeled,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else None

    print(
        "teacher_specs="
        + ",".join(f"{spec.channels}x{spec.height}x{spec.width}" for spec in target_specs),
        flush=True,
    )
    print(
        f"student=tiny_depthwise_cnn params={count_params(student)} "
        f"widths={widths} depths={depths} stage_strides={student.stage_strides} "
        f"level_stage_indices={student.level_stage_indices}",
        flush=True,
    )

    best_loss = float("inf")
    best_epoch = -1
    stale_epochs = 0
    use_early_stop = args.early_stop_patience > 0

    for epoch in range(args.epochs):
        student.train()
        epoch_loss_sum = 0.0
        epoch_cos_sum = 0.0
        epoch_l1_sum = 0.0
        epoch_samples = 0
        for step, batch in enumerate(loader):
            images = batch["images"].to(device)
            teacher_features = get_teacher_features(
                teacher,
                images,
                batch["paths"],
                cache_dir=cache_dir,
                device=device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                student_features = student(images)
                losses = distill_feature_loss(
                    student_features,
                    teacher_features,
                    cosine_weight=args.cosine_weight,
                    l1_weight=args.l1_weight,
                )

            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()

            batch_size = int(images.shape[0])
            epoch_loss_sum += float(losses["loss"].item()) * batch_size
            epoch_cos_sum += float(losses["loss_cos"].item()) * batch_size
            epoch_l1_sum += float(losses["loss_l1"].item()) * batch_size
            epoch_samples += batch_size

            if step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={step} loss={losses['loss'].item():.4f} "
                    f"cos={losses['loss_cos'].item():.4f} "
                    f"l1={losses['loss_l1'].item():.4f}",
                    flush=True,
                )

        normalizer = max(epoch_samples, 1)
        metrics = {
            "avg_loss": epoch_loss_sum / normalizer,
            "avg_cos": epoch_cos_sum / normalizer,
            "avg_l1": epoch_l1_sum / normalizer,
        }
        print(
            f"epoch={epoch} avg_loss={metrics['avg_loss']:.4f} "
            f"avg_cos={metrics['avg_cos']:.4f} avg_l1={metrics['avg_l1']:.4f}",
            flush=True,
        )

        improved = metrics["avg_loss"] < best_loss - args.early_stop_min_delta
        if improved:
            best_loss = metrics["avg_loss"]
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if not use_early_stop or improved:
            save_checkpoint(student, args, resolution, feature_levels, epoch, metrics)
            suffix = "best" if use_early_stop else "latest"
            print(f"saved {suffix} {args.output}", flush=True)

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


if __name__ == "__main__":
    main()
