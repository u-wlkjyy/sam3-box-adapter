# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import argparse
import random
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_clean_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_name(prefix: str, idx: int, original_name: str) -> str:
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return f"{prefix}_{idx:06d}_{stem}{suffix}"


def sample_items(items: Sequence, count: int, rng: random.Random, shuffle: bool = True):
    items = list(items)
    if count == 0:
        return []
    if count < 0 or count >= len(items):
        if shuffle:
            rng.shuffle(items)
        return items
    if not shuffle:
        return items[:count]
    return rng.sample(items, count)


def maybe_shuffle(items: List, rng: random.Random, shuffle: bool):
    if shuffle:
        rng.shuffle(items)
    return items


def zip_entries(zip_path: Path, pattern: str) -> List[str]:
    regex = re.compile(pattern)
    with zipfile.ZipFile(zip_path) as zf:
        return [name for name in zf.namelist() if regex.search(name)]


def tar_entries(tar_path: Path, pattern: str) -> List[str]:
    regex = re.compile(pattern)
    with tarfile.open(tar_path, "r:*") as tf:
        return [
            member.name
            for member in tf.getmembers()
            if member.isfile() and regex.search(member.name)
        ]


def extract_zip_images(zip_path: Path, entries: Sequence[str], out_dir: Path, prefix: str) -> int:
    written = 0
    with zipfile.ZipFile(zip_path) as zf:
        for idx, entry in enumerate(entries):
            out_path = out_dir / safe_name(prefix, idx, entry)
            with zf.open(entry) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            written += 1
    return written


def extract_tar_images(tar_path: Path, entries: Sequence[str], out_dir: Path, prefix: str) -> int:
    entry_set = set(entries)
    written = 0
    with tarfile.open(tar_path, "r:*") as tf:
        for member in tf.getmembers():
            if member.name not in entry_set:
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            out_path = out_dir / safe_name(prefix, written, member.name)
            with src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            written += 1
    return written


def copy_images_from_dirs(
    image_dirs: Iterable[Path],
    out_dir: Path,
    prefix: str,
    limit: int,
    rng: random.Random,
) -> int:
    paths = []
    for image_dir in image_dirs:
        if not image_dir.exists():
            continue
        paths.extend(
            p
            for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS and not any(part.startswith(".") for part in p.parts)
        )
    selected = sample_items(paths, limit, rng)
    written = 0
    for idx, path in enumerate(selected):
        out_path = out_dir / safe_name(prefix, idx, path.name)
        shutil.copy2(path, out_path)
        written += 1
    return written


def maybe_extract_zip(
    name: str,
    zip_path: Path,
    pattern: str,
    count: int,
    out_dir: Path,
    rng: random.Random,
    shuffle: bool,
) -> Tuple[str, int, int]:
    if count == 0:
        return name, 0, 0
    if not zip_path.exists():
        print(f"skip {name}: missing {zip_path}", flush=True)
        return name, 0, 0
    entries = zip_entries(zip_path, pattern)
    selected = sample_items(entries, count, rng, shuffle=shuffle)
    written = extract_zip_images(zip_path, selected, out_dir, name)
    return name, len(entries), written


def maybe_extract_tar_many(
    name: str,
    tar_paths: Sequence[Path],
    pattern: str,
    count: int,
    out_dir: Path,
    rng: random.Random,
    shuffle: bool,
) -> Tuple[str, int, int]:
    if count == 0:
        return name, 0, 0
    available = []
    for tar_path in tar_paths:
        if not tar_path.exists():
            continue
        for entry in tar_entries(tar_path, pattern):
            available.append((tar_path, entry))
    if not available:
        print(f"skip {name}: no matching images found", flush=True)
        return name, 0, 0

    selected = sample_items(available, count, rng, shuffle=shuffle)
    by_tar = {}
    for tar_path, entry in selected:
        by_tar.setdefault(tar_path, []).append(entry)

    written = 0
    for tar_idx, (tar_path, entries) in enumerate(by_tar.items()):
        written += extract_tar_images(tar_path, entries, out_dir, f"{name}{tar_idx:02d}")
    return name, len(available), written


def main():
    parser = argparse.ArgumentParser(
        description="Build one unlabeled road-image mix from AutoDL public datasets and local synthetic images."
    )
    parser.add_argument("--pub-root", default="/autodl-pub/data")
    parser.add_argument("--out", default="unlabeled_road_mix/images")
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Take the first N images in archive order. This is faster for large zip files.",
    )

    parser.add_argument("--kitti-count", type=int, default=3000)
    parser.add_argument("--cityscapes-count", type=int, default=3000)
    parser.add_argument("--culane-count", type=int, default=0)
    parser.add_argument("--synthetic-count", type=int, default=-1, help="-1 means copy all available synthetic images.")

    parser.add_argument("--kitti-zip", default="KITTI_Object/raw/data_object_image_2.zip")
    parser.add_argument("--cityscapes-zip", default="cityscapes/leftImg8bit_trainvaltest.zip")
    parser.add_argument(
        "--culane-tars",
        nargs="*",
        default=[
            "CULane/driver_100_30frame.tar.gz",
            "CULane/driver_161_90frame.tar.gz",
            "CULane/driver_182_30frame.tar.gz",
            "CULane/driver_193_90frame.tar.gz",
            "CULane/driver_23_30frame_part1.tar.gz",
            "CULane/driver_23_30frame_part2.tar.gz",
            "CULane/driver_37_30frame.tar.gz",
        ],
    )
    parser.add_argument(
        "--synthetic-dirs",
        nargs="*",
        default=[
            "synthetic_highway_foreign_objects/images",
            "synthetic_highway_foreign_objects_test30/images",
            "synthetic_highway_foreign_objects_newprompt30/images",
            "synthetic_highway_hard_small_objects30/images",
        ],
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    shuffle = not args.no_shuffle
    pub_root = Path(args.pub_root)
    out_dir = Path(args.out)
    ensure_clean_dir(out_dir, overwrite=args.overwrite)

    summaries = []
    summaries.append(
        maybe_extract_zip(
            "kitti",
            pub_root / args.kitti_zip,
            r"(training|testing)/image_2/.*\.png$",
            args.kitti_count,
            out_dir,
            rng,
            shuffle,
        )
    )
    summaries.append(
        maybe_extract_zip(
            "cityscapes",
            pub_root / args.cityscapes_zip,
            r"leftImg8bit/(train|val|test)/.*_leftImg8bit\.png$",
            args.cityscapes_count,
            out_dir,
            rng,
            shuffle,
        )
    )
    summaries.append(
        maybe_extract_tar_many(
            "culane",
            [pub_root / path for path in args.culane_tars],
            r"\.(jpg|jpeg|png)$",
            args.culane_count,
            out_dir,
            rng,
            shuffle,
        )
    )

    synthetic_written = copy_images_from_dirs(
        [Path(path) for path in args.synthetic_dirs],
        out_dir,
        "synthetic",
        args.synthetic_count,
        rng,
    )
    summaries.append(("synthetic", synthetic_written, synthetic_written))

    total = sum(1 for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    print("summary", flush=True)
    for name, available, written in summaries:
        print(f"{name}: available={available} written={written}", flush=True)
    print(f"out={out_dir}", flush=True)
    print(f"total_images={total}", flush=True)


if __name__ == "__main__":
    main()
