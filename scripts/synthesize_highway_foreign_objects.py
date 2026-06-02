# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import argparse
import base64
import json
import http.client
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


OBJECTS = [
    ("dark tire debris fragment", 0.13, 0.08),
    ("fallen cardboard box", 0.16, 0.11),
    ("black garbage bag", 0.15, 0.10),
    ("orange traffic cone lying on its side", 0.12, 0.10),
    ("plastic bucket", 0.12, 0.10),
    ("flat metal sheet", 0.18, 0.08),
    ("wooden pallet fragment", 0.18, 0.11),
    ("broken vehicle bumper piece", 0.18, 0.09),
    ("loose cargo bag", 0.16, 0.11),
    ("small blue tarp bundle", 0.18, 0.10),
]

WEATHER = [
    "clear daytime",
    "overcast daylight",
    "slightly hazy afternoon",
    "early morning soft light",
    "late afternoon warm light",
    "light rain with wet asphalt",
]

TRAFFIC = [
    "light traffic with a few distant vehicles",
    "moderate traffic in adjacent lanes",
    "sparse traffic with long empty lane sections",
    "several cars and trucks far from the object",
    "normal highway traffic in the background",
]

CAMERA = [
    "elevated roadside CCTV viewpoint",
    "fixed gantry-mounted traffic camera viewpoint",
    "high pole surveillance camera viewpoint",
    "wide-angle traffic monitoring camera viewpoint",
]

LANE_POSITIONS = [
    # cx, cy, width jitter ranges are normalized in the generated image.
    (0.35, 0.64),
    (0.47, 0.61),
    (0.58, 0.66),
    (0.68, 0.59),
    (0.43, 0.73),
    (0.62, 0.74),
]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def choose_box(obj_w, obj_h):
    base_x, base_y = random.choice(LANE_POSITIONS)
    cx = clamp(random.gauss(base_x, 0.035), 0.15, 0.85)
    cy = clamp(random.gauss(base_y, 0.035), 0.42, 0.86)
    scale = random.uniform(0.75, 1.25)
    w = clamp(obj_w * scale, 0.06, 0.24)
    h = clamp(obj_h * scale, 0.04, 0.18)
    return cx, cy, w, h


def region_words(cx, cy):
    horizontal = "center"
    if cx < 0.4:
        horizontal = "left"
    elif cx > 0.6:
        horizontal = "right"

    vertical = "middle"
    if cy > 0.68:
        vertical = "foreground"
    elif cy < 0.52:
        vertical = "upper middle"
    return f"{vertical} {horizontal}"


def positive_prompt(obj_name, cx, cy):
    return (
        "Realistic highway CCTV surveillance frame from a fixed elevated traffic "
        f"camera, {random.choice(CAMERA)}, {random.choice(WEATHER)}, "
        f"{random.choice(TRAFFIC)}. Multi-lane asphalt highway with lane markings, "
        "guardrails, perspective distortion, mild compression artifacts, and real "
        "traffic-camera image quality. "
        f"One clearly visible foreign object is lying on the road surface: {obj_name}. "
        f"Place the foreign object in the {region_words(cx, cy)} area of one lane, "
        "fully inside the frame, separate from all vehicles, not attached to any car, "
        "with natural lighting and a small realistic shadow. The object should be "
        "detectable but not huge, roughly small-to-medium size relative to the lane. "
        "No pedestrians, no text overlay, no timestamp, no watermark, no bounding boxes."
    )


def negative_prompt():
    return (
        "Realistic highway CCTV surveillance frame from a fixed elevated traffic "
        f"camera, {random.choice(CAMERA)}, {random.choice(WEATHER)}, "
        f"{random.choice(TRAFFIC)}. Clean multi-lane asphalt highway with lane "
        "markings, guardrails, perspective distortion, mild compression artifacts, "
        "and real traffic-camera image quality. No foreign objects on the road "
        "surface, no debris, no fallen cargo, no pedestrians, no stopped vehicles, "
        "no text overlay, no timestamp, no watermark, no bounding boxes."
    )


def post_json(url, api_key, payload, timeout=240):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def save_image_response(resp, out_path):
    item = resp["data"][0]
    if "b64_json" in item:
        out_path.write_bytes(base64.b64decode(item["b64_json"]))
        return
    if "url" in item:
        with urllib.request.urlopen(item["url"], timeout=240) as resp:
            out_path.write_bytes(resp.read())
        return
    raise RuntimeError("Image response has neither b64_json nor url")


def generate_one(job, args, endpoint, images_dir, labels_dir, metadata_path, lock):
    stem = f"highway_{job['idx']:04d}"
    img_path = images_dir / f"{stem}.png"
    label_path = labels_dir / f"{stem}.txt"
    if img_path.exists() and label_path.exists():
        return f"skip {stem}"

    payload = {
        "model": args.model,
        "prompt": job["prompt"],
        "size": args.size,
    }

    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            resp = post_json(endpoint, args.api_key, payload)
            save_image_response(resp, img_path)
            if job["positive"]:
                cx, cy, w, h = job["box"]
                label_path.write_text(
                    f"{args.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n",
                    encoding="utf-8",
                )
            else:
                label_path.write_text("", encoding="utf-8")
            with lock:
                with metadata_path.open("a", encoding="utf-8") as meta:
                    meta.write(
                        json.dumps(
                            {**job, "image": str(img_path), "label": str(label_path)}
                        )
                        + "\n"
                    )
            time.sleep(args.sleep)
            return f"wrote {stem}"
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            RuntimeError,
        ) as exc:
            last_error = exc
            wait = args.sleep * attempt * 4
            time.sleep(wait)
    raise RuntimeError(f"Failed to generate {stem}: {last_error}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic highway foreign-object detection data."
    )
    parser.add_argument("--out", default="synthetic_highway_foreign_objects")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--negative-count", type=int, default=20)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--model", default="gpt-image-2")
    parser.add_argument("--base-url", default=os.getenv("IMAGE_API_BASE_URL"))
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--api-key", default=os.getenv("IMAGE_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        raise RuntimeError("Set IMAGE_API_KEY or pass --api-key")
    if not args.base_url:
        raise RuntimeError("Set IMAGE_API_BASE_URL or pass --base-url")
    if args.negative_count > args.count:
        raise ValueError("--negative-count cannot exceed --count")

    random.seed(args.seed)
    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.jsonl"

    positive_count = args.count - args.negative_count
    jobs = []
    for idx in range(positive_count):
        obj_name, obj_w, obj_h = random.choice(OBJECTS)
        cx, cy, w, h = choose_box(obj_w, obj_h)
        jobs.append(
            {
                "idx": idx,
                "positive": True,
                "object": obj_name,
                "box": [cx, cy, w, h],
                "prompt": positive_prompt(obj_name, cx, cy),
            }
        )
    for idx in range(positive_count, args.count):
        jobs.append(
            {
                "idx": idx,
                "positive": False,
                "object": None,
                "box": None,
                "prompt": negative_prompt(),
            }
        )
    random.shuffle(jobs)

    endpoint = args.base_url.rstrip("/") + "/images/generations"
    lock = threading.Lock()
    if args.workers <= 1:
        for n, job in enumerate(jobs, 1):
            msg = generate_one(job, args, endpoint, images_dir, labels_dir, metadata_path, lock)
            print(f"[{n}/{len(jobs)}] {msg}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    generate_one,
                    job,
                    args,
                    endpoint,
                    images_dir,
                    labels_dir,
                    metadata_path,
                    lock,
                )
                for job in jobs
            ]
            for n, fut in enumerate(as_completed(futures), 1):
                print(f"[{n}/{len(jobs)}] {fut.result()}", flush=True)

    print(f"Done: {out_dir}")
    print("Class 0: foreign_object")
    print("Labels are approximate prompt-constrained boxes; inspect before real training.")


if __name__ == "__main__":
    main()
