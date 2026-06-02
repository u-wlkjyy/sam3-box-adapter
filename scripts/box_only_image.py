# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import argparse
from pathlib import Path

import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def main():
    parser = argparse.ArgumentParser(description="Run SAM3 text grounding as boxes only.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--prompt", required=True, help="Text prompt, for example 'shoe'.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam3.pt",
        help="Path to a SAM3 checkpoint.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Pass --device cpu for CPU testing.")

    checkpoint = Path(args.checkpoint)
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        eval_mode=True,
        device=args.device,
    )
    processor = Sam3Processor(
        model,
        device=args.device,
        confidence_threshold=args.threshold,
    )

    image = Image.open(args.image).convert("RGB")
    amp_enabled = args.device.startswith("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        state = processor.set_image(image)
        state = processor.set_text_prompt(args.prompt, state, return_masks=False)

    boxes = state["boxes"].detach().cpu()
    scores = state["scores"].detach().cpu()
    for score, box in zip(scores.tolist(), boxes.tolist()):
        x0, y0, x1, y1 = box
        print(f"{score:.4f}\t{x0:.1f}\t{y0:.1f}\t{x1:.1f}\t{y1:.1f}")


if __name__ == "__main__":
    main()
