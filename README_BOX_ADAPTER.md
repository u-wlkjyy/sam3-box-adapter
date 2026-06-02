# SAM3-BoxAdapter

SAM3-BoxAdapter is a parameter-efficient adaptation prototype that turns a frozen
SAM3 image backbone into a fixed-class, box-only detector for few-shot industrial
inspection scenarios such as highway foreign-object detection.

The main idea is to use SAM3 as a frozen visual foundation backbone rather than
as a full promptable segmentation pipeline. Text prompting and mask decoding are
removed from the detector path; only lightweight residual adapters and an FCOS-
style detection head are trained from YOLO-format box labels.

## Innovation

本文提出了一种面向工业小样本检测场景的 SAM3 参数高效适配方法。该方法冻结
SAM3 的大规模视觉主干，移除文本提示与掩码预测依赖，仅在视觉特征上训练轻量
级检测头，并在 ViT 主干后部引入少量残差 Adapter 模块，实现从通用可提示分割
模型到固定类别 box-only 检测器的快速迁移。

More formally, SAM3-BoxAdapter adapts a promptable segmentation foundation model
to fixed-category industrial object detection by:

1. using SAM3 image features as a frozen representation source;
2. inserting low-rank residual adapters into the last ViT blocks;
3. training a compact FCOS-style box head on top of SAM3 FPN features;
4. saving only the detector head and adapter weights, not the SAM3 backbone.

This is lightweight in trainable parameters and annotation requirements, not in
full inference compute. The SAM3 vision backbone remains the latency bottleneck.

## Contributions

1. **Box-only SAM3 adaptation.** The detector path avoids text prompts and mask
   supervision, converting SAM3 image features directly into fixed-class boxes.

2. **Parameter-efficient small-sample tuning.** The SAM3 backbone is frozen; only
   the detection head and optional residual adapters are trainable. This supports
   quick adaptation from tens to hundreds of box-labeled industrial samples.

3. **Adapter feature fusion.** Low-rank residual adapters can be attached to the
   last `N` ViT blocks. The adapter outputs are fused back into the backbone
   feature stream while the original SAM3 weights remain frozen.

4. **Industrial detection workflow.** The repository includes scripts for
   synthetic highway foreign-object data generation, YOLO-label training,
   batch prediction, contact-sheet visualization, and box-only SAM3 prompt
   inference.

5. **Latency analysis.** On an RTX 4090 at 1008px input resolution, the prototype
   measured about 58.75 ms per image for pure model + decode and about 57.17 ms
   inside the SAM3 backbone, showing that the detector head is not the bottleneck.

## Files

- `scripts/sam3_frozen_detector.py`: train and predict with frozen SAM3 features,
  optional residual adapters, and a lightweight detector head.
- `scripts/batch_predict_detector.py`: batch inference, visualization, contact
  sheet creation, and coarse YOLO-label evaluation.
- `scripts/box_only_image.py`: run SAM3 text grounding while skipping mask output.
- `scripts/synthesize_highway_foreign_objects.py`: generate synthetic highway
  foreign-object data with an OpenAI-compatible image API.
- `sam3/model/sam3_image.py` and `sam3/model/sam3_image_processor.py`: optional
  `return_masks=False` path for box-only prompt inference.

## Training

The detector expects YOLO labels:

```text
class_id center_x center_y width height
```

All coordinates are normalized to `[0, 1]`.

Example:

```bash
python scripts/sam3_frozen_detector.py train \
  --images synthetic_highway_foreign_objects/images \
  --labels synthetic_highway_foreign_objects/labels \
  --sam3-checkpoint checkpoints/sam3.pt \
  --output checkpoints/sam3_adapter_detector_highway.pt \
  --num-classes 1 \
  --resolution 1008 \
  --feature-levels 1 \
  --batch-size 1 \
  --epochs 20 \
  --workers 0 \
  --lr 1e-4 \
  --adapter-layers 6 \
  --adapter-rank 32 \
  --adapter-lr 5e-5 \
  --weight-decay 1e-4 \
  --device cuda
```

The saved detector checkpoint contains:

- `head`: detector head weights;
- `adapter_state`: adapter weights;
- adapter configuration;
- `num_classes`, `feature_levels`, and `resolution`.

It does not contain the SAM3 checkpoint.

## Prediction

Single image:

```bash
python scripts/sam3_frozen_detector.py predict \
  --image path/to/image.png \
  --sam3-checkpoint checkpoints/sam3.pt \
  --detector-checkpoint checkpoints/sam3_adapter_detector_highway.pt \
  --score-thresh 0.3 \
  --output-image prediction.png \
  --device cuda
```

Batch prediction:

```bash
python scripts/batch_predict_detector.py \
  --images synthetic_highway_foreign_objects_test30/images \
  --labels synthetic_highway_foreign_objects_test30/labels \
  --sam3-checkpoint checkpoints/sam3.pt \
  --detector-checkpoint checkpoints/sam3_adapter_detector_highway.pt \
  --output-dir pred_adapter_highway_test30_pred_only \
  --score-thresh 0.3 \
  --hide-text \
  --device cuda
```

## Synthetic Data Generation

The generator uses an OpenAI-compatible image generation endpoint. No API key or
base URL is stored in this repository. Provide them through environment variables
or command-line arguments:

```bash
export IMAGE_API_KEY=...
export IMAGE_API_BASE_URL=...

python scripts/synthesize_highway_foreign_objects.py \
  --out synthetic_highway_foreign_objects \
  --count 100 \
  --negative-count 20 \
  --model gpt-image-2 \
  --workers 8
```

Generated labels are approximate prompt-conditioned boxes and should be inspected
before real training.

## Limitations

- The method is parameter-efficient, but not yet compute-efficient. The frozen
  SAM3 ViT backbone dominates inference latency.
- Current experiments use synthetic highway images; real industrial deployment
  needs real data, stricter labeling, and baselines such as YOLO or RT-DETR.
- Hard scenes such as rain, tunnels, night, and tiny objects require more data or
  stronger adaptation.

## Paper Direction

A suitable paper framing is:

> Parameter-Efficient Adaptation of SAM3 for Few-Shot Box-Only Industrial Object
> Detection

The central claim should be small-sample, parameter-efficient task adaptation,
not real-time lightweight inference.
