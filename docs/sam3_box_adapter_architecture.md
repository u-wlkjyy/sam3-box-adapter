# SAM3-BoxAdapter Network Structure

This diagram summarizes the prototype implemented in `scripts/sam3_frozen_detector.py`.
The SAM3 image backbone is frozen. Only the FCOS-style detector head and optional
low-rank residual adapters are trainable.

```mermaid
flowchart LR
    img["Input image"] --> prep["Resize + Normalize<br/>1008 x 1008"]
    prep --> sam3["SAM3 image model<br/>segmentation disabled"]

    subgraph frozen["Frozen SAM3 vision backbone"]
        patch["Patch embedding"] --> early["ViT blocks 0..K-1<br/>frozen no-grad"]
        early --> late["Last N ViT blocks<br/>frozen original weights"]
        late --> neck["SAM3 SimpleFPN neck<br/>multi-scale 256-ch features"]
    end

    subgraph adapters["Trainable adapter path"]
        a1["Low-rank residual adapters<br/>Linear 1024->rank->1024"]
    end

    late -. residual fusion .-> a1
    a1 -. added back .-> late
    sam3 --> patch
    neck --> select["Select feature levels<br/>usually last 1 level"]

    subgraph head["Trainable detector head"]
        cls["Classification tower<br/>Conv + GN + SiLU"]
        reg["Box tower<br/>Conv + GN + SiLU"]
        logits["Class logits"]
        ltrb["LTRB regression"]
    end

    select --> cls --> logits
    select --> reg --> ltrb
    logits --> decode["Score threshold + NMS"]
    ltrb --> decode
    decode --> boxes["Fixed-class boxes"]
```

## Training Objective

```mermaid
flowchart LR
    labels["YOLO box labels<br/>class cx cy w h"] --> assign["FCOS point assignment"]
    logits["Predicted class logits"] --> focal["Sigmoid focal loss"]
    ltrb["Predicted LTRB boxes"] --> l1["L1 loss"]
    ltrb --> giou["GIoU loss"]
    assign --> focal
    assign --> l1
    assign --> giou
    focal --> total["Total loss"]
    l1 --> total
    giou --> total
```

## Checkpoint Contents

The detector checkpoint stores only task-specific weights:

- detector head state dict;
- adapter state dict;
- adapter configuration;
- number of classes, feature level count, and input resolution.

It does not include the SAM3 checkpoint.
