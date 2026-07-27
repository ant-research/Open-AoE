# AoE Dataset Selection, Reconstruction, and Retargeting Practices

Language: **English** | [中文](AOE_DATASET_PRACTICES.zh-CN.md)

This guide focuses on scene selection, instance binding, and data quality.
For clean-machine setup, third-party compatibility, 12-cell admission, and
failure classification, see
[Clean-Machine Deployment and Reproduction Practices](REPRODUCTION_PRACTICES.md).

The goal is to establish correct inputs from real AoE video before interpreting
the behavior of EgoInfinity, DAI, or SPIDER.

## 1. Bind the Same Physical Object Instance

A text prompt identifies a category, not a unique bottle among several bottles
in one frame. For every scene:

1. Confirm on the original RGB that the annotation box refers to the intended
   physical instance.
2. Choose a reference frame that shows both the object and the operating hand.
3. Verify that the object point cloud or mesh covers the same instance in the
   RGB overlay.
4. Record `segment + annotation_id + absolute ref-source-frame + bbox + prompt`.
5. When multiple instances share a class, bind the target with a point or box
   and temporal continuity instead of prompt ranking alone.

Consistent file provenance does not prove semantic instance correctness. The
RGB overlay must still show that the annotated physical object is being tracked.

AoE action boxes use normalized-to-1000 `[x1, y1, x2, y2]` coordinates. Scale
both corners by video width and height; do not interpret the final two values as
width and height. Before screening, save a frame with the original action box
and verify that:

- the pixel box stays inside the image;
- its center lies on the intended object;
- its area is plausible for that object;
- the prompt gate, contact sheet, and formal packager use the same convention.

An incorrectly interpreted box can remain numerically in bounds, so range
checks alone do not detect this error.

## 2. Automatic Windows and Direct Entry

Automatic entry is not a separate reconstruction backend. It selects a clean,
continuous window around the same annotation and then invokes the same formal
pipeline as direct entry.

- Prefer the longest window and start a full run only after the unchanged SAM3
  temporal and interaction gates pass.
- The current strategy preserves the left-side anchor and trims only the right
  endpoint, avoiding a silent switch to another action phase.
- Automatic entry is useful when later frames contain occlusion, the instance
  leaves view, or tracking switches identity.
- Direct entry is for a manually audited clip, reference frame, and box, and
  provides a reproducible baseline.
- Neither mode may bypass reconstruction, route binding, or backend input QC.

If a long window switches instance but a shorter continuous window with the
same anchor passes, classify that as a frontend window problem, not a DAI or
SPIDER optimization failure.

## 3. Optional Candidate Screening

Before a large batch, the diagnostic screener can rank candidate annotations.
Screening is optional and is not a formal reconstruction or retargeting entry.

```bash
python scripts/diagnostics/screen_aoe_retarget_candidates.py \
  --dataset-root "$AOE_DATA_ROOT" \
  --existing-experiments-root experiments \
  --exclude-existing \
  --require-hand-object-near \
  --top-k 100 \
  --date-tag YYYYMMDD \
  --output-json experiments/screening/candidates.json \
  --output-csv experiments/screening/candidates.csv
```

Use the output only to select annotations and windows worth running. Formal
results must still be regenerated from the original AoE video through the
automatic or direct production entry.

## 4. Normalize Camera Intrinsics Before Judging HOI

AoE/HaWoR hands and Ego objects may originate from different crops or camera
intrinsics. Placing both XYZ streams directly in one camera produces incorrect
overlays and contact geometry. The adapter must perform an auditable ray/depth
conversion that:

- preserves projected pixels;
- preserves Z depth;
- emits a camera matrix matching the actual video resolution;
- records the transform, source K, target K, and hashes in the manifest;
- never compensates projection error by translating or scaling the object.

Render RGB overlays for both AoE annotated hands and estimated hands before
running a backend.

## 5. Separate Reconstruction, IK, and Physics Failures

Inspect boundaries in order:

1. Original RGB plus hand/object overlay: errors indicate annotation, camera,
   instance, or reconstruction problems.
2. Pure-mesh HOI: inspect relative hand/object pose, thumbs, scale, and object
   orientation.
3. Backend initialization or kinematic IK: if the reference is correct but the
   robot palm is offset, inspect morphology and IK mapping.
4. Before and after MJWP optimization: if IK is plausible but contact is lost
   after warmup, preserve it as native backend behavior.
5. DAI/SPIDER plain video: admit a demo only from the real backend output.

Do not hide failures in overlays or triptychs. Do not use post-hoc object scale
or translation to present a diagnostic result as backend success.

## 6. Common Fail-Closed Outcomes

| Failure | Meaning | Correct response |
| --- | --- | --- |
| Multi-instance temporal switching | SAM3 switches physical instances | Shorten to a continuous window or bind the target box again |
| Prompt frame has no final mask | Other frames may have masks, but the formal reference has no target | Preserve the failure; use a direct fresh entry only after manually validating another reference frame for the same annotation |
| Area-ratio jump | The mask abruptly expands or contracts | Inspect occlusion, instance identity, and reference frame |
| Low interaction containment | The mask disagrees with the hand-object interaction region | Correct the real box/reference frame; do not lower thresholds |
| Oversized or shifted action box | AoE `xyxy` was interpreted as `xywh` | Convert normalized-to-1000 `xyxy` and verify on RGB |
| Camera-intrinsics mismatch | Hand, object, and RGB use different cameras | Apply an auditable K-to-K ray conversion |
| Only one hand passes bimanual overlap | The other annotated hand lacks sufficient visual interaction | Fail each hand independently; one hand cannot substitute for a bimanual task |
| HOI coordinate offset exceeds maximum | Preserving source HOI requires excessive rigid compensation | Classify as input/reconstruction quality; do not translate the object or relax the cap |
| Missing layout/timestamp binding | The route timeline is incomplete | Reconstruct fresh or convert the exact timebase; never fabricate frames |
| DAI/SPIDER hand-object separation | Backend output loses the source HOI | Preserve native output, verify the input boundary, then classify backend behavior |
| Backend returns zero but post-QC fails | Optimization completion does not imply display quality | Keep native trajectory/video for diagnosis but do not admit it as a demo |
| Native rollout ends before the source | Valid post-warmup motion does not cover the action | Do not extrapolate or repeat the last frame; materialization must fail closed |

## 7. Evidence Required for Every Scene

```text
original_rgb.mp4
rgb_overlay.mp4
mesh_hoi.mp4
dai_*.mp4 and/or spider_*.mp4
triptych.mp4 (optional review packaging)
fresh_input_manifest.json
prompt_gate.json / mask_qc_summary.json
adapter + route + backend manifests
native logs and return codes
SHA256SUMS.txt
```

If no DAI/SPIDER video exists, state whether the run never reached the backend
or the backend failed. Do not substitute an Ego robot video. A candidate may
have known limitations, but those limitations must be recorded next to its
conclusion.
