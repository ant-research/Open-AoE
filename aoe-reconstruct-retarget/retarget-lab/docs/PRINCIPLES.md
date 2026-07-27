# Pipeline Principles

Language: **English** | [中文](PRINCIPLES.zh-CN.md)

Navigation: [README](../README.md) | [Installation](INSTALL.md) | [API](API.md)

This document explains the pipeline split, data transformations, scale and
coordinate conventions, and visualization/debugging rules.

## 1. Goal

Input:

```text
Open-AoE egocentric RGB video
target object description
optional Open-AoE hand annotation/reconstruction
```

Output:

```text
raw RGB + reconstructed hand/object mesh overlay
depth / input geometry
robot retargeting render
```

The project should preserve physically meaningful intermediate assets instead
of tuning videos to look roughly correct. Keep masks, meshes, 6DoF poses, hand
trajectories, robot scenes, robot trajectories, videos, and manifests.

## 2. Two Main Pipeline Parts

### Trajectory 6DoF Reconstruction

This stage estimates:

```text
object mesh
object 6DoF trajectory
hand motion / hand mesh
depth / pointmap
RGB mesh overlay
diagnostic mask/mesh videos
```

Supported options:

```text
egoinfinity
do_as_i_do
```

### Retargeting

This stage maps human hand/object motion into a robot embodiment and renders it:

```text
egoinfinity -> G1
do_as_i_do -> Sharpa
spider      -> XHand / MJWP
```

`hand_source=aoe|estimated` decides whether the hand input comes from Open-AoE
hand data or from the selected reconstruction pipeline.

## 3. Why v4 Runs Two Full Pipelines

The full matrix has 12 cells:

```text
2 trajectory_6dof x 2 hand_source x 3 retargeting
```

Running every cell from raw RGB would repeat heavy EgoInfinity/Do-as-I-Do
reconstruction and Do-as-I-Do physics optimization. A short clip can therefore
take hours if every cell starts from scratch.

The recommended policy is:

1. Run EgoInfinity once.
2. Run Do-as-I-Do once.
3. Save reusable assets under `experiments/<run>/intermediates` and
   `experiments/<run>/assets`.
4. Reuse those assets for downstream hand-source and retargeter comparisons.

## 4. EgoInfinity Flow

```text
AoE raw RGB mp4
  -> frame extraction
  -> depth / gravity / hand reconstruction
  -> SAM3.1 video segmentation
  -> SAM3D object mesh
  -> object pose tracking / scale sanity
  -> RGB mesh overlay + pure mesh camera render
  -> G1 retargeting + robot_sim.mp4
```

Key diagnostics:

```text
mask_overlay.mp4
rgb_mesh_overlay.mp4
mesh_pure_camera.mp4
pipeline_result.pkl.gz
retarget/g1/robot_sim.mp4
```

EgoInfinity outputs are kept as produced by the upstream pipeline. Do not
post-filter `pipeline_result.pkl.gz`; object identity issues should be debugged
in the segmentation / SAM3D / pose stages.

Scale issues should be diagnosed through SAM3D canonical scale, mask+depth
physical bounding boxes, pose scale, and `scale_sanity` outputs. Do not use a
hard visual shrink as the final fix.

## 5. Do-as-I-Do Flow

```text
Do-as-I-Do reconstruction raw_dir
  -> config/gravity/layout/hand meshes/object masks
  -> object_6dof.npz + visual.obj + source_trajectory_keypoints.npz
  -> mesh overlay / depth / pure mesh camera render
  -> official retargeting/launch.py
  -> scene.xml + trajectory_mjwp.npz
  -> Sharpa visualization_mjwp.mp4
```

`raw_dir` must be a complete Do-as-I-Do reconstruction output, not a directory
containing only retargeting products.

Scale should be checked in:

```text
obj_tracking_out/<object>/combined_visualization/layout_camera_frame_optimized.json
translation_scale_optimization.method
translation_scale_optimization.mesh_scale
```

The intended scale source is a hand-mask/hand-mesh/pointmap-supported
`hand_anchored_pointmap` optimization. A `shim_from_local_to_scene_scale`
fallback is diagnostic only and should not be accepted as final demo scale.

## 6. SPIDER Flow

```text
experiments/<run>/assets/trajectory_6dof/<pipeline>/<task>/
  source_trajectory_keypoints.npz
  object_meshes/visual.obj
    -> experiment-local SPIDER dataset_dir
    -> generate_xml.py
    -> ik_fast.py
    -> examples/run_mjwp.py
    -> visualization_ik.mp4 / visualization_mjwp.mp4
```

The SPIDER wrapper performs input staging only, then invokes the three original
entry points on a pinned, tracked-clean checkout. Runtime patchers and physical
optimizer tuning are forbidden. Hash the source and staged
`trajectory_keypoints.npz`, robot assets, and object assets. Do not symlink an
entire historical `mano/` directory from another workspace, because SPIDER
metadata writes should not leak back into external outputs.

Keep `ref_dt` on the exact DAI input frame grid. If the original MJWP `sim_dt`
does not divide it, select the largest exact substep no coarser than the
upstream `0.01 s` default and record the policy in the manifest. For example,
30 FPS uses `sim_dt=1/120 s`. Never hide a time-grid conflict by resampling
keypoints or changing backend optimization parameters.

## 7. Visualization Rules

The final triptych rows are:

```text
overlay: raw RGB + reconstructed hand mesh + reconstructed object mesh/point cloud
depth:   depth / input geometry
robot:   robot body + dexterous hand render
```

Invalid final outputs:

```text
segmentation mask only
2D skeleton only
five fingertips
pose dots
object as a single point
borrowed video from another scene or cell
```

These views may be useful diagnostics, but they are not final demo outputs.

## 8. Coordinate and Scale Debugging

Typical transform chain:

```text
camera frame (x-right, y-down, z-fwd)
gravity alignment (camera-frame world-up -> MuJoCo +Z)
object canonical mesh frame
local object frame
scene/world frame (Z-up)
robot/MuJoCo frame (gravity 0 0 -9.81)
render camera frame
```

For Do-as-I-Do, `gravity.json["vec3d"]` is the camera-frame world-up direction.
For an upright camera fallback this is `[0, -1, 0]`, not `[0, 0, 1]`. A
legacy `[0, 0, 1]` shim makes camera-forward become world-up, which corrupts
the DAI and SPIDER physical retargeting frames even if the final render camera
is later adjusted.

When overlay or robot scale is wrong, debug in this order:

1. Segmentation: does the mask include hands, table, background, or another
   object?
2. Reconstruction: are mesh bbox, vertex range, and canonical scale already
   abnormal?
3. 6DoF/layout: are pose translation, layout scale, or optimized mesh scale
   abnormal?
4. Retarget adapter: if projection is correct but MuJoCo is huge or displaced,
   inspect units and coordinate transforms.

Do not use these as final fixes:

```text
hard scale
scale clamp for visual appearance
manual object shrinking
replacing object with a point marker
replacing robot hand with skeleton or fingertips
```

## 9. Integration Practices

| Symptom | Common cause | Recommended handling |
| --- | --- | --- |
| Do-as-I-Do hand mesh did not overlap RGB hand | hand mesh projected with object/MoGe intrinsics | use AoE hand camera intrinsics and clip-resolution scaling |
| Do-as-I-Do frame mapping was wrong | `source_frame_ids` are original full-video ids while clips may use local indices | prefer local sequential indices unless source-id files exist |
| Do-as-I-Do object became a point | overlay applied an extra hard-coded scale | use layout/optimized scale by default |
| Do-as-I-Do object asset was mixed up | physical retarget assets used as RGB reconstruction assets | prefer `clip_dir/obj_tracking_out` and `video_segmentation/masks` |
| Projection failed with `Meshes does not have textures` | reconstructed OBJ contains geometry but no MTL/texture | use the projector's `--object-color` for a deterministic diagnostic-only vertex texture; never rewrite geometry or physical assets |
| Do-as-I-Do scale reached meter level | scale optimization fell back to shim because hand masks/meshes were missing | render hand masks, pass hand meshes, and reject shim by default |
| EgoInfinity selected the wrong same-prompt bottle | multiple candidates existed | object filter supports `all`, `best`, and target point hints |
| EgoInfinity mesh was much larger than mask | SAM3D monocular canonical/pose scale was wrong | `scale_sanity.py` checks mask+depth physical bbox |
| EgoInfinity overlay lacked `T_seq` | newer pkl stores pose in `frame_data[*].sam3_obj_data` | renderer assembles 4x4 transforms from frame data |
| SAM3D worker OOM while task looked idle | interrupted run left an orphan worker | check and clean only verified stale socket workers |
| SPIDER produced no robot output | early runner was import-only | pristine wrapper invokes the original generate_xml -> ik_fast -> run_mjwp and preserves native return codes |
| SPIDER reported `trace_dt must be divisible by sim_dt` | 30 FPS `ref_dt=1/30` is not exactly divisible by the upstream `0.01` default | preserve keypoints/ref_dt and deterministically use the largest exact substep; 30 FPS uses `sim_dt=1/120` |
| SPIDER `generate_xml` could not open object `visual.obj` | staged task info did not bind object assets inside the experiment-local processed dataset | copy hash-verified object assets and rewrite only the staged task-info copy; never mutate the source DAI output |
| DAI/SPIDER robot row did not match the RGB camera aspect/view | official MJWP videos concatenate ref/sim views and use default offscreen resolution | re-render `scene.xml + trajectory_mjwp.npz` with `render_mujoco_trajectory.py --camera front --width 1280 --height 720` |

## 10. Run Artifact Check

- `experiments/<run>/run_env.txt` records input video, time range, GPU ids, and
  task name.
- `logs/` contains commands and logs for EgoInfinity, Do-as-I-Do, and SPIDER.
- `intermediates/` stores native third-party outputs.
- `assets/` stores stable reusable meshes, `npz`, task info, and manifests.
- `cells/<cell>/manifest.json` records the exact overlay/depth/robot sources.
- `videos/<cell>__triptych.mp4` is synchronized; top row is mesh overlay and
  bottom row is a robot-body render.
