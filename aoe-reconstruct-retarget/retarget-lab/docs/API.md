# Script API

Language: **English** | [中文](API.zh-CN.md)

Navigation: [README](../README.md) | [Installation](INSTALL.md) | [Principles](PRINCIPLES.md)

This document describes the runnable script surface, inputs, outputs, and
experiment asset layout. Run commands from the repository root:

```bash
cd "$AOE_RETARGET_LAB_ROOT"
source local_env.sh
```

Before the first run on a new machine:

```bash
python3 scripts/check_third_party_config.py --json third_party_config_report.json
```

The fresh AoE runner accepts either `AOE_DATA_ROOT` from `local_env.sh` or an
explicit `--dataset-root /path/to/poc_deliver`.

All generated outputs must stay under:

```text
experiments/<run_name>/
```

## 1. Cell Keys

```text
traj_<egoinfinity|do_as_i_do>__hand_<aoe|estimated>__retarget_<egoinfinity|do_as_i_do|spider>
```

| Axis | Options | Meaning |
| --- | --- | --- |
| `trajectory_6dof` | `egoinfinity`, `do_as_i_do` | object 6DoF, mesh, depth, and overlay source |
| `hand_source` | `aoe`, `estimated` | Open-AoE hand data or hand motion estimated by the selected pipeline |
| `retargeting` | `egoinfinity`, `do_as_i_do`, `spider` | robot retargeting/rendering backend |

`hand_source` belongs to the shared retargeting input layer. It is not owned by
any single retargeter. Each cell manifest should record the actual hand source
asset used by that cell.

### 1.1 Fresh AoE automatic and direct entry

```bash
python3 scripts/run_fresh_aoe_auto_window.py \
  --runner scripts/run_fresh_aoe_scene_12_demos.sh \
  --experiments-root experiments \
  --source-run <source_run> --matrix-run <matrix_run> \
  --window-selection-mode auto \
  --candidate-durations-sec 6.0,5.0,4.0,3.0 \
  --manifest experiments/<source_run>_window.json -- \
  --dataset-root "$AOE_DATA_ROOT" \
  --scene <scene> --segment <segment> --annotation-id <id> \
  --object-name "<object prompt>" --task <task> \
  --hand-type <left|right|bimanual> \
  --anchor-hand <left|right|bimanual> \
  --ref-source-frame <absolute_frame> --run-spider all --compose
```

Replace `auto` with `direct`, remove `--candidate-durations-sec`, and add
`--direct-duration-sec <seconds>` to use a user-audited clip window. Direct
mode bypasses only automatic window search, not reconstruction or route/input
validation.

## 2. Recommended Entrypoint: Full 12 Demos

```bash
scripts/run_full_12_demos.sh \
  --run-name <source_run_name> \
  --matrix-run-name <matrix_run_name> \
  --ego-video /path/to/raw_video_undistorted.mp4 \
  --ego-clip-id <clip_id> \
  --ego-objects "bottle of vinegar" \
  --ego-start 0.0 \
  --ego-end 3.0 \
  --dai-raw-dir /path/to/do_as_i_do_raw_dir \
  --dai-clip-dir /path/to/do_as_i_do_clip_dir \
  --task <task_name> \
  --hand-type bimanual \
  --main-cuda 0 \
  --sam3-cuda 1 \
  --sam3d-cuda 2 \
  --spider-cuda-visible-devices 3
```

The script:

1. Runs `scripts/run_v4_two_full_pipelines.sh` once to produce a source run.
2. Runs `scripts/reuse_v4_for_12_demos.py` to materialize the 12-cell matrix.
3. Runs SPIDER cells for Do-as-I-Do trajectory inputs by default.
4. Writes triptych review videos and manifests under `experiments/`.

Expected outputs:

```text
experiments/<source_run_name>/videos/v4_egoinfinity_full__triptych.mp4
experiments/<source_run_name>/videos/v4_do_as_i_do_full__triptych.mp4
experiments/<source_run_name>/reuse/reuse_manifest.json
experiments/<matrix_run_name>/videos/<cell>__triptych.mp4
experiments/<matrix_run_name>/reuse_12_demo_manifest.json
experiments/<matrix_run_name>/cells/<cell>/manifest.json
```

To reuse an existing source run without rerunning SAM3D, EgoInfinity, or
Do-as-I-Do:

```bash
scripts/run_full_12_demos.sh \
  --source-run <source_run_name> \
  --matrix-run-name <matrix_run_name> \
  --task <task_name> \
  --hand-type bimanual \
  --spider-cuda-visible-devices 3
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--source-run` | skip the two full pipelines and reuse `experiments/<source_run>/` |
| `--matrix-run-name` | output run for the 12 demos |
| `--run-spider none\|do_as_i_do\|all` | control SPIDER cells |
| `--duration <sec>` | trim composed videos; `0` keeps source duration |
| `--mode symlink\|hardlink\|copy` | how reused assets are materialized |
| `--force-spider` | rerun existing SPIDER cells |
| `--no-spider-fallback` | do not reuse Do-as-I-Do SPIDER robot videos for EgoInfinity+SPIDER display cells |
| `--allow-spider-mjwp-fallback-video` | diagnostic only; allow SPIDER IK/object-reference fallback videos into composed cells |

### 2.1 Run one selected combination

DAI/Sharpa, using the route-specific prepared raw directory:

```bash
cell=traj_<egoinfinity|do_as_i_do>__hand_<aoe|estimated>__retarget_do_as_i_do
output_root=experiments/<matrix_run>/intermediates/retargeting/do_as_i_do/$cell/retargeting_outputs

scripts/run_do_as_i_do_official_retarget.sh \
  --raw-dir <route_specific_raw_dir> \
  --task <task> \
  --output-root-dir "$output_root" \
  --cuda-visible-devices <gpu> \
  --egl-device-id 0 --headless --no-wait

$RETARGETING_PYTHON scripts/index_cell_assets.py \
  --run-name <matrix_run> \
  --trajectory-6dof <egoinfinity|do_as_i_do> \
  --hand-source <aoe|estimated> \
  --retargeting do_as_i_do \
  --task <task> --hand-type <left|right|bimanual> --robot sharpa
```

The pristine native wrapper deliberately does not accept matrix axes. The
prepared input binds the route; `index_cell_assets.py` records that binding
only after native DAI succeeds.

SPIDER/XHand, after keypoints, object mesh, adapter provenance, and robot assets
have been indexed under the same run:

```bash
scripts/run_spider_retarget.sh \
  --run-name <matrix_run> \
  --trajectory-6dof <egoinfinity|do_as_i_do> \
  --hand-source <aoe|estimated> \
  --task <task> --hand-type <left|right|bimanual> \
  --robot-type xhand --max-sim-steps -1
```

The EgoInfinity/G1 native combination uses `run_egoinfinity_retarget.sh`, as
documented below. Do not label an externally assembled hand route as a native
EgoInfinity route without an explicit matrix-adapter asset binding.

## 3. Two Full Pipelines Only

```bash
V4_RUN_NAME=<run_name> \
V4_EGO_VIDEO=/path/to/raw_video_undistorted.mp4 \
V4_EGO_CLIP_ID=<clip_id> \
V4_EGO_OBJECTS="bottle of vinegar" \
V4_EGO_START=0.0 \
V4_EGO_END=3.0 \
V4_DAI_RAW_DIR=/path/to/do_as_i_do_raw_dir \
V4_DAI_CLIP_DIR=/path/to/do_as_i_do_clip_dir \
V4_TASK=<task_name> \
V4_HAND_TYPE=bimanual \
MAIN_CUDA=0 \
SAM3_WORKER_CUDA=1 \
SAM3D_WORKER_CUDA=2 \
scripts/run_v4_two_full_pipelines.sh
```

The script:

1. Starts SAM3/SAM3.1 and SAM3D workers.
2. Runs one full EgoInfinity pass.
3. Runs one full Do-as-I-Do pass with the official Sharpa entrypoint.
4. Writes reusable assets and a reuse manifest.

Important variables:

| Variable | Meaning |
| --- | --- |
| `V4_RUN_NAME` | experiment directory under `experiments/` |
| `V4_EGO_VIDEO` | AoE undistorted RGB mp4 for EgoInfinity |
| `V4_EGO_CLIP_ID` | clip id written into EgoInfinity metadata |
| `V4_EGO_OBJECTS` | EgoInfinity object text prompt |
| `V4_EGO_START`, `V4_EGO_END`, `V4_EGO_FPS` | EgoInfinity crop time and frame rate |
| `V4_DAI_RAW_DIR` | complete Do-as-I-Do reconstruction/retarget raw directory |
| `V4_DAI_CLIP_DIR` | Do-as-I-Do clip directory used for visualization |
| `V4_TASK` | stable task name used in output paths |
| `V4_HAND_TYPE` | `right`, `left`, or `bimanual` |
| `MAIN_CUDA` | GPU for main processes |
| `SAM3_WORKER_CUDA` | GPU for segmentation worker |
| `SAM3D_WORKER_CUDA` | GPU for SAM3D worker |

Expected outputs:

```text
experiments/<run>/videos/v4_egoinfinity_full__triptych.mp4
experiments/<run>/videos/v4_do_as_i_do_full__triptych.mp4
experiments/<run>/reuse/reuse_manifest.json
experiments/<run>/run_env.txt
experiments/<run>/logs/
```

## 4. EgoInfinity 6DoF + G1 Retargeting

```bash
scripts/run_egoinfinity_retarget.sh \
  --clip-dir /path/to/raw_video_undistorted.mp4 \
  --clip-id <clip_id> \
  --objects "<object prompt>" \
  --start <sec> \
  --end <sec> \
  --fps 15 \
  --run-name <run_name> \
  --robot g1
```

Inputs:

```text
AoE undistorted RGB mp4
object text prompt
start/end/fps
```

Useful environment variables:

| Variable | Meaning |
| --- | --- |
| `EGOINFINITY_OBJECT_SELECTION_MODE=all\|best` | keep all stable instances or select one |
| `EGOINFINITY_TARGET_POINT=x,y` | pixel hint for same-prompt object ranking |
| `EGOINFINITY_MAX_CENTROID_JUMP_PX` | object continuity threshold |
| `EGOINFINITY_POST_SCALE_SANITY=0\|1` | run post scale sanity |
| `EGOINFINITY_SCALE_SANITY_THRESHOLD` | mask/depth scale sanity threshold |
| `EGOINFINITY_RESET_OUTPUT=1` | delete current clip cache and rerun |
| `SAM3_WORKER_SOCKET`, `SAM3D_WORKER_SOCKET` | reuse external workers |

Main outputs:

```text
experiments/<run>/intermediates/egoinfinity/clip/pipeline_result.pkl.gz
experiments/<run>/intermediates/egoinfinity/clip/mask_overlay.mp4
experiments/<run>/intermediates/egoinfinity/clip/rgb_mesh_overlay.mp4
experiments/<run>/intermediates/egoinfinity/clip/mesh_pure_camera.mp4
experiments/<run>/intermediates/egoinfinity/clip/retarget_samples/depth.mp4
experiments/<run>/intermediates/egoinfinity/clip/retarget/g1/robot_sim.mp4
experiments/<run>/intermediates/egoinfinity/clip/object_filter_report.json
experiments/<run>/logs/retarget_egoinfinity_g1.log
```

Use `retarget/g1/robot_sim.mp4` as the robot render. Do not use input
visualization, fingertip markers, or skeleton debug views as final robot output.

## 5. Do-as-I-Do 6DoF Preparation

```bash
scripts/prepare_do_as_i_do_trajectory_6dof.sh \
  --run-name <run_name> \
  --task <task_name> \
  --raw-dir <complete_do_as_i_do_raw_dir>
```

The raw directory must contain:

```text
config.json
gravity.json
obj_tracking_out/<object>/combined_visualization/layout_camera_frame_optimized.json
video_segmentation/masks/frame_*_masks/<object>/<object>.obj
*/all_hand_meshes.npz
```

`left_joints` / `right_joints` inside `all_hand_meshes.npz` must use the
`N x 21 x 3` OpenPose/WiLoR hand-joint layout. Some Do-as-I-Do trials only
store standard MANO `N x 16 x 3` joints; adapt those inputs in the current
experiment's prepared raw directory before calling the official
`retargeting/launch.py`. The adapter should preserve the wrist and three
joints per finger from MANO-16, then add five fingertip joints from the
same-frame MANO vertices. This is hand-joint input-format adaptation; it must
not change object scale, pose, or mesh geometry.

Outputs:

```text
experiments/<run>/intermediates/trajectory_6dof/do_as_i_do/reconstruction/raw_dir
experiments/<run>/assets/trajectory_6dof/do_as_i_do/<task>/object_6dof.npz
experiments/<run>/assets/trajectory_6dof/do_as_i_do/<task>/object_meshes/visual.obj
experiments/<run>/assets/trajectory_6dof/do_as_i_do/<task>/source_trajectory_keypoints.npz
experiments/<run>/assets/trajectory_6dof/do_as_i_do/<task>/task_info.json
experiments/<run>/logs/trajectory_6dof_do_as_i_do.log
```

To generate Do-as-I-Do overlay/depth/pure-mesh diagnostics:

```bash
$RETARGETING_PYTHON scripts/review/materialize_do_as_i_do_visuals.py \
  --run-name <run_name> \
  --clip-dir <do_as_i_do_clip_dir> \
  --task <task_name> \
  --fps 15
```

Common diagnostics:

```text
experiments/<run>/intermediates/trajectory_6dof/do_as_i_do/reconstruction/mask_overlay.mp4
experiments/<run>/intermediates/trajectory_6dof/do_as_i_do/reconstruction/mesh_overlay.mp4
experiments/<run>/intermediates/trajectory_6dof/do_as_i_do/reconstruction/depth.mp4
experiments/<run>/intermediates/trajectory_6dof/do_as_i_do/reconstruction/mesh_pure_camera.mp4
```

## 6. Do-as-I-Do Official Sharpa Retargeting

```bash
scripts/run_do_as_i_do_official_retarget.sh \
  --raw-dir <complete_do_as_i_do_raw_dir> \
  --task <task_name> \
  --output-root-dir experiments/<run_name>/intermediates/retargeting/do_as_i_do/<cell>/retargeting_outputs \
  --cuda-visible-devices <gpu> \
  --egl-device-id 0 \
  --headless \
  --no-wait
```

Main upstream entrypoint:

```text
third_party/do-as-i-do/retargeting/launch.py
```

Do not use direct wrapper calls to `decompose_mesh`, `generate_scene`,
`solve_ik`, and `optimize_physics` as the main path. The output path must not
already exist; the wrapper stages a disposable clean pinned checkout, runs the
unchanged native `launch.py`, and atomically publishes its output.

Outputs:

```text
experiments/<run>/intermediates/retargeting/do_as_i_do/<cell>/retargeting_outputs/sharpa/<hand_type>/<task>/0/scene.xml
experiments/<run>/intermediates/retargeting/do_as_i_do/<cell>/retargeting_outputs/sharpa/<hand_type>/<task>/0/trajectory_mjwp.npz
experiments/<run>/intermediates/retargeting/do_as_i_do/<cell>/retargeting_outputs/sharpa/<hand_type>/<task>/0/visualization_mjwp.mp4
experiments/<run>/logs/do_as_i_do_official_retarget.log
```

If the official entrypoint does not emit mp4, the wrapper renders from:

```text
scene.xml + trajectory_mjwp.npz
```

using `scripts/diagnostics/render_mujoco_trajectory.py`.

## 7. SPIDER / XHand Retargeting

```bash
scripts/run_spider_retarget.sh \
  --run-name <run_name> \
  --trajectory-6dof do_as_i_do \
  --hand-source estimated \
  --task <task_name> \
  --hand-type bimanual \
  --robot-type xhand \
  --max-sim-steps -1 \
  --num-samples 1024 \
  --max-num-iterations 16
```

Required input assets:

```text
experiments/<run>/assets/trajectory_6dof/<pipeline>/<task>/source_trajectory_keypoints.npz
experiments/<run>/assets/trajectory_6dof/<pipeline>/<task>/object_meshes/visual.obj
```

The wrapper follows the official SPIDER path:

```text
spider/preprocess/generate_xml.py
spider/preprocess/ik_fast.py
examples/run_mjwp.py
```

Outputs:

```text
experiments/<run>/intermediates/retargeting/spider/<cell>/dataset/
experiments/<run>/intermediates/retargeting/spider/<cell>/spider_input_manifest.json
experiments/<run>/intermediates/retargeting/spider/<cell>/robot/scene.xml
experiments/<run>/intermediates/retargeting/spider/<cell>/robot/trajectory_kinematic.npz
experiments/<run>/intermediates/retargeting/spider/<cell>/robot/trajectory_mjwp.npz
experiments/<run>/intermediates/retargeting/spider/<cell>/robot/visualization_ik.mp4
experiments/<run>/intermediates/retargeting/spider/<cell>/robot/visualization_mjwp.mp4
```

Fast smoke settings:

```bash
SPIDER_DEVICE=cuda:0 \
scripts/run_spider_retarget.sh \
  --run-name <run_name> \
  --trajectory-6dof do_as_i_do \
  --hand-source estimated \
  --task <task_name> \
  --hand-type bimanual \
  --robot-type xhand \
  --max-sim-steps 3 \
  --num-samples 128 \
  --max-num-iterations 4
```

## 8. Indexing and Triptych Composition

Index a cell into stable review paths:

```bash
$EGOINFINITY_PYTHON scripts/index_cell_assets.py \
  --run-name <run_name> \
  --trajectory-6dof <egoinfinity|do_as_i_do> \
  --hand-source <aoe|estimated> \
  --retargeting <egoinfinity|do_as_i_do|spider> \
  --task <task_name> \
  --hand-type <right|left|bimanual>
```

Outputs:

```text
experiments/<run>/cells/<cell>/overlay.mp4
experiments/<run>/cells/<cell>/depth.mp4
experiments/<run>/cells/<cell>/robot.mp4
experiments/<run>/cells/<cell>/manifest.json
experiments/<run>/assets/cells/<cell>/asset_manifest.json
```

Compose the synchronized video:

```bash
$EGOINFINITY_PYTHON scripts/review/compose_triptych.py \
  --overlay experiments/<run>/cells/<cell>/overlay.mp4 \
  --depth experiments/<run>/cells/<cell>/depth.mp4 \
  --robot experiments/<run>/cells/<cell>/robot.mp4 \
  --output experiments/<run>/videos/<cell>__triptych.mp4 \
  --duration 0
```

## 9. Asset Layout

```text
experiments/<run>/
  run_env.txt
  logs/
  intermediates/
    egoinfinity/clip/
    trajectory_6dof/
      egoinfinity/
      do_as_i_do/
    retargeting/
      egoinfinity/<cell>/
      do_as_i_do/<cell>/
      spider/<cell>/
  assets/
    trajectory_6dof/
      egoinfinity/<task>/
      do_as_i_do/<task>/
    cells/<cell>/
  cells/<cell>/
    overlay.mp4
    depth.mp4
    robot.mp4
    manifest.json
  videos/
    <cell>__triptych.mp4
  reuse/
    reuse_manifest.json
```
