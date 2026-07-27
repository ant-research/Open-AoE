# Installation Guide

Language: **English** | [中文](INSTALL.zh-CN.md)

Navigation: [README](../README.md) | [API](API.md) | [Principles](PRINCIPLES.md)

This guide is for public users who want to clone AoE Retarget Lab, connect
their own Open-AoE data, install third-party packages, and run the demo scripts.

The repository does not vendor third-party source trees, datasets, model
weights, Hugging Face caches, MANO assets, conda environments, or generated
experiment outputs. Obtain those assets according to the licenses of Open-AoE
and each upstream project.

## 1. System Requirements

Recommended:

```text
Linux x86_64
NVIDIA GPU with CUDA 12.x
conda or mamba
git
ffmpeg
gcc/g++, cmake, ninja, pkg-config
EGL-capable NVIDIA driver for headless MuJoCo rendering
```

Use separate environments because CUDA, MuJoCo, Warp, JAX, PyTorch, and Python
pins differ across upstream packages.

| Environment | Python | Used for |
| --- | --- | --- |
| `aoe-base` | 3.10+ | lightweight repo tools |
| `egoinfinity` | 3.10 | EgoInfinity and G1 retargeting |
| `sam3` | upstream-defined | SAM3/SAM3.1 segmentation worker |
| `sam3d` | upstream-defined | SAM3D Objects, Fast-SAM3D, MoGe |
| `hawor` | upstream-defined | HaWoR and projection diagnostics |
| `dai-retarget` | 3.12 | Do-as-I-Do Sharpa / MuJoCo-Warp |
| `spider` | 3.12 | SPIDER / MJWP / XHand |

Install only the environments needed for the pipelines you plan to run.

## 2. Clone This Repo

```bash
git clone <AOE_RETARGET_LAB_GIT_URL> aoe-retarget-lab
cd aoe-retarget-lab

conda create -y -n aoe-base python=3.10 pip
conda activate aoe-base
pip install -e .

export AOE_RETARGET_LAB_ROOT="$PWD"
```

## 3. Configure Local Paths

Copy the supplied template and create a machine-local `local_env.sh`; do not
commit the edited file.

```bash
cp local_env.example.sh local_env.sh
```

```bash
export AOE_RETARGET_LAB_ROOT=/path/to/aoe-retarget-lab
export AOE_DATA_ROOT=/path/to/open-aoe
export AOE_MODEL_ROOT=/path/to/aoe-retarget-models

export HF_HOME=/path/to/huggingface_cache
export TORCH_HOME=/path/to/torch_cache
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

# Python executables used by the script API. Keep these machine-local.
export EGOINFINITY_PYTHON=/path/to/egoinfinity_env/bin/python
export RETARGETING_PYTHON=/path/to/dai_or_shared_retargeting_env/bin/python
export SPIDER_PYTHON=/path/to/spider_or_shared_retargeting_env/bin/python

# Optional defaults for front-camera robot re-rendering.
export ROBOT_RENDER_WIDTH=1280
export ROBOT_RENDER_HEIGHT=720
export ROBOT_RENDER_FPS=25
export ROBOT_RENDER_STRIDE=8
```

Production entrypoints use only the configured local paths. Unset Python
variables fall back only to `python3`; the configuration
preflight below therefore catches missing environments before a long run.

After caches are populated, offline runs can use:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Run the read-only configuration preflight after installing all environments:

```bash
source local_env.sh
python3 scripts/check_third_party_config.py \
  --json third_party_config_report.json
```

`status=ok` means the configured executables and source/data directories are
resolvable and the pinned SAM3D Objects checkpoint bundle is complete. It does
not claim that every other licensed weight is available or that a GPU has
sufficient memory; retain the report with each machine's first experiment.

For a portable deployment whose Python environments must remain under one
runtime directory, also verify the final executable targets (after resolving
symlinks):

```bash
python3 scripts/check_third_party_config.py \
  --required-executable-root /path/to/runtime-root \
  --json third_party_config_report.json
```

The equivalent environment variable is `AOE_REQUIRED_EXECUTABLE_ROOT`. The
option is deliberately optional: use it when the deployment promises colocated
environments. A configured Python symlink that escapes the required root makes
the audit fail closed.


### Gravity Fallback

`gravity.json` stores the world-up direction in the Do-as-I-Do camera frame
(`x-right, y-down, z-fwd`), not the MuJoCo gravity vector. GeoCalib output is
preferred. When GeoCalib is unavailable and the egocentric camera is assumed
upright, use `[0.0, -1.0, 0.0]` with `semantics=world_up_direction_in_camera_frame`.
The retargeting stage rotates that vector to MuJoCo `+Z`; MuJoCo scene gravity
should remain `0 0 -9.81`.

After a reconstruction or before reusing historical runs, check:

```bash
$RETARGETING_PYTHON scripts/diagnostics/audit_gravity_frames.py \
  --run-root experiments/<run_or_matrix> \
  --source-run-root experiments/<source_run> \
  --task <task_name> \
  --hand-type left
```

## 4. External Assets

Do-as-I-Do and EgoInfinity use different SAM versions:

```text
Do-as-I-Do reconstruction Stage 1 -> facebook/sam3, sam3.pt
EgoInfinity object segmentation     -> facebook/sam3.1, sam3.1_multiplex.pt
```

Do not substitute SAM3.1 for Do-as-I-Do Stage 1. Check local caches with:

```bash
find -L "$HF_HOME/hub/models--facebook--sam3" -name sam3.pt -print
find -L "$HF_HOME/hub/models--facebook--sam3.1" -name sam3.1_multiplex.pt -print
```

Other required assets may include:

```text
EgoInfinity retarget checkpoints, e.g. retarget/ckpts/g1.pt
MANO_LEFT.pkl / MANO_RIGHT.pkl
MANO_LEFT.npz / MANO_RIGHT.npz
MoGe weights
Fast-SAM3D / SAM3D Objects configs and checkpoints
TAPIR / BootsTAPIR checkpoint
HaWoR and Metric3D weights
```

Keep all of these outside the Git repo.

TAPIR is an optional Fast-SAM3D rotation-velocity prior, but its Python entry
imports the complete pinned inference dependency set, not only `tapir_model`.
Install that set into the environment used by `SAM3D_PYTHON`:

```bash
"$SAM3D_PYTHON" -m pip install -r \
  third_party/do-as-i-do/reconstruction/modules/tapnet/requirements_inference.txt
"$SAM3D_PYTHON" -c \
  "import chex, jax, jaxline, haiku, tree, einshape; from tapnet.utils import transforms"
```

If TAPIR is unavailable, the reconstruction wrapper records
`tapir_status.json` and continues without the adaptive rotation-velocity
prior. This explicit fallback is not evidence that TAPIR succeeded.

## 5. EgoInfinity Environment

```bash
conda env create -f third_party/EgoInfinity/retarget/environment.yml
conda activate egoinfinity
pip install -e "$AOE_RETARGET_LAB_ROOT"

export EGOINFINITY_PYTHON=$(which python)
```

Verify:

```bash
python - <<'PY'
import mujoco, torch, trimesh
print("mujoco", mujoco.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("trimesh", trimesh.__version__)
PY
```

## 6. SAM3 and SAM3D Workers

Install SAM3 and SAM3D Objects according to their upstream instructions, then
export the Python binaries and source/model roots:

```bash
export SAM3_PYTHON=/path/to/sam3_env/bin/python
export SAM3_REPO=/path/to/sam3_source

export SAM3D_PYTHON=/path/to/sam3d_env/bin/python
export SAM3D_REPO=/path/to/sam-3d-objects
export SAM3D_CHECKPOINT_DIR="$SAM3D_REPO/checkpoints/hf"
# Optional strict integrity audit (sha256sum format, relative filenames):
export SAM3D_CHECKPOINT_SHA256_MANIFEST=/path/to/sam3d-checkpoints.sha256
export FASTSAM3D_DIR=/path/to/Fast-SAM3D
export FASTSAM3D_CHECKPOINT_DIR="$FASTSAM3D_DIR/checkpoints/hf"
# The official checkpoint filenames are shared, so one manifest may validate
# both directories when they contain the same verified bundle.
export FASTSAM3D_CHECKPOINT_SHA256_MANIFEST=/path/to/sam3d-checkpoints.sha256

# Optional isolated package root for a validated Open3D binary compatibility
# wheel. Do not point this at an entire unrelated site-packages directory.
# export SAM3D_OPEN3D_COMPAT_ROOT=/path/to/open3d-compat-target

export DINOV2_LOCAL_REPO=/path/to/facebookresearch_dinov2_main
export DINOV2_REPO_DIR=$DINOV2_LOCAL_REPO
```

Download the gated `facebook/sam-3d-objects` bundle with the upstream command
and keep its `checkpoints/hf` directory intact. The configuration audit checks
all generator/decoder YAML and checkpoint sizes for the pinned bundle. When
`SAM3D_CHECKPOINT_SHA256_MANIFEST` is set, it also hashes every required file
and detects same-size corruption. The audit independently checks the six model
checkpoint files visible to Fast-SAM3D; a complete SAM3D Objects directory does
not make a missing or truncated Fast-SAM3D binding acceptable.

When `SAM3D_OPEN3D_COMPAT_ROOT` is set, both pristine SAM3D launchers prepend
that directory immediately before importing Open3D. The directory must contain
`open3d/__init__.py`; the configuration audit and launchers fail closed when it
is incomplete. This keeps a machine-specific binary wheel outside both the
backend checkout and the main environment.

Verify:

```bash
$SAM3_PYTHON -c "import torch; print('sam3 cuda', torch.cuda.is_available())"
$SAM3D_PYTHON -c "import torch; print('sam3d cuda', torch.cuda.is_available())"
```

Fresh Do-as-I-Do reconstruction also requires GeoCalib for camera-frame gravity.
Use the pinned upstream checkout and v1.0 pinhole weight used by the audited
runs:

```bash
git clone https://github.com/cvg/GeoCalib.git /path/to/GeoCalib
git -C /path/to/GeoCalib checkout 97b8968e7798a66bf04fcf791fb535624241bda7
export GEOCALIB_DIR=/path/to/GeoCalib
export TORCH_HOME=/path/to/torch_cache
# Store geocalib-pinhole.tar as $TORCH_HOME/hub/checkpoints/pinhole.tar
```

The expected pinhole weight SHA-256 is
`86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac`.
Verify `from geocalib import GeoCalib` in `$SAM3D_PYTHON`. Production fresh
runs stop if gravity estimation fails; `DAI_REQUIRE_GEOCALIB=0` is only for
explicitly invalid diagnostic runs.

If a run is interrupted, a stale SAM3D worker may keep GPU memory. Check before
starting a long run:

```bash
pgrep -af "egoinfinity_sam3d|sam3d_worker.py|sam3_worker.py"
nvidia-smi
```

Only kill a process after confirming it is an orphan from an old run.

## 7. HaWoR / Projection Diagnostics

Follow the upstream HaWoR and Do-as-I-Do reconstruction setup, then set:

```bash
export HAWOR_PYTHON=/path/to/hawor_env/bin/python
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

Verify:

```bash
$HAWOR_PYTHON -c "import torch; print('hawor cuda', torch.cuda.is_available())"
```

Use an environment with `pytorch3d` for Do-as-I-Do mesh projection diagnostics;
the Do-as-I-Do retargeting environment does not necessarily provide it.

Some reconstructed OBJ files intentionally contain geometry only and have no
MTL or texture atlas. The fresh reconstruction runner binds the pristine
projector's existing `--object-color 0.6,0.9,0.6` option so these meshes receive
a deterministic vertex color in diagnostic renders. This changes only the
projection video's appearance; vertices, faces, poses, camera intrinsics, and
the physical retarget asset remain untouched.

## 8. Do-as-I-Do Retargeting / Sharpa

```bash
conda create -y -n dai-retarget python=3.12 pip
conda activate dai-retarget

cd "$AOE_RETARGET_LAB_ROOT/third_party/do-as-i-do/retargeting"
pip install -e .
cd "$AOE_RETARGET_LAB_ROOT"

export RETARGETING_PYTHON=$(which python)
```

This package pins `mujoco==3.4.0`, `warp-lang==1.10.1`, and a pinned
`mujoco-warp` revision. Verify:

```bash
python - <<'PY'
import mujoco, torch, warp
print("mujoco", mujoco.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("warp", warp.__version__)
PY
```

The wrapper uses the official entrypoint:

```text
third_party/do-as-i-do/retargeting/launch.py
```

MJWP troubleshooting:

- If logs stop after `Warmup: set initial hand to open`, enable the built-in
  timing logs and check `MJWP setup_env` stages.
- `num_samples * num_perturb_samples` is the number of parallel MJWarp worlds.
  On the NAS cooking-wine scene, `1024 * 4 = 4096` worlds stalled inside
  `mjwarp.put_data`; `1024 * 1` completed normally.
- The analytical warmup distance solve is capped by
  `DAI_MJWP_WARMUP_PAIR_MAX_POINTS` (default `1000`) to avoid constructing huge
  hand/object pairwise matrices.

## 9. SPIDER / MJWP

```bash
conda create -y -n spider python=3.12 pip
conda activate spider

cd "$AOE_RETARGET_LAB_ROOT/third_party/SPIDER"
pip install -e .
cd "$AOE_RETARGET_LAB_ROOT"

export SPIDER_PYTHON=$(which python)
```

SPIDER uses a newer MJWP stack than Do-as-I-Do:

```text
mujoco == 3.7.0
mujoco-warp == 3.7.0.1
warp-lang == 1.12.1
```

If heavy GPU dependencies are already installed, re-apply only the editable
source link:

```bash
cd "$AOE_RETARGET_LAB_ROOT/third_party/SPIDER"
pip install --no-deps -e .
```

Verify:

```bash
python - <<'PY'
import pathlib, spider, mujoco, mujoco_warp, warp, mink
print("spider", pathlib.Path(spider.__file__).resolve())
print("mujoco", mujoco.__version__)
print("mujoco_warp", getattr(mujoco_warp, "__version__", "unknown"))
print("warp", warp.__version__)
print("mink", getattr(mink, "__version__", "unknown"))
PY
```

The `spider` path should resolve inside `third_party/SPIDER/spider/`.

When Do-as-I-Do and SPIDER share one compatible environment, configure both
executables explicitly:

```bash
export RETARGETING_PYTHON=/path/to/retargeting/bin/python
export SPIDER_PYTHON=/path/to/retargeting/bin/python
```

That environment has been used successfully for Do-as-I-Do/Sharpa rendering,
SPIDER/MJWP, and the front-camera re-render scripts. If you split Do-as-I-Do
and SPIDER into separate conda envs, keep both variables explicit instead of
relying on whichever `python` is active.

## 9.1. Headless MuJoCo Rendering Configuration

Server runs must initialize EGL before importing MuJoCo:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
```

`scripts/diagnostics/render_mujoco_trajectory.py` also sets these defaults internally, but
they should still be present in `local_env.sh` so upstream Do-as-I-Do and
SPIDER renderers behave consistently.

Smoke-test pure robot rendering from an existing `scene.xml +
trajectory_mjwp.npz`:

```bash
$RETARGETING_PYTHON scripts/diagnostics/render_mujoco_trajectory.py \
  --scene experiments/<run>/intermediates/retargeting/do_as_i_do/<cell>/retargeting_outputs/sharpa/<hand_type>/<task>/0/scene.xml \
  --trajectory experiments/<run>/intermediates/retargeting/do_as_i_do/<cell>/retargeting_outputs/sharpa/<hand_type>/<task>/0/trajectory_mjwp.npz \
  --output experiments/smoke/aoe_robot_render_smoke.mp4 \
  --width 320 \
  --height 180 \
  --fps 10 \
  --stride 64 \
  --camera front
```

The renderer loads MuJoCo scenes from the scene directory so relative mesh
paths such as `../../../assets/robots/...` resolve correctly. It also raises
the MuJoCo offscreen framebuffer size before creating a 1280x720 renderer.

## 10. Prepare Open-AoE Inputs

For EgoInfinity:

```text
raw_video_undistorted.mp4
```

For Do-as-I-Do, provide a complete reconstruction directory containing:

```text
config.json
gravity.json
obj_tracking_out/<object>/combined_visualization/layout_camera_frame_optimized.json
video_segmentation/masks/frame_*_masks/<object>/<object>.obj
*/all_hand_meshes.npz
```

For `hand_source=aoe`, provide Open-AoE hand reconstruction such as
`ego_hands_reconstruction/hands.npz` and make sure the selected adapter can
convert it into the target retargeter format.

## 11. Preflight

```bash
source local_env.sh

$EGOINFINITY_PYTHON -c "import torch, mujoco; print('egoinfinity ok', torch.cuda.is_available())"
$SAM3_PYTHON -c "import torch; print('sam3 ok', torch.cuda.is_available())"
$SAM3D_PYTHON -c "import torch; print('sam3d ok', torch.cuda.is_available())"
$RETARGETING_PYTHON -c "import mujoco, warp; print('dai-retarget ok')"
$SPIDER_PYTHON -c "import spider, mujoco, mujoco_warp, warp; print('spider ok')"
```

## 12. Smoke Demo

```bash
source local_env.sh

V4_RUN_NAME=smoke_v4 \
V4_EGO_VIDEO=/path/to/raw_video_undistorted.mp4 \
V4_EGO_CLIP_ID=my_clip \
V4_EGO_OBJECTS="bottle of vinegar" \
V4_EGO_START=0.0 \
V4_EGO_END=3.0 \
V4_DAI_RAW_DIR=/path/to/do_as_i_do_raw_dir \
V4_DAI_CLIP_DIR=/path/to/do_as_i_do_clip_dir \
V4_TASK=vinegar_demo \
V4_HAND_TYPE=bimanual \
MAIN_CUDA=0 \
SAM3_WORKER_CUDA=1 \
SAM3D_WORKER_CUDA=2 \
scripts/run_v4_two_full_pipelines.sh
```

Expected:

```text
experiments/smoke_v4/videos/v4_egoinfinity_full__triptych.mp4
experiments/smoke_v4/videos/v4_do_as_i_do_full__triptych.mp4
experiments/smoke_v4/reuse/reuse_manifest.json
```

## 13. Git Hygiene

Do not commit Open-AoE raw data, generated `experiments/`, conda or venv
directories, Hugging Face caches, model weights, generated `npz` files, or
videos.
