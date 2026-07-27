#!/usr/bin/env bash
# Copy to local_env.sh, edit machine-local paths, then: source local_env.sh

export AOE_RETARGET_LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AOE_DATA_ROOT=/path/to/open-aoe/extracted/poc_deliver

export EGOINFINITY_PYTHON=/path/to/egoinfinity/bin/python
export RETARGETING_PYTHON=/path/to/dai-retarget/bin/python
export SPIDER_PYTHON=/path/to/spider/bin/python
export SAM3_PYTHON=/path/to/sam3/bin/python
export SAM3D_PYTHON=/path/to/sam3d/bin/python

export EGOINFINITY_ROOT="$AOE_RETARGET_LAB_ROOT/third_party/EgoInfinity"
export SAM3_REPO="$AOE_RETARGET_LAB_ROOT/third_party/sam3"
export SAM3D_REPO="$AOE_RETARGET_LAB_ROOT/third_party/sam-3d-objects"
export SAM3D_CHECKPOINT_DIR="$SAM3D_REPO/checkpoints/hf"
export FASTSAM3D_DIR="$AOE_RETARGET_LAB_ROOT/third_party/Fast-SAM3D"
export FASTSAM3D_CHECKPOINT_DIR="$FASTSAM3D_DIR/checkpoints/hf"
# Optional isolated Python package root for a machine-specific Open3D binary
# compatibility wheel. It must contain open3d/__init__.py.
# export SAM3D_OPEN3D_COMPAT_ROOT=/path/to/open3d-compat-target
# Optional but recommended for a one-time strict integrity audit. Use
# sha256sum format with paths relative to SAM3D_CHECKPOINT_DIR.
# export SAM3D_CHECKPOINT_SHA256_MANIFEST="/path/to/sam3d-checkpoints.sha256"
# Fast-SAM3D uses the same six official checkpoint files, so the same
# relative-name manifest can normally validate both directories.
# export FASTSAM3D_CHECKPOINT_SHA256_MANIFEST="/path/to/sam3d-checkpoints.sha256"
export DINOV2_LOCAL_REPO="$AOE_RETARGET_LAB_ROOT/third_party/dinov2"
export GEOCALIB_DIR="$AOE_RETARGET_LAB_ROOT/third_party/GeoCalib"

export HF_HOME="$HOME/.cache/huggingface"
export TORCH_HOME="$HOME/.cache/torch"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

# Select GPUs per machine. These are examples, not admission thresholds.
export MAIN_CUDA=0
export SAM3_WORKER_CUDA=0
export SAM3D_WORKER_CUDA=0
export DAI_CUDA_VISIBLE_DEVICES=0
export SPIDER_CUDA_VISIBLE_DEVICES=0
export SPIDER_DEVICE=cuda:0
