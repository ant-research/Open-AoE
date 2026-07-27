# 安装文档

语言： [English](INSTALL.md) | **中文**

导航： [README](README.zh-CN.md) | [API](API.zh-CN.md) | [原理](PRINCIPLES.zh-CN.md)

本文档面向 public 用户，说明如何 clone AoE Retarget Lab、接入自己的
Open-AoE 数据、安装第三方包并运行 demo。

repo 会保留 `third_party/` 下的源码快照，但不会重新分发数据集、模型权重、
Hugging Face cache、MANO 文件、conda 环境或生成结果。请按照 Open-AoE 和各
上游项目的 license 自行准备这些外部资产。

## 1. 系统要求

推荐环境：

```text
Linux x86_64
NVIDIA GPU + CUDA 12.x
conda 或 mamba
git
ffmpeg
gcc/g++, cmake, ninja, pkg-config
支持 EGL headless MuJoCo 渲染的 NVIDIA driver
```

不同上游包的 CUDA、MuJoCo、Warp、JAX、PyTorch 和 Python 版本约束会冲突，
建议拆成多个环境：

| 环境 | Python | 用途 |
| --- | --- | --- |
| `aoe-base` | 3.10+ | repo 轻量工具 |
| `egoinfinity` | 3.10 | EgoInfinity 和 G1 retargeting |
| `sam3` | 以上游为准 | SAM3/SAM3.1 segmentation worker |
| `sam3d` | 以上游为准 | SAM3D Objects、Fast-SAM3D、MoGe |
| `hawor` | 以上游为准 | HaWoR 和 projection 诊断 |
| `dai-retarget` | 3.12 | Do-as-I-Do Sharpa / MuJoCo-Warp |
| `spider` | 3.12 | SPIDER / MJWP / XHand |

只安装你需要运行的管线对应环境即可。

## 2. Clone 本 repo

```bash
git clone <AOE_RETARGET_LAB_GIT_URL> aoe-retarget-lab
cd aoe-retarget-lab

conda create -y -n aoe-base python=3.10 pip
conda activate aoe-base
pip install -e .

export AOE_RETARGET_LAB_ROOT="$PWD"
```

## 3. 配置本机路径

创建一个本机专用的 `local_env.sh`，不要提交它。

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

# 脚本 API 使用的 Python 路径。保持为本机 local 配置，不要提交。
export EGOINFINITY_PYTHON=/path/to/egoinfinity_env/bin/python
export RETARGETING_PYTHON=/path/to/dai_or_shared_retargeting_env/bin/python
export SPIDER_PYTHON=/path/to/spider_or_shared_retargeting_env/bin/python

# front-camera robot 重渲染的可选默认参数。
export ROBOT_RENDER_WIDTH=1280
export ROBOT_RENDER_HEIGHT=720
export ROBOT_RENDER_FPS=25
export ROBOT_RENDER_STRIDE=8
```

cache 准备好之后，可以切到离线模式：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 4. 外部模型资产

Do-as-I-Do 和 EgoInfinity 使用不同 SAM 版本：

```text
Do-as-I-Do reconstruction Stage 1 -> facebook/sam3, sam3.pt
EgoInfinity object segmentation     -> facebook/sam3.1, sam3.1_multiplex.pt
```

不要把 Do-as-I-Do Stage 1 静默替换成 SAM3.1。检查 cache：

```bash
find -L "$HF_HOME/hub/models--facebook--sam3" -name sam3.pt -print
find -L "$HF_HOME/hub/models--facebook--sam3.1" -name sam3.1_multiplex.pt -print
```

其他常见必需资产：

```text
EgoInfinity retarget checkpoints，例如 retarget/ckpts/g1.pt
MANO_LEFT.pkl / MANO_RIGHT.pkl
MANO_LEFT.npz / MANO_RIGHT.npz
MoGe weights
Fast-SAM3D / SAM3D Objects configs and checkpoints
TAPIR / BootsTAPIR checkpoint
HaWoR and Metric3D weights
```

这些文件都应放在 Git repo 外部。

TAPIR 是 Fast-SAM3D 的可选旋转速度先验，但它的运行入口需要 pinned
`requirements_inference.txt` 中的完整依赖，而不只是能够 import `tapir_model`。
在 `SAM3D_PYTHON` 对应环境中安装并验证：

```bash
"$SAM3D_PYTHON" -m pip install -r \
  third_party/do-as-i-do/reconstruction/modules/tapnet/requirements_inference.txt
"$SAM3D_PYTHON" -c \
  "import chex, jax, jaxline, haiku, tree, einshape; from tapnet.utils import transforms"
```

若 TAPIR 不可用，wrapper 会写入 `tapir_status.json`，并明确在没有自适应旋转
速度先验的情况下继续；该 fallback 不能记为 TAPIR 成功。

## 5. EgoInfinity 环境

```bash
conda env create -f third_party/EgoInfinity/retarget/environment.yml
conda activate egoinfinity
pip install -e "$AOE_RETARGET_LAB_ROOT"

export EGOINFINITY_PYTHON=$(which python)
```

验证：

```bash
python - <<'PY'
import mujoco, torch, trimesh
print("mujoco", mujoco.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("trimesh", trimesh.__version__)
PY
```

## 6. SAM3 和 SAM3D worker

按 SAM3 和 SAM3D Objects 上游说明安装，然后导出 Python 和源码/模型路径：

```bash
export SAM3_PYTHON=/path/to/sam3_env/bin/python
export SAM3_REPO=/path/to/sam3_source

export SAM3D_PYTHON=/path/to/sam3d_env/bin/python
export SAM3D_REPO=/path/to/sam-3d-objects
export SAM3D_CHECKPOINT_DIR="$SAM3D_REPO/checkpoints/hf"
# 可选的严格完整性审计（sha256sum 格式，文件名相对 checkpoint 目录）：
export SAM3D_CHECKPOINT_SHA256_MANIFEST=/path/to/sam3d-checkpoints.sha256
export FASTSAM3D_DIR=/path/to/Fast-SAM3D
export FASTSAM3D_CHECKPOINT_DIR="$FASTSAM3D_DIR/checkpoints/hf"
# 两个目录使用同一套已验证官方文件时，可复用同一份相对文件名 manifest：
export FASTSAM3D_CHECKPOINT_SHA256_MANIFEST=/path/to/sam3d-checkpoints.sha256

# 可选：经验证的 Open3D 二进制兼容 wheel 独立安装根目录。不要指向其他环境的整套
# site-packages。
# export SAM3D_OPEN3D_COMPAT_ROOT=/path/to/open3d-compat-target

export DINOV2_LOCAL_REPO=/path/to/facebookresearch_dinov2_main
export DINOV2_REPO_DIR=$DINOV2_LOCAL_REPO
```

按照上游命令下载受权限控制的 `facebook/sam-3d-objects` 完整 bundle，并保持
其 `checkpoints/hf` 目录结构不变。配置审计会检查当前 pinned bundle 的全部
generator/decoder YAML 与 checkpoint 大小；设置
`SAM3D_CHECKPOINT_SHA256_MANIFEST` 后还会验证每个必需文件的 SHA256，从而发现
文件大小不变但内容损坏的情况。审计还会独立检查 Fast-SAM3D 实际可见的六个模型
checkpoint；SAM3D Objects 目录完整，不能替代 Fast-SAM3D 目录自身的完整绑定。

设置 `SAM3D_OPEN3D_COMPAT_ROOT` 后，两个 pristine SAM3D launcher 都会在导入
Open3D 前临时把该目录放到模块搜索路径最前面。目录必须包含
`open3d/__init__.py`；配置审计和 launcher 对不完整目录均 fail-closed。这样机器特定
二进制 wheel 不会写入后端 checkout，也不会覆盖主 SAM3D 环境。

验证：

```bash
$SAM3_PYTHON -c "import torch; print('sam3 cuda', torch.cuda.is_available())"
$SAM3D_PYTHON -c "import torch; print('sam3d cuda', torch.cuda.is_available())"
```

全部环境安装完成后，先运行只读配置审计：

```bash
source local_env.sh
python3 scripts/check_third_party_config.py \
  --json third_party_config_report.json
```

如果可移植部署要求所有 Python 环境都位于同一个运行时目录，还应校验解释器解析
symlink 后的最终路径：

```bash
python3 scripts/check_third_party_config.py \
  --required-executable-root /path/to/runtime-root \
  --json third_party_config_report.json
```

也可以设置等价的环境变量 `AOE_REQUIRED_EXECUTABLE_ROOT`。该检查默认不启用；当部署
承诺环境集中存放时再开启。只要任一已配置 Python 的最终 `realpath` 逃逸出指定根目录，
审计就会 fail-closed。

Fresh Do-as-I-Do reconstruction 还必须用 GeoCalib 估计相机坐标系中的重力。
审计过的实验固定使用以下上游提交和 v1.0 pinhole 权重：

```bash
git clone https://github.com/cvg/GeoCalib.git /path/to/GeoCalib
git -C /path/to/GeoCalib checkout 97b8968e7798a66bf04fcf791fb535624241bda7
export GEOCALIB_DIR=/path/to/GeoCalib
export TORCH_HOME=/path/to/torch_cache
# 将 geocalib-pinhole.tar 放到 $TORCH_HOME/hub/checkpoints/pinhole.tar
```

pinhole 权重的预期 SHA-256 是
`86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac`。
使用 `$SAM3D_PYTHON` 验证 `from geocalib import GeoCalib`。生产 fresh run
在重力估计失败时直接停止；`DAI_REQUIRE_GEOCALIB=0` 只允许用于明确标记为
无效的诊断 run。

如果任务中断，旧 SAM3D worker 可能继续占 GPU。长任务前检查：

```bash
pgrep -af "egoinfinity_sam3d|sam3d_worker.py|sam3_worker.py"
nvidia-smi
```

只清理确认是旧 run 遗留的 orphan 进程。

## 7. HaWoR / projection 诊断环境

按 HaWoR 和 Do-as-I-Do reconstruction 上游说明安装，然后设置：

```bash
export HAWOR_PYTHON=/path/to/hawor_env/bin/python
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

验证：

```bash
$HAWOR_PYTHON -c "import torch; print('hawor cuda', torch.cuda.is_available())"
```

Do-as-I-Do mesh projection 诊断需要带 `pytorch3d` 的环境，不要默认认为
Do-as-I-Do retargeting 环境里有它。

部分重建 OBJ 只有几何，没有 MTL 或纹理图。fresh reconstruction runner 会绑定
原始 projector 已提供的 `--object-color 0.6,0.9,0.6`，仅为诊断视频补充固定的
顶点颜色。该处理不修改顶点、面、位姿、相机内参或送入重定向的物理资产。

## 8. Do-as-I-Do Retargeting / Sharpa

```bash
conda create -y -n dai-retarget python=3.12 pip
conda activate dai-retarget

cd "$AOE_RETARGET_LAB_ROOT/third_party/do-as-i-do/retargeting"
pip install -e .
cd "$AOE_RETARGET_LAB_ROOT"

export RETARGETING_PYTHON=$(which python)
```

该包固定使用 `mujoco==3.4.0`、`warp-lang==1.10.1` 和指定 revision 的
`mujoco-warp`。验证：

```bash
python - <<'PY'
import mujoco, torch, warp
print("mujoco", mujoco.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("warp", warp.__version__)
PY
```

本 repo 的 wrapper 使用官方入口：

```text
third_party/do-as-i-do/retargeting/launch.py
```

## 9. SPIDER / MJWP

```bash
conda create -y -n spider python=3.12 pip
conda activate spider

cd "$AOE_RETARGET_LAB_ROOT/third_party/SPIDER"
pip install -e .
cd "$AOE_RETARGET_LAB_ROOT"

export SPIDER_PYTHON=$(which python)
```

SPIDER 使用比 Do-as-I-Do 更新的 MJWP stack：

```text
mujoco == 3.7.0
mujoco-warp == 3.7.0.1
warp-lang == 1.12.1
```

如果重 GPU 依赖已经装好，只需要重新指向 editable 源码：

```bash
cd "$AOE_RETARGET_LAB_ROOT/third_party/SPIDER"
pip install --no-deps -e .
```

验证：

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

`spider` 路径应指向 `third_party/SPIDER/spider/`。

当 Do-as-I-Do 和 SPIDER 共用一个兼容环境时，显式配置两个可执行文件：

```bash
export RETARGETING_PYTHON=/path/to/retargeting/bin/python
export SPIDER_PYTHON=/path/to/retargeting/bin/python
```

这个环境已经用于 Do-as-I-Do/Sharpa 渲染、SPIDER/MJWP 和 front-camera
重渲染脚本。如果将 Do-as-I-Do 和 SPIDER 拆成两个 conda 环境，请显式设置
两个变量，不要依赖当前 shell 里碰巧激活的 `python`。

## 9.1. Headless MuJoCo 渲染配置

服务器运行时，必须在 import MuJoCo 之前初始化 EGL：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
```

`scripts/diagnostics/render_mujoco_trajectory.py` 内部也会设置这些默认值，但仍建议写进
`local_env.sh`，这样上游 Do-as-I-Do 和 SPIDER renderer 也能保持一致。

用已有 `scene.xml + trajectory_mjwp.npz` 做纯 robot 渲染 smoke test：

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

renderer 会从 scene 所在目录加载 MuJoCo scene，因此
`../../../assets/robots/...` 这类相对 mesh 路径可以正确解析。渲染
1280x720 前，它也会先扩大 MuJoCo offscreen framebuffer。

## 10. 准备 Open-AoE 输入

EgoInfinity 需要：

```text
raw_video_undistorted.mp4
```

Do-as-I-Do 需要完整 reconstruction 目录，至少包含：

```text
config.json
gravity.json
obj_tracking_out/<object>/combined_visualization/layout_camera_frame_optimized.json
video_segmentation/masks/frame_*_masks/<object>/<object>.obj
*/all_hand_meshes.npz
```

如果使用 `hand_source=aoe`，需要提供 Open-AoE hand reconstruction，例如
`ego_hands_reconstruction/hands.npz`，并确认所选 adapter 能转成目标
retargeter 所需格式。

## 11. Preflight

```bash
source local_env.sh

$EGOINFINITY_PYTHON -c "import torch, mujoco; print('egoinfinity ok', torch.cuda.is_available())"
$SAM3_PYTHON -c "import torch; print('sam3 ok', torch.cuda.is_available())"
$SAM3D_PYTHON -c "import torch; print('sam3d ok', torch.cuda.is_available())"
$RETARGETING_PYTHON -c "import mujoco, warp; print('dai-retarget ok')"
$SPIDER_PYTHON -c "import spider, mujoco, mujoco_warp, warp; print('spider ok')"
```

## 12. Smoke demo

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

预期输出：

```text
experiments/smoke_v4/videos/v4_egoinfinity_full__triptych.mp4
experiments/smoke_v4/videos/v4_do_as_i_do_full__triptych.mp4
experiments/smoke_v4/reuse/reuse_manifest.json
```

## 13. Git hygiene

不要提交 Open-AoE raw data、生成的 `experiments/`、conda/venv 目录、
Hugging Face cache、模型权重、生成的 `npz` 和视频文件。
