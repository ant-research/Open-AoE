# AoE Retarget Lab

语言： [English](../README.md) | **中文**

AoE Retarget Lab 用来把 Open-AoE 第一视角 RGB 视频接入两个物体 6DoF
重建管线和三个机器人重定向/渲染后端。主要检查结果是一段同步视频：

```text
原始 RGB + 重建 mesh overlay
深度 / 输入几何
机器人 retargeting 渲染
```

所有新生成的资产都应写入 `experiments/<run_name>/`。不要把 mesh、轨迹、
视频、日志或 manifest 写回 Open-AoE 数据集目录或第三方源码目录。

## 文档入口

| 文档 | English | 中文 |
| --- | --- | --- |
| README | [../README.md](../README.md) | 本文档 |
| 安装 | [INSTALL.md](INSTALL.md) | [INSTALL.zh-CN.md](INSTALL.zh-CN.md) |
| 脚本 API | [API.md](API.md) | [API.zh-CN.md](API.zh-CN.md) |
| 原理介绍 | [PRINCIPLES.md](PRINCIPLES.md) | [PRINCIPLES.zh-CN.md](PRINCIPLES.zh-CN.md) |
| AoE 数据集筛选与调试经验 | [AOE_DATASET_PRACTICES.md](AOE_DATASET_PRACTICES.md) | [AOE_DATASET_PRACTICES.zh-CN.md](AOE_DATASET_PRACTICES.zh-CN.md) |
| 干净机器部署与复现经验 | [REPRODUCTION_PRACTICES.md](REPRODUCTION_PRACTICES.md) | [REPRODUCTION_PRACTICES.zh-CN.md](REPRODUCTION_PRACTICES.zh-CN.md) |

建议先看 [安装文档](INSTALL.zh-CN.md)，再看 [API 文档](API.zh-CN.md)。
管线拆分、尺度和可视化约定见 [原理介绍](PRINCIPLES.zh-CN.md)。
新机器部署、失败分类和复现检查顺序见
[干净机器部署与复现经验](REPRODUCTION_PRACTICES.zh-CN.md)。

## 项目结构

```text
scripts/                     可运行的 shell 和 Python 入口
scripts/maintenance/setup_third_party.sh 本地拉取第三方上游包的脚本
src/aoe_retarget_lab/        AoE 适配器和工具
configs/                     示例配置
experiments/                 本地实验输出
docs/                        中英双语文档
third_party/                 setup 后生成的本地上游 clone，已 gitignore
```

这个 public 导入包不会 vendor EgoInfinity、Do-as-I-Do、SPIDER、机器人资产、
模型权重、数据集或生成视频。clone 后运行 `scripts/maintenance/setup_third_party.sh`，
把第三方包拉到已 gitignore 的 `third_party/`。

## 管线矩阵

项目有三条轴：

```text
trajectory_6dof = egoinfinity | do_as_i_do
hand_source     = aoe | estimated
retargeting     = egoinfinity | do_as_i_do | spider
```

完整比较矩阵是：

```text
2 trajectory_6dof x 2 hand_source x 3 retargeting = 12 cells
```

cell 名称固定为：

```text
traj_<trajectory_6dof>__hand_<hand_source>__retarget_<retargeting>
```

## 快速开始

先完成环境安装、拉取第三方包，并配置外部资产路径：

```bash
bash scripts/maintenance/setup_third_party.sh
source local_env.sh
```

### AoE 数据集输入

自动选窗和人工指定窗口统一使用 `run_fresh_aoe_auto_window.py`，两种模式
最终进入完全相同的 fresh reconstruction、DAI 和 SPIDER 重定向链路。

自动模式按从长到短的候选时长执行生产 SAM3 门禁，只对第一个通过的窗口
启动一次正式 full run：

```bash
python scripts/run_fresh_aoe_auto_window.py \
  --runner scripts/run_fresh_aoe_scene_12_demos.sh \
  --experiments-root experiments \
  --source-run <source_run> --matrix-run <matrix_run> \
  --window-selection-mode auto \
  --candidate-durations-sec 3.0,2.6,2.4,2.2,2.0 \
  --manifest experiments/<source_run>_window.json -- \
  --scene <scene> --segment <segment> --annotation-id <id> \
  --object-name <object_prompt> --task <task> \
  --hand-type left --anchor-hand left --ref-source-frame <absolute_frame>
```

显式模式跳过自动选窗，直接使用人工审计过的时长和绝对参考帧。它仍会重新
执行重建和后端校验，不会静默复用旧 adapter，也不会放宽质量门禁：

```bash
python scripts/run_fresh_aoe_auto_window.py \
  --runner scripts/run_fresh_aoe_scene_12_demos.sh \
  --experiments-root experiments \
  --source-run <source_run> --matrix-run <matrix_run> \
  --window-selection-mode direct --direct-duration-sec 3.0 \
  --manifest experiments/<source_run>_window.json -- \
  --scene <scene> --segment <segment> --annotation-id <id> \
  --object-name <object_prompt> --task <task> \
  --hand-type left --anchor-hand left --ref-source-frame <absolute_frame>
```

输入证据包括 `fresh_input_manifest.json`、`raw_video_materialization.json`、
`prompt_gate.json` 和 `mask_qc_summary.json`。重定向视频位于 source/matrix
run 的 `videos/` 目录和 exact-route cell 目录。

推荐运行 full-12 主入口。它先完整跑一次 EgoInfinity 和一次 Do-as-I-Do，
再复用这两条大链路的中间结果，合成 12 个对比 cell。

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

预期输出：

```text
experiments/<source_run_name>/videos/v4_egoinfinity_full__triptych.mp4
experiments/<source_run_name>/videos/v4_do_as_i_do_full__triptych.mp4
experiments/<source_run_name>/reuse/reuse_manifest.json
experiments/<matrix_run_name>/videos/<cell>__triptych.mp4
experiments/<matrix_run_name>/reuse_12_demo_manifest.json
experiments/<matrix_run_name>/cells/
experiments/<matrix_run_name>/assets/
```
