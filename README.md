# Open-AoE

<h2 align="center">
  2,000 Hours of Smartphone-Collected Egocentric Manipulation Data<br>with a Complete Data-to-Model Toolchain
</h2>

<p align="center">
  <sub>by Ant Digital Technologies’ Phecda Lab</sub>
</p>

<p align="center">
  English | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.14183"><img src="https://img.shields.io/badge/arXiv-2607.14183-b31b1b.svg" alt="Open-AoE Technical Report on arXiv"></a>
  <a href="https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h"><img src="https://img.shields.io/badge/Hugging%20Face-Dataset-FFD21E.svg" alt="Open-AoE Dataset on Hugging Face"></a>
  <a href="https://www.modelscope.cn/datasets/inclusionAI/OpenAoE-2000h"><img src="https://img.shields.io/badge/ModelScope-Dataset-624AFF.svg" alt="Open-AoE Dataset on ModelScope"></a>
  <img src="https://img.shields.io/static/v1?label=Capture%20App&amp;message=%E5%85%B7%E8%BA%AB%E6%99%BA%E8%83%BD%E6%95%B0%E9%87%87%E5%8A%A9%E6%89%8B&amp;color=2ea44f" alt="Search for 具身智能数采助手 in major app stores">
  <a href="https://dtcoder.antdigital.com/home"><img src="https://img.shields.io/badge/Developer%20Tool-AiWork-1677FF.svg" alt="Develop Open-AoE experiments with AiWork"></a>
</p>

> [!TIP]
> New to Open-AoE? Start with the [data specification](open-aoe-2000h/README.md), render one segment with [AoE-Visualization](aoe-visualization/README.md), then choose a model recipe from [AoE-Training-Ready](aoe-training-ready/README.md). Use [AiWork](https://dtcoder.antdigital.com/home) to start your experiment.

<p align="center">
  <img src="docs/fig1-open-aoe-overview.png" width="100%" alt="Open-AoE dataset, processing pipeline, and open-source toolchain overview">
</p>

Open-AoE is a large-scale, real-world egocentric manipulation dataset collected entirely with consumer smartphones. It provides approximately **2,000 hours** of first-person video with synchronized hand motion, camera motion, and bilingual atomic-action annotations. The repository connects those signals to visualization, human-to-robot retargeting, robot replay, and model-specific training recipes.

## Get the Release

| Resource | Access | What you get |
|---|---|---|
| **Technical report** | [Read on arXiv](https://arxiv.org/abs/2607.14183) | Dataset design, processing, analysis, toolchain, and experiments |
| **Open-AoE-2000H** | [Hugging Face](https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h) · [ModelScope](https://www.modelscope.cn/datasets/inclusionAI/OpenAoE-2000h) | Dataset files and distribution information |
| **Data specification** | [Field-level documentation](open-aoe-2000h/README.md) | Directory layout, schemas, coordinate systems, and validation notes |
| **Capture app** | Search for **“具身智能数采助手”** in major app stores | Smartphone data collection client |

The technical report is available on [arXiv](https://arxiv.org/abs/2607.14183).

## Choose Your Starting Point

| I want to… | Start here | Result |
|---|---|---|
| **Understand what is in one segment** | [Dataset specification](open-aoe-2000h/README.md) | Learn the video, calibration, MANO, camera-trajectory, and annotation fields |
| **See the data before writing code** | [AoE-Visualization](aoe-visualization/README.md) | Render an end-to-end review video with hands, trajectories, actions, and a 3D view |
| **Train a VLA policy** | [LeRobot recipes](aoe-training-ready/lerobot/README_OPEN_AOE.md) · [GR00T N1.7](aoe-training-ready/gr00t_n1d7/README.md) · [H-RDT](aoe-training-ready/H-RDT/README.md) · [VITRA](aoe-training-ready/vitra/README.md) | Convert Open-AoE signals to model-specific state/action semantics and launch training |
| **Train a world or video-action model** | [Training recipe index](aoe-training-ready/README.md) | Use DreamZero, LingBot-VA, Ctrl-World, iVideoGPT, GenieRedux, LAOM, AdaWorld, or DreamDojo integrations |
| **Reconstruct scenes or retarget human motion** | [AoE-Reconstruct-Retarget](aoe-reconstruct-retarget/README.md) | Reconstruct interaction assets and produce robot trajectories, simulation renders, or robotized video |
| **Add a model or robot integration** | [Contributing guide](CONTRIBUTING.md) | Follow the repository structure and dependency rules |

## Explore Open-AoE with AiWork

[AiWork](https://dtcoder.antdigital.com/home) is a desktop AI workspace where you can select a project, choose a Skill or Expert, and describe a task in a conversation. For Open-AoE, start with the data you need or the robot/model you want to work with; review the returned files, charts, videos, and commands as the work progresses.

![AiWork desktop task window with the Human-to-Robot Data Conversion Expert selected](docs/assets/aiwork/aiwork-conversation-window.png)

*AiWork's desktop interface: projects and “Experts · Skills” on the left, with a task prompt and selected Expert in the conversation area. Compose a request, select the Expert, and start the conversation.*

### Choose a guide

| Goal | Guide | What to expect |
| --- | --- | --- |
| Find suitable OpenAoE samples, inspect quality, and download a subset | **Skill best practices** — [English](docs/aiwork-skill-best-practices.md) · [中文](docs/aiwork-skill-best-practices.zh-CN.md) | Six conversational steps with sample lists, quality reports, charts, provenance, and review-video examples |
| Convert demonstrations for a robot or prepare a model experiment | **Expert best practices** — [English](docs/aiwork-expert-best-practices.md) · [中文](docs/aiwork-expert-best-practices.zh-CN.md) | Two Expert workflows, capability illustrations, conversion outputs, QA plots, and training curves |

The **Human-to-Robot Data Conversion Expert** prepares robot-specific trajectories and data. The **Embodied Model Training Expert** checks compatibility and generates commands; you run training and evaluation on your compute and return the results for analysis.

Find `openaoe数据助手` in the Skill Center Marketplace, or `人类数据转换专家` and `具身模型训练专家` in the Expert Community. For manual upload, download the [AiWork installation packages](packages/aiwork/README.md).

## Quick Start: Inspect One Segment

### 1. Clone the repository

```bash
git clone https://github.com/ant-research/Open-AoE.git
cd Open-AoE
```

### 2. Download the data

Choose either [Hugging Face](https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h) or [ModelScope](https://www.modelscope.cn/datasets/inclusionAI/OpenAoE-2000h), then locate one extracted segment directory. Its expected layout is documented in the [data specification](open-aoe-2000h/README.md).

### 3. Render the segment

Some visualization and retargeting workflows require MANO model files. Register and download `MANO_RIGHT.pkl` and `MANO_LEFT.pkl` from the [MANO website](https://mano.is.tue.mpg.de/), then install them into the repository's shared asset directory:

```bash
bash assets/mano/download_mano.sh \
  ~/Downloads/MANO_RIGHT.pkl \
  ~/Downloads/MANO_LEFT.pkl

cd aoe-visualization
pip install -r requirements.txt
python visualize.py --sample /path/to/open_aoe_segment
```

The output is `output/<segment-name>/AoE_output_vis.mp4`, a synchronized review of the ego video, reconstructed hands, wrist trajectories, atomic actions, world-frame motion, and timeline. See the [visualization guide](aoe-visualization/README.md) for EGL/OpenGL requirements and batch rendering.

## From Raw Data to Model Input

Each segment is a synchronized multimodal record rather than a standalone video:

| Signal | Main artifact | Typical use |
|---|---|---|
| Raw and undistorted RGB | `raw_video.mp4`, `raw_video_undistorted.mp4` | Visual observation, video modeling, and overlays |
| Camera metadata | `video_info.json`, `undistorted_video_info.json` | Intrinsics, distortion, device information, resolution, and frame rate |
| Camera motion | `camera_traj.npz` and transforms in `hands.npz` | Metric-scale 6-DoF trajectories and world/camera transforms |
| Hand reconstruction | `hands.npz` | Per-frame bilateral MANO pose, shape, root transform, and validity |
| Atomic actions | `ego_action_annotation.json` | Temporally aligned actions with hand, verb, object, and bilingual description |

The toolchain then maps those synchronized signals into task-specific representations:

| Stage | Component | Output |
|---|---|---|
| **Inspect** | [AoE-Visualization](aoe-visualization/README.md) | One review video per segment for visual and temporal quality checks |
| **Reconstruct / retarget** | [AoE-Reconstruct-Retarget](aoe-reconstruct-retarget/README.md) | Reconstructed assets, robot trajectories, simulation validation, and robotized video |
| **Convert / train** | [AoE-Training-Ready](aoe-training-ready/README.md) | Model-specific datasets, actions, patches, launchers, and training recipes |

> [!IMPORTANT]
> Training conversion is an **action-semantics adaptation**, not just file-format conversion. Start with the [shared action specification](aoe-training-ready/ACTION_SPEC.md), then follow the README for your target model.

## Training Recipe Map

<p align="center">
  <img src="docs/fig4-training-ready.png" width="100%" alt="Open-AoE training-ready annotation spectrum for VLA, world-action, and world models">
</p>

| Target | Included recipes | Recommended entry |
|---|---|---|
| **VLA policies** | ACT, Diffusion Policy, π0.5, SmolVLA, GR00T N1.7, H-RDT, VITRA | [Training-Ready index](aoe-training-ready/README.md) |
| **World / video-action models** | DreamZero, LingBot-VA, Ctrl-World, iVideoGPT | [Training-Ready index](aoe-training-ready/README.md) |
| **Latent-action / world models** | GenieRedux, LAOM, AdaWorld, DreamDojo | [Training-Ready index](aoe-training-ready/README.md) |

Each recipe is self-contained and documents its upstream repository and verified commit, data conversion, environment variables, training command, outputs, and any required patch. Upstream projects and checkpoints are not vendored into this repository.

## Reconstruction and Retargeting Map

<p align="center">
  <img src="docs/fig3-reconstruct-retarget.png" width="100%" alt="Open-AoE reconstruction, motion retargeting, and robot-overlay routes">
</p>

| Subproject | Coverage | Main capabilities |
|---|---|---|
| [**Phantom**](aoe-reconstruct-retarget/phantom/) | Unitree G1 + Dex3 / Inspire | Arm IK, dexterous-hand retargeting, MuJoCo visualization, and robot overlay |
| [**Retarget Galbot**](aoe-reconstruct-retarget/retarget_galbot/) | Galbot / Galaxea bimanual platforms | Palm-to-TCP IK, gripper mapping, egoview synthesis, and LeRobot/Rerun export |
| [**AoE Retarget Lab**](aoe-reconstruct-retarget/retarget-lab/) | EgoInfinity/G1, Do-as-I-Do/Sharpa, SPIDER/XHand | External-method adapters, 6-DoF reconstruction routes, and a 12-cell comparison matrix |

Third-party repositories, model weights, robot assets, and generated videos are not included. Follow each subproject's setup guide to obtain its external dependencies.

## Data Processing and Quality Control

<p align="center">
  <img src="docs/fig2-data-pipline.png" width="100%" alt="Open-AoE capture, processing, reconstruction, annotation, and quality-control pipeline">
</p>

1. **On-device capture control** checks hand visibility, wearing conditions, lighting, motion quality, and device health before upload.
2. **Offline quality control and scene labeling** filters invalid or sensitive content, standardizes frame rate, slices videos, and assigns scene/task metadata.
3. **Reconstruction and annotation** estimates camera trajectories, reconstructs MANO hands, and produces atomic-action segments.
4. **Quality inspection and delivery** applies completeness, correctness, and temporal-consistency gates followed by human review.

## Repository Layout

```text
Open-AoE/
├── Open-AoE-tech-report.pdf  # Current technical report
├── open-aoe-2000h/           # Dataset format and field-level documentation
├── aoe-visualization/        # Synchronized data review and rendering
├── aoe-reconstruct-retarget/ # Reconstruction, retargeting, replay, and overlays
├── aoe-training-ready/       # Model adapters, converters, launchers, and patches
├── assets/mano/              # Shared MANO setup helper; model files are not tracked
├── docs/                     # Overview figures and AiWork best-practice guides
├── CONTRIBUTING.md
├── LEGAL.md
└── LICENSE
```

## Why Open-AoE?

- **Smartphone-first collection:** consumer devices make real-world egocentric capture accessible and scalable.
- **Manipulation-aligned signals:** calibrated RGB is synchronized with camera motion, MANO hand reconstruction, validity masks, and atomic actions.
- **Complete data-to-model path:** the repository covers inspection, retargeting, representation conversion, and model-specific training integration.
- **Modular and extensible:** components can be used independently, and new robot embodiments or model recipes can be added as self-contained integrations.

## Citation

The technical report is available on [arXiv](https://arxiv.org/abs/2607.14183).

## Contact

- **Data processing pipeline applications:** [kaile.zk@antgroup.com](mailto:kaile.zk@antgroup.com)
- **Industry collaboration:** [kaile.zk@antgroup.com](mailto:kaile.zk@antgroup.com)

## Contributing

Contributions of robot embodiments, reconstruction or retargeting backends, visualization features, data converters, and training recipes are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License and Third-Party Components

Original source code in this repository is released under the [Apache License 2.0](LICENSE). Dataset distribution terms, model weights, robot assets, MANO files, and third-party components may use different licenses. Read [LEGAL.md](LEGAL.md) and the relevant subproject README before redistribution or commercial use.

## Acknowledgements

Open-AoE is built by the AoE community. We gratefully acknowledge the following contributors from industry, universities, and research institutions.

### Dataset and Visualization

- **National University of Singapore:** Qingze Guan
- **University of Chinese Academy of Sciences:** Zhengxing Wu
- **Ant Digital Technology, Ant Group:** Zishuo Li, Wanke Zhan, Yang Sun, Zhiyi Huang, Zitong Shan

### Open-Source Code, Toolchain, and Experiments

- **Zhejiang University:** Jiadong Hong
- **The University of Hong Kong:** Zhenchao Jin, Yushi Feng
- **The Hong Kong University of Science and Technology (Guangzhou):** Taowen Wang
- **Beijing Academy of Artificial Intelligence:** You Liu, Yibo Wang
- **University of Chinese Academy of Sciences:** Yifan Yang
- **The Hong Kong University of Science and Technology:** Hao Cheng
- **Ant Digital Technology, Ant Group:** Bowen Yang, Changtao Miao, Zhaowen Zhou, Man Luo

We also thank all data contributors, maintainers, and upstream open-source and research communities whose work made this release possible.
