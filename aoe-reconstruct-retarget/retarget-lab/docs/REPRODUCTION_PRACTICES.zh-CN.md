# Retarget Lab 干净机器部署与复现经验

语言： [English](REPRODUCTION_PRACTICES.md) | **中文**

本文把干净机器复现中可迁移到其他机器的经验整理为操作指南。本文不收录机器名、
逐次 run、具体指标或归档哈希；数据集选场经验见
[AoE 数据集筛选、重建与重定向经验](AOE_DATASET_PRACTICES.zh-CN.md)。

## 1. 先区分三种“成功”

复现时不要把下面三件事混为一谈：

1. **环境成功**：解释器、CUDA、checkpoint 和第三方原生入口可用。
2. **管线成功**：SAM3 → SAM3D/6DoF → Ego → DAI/SPIDER 能运行并产生结果。
3. **Demo 成功**：DAI 或 SPIDER 结果通过输入审计、后端后置 QC 和人工视频审阅。

环境成功和管线成功只能证明部署可用，不能代替 Demo 质量结论。run 编号通常还会
包含环境定位、fresh rerun 和同场景不同 annotation，不能直接当成独立场景数量。

## 2. 推荐部署顺序

### 2.1 固定代码和第三方版本

- 先记录 Retarget Lab commit，再按安装文档固定第三方 commit。
- 第三方 checkout 必须 tracked-clean；兼容逻辑放在 Lab launcher，不在运行时改后端。
- 环境、权重、数据、cache 和实验输出放在工作盘；`local_env.sh` 保持机器私有。
- 不把 `third_party/`、环境、权重、实验目录或机器路径提交到 Git。

### 2.2 分离 Python 环境

SAM3/SAM3D、EgoInfinity、DAI 和 SPIDER 的依赖跨度较大，推荐使用独立环境，
不要为了“一个环境全装下”而覆盖已验证的 Torch/CUDA 组合。一次干净机器复现中
验证过的兼容组合包括：

- SAM3D：Torch 2.8、CUDA 12.8、Kaolin 0.18、PyTorch3D 0.7.8；
- SPIDER：Python 3.12、MuJoCo 3.7.0、MuJoCo Warp 3.7.0.1。

上游旧 Conda 文件在当前索引上不可满足，不等于后端算法不可运行。应保留求解失败
证据，并把实际兼容 runtime 和偏差写入环境报告。

### 2.3 权重必须做真实加载

只检查文件存在会漏掉 partial download、错误 snapshot 和缺失子依赖。每个资产至少
完成：

1. 文件大小和 SHA-256；
2. 离线 import / `from_pretrained`；
3. 能触发权重反序列化的最小 smoke；
4. snapshot、cache 根和模型 commit 记录。

SAM3D Objects 与 Fast-SAM3D 即使文件名相同，也要分别审计它们实际读取的目录。
EgoInfinity 还会在后续阶段依次需要 MoGe、WiLoR、SAM2、infiller、MEMFOF 和
ResNet34；首阶段通过不能证明整条 Ego 链路资产完整。

### 2.4 配置审计必须看 resolved path

运行前执行：

```bash
source local_env.sh
python scripts/check_third_party_config.py --json third_party_config_report.json
```

除配置字符串外，还要检查解释器 symlink 的最终 `realpath`，防止表面指向新工作盘、
实际仍落在旧环境。需要严格迁移时使用 `AOE_REQUIRED_EXECUTABLE_ROOT`。

## 3. 干净机器最容易暴露的集成问题

### 3.1 Python 命名空间遮蔽

SAM3D 和 Fast-SAM3D 都有无 `__init__.py` 的顶层 `notebook/`。已安装的 Jupyter
`notebook` 会优先占用该名字。正式入口通过 compatibility launcher 显式绑定上游
模块命名空间，不修改第三方源码。

### 3.2 上下游 API 版本不一致

SAM3 checkpoint 参数、EgoInfinity 的旧 SAM3D worker 与当前 `Inference` 调用签名
都出现过不一致。处理原则是：

- 保留原参数和推理步数；
- 在 Lab 代理中映射到当前公开或实际 pipeline 入口；
- 用真实图像/mask 做最小重建 smoke；
- 不把 mock import 通过当作后端可用。

### 3.3 Open3D CUDA 原生崩溃

Open3D 0.18 CUDA 在部分机器上会对特定 dtype 和 voxelization 稳定 SIGSEGV。
定位时应：

1. 用 GDB 确认崩溃边界；
2. 用同一 mesh 做最小复现；
3. 分别测试 dtype/contiguity 和 voxelization；
4. 在隔离目录验证兼容 wheel；
5. 让 mesh generation 与 tracking 两个独立入口都使用同一兼容边界。

输入边界只允许规范化数组 dtype/contiguity，不能改坐标、索引、优化参数或质量阈值。

### 3.4 DAI 与 reconstruction root 不应混绑

AoE hand 导出、hand-mask rasterization 等属于 Lab 输入包装，不属于 pristine DAI
后端。正式配置应分别绑定：

- reconstruction/helper scripts root；
- pristine DAI retarget root。

`DAI_RUNTIME_PATCH_MODE=pristine` 必须跳过历史 runtime patcher，并检查 DAI checkout
仍为 pinned clean。

### 3.5 SPIDER 时间网格和 headless 渲染

SPIDER 原生 MJWP 要求 `trace_dt` 可被 `sim_dt` 整除。30 FPS 输入可选
`sim_dt=1/120`，保持源帧时间网格不变。该处理只是确定性的输入时间包装，不应调整
reward、noise、采样数或 optimizer。

正式 SPIDER 路径只运行原版 `generate_xml.py → ik_fast.py → run_mjwp.py`，输入复制
到实验私有目录，后端返回码原样传播。EGL/CUDA 驱动版本不一致应先通过重启或机器
环境修复解决，不要用算法参数掩盖。

### 3.6 AoE 输入包装的坐标与真实帧数

- action annotation 的 bbox 是归一化到 1000 的 `[x1,y1,x2,y2]`，默认必须按
  `xyxy` 解释；
- 靠近视频末尾的 clip 可能短于请求时长，reference frame 必须以编码后的
  `ffprobe -count_frames` 为准；
- direct-entry 只绕过自动选窗，仍需 fresh SAM3/SAM3D 和全部后端审计。

## 4. 推荐验证阶梯

按从便宜到昂贵的顺序验证，避免用 full run 排查基础安装：

1. Python 路径、commit、tracked-clean 和 checkpoint 审计；
2. 各环境关键 import 与 CUDA smoke；
3. 单图 SAM3、单物体 SAM3D、单帧 Ego/DAI/SPIDER 原生入口；
4. automatic 与 direct 各一个 fresh 最小场景；
5. 单场景完整 reconstruction；
6. DAI/SPIDER 原生后端；
7. 12-cell matrix；
8. 多场景串行筛选。

代码提交前至少执行：

```bash
python -m pytest -q
git diff --check
```

机器私有 DAI 默认变量可能污染部分 CLI 单测。代码审计应显式清除这些变量后运行，
并在提交说明中记录环境中立的测试结果。

## 5. 失败分类

| 分类 | 典型证据 | 是否应修 Lab |
| --- | --- | --- |
| 环境/资产 | import、checkpoint load、CUDA、驱动失败 | 是 |
| Lab 集成 | 参数映射、路径、帧号、bbox、输入布局错误 | 是 |
| 上游原生 runtime | 可缩小到第三方二进制/API 的稳定崩溃 | 只做边界兼容并记录 |
| SAM3 数据质量 | 无 prompt mask、valid ratio、身份跳变、错误实例 | 否 |
| 重建质量 | 位姿跳变、mesh 漂移、HOI offset 过大 | 否 |
| 后端质量 | DAI/SPIDER 完成但 post-QC 或视觉审阅失败 | 否 |
| 连带失败 | 上游没有合法 exact-route 输入 | 不算后端独立故障 |

修复只能针对前三类中的环境、集成或运行时包装；后四类必须 fail-closed。不得改
SAM3 选择逻辑、生产阈值、DAI/SPIDER 算法或物理优化参数来增加 Demo 数。

## 6. 如何理解 12 个组合

矩阵为：

```text
trajectory_6dof = egoinfinity | do_as_i_do
hand_source     = aoe | estimated
retargeting     = egoinfinity | do_as_i_do | spider
```

共 `2 × 2 × 3 = 12` cells。一个 cell 只有同时满足以下条件才算成功：

1. exact-route 输入存在且 provenance 正确；
2. 输入 QC 通过；
3. 原生后端返回成功并覆盖完整源时长；
4. post-retarget QC 通过；
5. plain robot 视频人工审阅合理。

Ego-only、compatibility smoke、短 rollout 或仅生成视频均不能补作 DAI/SPIDER
Demo。原生优化或 smoke 完成只说明部署链路可运行，不说明最终质量已经达标。

## 7. 多场景复现的停止与资源策略

- 正式场景必须从原始视频 fresh 运行，不复用 reconstruction、adapter 或 trajectory。
- 单 GPU 串行，启动前核验 GPU PID 和父子关系。
- 自动窗口按 6/5/4/3 秒最长优先；只有人工视频明确可用且 automatic 的失败可由
  参考帧解释时，才使用 direct。
- run ID 只是尝试序号；报告同时给出独立 segment/annotation 数。
- 工作盘保留安全空间。归档期间不启动新 run；只有审阅包复制并校验 SHA 后，才可
  清理可再生成中间缓存。
- 若大量失败集中在同一质量门禁，应先汇总失败分布和代表性视频，再决定是否继续
  扩展候选，而不是无界增加 GPU 消耗。

## 8. 每个终态必须保存的证据

```text
原始 RGB
SAM3 mask / RGB overlay
mesh HOI / projected overlay
Ego、DAI、SPIDER plain robot render（若产生）
triptych
window、input、route、backend 和 QC manifests
原生命令、返回码和日志
输入/输出 SHA256SUMS
失败阶段与失败分类
```

审阅包必须同时验证外层归档 SHA 和内层文件清单。没有生成某个后端结果时，应明确
记录首个失败边界，不能用 Ego 视频或其他 route 的结果代替。

## 9. 对外报告原则

对外报告应分别给出环境、管线和 Demo 三层结论，并为每层提供相应证据。不能因为
完整原生链路或 compatibility smoke 已运行，就声称 12 个组合已经产生可用 Demo；
也不能把 SAM3 数据质量、重建漂移或后端 post-QC 拒绝笼统写成“部署失败”。

机器名、逐次 run、候选清单、失败指标、视频审阅和归档哈希属于本地实验记录，
不应放入通用经验文档或公开 release。
