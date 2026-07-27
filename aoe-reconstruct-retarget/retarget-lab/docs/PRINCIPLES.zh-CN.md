# 原理介绍

语言： [English](PRINCIPLES.md) | **中文**

导航： [README](README.zh-CN.md) | [安装](INSTALL.zh-CN.md) | [API](API.zh-CN.md)

本文档解释 AoE Retarget Lab 的管线拆分、数据变换、尺度/坐标约定和可视化调试原则。

## 1. 总体目标

输入：

```text
Open-AoE 第一视角 RGB 视频
目标物体描述
可选 Open-AoE hand annotation/reconstruction
```

输出：

```text
raw RGB + reconstructed hand/object mesh overlay
depth / input geometry
robot retargeting render
```

项目不追求“把 demo 调得看起来差不多”，而是尽量保留物理一致的中间资产：
mask、mesh、6DoF pose、hand trajectory、robot scene、robot trajectory、视频和 manifest。

## 2. 两类主任务

### Trajectory 6DoF Reconstruction

这一阶段估计：

```text
object mesh
object 6DoF trajectory
hand motion / hand mesh
depth / pointmap
RGB mesh overlay
diagnostic mask/mesh videos
```

当前支持：

```text
egoinfinity
do_as_i_do
```

### Retargeting

这一阶段把人手/物体动作映射到机器人 embodiment，并渲染机器人：

```text
egoinfinity -> G1
do_as_i_do -> Sharpa
spider      -> XHand / MJWP
```

`hand_source=aoe|estimated` 决定手部输入来自 Open-AoE 手部数据，还是来自所选重建管线估计结果。

## 3. 为什么 v4 只跑两条完整管线

完整矩阵有 12 个 cell：

```text
2 trajectory_6dof x 2 hand_source x 3 retargeting
```

如果每个 cell 都从 raw RGB 开始，会重复运行很重的 EgoInfinity/Do-as-I-Do
重建和 Do-as-I-Do physics optimization。一个短 clip 也可能因为重复全链路而跑数小时。

当前推荐策略：

1. EgoInfinity 完整跑一次。
2. Do-as-I-Do 完整跑一次。
3. 将可复用资产保存到 `experiments/<run>/intermediates` 和 `experiments/<run>/assets`。
4. 后续 hand-source 和 retargeter 对比复用这些资产。

## 4. EgoInfinity 数据流

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

关键诊断：

```text
mask_overlay.mp4
rgb_mesh_overlay.mp4
mesh_pure_camera.mp4
pipeline_result.pkl.gz
retarget/g1/robot_sim.mp4
```

EgoInfinity 输出保持上游 pipeline 原样，不再 post-filter
`pipeline_result.pkl.gz`；物体身份错误应回到 segmentation / SAM3D /
pose 阶段诊断。

尺度问题应优先看 SAM3D canonical scale、mask+depth 物理 bbox、pose scale
和 `scale_sanity` 输出。不要把硬缩放当最终修复。

## 5. Do-as-I-Do 数据流

```text
Do-as-I-Do reconstruction raw_dir
  -> config/gravity/layout/hand meshes/object masks
  -> object_6dof.npz + visual.obj + source_trajectory_keypoints.npz
  -> mesh overlay / depth / pure mesh camera render
  -> official retargeting/launch.py
  -> scene.xml + trajectory_mjwp.npz
  -> Sharpa visualization_mjwp.mp4
```

`raw_dir` 必须是完整 Do-as-I-Do reconstruction output，不能只是 retargeting 结果目录。

尺度检查重点：

```text
obj_tracking_out/<object>/combined_visualization/layout_camera_frame_optimized.json
translation_scale_optimization.method
translation_scale_optimization.mesh_scale
```

理想尺度来源是由 hand mask / hand mesh / pointmap 支持的
`hand_anchored_pointmap` 优化结果。`shim_from_local_to_scene_scale` 只是诊断
fallback，不能作为最终 demo 尺度。

## 6. SPIDER 数据流

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

SPIDER wrapper 只做输入 staging，并在 pinned、tracked-clean checkout 上依次调用
三个原版入口。它不允许 runtime patcher 或物理优化调参。源 `trajectory_keypoints.npz`
与 robot/object assets 在复制前后都要做哈希校验；不要整目录 symlink 其他 workspace
的历史 `mano/` 目录，避免 metadata 写回外部输出。

`ref_dt` 必须保持 DAI 输入的原始帧网格。若原版 MJWP 的 `sim_dt` 不能整除
`ref_dt`，选择不粗于上游默认 `0.01 s` 的最大精确子步，并把策略写入 manifest；
例如 30 FPS 使用 `sim_dt=1/120 s`。不得通过重采样 keypoints 或修改后端优化参数
来掩盖时间网格冲突。

## 7. 可视化规则

最终 triptych 三行：

```text
overlay: raw RGB + reconstructed hand mesh + reconstructed object mesh/point cloud
depth:   depth / input geometry
robot:   robot body + dexterous hand render
```

不合格最终输出：

```text
segmentation mask only
2D skeleton only
five fingertips
pose dots
object as a single point
borrowed video from another scene or cell
```

这些视图可以作为诊断中间结果，但不能作为最终 demo 输出。

## 8. 坐标和尺度调试

常见变换链路：

```text
camera frame
object canonical mesh frame
local object frame
scene/world frame
robot/MuJoCo frame
render camera frame
```

overlay 或 robot 尺度不对时，按顺序排查：

1. 分割：mask 是否吞进手、桌面、背景或另一个物体。
2. 重建：mesh bbox、顶点范围、canonical scale 是否已经异常。
3. 6DoF/layout：pose translation、layout scale、optimized mesh scale 是否异常。
4. retarget adapter：如果 projection 正常但 MuJoCo 中巨大或漂移，再查单位和坐标变换。

不要把这些做法当最终修复：

```text
hard scale
scale clamp for visual appearance
manual object shrinking
replacing object with a point marker
replacing robot hand with skeleton or fingertips
```

## 9. 集成处理经验

| 现象 | 常见原因 | 建议处理 |
| --- | --- | --- |
| Do-as-I-Do hand mesh 和 RGB 手不重合 | 用 object/MoGe intrinsics 投影 hand mesh | 使用 AoE hand camera intrinsics，并按 clip resolution 缩放 |
| Do-as-I-Do frame 映射错误 | `source_frame_ids` 是原始全视频 id，而 clip 可能使用本地连续 index | 优先本地连续 index，只有 source-id 文件存在时才用 source id |
| Do-as-I-Do object 变成点 | overlay 额外用了硬编码 scale | 默认使用 layout/optimized scale |
| Do-as-I-Do object 资产混用 | physical retarget assets 被当成 RGB reconstruction assets | 优先用 `clip_dir/obj_tracking_out` 和 `video_segmentation/masks` |
| projection 报 `Meshes does not have textures` | 重建 OBJ 只有几何，没有 MTL/纹理 | 使用 projector 的 `--object-color` 只补固定诊断顶点色；不重写几何或物理资产 |
| Do-as-I-Do 尺度到米级 | scale optimization 因缺 hand mask/mesh 落入 shim | 渲染 hand masks，传入 hand meshes，并默认拒绝 shim |
| EgoInfinity 选错同 prompt 瓶子 | 场景中有多个候选 | object filter 支持 `all`、`best` 和 target point |
| EgoInfinity mesh 远大于 mask | SAM3D monocular canonical/pose scale 错 | `scale_sanity.py` 检查 mask+depth 物理 bbox |
| EgoInfinity overlay 缺 `T_seq` | 新 pkl 把 pose 放在 `frame_data[*].sam3_obj_data` | renderer 从 frame data 组装 4x4 transform |
| SAM3D worker OOM 但任务看似空闲 | 中断 run 留下 orphan worker | 只清理确认过的旧 socket worker |
| SPIDER 没有机器人输出 | 早期 runner 只是 import check | pristine wrapper 依次调用原版 generate_xml -> ik_fast -> run_mjwp，并以原生返回码判定结果 |
| SPIDER `trace_dt must be divisible by sim_dt` | 30 FPS 的 `ref_dt=1/30` 不能被上游默认 `0.01` 精确整除 | 保持 keypoints/ref_dt 不变，确定性选择最大精确子步；30 FPS 使用 `sim_dt=1/120` |
| SPIDER `generate_xml` 找不到 object `visual.obj` | staged task info 的 object asset 路径没有绑定到实验私有 processed dataset | 复制经过哈希校验的 object assets 并改写实验副本 task info；不修改源 DAI 输出 |
| DAI/SPIDER robot 行和 RGB 相机比例/视角不一致 | 官方 MJWP 视频拼接 ref/sim，且使用默认 offscreen 分辨率 | 用 `render_mujoco_trajectory.py --camera front --width 1280 --height 720` 从 `scene.xml + trajectory_mjwp.npz` 重渲染 |

## 10. 运行产物检查

- `experiments/<run>/run_env.txt` 记录输入视频、时间段、GPU 和 task 名。
- `logs/` 中有 EgoInfinity、Do-as-I-Do、SPIDER 的命令和日志。
- `intermediates/` 保存第三方原始输出。
- `assets/` 保存稳定可复用 mesh、`npz`、task info 和 manifest。
- `cells/<cell>/manifest.json` 记录 overlay/depth/robot 的真实来源。
- `videos/<cell>__triptych.mp4` 三行同步，top row 是 mesh overlay，bottom row 是机器人本体渲染。
