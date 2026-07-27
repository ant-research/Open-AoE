# AoE 数据集筛选、重建与重定向经验

语言： [English](AOE_DATASET_PRACTICES.md) | **中文**

本文集中讨论选场、实例绑定和数据质量。新机器环境搭建、第三方兼容、12-cell
验收和失败分类见
[Retarget Lab 干净机器部署与复现经验](REPRODUCTION_PRACTICES.zh-CN.md)。

本文总结从真实 AoE 视频接入 EgoInfinity、DAI 和 SPIDER 时反复出现的问题，
目标是帮助第三方先获得正确输入，再判断后端行为。

## 1. 先确定“同一物体实例”

文本 prompt 只说明类别，不能在同一帧多个瓶子中唯一定位目标。对每个场景：

1. 在原始 RGB 上确认 annotation bbox 对应的物理实例。
2. 参考帧必须同时看见目标和执行操作的手。
3. RGB overlay 中物体点云/mesh 必须覆盖同一实例。
4. 记录 `segment + annotation_id + absolute ref-source-frame + bbox + prompt`。
5. 若同类实例并列，使用目标点/框和时序连续性绑定，不能只依赖 prompt 排名。

文件 provenance 全部一致，也不能单独证明语义实例正确；仍需要通过
RGB overlay 确认跟踪的是标注中的同一物体。

AoE action annotation 的框坐标使用归一化到 1000 的
`[x1, y1, x2, y2]`。转换到视频像素时应分别按宽高缩放两个角点，不能把后两个值
当作 width/height。建议在正式筛选前保存一张“原始 RGB + action bbox”抽帧，并检查：

- 像素框没有超出画面；
- 框中心落在目标实例上；
- 框面积与目标大小相符；
- prompt gate、contact sheet 和正式打包器使用同一种坐标解释。

坐标格式错误可能仍产生合法范围内的框，因此仅做数值边界检查不足以发现问题。

## 2. 自动窗口与直接入口

自动入口不是新的重建后端。它只负责在同一标注附近选择一个干净连续窗口，
然后调用与直接入口相同的正式管线。

- 优先最长窗口；只有 SAM3 temporal/interaction gate 通过才进入 full run。
- 当前策略仅向左保留 anchor 并裁短右端，避免悄悄换动作阶段。
- 自动入口适合存在遮挡、实例离开画面、后半段 ID switch 的片段。
- 直接入口用于人工已经确认 clip/ref frame/bbox 的输入，并作为可复现基线。
- 两者均不能绕过重建、route binding 或后端输入审计。

若长窗口中发生实例切换，而同一 anchor 的较短连续窗口通过，这属于前端
数据窗口问题，不应归因于 DAI/SPIDER 优化。

## 3. 可选的数据集候选筛选

在大批量数据上运行前，可以用诊断工具生成候选列表；这一步不是正式
重建或重定向的必需入口。

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

输出仅用于选择值得运行的 annotation/window。正式结果仍须由自动入口或直接
入口从原始 AoE 视频重新生成。

## 4. 相机内参必须统一后再判断 HOI

AoE/HaWoR 手和 Ego object 可能来自不同裁剪或内参。直接把两套 XYZ 放在一个
相机里会造成 overlay 和接触关系错误。应在 adapter 中做射线/深度坐标换算，
并满足：

- 投影像素保持一致；
- Z 深度保持一致；
- 输出 K 与实际视频分辨率一致；
- 换算矩阵、原始 K、目标 K 和 hash 写入 manifest；
- 不用物体平移/缩放来补投影误差。

先分别渲染 AoE annotated hand 与 estimated hand 的 RGB overlay，再运行后端。

## 5. 如何区分重建、IK 与物理优化问题

按边界逐级检查：

1. 原始 RGB + hand/object overlay：错则是标注、相机、实例或重建问题。
2. 纯 mesh HOI：检查手物相对位置、拇指、尺度和物体朝向。
3. 后端初始/kinematic IK：若 reference 正确但机器人掌面偏移，是形态/IK 映射。
4. MJWP 优化前后：若 IK 尚可而 warmup 后失联，是后端原生优化行为。
5. DAI/SPIDER plain video：最终只据真实后端输出判断可用性。

不要用 overlay/triptych 隐藏失败，也不要用 post-hoc object scale/translation
把诊断结果伪装成后端成功。

## 6. 常见 fail-closed 及处理

| 失败 | 含义 | 正确处理 |
| --- | --- | --- |
| multi-instance temporal switching | SAM3 跨帧换了实例 | 缩短连续窗口或重新绑定目标框 |
| prompt frame has no final mask | 序列中可能有 mask，但正式参考帧没有目标实例 | 保留失败；只有人工确认同一标注的其他参考帧有效时才用 direct fresh 入口，仍执行正式 SAM3/SAM3D |
| area ratio jump | mask 突然扩张/收缩 | 检查遮挡、错实例和参考帧 |
| interaction containment low | mask 与手物交互区域不一致 | 调整真实 bbox/ref frame，不降阈值 |
| action bbox 偏大或偏移 | 将 AoE 的 xyxy 误作 xywh | 按 normalized-1000 xyxy 转为像素并用 RGB 抽帧核对 |
| camera intrinsics mismatch | 手、物、RGB 投影相机不同 | 做可审计的 K-to-K 射线换算 |
| bimanual overlap 只有单手通过 | 另一只标注手没有与物体形成足够视觉交互 | 按每手门槛 fail-closed；不能用通过的一只手替代双手任务 |
| HOI coordinate offset exceeds maximum | 手、物输入各自看似有效，但保持源 HOI 需要过大刚体补偿 | 归为输入/重建质量失败；不通过平移物体或放宽补偿上限修正 |
| missing layout/timestamp binding | route 时间轴不完整 | fresh 重建或精确 timebase 转换，不伪造帧 |
| DAI/SPIDER hand-object separation | 后端输出失去 HOI | 保留原生结果；先证实输入边界，再归因后端 |
| native backend rc=0 but post-retarget QC fails | 优化完成不等于可展示；物体跟踪或腕—物相对关系超过门槛 | 保存原生轨迹和视频用于诊断，但不得晋升 Demo |
| native rollout shorter than source endpoint | warmup 后有效轨迹未覆盖完整源动作 | 不外推或重复末帧；materializer 应拒绝生成伪完整视频 |

## 7. 每个场景应交付的证据

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

缺失 DAI/SPIDER 视频时必须明确写“未到达后端”或“后端失败”，不能用 Ego robot
video 代替。候选可以有已知限制，但限制必须紧邻候选结论记录。
