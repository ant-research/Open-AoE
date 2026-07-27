# Retarget Lab Clean-Machine Deployment and Reproduction Practices

Language: **English** | [中文](REPRODUCTION_PRACTICES.zh-CN.md)

This guide distills practices that transfer across clean-machine reproductions.
It intentionally excludes machine names, individual run histories, measured
failure values, and archive hashes. For scene selection and data quality, see
[AoE Dataset Selection, Reconstruction, and Retargeting Practices](AOE_DATASET_PRACTICES.md).

## 1. Distinguish Three Kinds of Success

Do not conflate:

1. **Environment success**: interpreters, CUDA, checkpoints, and native
   third-party entries work.
2. **Pipeline success**: SAM3 → SAM3D/6DoF → Ego → DAI/SPIDER executes and
   produces outputs.
3. **Demo success**: a DAI or SPIDER result passes input audit, post-backend QC,
   and manual video review.

Environment and pipeline success prove deployability, not demo quality. Run
numbers may include environment diagnosis, fresh reruns, and multiple
annotations from one scene; they are not counts of independent scenes.

## 2. Recommended Deployment Order

### 2.1 Pin Source and Third-Party Versions

- Record the Retarget Lab commit, then pin every third-party commit listed by
  the installation guide.
- Keep third-party checkouts tracked-clean. Put compatibility in Lab launchers,
  never in runtime edits to backend source.
- Store environments, weights, data, caches, and experiments on the work disk.
  Keep `local_env.sh` machine-private.
- Never commit `third_party/`, environments, weights, experiment outputs, or
  machine paths.

### 2.2 Separate Python Environments

SAM3/SAM3D, EgoInfinity, DAI, and SPIDER span incompatible dependency ranges.
Use separate environments instead of forcing all packages into one interpreter
and overwriting a validated Torch/CUDA combination. One clean-machine audit
validated, for example:

- SAM3D: Torch 2.8, CUDA 12.8, Kaolin 0.18, PyTorch3D 0.7.8;
- SPIDER: Python 3.12, MuJoCo 3.7.0, MuJoCo Warp 3.7.0.1.

An old upstream Conda file being unsatisfiable against current indices does not
prove that the backend algorithm cannot run. Preserve the solver failure and
record the actual compatibility runtime and deviations in the environment
report.

### 2.3 Actually Load Every Weight

Existence checks miss partial downloads, incorrect snapshots, and missing
transitive dependencies. For every model asset, record:

1. file size and SHA-256;
2. offline import or `from_pretrained`;
3. a minimal smoke that triggers weight deserialization;
4. snapshot, cache root, and model commit.

Audit the directories actually read by SAM3D Objects and Fast-SAM3D separately,
even when their checkpoint filenames match. EgoInfinity may require MoGe,
WiLoR, SAM2, infiller, MEMFOF, and ResNet34 only in later stages; passing its
first stage does not prove that the complete asset set is available.

### 2.4 Audit Resolved Paths

Before a run:

```bash
source local_env.sh
python scripts/check_third_party_config.py --json third_party_config_report.json
```

In addition to configured strings, audit the final `realpath` of interpreter
symlinks. A path may appear to live on the new work disk while resolving to an
old environment. Use `AOE_REQUIRED_EXECUTABLE_ROOT` for strict migrations.

## 3. Integration Problems Exposed by Clean Machines

### 3.1 Python Namespace Shadowing

SAM3D and Fast-SAM3D both contain a top-level `notebook/` without
`__init__.py`. An installed Jupyter `notebook` package may capture that name.
Production compatibility launchers explicitly bind the upstream namespace in
memory without changing third-party source.

### 3.2 Upstream/Downstream API Drift

SAM3 checkpoint arguments and an older EgoInfinity SAM3D worker may not match a
current `Inference` signature. The compatibility policy is:

- preserve original parameters and inference-step counts;
- map them in a Lab proxy to the current pipeline entry;
- run a real image/mask reconstruction smoke;
- never treat a mocked import as backend validation.

### 3.3 Native Open3D CUDA Crashes

Open3D 0.18 CUDA can reproducibly crash on certain dtypes and voxelization
operations on some machines. Diagnose it by:

1. locating the native boundary with GDB;
2. minimizing with the same mesh;
3. testing dtype/contiguity and voxelization independently;
4. validating a compatible wheel in an isolated directory;
5. applying the same boundary compatibility to both mesh generation and
   tracking entries.

At the input boundary, normalize only dtype and contiguity. Do not modify
coordinates, indices, optimizer behavior, or quality thresholds.

### 3.4 Keep DAI and Reconstruction Roots Separate

AoE hand export and hand-mask rasterization are Lab input adapters, not part of
the pristine DAI backend. Bind the reconstruction/helper scripts root
separately from the pristine DAI retarget root.

`DAI_RUNTIME_PATCH_MODE=pristine` must skip historical runtime patchers and
verify that the DAI checkout remains pinned and clean.

### 3.5 SPIDER Timebase and Headless Rendering

Native SPIDER MJWP requires `trace_dt` to be divisible by `sim_dt`. For 30 FPS,
`sim_dt=1/120` preserves the source-frame grid while providing an exact
sub-step. This is deterministic input packaging and must not alter rewards,
noise, sample counts, or optimizer behavior.

The production route runs the original
`generate_xml.py → ik_fast.py → run_mjwp.py`, copies inputs into a private
experiment directory, and propagates native return codes. Resolve EGL/CUDA
driver mismatches through machine setup or restart, not algorithm changes.

### 3.6 AoE Coordinates and Encoded Frame Counts

- Action boxes are normalized-to-1000 `[x1,y1,x2,y2]` and default to `xyxy`.
- A clip near the end of a source video may encode fewer frames than requested.
  Bind the reference frame to the post-encode
  `ffprobe -count_frames` result.
- Direct entry bypasses only automatic window search. It still performs fresh
  SAM3/SAM3D reconstruction and all backend audits.

## 4. Recommended Validation Ladder

Validate from inexpensive to expensive:

1. Python paths, commits, tracked-clean status, and checkpoints;
2. key imports and CUDA smoke in every environment;
3. single-image SAM3, single-object SAM3D, and minimal native
   Ego/DAI/SPIDER entries;
4. one fresh automatic and one fresh direct scene;
5. full reconstruction for one scene;
6. native DAI/SPIDER execution;
7. the 12-cell matrix;
8. serial multi-scene screening.

Before committing:

```bash
python -m pytest -q
git diff --check
```

Machine-private DAI defaults can contaminate CLI tests. Clear them explicitly
and report the environment-neutral test result.

## 5. Failure Classification

| Class | Typical evidence | Fix in Lab? |
| --- | --- | --- |
| Environment/asset | import, checkpoint load, CUDA, or driver failure | Yes |
| Lab integration | incorrect argument mapping, path, frame, box, or layout | Yes |
| Native upstream runtime | stable crash minimized to a third-party binary/API | Boundary compatibility only, with evidence |
| SAM3 data quality | missing prompt mask, low valid ratio, identity switch | No |
| Reconstruction quality | pose jumps, mesh drift, excessive HOI offset | No |
| Backend quality | DAI/SPIDER completes but post-QC or review fails | No |
| Cascaded failure | no valid exact-route input from an earlier stage | Not an independent backend failure |

Only environment, integration, and narrowly scoped runtime packaging may be
fixed. The remaining classes must fail closed. Do not alter SAM3 selection,
production thresholds, DAI/SPIDER algorithms, or physics parameters to inflate
the demo count.

## 6. Interpreting the 12 Cells

The matrix is:

```text
trajectory_6dof = egoinfinity | do_as_i_do
hand_source     = aoe | estimated
retargeting     = egoinfinity | do_as_i_do | spider
```

This yields `2 × 2 × 3 = 12` cells. A cell succeeds only when:

1. exact-route input exists with correct provenance;
2. input QC passes;
3. the native backend succeeds and covers the complete source duration;
4. post-retarget QC passes;
5. manual review finds the plain robot video reasonable.

Ego-only output, a compatibility smoke, a short rollout, or mere video
generation cannot substitute for a DAI/SPIDER demo. Native optimization or a
smoke proves execution, not final quality.

## 7. Multi-Scene Stop and Resource Policy

- Run every formal scene fresh from original video. Do not reuse
  reconstruction, adapter, or trajectory artifacts.
- Serialize work on a single GPU and inspect GPU PID ancestry before launch.
- Try 6/5/4/3-second automatic windows longest-first. Use direct entry only
  when manual review shows a valid video and the automatic failure is
  specifically attributable to reference-frame choice.
- Treat run IDs as attempt identifiers and report independent
  segment/annotation counts separately.
- Keep safe free space. Do not start a run during archive creation. Delete
  regenerable intermediates only after the review bundle has been copied and
  its SHA verified.
- If failures concentrate at one quality gate, summarize their distribution
  and representative videos before spending unbounded GPU time on more scenes.

## 8. Evidence Required at Every Terminal State

```text
original RGB
SAM3 mask / RGB overlay
mesh HOI / projected overlay
Ego, DAI, and SPIDER plain robot renders (when produced)
triptych
window, input, route, backend, and QC manifests
native commands, return codes, and logs
input/output SHA256SUMS
terminal stage and failure classification
```

Verify both the outer archive hash and the inner file manifest. If a backend
produces no result, record the first failure boundary; do not substitute an Ego
video or another route.

## 9. Public Reporting Policy

Report environment, pipeline, and demo conclusions separately, with evidence
for each layer. A complete native chain or compatibility smoke does not prove
that all 12 cells produced usable demos. Conversely, SAM3 data-quality failure,
reconstruction drift, or backend post-QC rejection should not be summarized as
“deployment failed.”

Machine names, individual runs, candidate lists, failure measurements, video
reviews, and archive hashes belong in private experiment records, not in this
general guide or a public release.
