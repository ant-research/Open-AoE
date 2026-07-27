#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${FULL12_PYTHON:-${EGOINFINITY_PYTHON:-python}}"
retargeting_python="${RETARGETING_PYTHON:-$python_bin}"

source_run=""
base_run_name="${V4_RUN_NAME:-}"
matrix_run_name=""
task="${V4_TASK:-foundation_jar_bimanual_leftscale}"
hand_type="${V4_HAND_TYPE:-bimanual}"
mode="${FULL12_REUSE_MODE:-symlink}"
duration="${FULL12_DURATION:-0}"
compose=1

ego_video="${V4_EGO_VIDEO:-}"
ego_clip_id="${V4_EGO_CLIP_ID:-}"
ego_objects="${V4_EGO_OBJECTS:-}"
ego_start="${V4_EGO_START:-0.0}"
ego_end="${V4_EGO_END:-3.5}"
ego_fps="${V4_EGO_FPS:-15}"
dai_raw_dir="${V4_DAI_RAW_DIR:-}"
dai_clip_dir="${V4_DAI_CLIP_DIR:-}"
dai_source_dir="${V4_DAI_SOURCE_DIR:-}"
dai_prepared_raw_dir="${V4_DAI_PREPARED_RAW_DIR:-}"
dai_object_id="${V4_DAI_OBJECT_ID:-}"
dai_hand_npz="${V4_DAI_HAND_NPZ:-}"
dai_fallback_gravity_json="${V4_DAI_FALLBACK_GRAVITY_JSON:-}"
dai_anchor_hand="${V4_DAI_ANCHOR_HAND:-}"
dai_ref_frame="${V4_DAI_REF_FRAME:-}"
dai_optimize_scale="${V4_DAI_OPTIMIZE_SCALE:-1}"

main_cuda="${MAIN_CUDA:-0}"
dai_cuda_visible_devices="${DAI_CUDA_VISIBLE_DEVICES:-$main_cuda}"
sam3_cuda="${SAM3_WORKER_CUDA:-1}"
sam3d_cuda="${SAM3D_WORKER_CUDA:-2}"

run_spider="${FULL12_RUN_SPIDER:-do_as_i_do}"
spider_python="${SPIDER_PYTHON:-}"
spider_cuda_visible_devices="${SPIDER_CUDA_VISIBLE_DEVICES:-}"
spider_device="${SPIDER_DEVICE:-cuda:0}"
spider_robot_type="${SPIDER_ROBOT_TYPE:-xhand}"
spider_dataset_name="${SPIDER_DATASET_NAME:-do_as_i_do}"
spider_data_id="${SPIDER_DATA_ID:-0}"
spider_max_sim_steps="${SPIDER_MAX_SIM_STEPS:--1}"
spider_num_samples="${SPIDER_NUM_SAMPLES:-1024}"
spider_max_num_iterations="${SPIDER_MAX_NUM_ITERATIONS:-16}"
skip_existing_spider=1
keep_going=1

usage() {
  cat <<'EOF'
Usage:
  scripts/run_full_12_demos.sh [options]

Default mode:
  Run the two heavy full pipelines once, then reuse their intermediate assets to
  compose the 12-cell demo matrix.

Reuse mode:
  Pass --source-run <run> to skip the heavy full pipelines and only build the
  12 demos from an existing experiments/<run>/ source run.

Required for full mode, either as options or environment variables:
  --ego-video PATH          AoE undistorted RGB mp4 (V4_EGO_VIDEO)
  --ego-clip-id ID          stable clip id (V4_EGO_CLIP_ID)
  --ego-objects TEXT        EgoInfinity object prompt (V4_EGO_OBJECTS)
  --dai-raw-dir PATH        Do-as-I-Do reconstruction raw dir (V4_DAI_RAW_DIR)
  --dai-clip-dir PATH       Do-as-I-Do visualization clip dir (V4_DAI_CLIP_DIR)

  Or pass --dai-source-dir PATH to adapt a Do-as-I-Do reconstruction scene into
  experiments/<run>/intermediates/adapters/do_as_i_do/prepared_raw_dir first.

Common options:
  --run-name NAME           source run name for full mode
  --source-run NAME         reuse an existing source run instead of running full mode
  --matrix-run-name NAME    12-demo output run name
  --task NAME               task name used in output paths
  --hand-type TYPE          right, left, or bimanual
  --ego-start SEC           EgoInfinity clip start time
  --ego-end SEC             EgoInfinity clip end time
  --ego-fps FPS             EgoInfinity clip fps
  --dai-source-dir PATH     source Do-as-I-Do scene to adapt locally
  --dai-prepared-raw-dir PATH
                            override adapter output dir
  --dai-object-id ID        override object id inferred from config.json
  --dai-hand-npz PATH       override source hand mesh npz
  --dai-fallback-gravity-json PATH
                            gravity.json used if source dir does not have one
  --dai-anchor-hand TYPE    hand used for Do-as-I-Do scale optimization
  --dai-ref-frame N         reference frame for scale optimization
  --no-dai-scale-optimize   skip official hand-anchored scale optimization
  --main-cuda ID            GPU for main processes
  --dai-cuda-visible-devices IDS
                            GPU visibility for Do-as-I-Do retarget/Mujoco
  --sam3-cuda ID            GPU for SAM3/SAM3.1 worker
  --sam3d-cuda ID           GPU for SAM3D worker
  --run-spider MODE         none, do_as_i_do, or all
  --skip-spider             shortcut for --run-spider none
  --spider-cuda-visible-devices IDS
  --spider-device DEVICE
  --spider-max-sim-steps N
  --spider-num-samples N
  --spider-max-num-iterations N
  --mode MODE               symlink, hardlink, or copy
  --duration SEC            triptych duration; 0 keeps source duration
  --no-compose              prepare assets without composing mp4 demos
  --allow-spider-fallback   rejected: production full runs require exact-route SPIDER output
  --no-spider-fallback      fail/miss EgoInfinity+SPIDER cells instead of fallback (default)
  --force-spider            rerun existing SPIDER cells
  --stop-on-error           stop on a failed SPIDER cell
  -h, --help
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

need_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a value"
}

resolve_repo_path() {
  local path="$1"
  if [[ -z "$path" ]]; then
    return 0
  fi
  if [[ "$path" == /* ]]; then
    readlink -f "$path"
  else
    readlink -f "$repo_root/$path"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name)
      need_value "$@"; base_run_name="$2"; shift 2 ;;
    --source-run)
      need_value "$@"; source_run="$2"; shift 2 ;;
    --matrix-run-name)
      need_value "$@"; matrix_run_name="$2"; shift 2 ;;
    --task)
      need_value "$@"; task="$2"; shift 2 ;;
    --hand-type)
      need_value "$@"; hand_type="$2"; shift 2 ;;
    --ego-video)
      need_value "$@"; ego_video="$2"; shift 2 ;;
    --ego-clip-id)
      need_value "$@"; ego_clip_id="$2"; shift 2 ;;
    --ego-objects)
      need_value "$@"; ego_objects="$2"; shift 2 ;;
    --ego-start)
      need_value "$@"; ego_start="$2"; shift 2 ;;
    --ego-end)
      need_value "$@"; ego_end="$2"; shift 2 ;;
    --ego-fps)
      need_value "$@"; ego_fps="$2"; shift 2 ;;
    --dai-raw-dir)
      need_value "$@"; dai_raw_dir="$2"; shift 2 ;;
    --dai-clip-dir)
      need_value "$@"; dai_clip_dir="$2"; shift 2 ;;
    --dai-source-dir)
      need_value "$@"; dai_source_dir="$2"; shift 2 ;;
    --dai-prepared-raw-dir)
      need_value "$@"; dai_prepared_raw_dir="$2"; shift 2 ;;
    --dai-object-id)
      need_value "$@"; dai_object_id="$2"; shift 2 ;;
    --dai-hand-npz)
      need_value "$@"; dai_hand_npz="$2"; shift 2 ;;
    --dai-fallback-gravity-json)
      need_value "$@"; dai_fallback_gravity_json="$2"; shift 2 ;;
    --dai-anchor-hand)
      need_value "$@"; dai_anchor_hand="$2"; shift 2 ;;
    --dai-ref-frame)
      need_value "$@"; dai_ref_frame="$2"; shift 2 ;;
    --no-dai-scale-optimize)
      dai_optimize_scale=0; shift ;;
    --main-cuda)
      need_value "$@"; main_cuda="$2"; shift 2 ;;
    --dai-cuda-visible-devices)
      need_value "$@"; dai_cuda_visible_devices="$2"; shift 2 ;;
    --sam3-cuda)
      need_value "$@"; sam3_cuda="$2"; shift 2 ;;
    --sam3d-cuda)
      need_value "$@"; sam3d_cuda="$2"; shift 2 ;;
    --run-spider)
      need_value "$@"; run_spider="$2"; shift 2 ;;
    --skip-spider)
      run_spider="none"; shift ;;
    --spider-python)
      need_value "$@"; spider_python="$2"; shift 2 ;;
    --spider-cuda-visible-devices)
      need_value "$@"; spider_cuda_visible_devices="$2"; shift 2 ;;
    --spider-device)
      need_value "$@"; spider_device="$2"; shift 2 ;;
    --spider-robot-type)
      need_value "$@"; spider_robot_type="$2"; shift 2 ;;
    --spider-dataset-name)
      need_value "$@"; spider_dataset_name="$2"; shift 2 ;;
    --spider-data-id)
      need_value "$@"; spider_data_id="$2"; shift 2 ;;
    --spider-max-sim-steps)
      need_value "$@"; spider_max_sim_steps="$2"; shift 2 ;;
    --spider-num-samples)
      need_value "$@"; spider_num_samples="$2"; shift 2 ;;
    --spider-max-num-iterations)
      need_value "$@"; spider_max_num_iterations="$2"; shift 2 ;;
    --mode)
      need_value "$@"; mode="$2"; shift 2 ;;
    --duration)
      need_value "$@"; duration="$2"; shift 2 ;;
    --no-compose)
      compose=0; shift ;;
    --allow-spider-fallback)
      die "--allow-spider-fallback is diagnostic-only and forbidden in production full runs" ;;
    --no-spider-fallback)
      shift ;;
    --force-spider)
      skip_existing_spider=0; shift ;;
    --stop-on-error)
      keep_going=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      die "unknown option: $1" ;;
  esac
done

case "$run_spider" in
  none|do_as_i_do|all) ;;
  *) die "--run-spider must be one of: none, do_as_i_do, all" ;;
esac

case "$mode" in
  symlink|hardlink|copy) ;;
  *) die "--mode must be one of: symlink, hardlink, copy" ;;
esac

if [[ -z "$source_run" ]]; then
  if [[ -z "$base_run_name" ]]; then
    base_run_name="full12_source_$(date +%Y%m%d_%H%M%S)"
  fi
  if [[ -n "$dai_source_dir" ]]; then
    if [[ -z "$dai_prepared_raw_dir" ]]; then
      dai_prepared_raw_dir="$repo_root/experiments/$base_run_name/intermediates/adapters/do_as_i_do/prepared_raw_dir"
    fi
    adapter_cmd=(
      "$retargeting_python"
      "$repo_root/scripts/prepare_do_as_i_do_scene_adapter.py"
      --source-dir "$dai_source_dir"
      --output-dir "$dai_prepared_raw_dir"
      --task "$task"
      --hand-source aoe
      --python-bin "$retargeting_python"
      --hoi-contact-alignment diagnostic
      --force
    )
    if [[ -n "$dai_object_id" ]]; then
      adapter_cmd+=(--object-id "$dai_object_id")
    fi
    if [[ -n "$dai_hand_npz" ]]; then
      adapter_cmd+=(--hand-npz "$dai_hand_npz")
    fi
    if [[ -n "$dai_fallback_gravity_json" ]]; then
      adapter_cmd+=(--fallback-gravity-json "$dai_fallback_gravity_json")
    fi
    resolved_dai_anchor_hand="$dai_anchor_hand"
    if [[ -z "$resolved_dai_anchor_hand" && "$hand_type" =~ ^(left|right)$ ]]; then
      resolved_dai_anchor_hand="$hand_type"
    fi
    if [[ -z "$resolved_dai_anchor_hand" && -s "$dai_source_dir/scale_anchor_hand_selection.json" ]]; then
      resolved_dai_anchor_hand="$("$retargeting_python" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("selected_anchor_hand", ""))' "$dai_source_dir/scale_anchor_hand_selection.json")"
    fi
    if [[ -z "$resolved_dai_anchor_hand" && -s "$dai_source_dir/config.json" ]]; then
      resolved_dai_anchor_hand="$("$retargeting_python" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("scale_anchor_hand", ""))' "$dai_source_dir/config.json")"
    fi
    case "$resolved_dai_anchor_hand" in
      left|right) ;;
      *) die "DAI scale optimization requires a resolved left/right anchor; pass --dai-anchor-hand or provide scale_anchor_hand_selection.json" ;;
    esac
    adapter_cmd+=(--anchor-hand "$resolved_dai_anchor_hand")
    if [[ -n "$dai_ref_frame" ]]; then
      adapter_cmd+=(--ref-frame "$dai_ref_frame")
    fi
    # Production route adapters preserve the source HOI exactly. The adapter
    # CLI still supports scale optimization for diagnostics, but matrix inputs
    # must opt out so hoi_refinement.applied remains false.
    adapter_cmd+=(--no-optimize-scale)
    echo "[full12] adapting Do-as-I-Do scene: $dai_source_dir"
    (cd "$repo_root" && "${adapter_cmd[@]}")
    dai_raw_dir="$dai_prepared_raw_dir"
    if [[ -z "$dai_clip_dir" ]]; then
      dai_clip_dir="$dai_prepared_raw_dir"
    fi
  fi
  [[ -n "$ego_video" ]] || die "set --ego-video or V4_EGO_VIDEO"
  [[ -n "$ego_clip_id" ]] || die "set --ego-clip-id or V4_EGO_CLIP_ID"
  [[ -n "$ego_objects" ]] || die "set --ego-objects or V4_EGO_OBJECTS"
  [[ -n "$dai_raw_dir" ]] || die "set --dai-raw-dir or V4_DAI_RAW_DIR"
  [[ -n "$dai_clip_dir" ]] || die "set --dai-clip-dir or V4_DAI_CLIP_DIR"
  ego_video="$(resolve_repo_path "$ego_video")"
  dai_raw_dir="$(resolve_repo_path "$dai_raw_dir")"
  dai_clip_dir="$(resolve_repo_path "$dai_clip_dir")"

  export V4_RUN_NAME="$base_run_name"
  export V4_TASK="$task"
  export V4_HAND_TYPE="$hand_type"
  export V4_EGO_VIDEO="$ego_video"
  export V4_EGO_CLIP_ID="$ego_clip_id"
  export V4_EGO_OBJECTS="$ego_objects"
  export V4_EGO_START="$ego_start"
  export V4_EGO_END="$ego_end"
  export V4_EGO_FPS="$ego_fps"
  export V4_DAI_RAW_DIR="$dai_raw_dir"
  export V4_DAI_CLIP_DIR="$dai_clip_dir"
  export MAIN_CUDA="$main_cuda"
  export DAI_CUDA_VISIBLE_DEVICES="$dai_cuda_visible_devices"
  export SAM3_WORKER_CUDA="$sam3_cuda"
  export SAM3D_WORKER_CUDA="$sam3d_cuda"

  echo "[full12] running two full pipelines: experiments/$base_run_name"
  (cd "$repo_root" && scripts/run_v4_two_full_pipelines.sh)
  source_run="$base_run_name"
else
  echo "[full12] reusing source run: experiments/$source_run"
fi

if [[ -z "$matrix_run_name" ]]; then
  matrix_run_name="${source_run}__12demos_$(date +%Y%m%d_%H%M%S)"
fi

reuse_cmd=(
  "$python_bin"
  "$repo_root/scripts/reuse_v4_for_12_demos.py"
  --source-run "$source_run"
  --run-name "$matrix_run_name"
  --task "$task"
  --hand-type "$hand_type"
  --mode "$mode"
  --run-spider "$run_spider"
  --spider-device "$spider_device"
  --spider-robot-type "$spider_robot_type"
  --spider-dataset-name "$spider_dataset_name"
  --spider-data-id "$spider_data_id"
  --spider-max-sim-steps "$spider_max_sim_steps"
  --spider-num-samples "$spider_num_samples"
  --spider-max-num-iterations "$spider_max_num_iterations"
  --no-allow-spider-fallback
  --no-allow-spider-mjwp-fallback-video
  --no-allow-dai-hand-fallback
  --no-allow-cross-trajectory-dai-robot-fallback
  --no-include-numerically-invalid-aligned-robot
)

if [[ "$compose" -eq 1 ]]; then
  reuse_cmd+=(--compose)
fi
if [[ "$duration" != "0" && "$duration" != "0.0" ]]; then
  reuse_cmd+=(--duration "$duration")
fi
if [[ -n "$spider_python" ]]; then
  reuse_cmd+=(--spider-python "$spider_python")
fi
if [[ -n "$spider_cuda_visible_devices" ]]; then
  reuse_cmd+=(--spider-cuda-visible-devices "$spider_cuda_visible_devices")
fi
if [[ "$skip_existing_spider" -eq 1 ]]; then
  reuse_cmd+=(--skip-existing-spider)
else
  reuse_cmd+=(--no-skip-existing-spider)
fi
if [[ "$keep_going" -eq 1 ]]; then
  reuse_cmd+=(--keep-going)
else
  reuse_cmd+=(--no-keep-going)
fi

echo "[full12] expanding 12 demos: experiments/$matrix_run_name"
(cd "$repo_root" && "${reuse_cmd[@]}")

matrix_root="$repo_root/experiments/$matrix_run_name"
matrix_audit_json="$matrix_root/audit_retarget_run.json"
matrix_audit_log="$matrix_root/logs/audit_retarget_run.log"
mkdir -p "$matrix_root/logs"
echo "[full12] strict matrix admission audit: experiments/$matrix_run_name"
matrix_audit_rc=0
"$python_bin" - \
  "$repo_root" "$matrix_run_name" "$source_run" "$matrix_audit_json" "$compose" <<'PY' \
  2>&1 | tee "$matrix_audit_log" || matrix_audit_rc=$?
import json
import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
matrix_run = sys.argv[2]
source_run = sys.argv[3]
output_json = Path(sys.argv[4]).resolve()
compose_enabled = bool(int(sys.argv[5]))
sys.path.insert(0, str(repo_root / "scripts"))

import audit_retarget_run as audit  # noqa: E402

matrix_root = repo_root / "experiments" / matrix_run
manifest_path = matrix_root / "reuse_12_demo_manifest.json"
manifest = audit.load_json(manifest_path)
errors = []
warnings = []
cell_report = {}
spider_report = {}
if manifest is None:
    errors.append(f"missing matrix manifest: {manifest_path}")
else:
    cell_report = audit.audit_cells(
        matrix_root,
        manifest,
        allow_legacy_unaligned=False,
        require_triptych=compose_enabled,
    )
    spider_report = audit.audit_spider(matrix_root, manifest, allow_legacy_unaligned=False)
    errors.extend(cell_report.get("errors") or [])
    errors.extend(spider_report.get("errors") or [])
    warnings.extend(cell_report.get("warnings") or [])
    warnings.extend(spider_report.get("warnings") or [])

admission = audit.summarize_route_admission(cell_report, spider_report)
errors.extend(admission.get("errors") or [])
candidate_routes = list(admission.get("admitted_routes") or [])
if not candidate_routes:
    errors.append("strict matrix audit admitted no exact numerical-valid retarget route")
candidate_routes_blocked_by_audit_errors = candidate_routes if errors else []
admitted_routes = [] if errors else candidate_routes
admitted_route_count = len(admitted_routes)

report = {
    "scope": "matrix_only",
    "matrix_run": matrix_run,
    "matrix_root": str(matrix_root),
    "source_run": source_run,
    "manifest": str(manifest_path),
    "source_audit_status": "deferred_to_fresh_full_run_audit",
    "cells": cell_report,
    "spider": spider_report,
    "admitted_routes": admitted_routes,
    "candidate_routes_blocked_by_audit_errors": candidate_routes_blocked_by_audit_errors,
    "unavailable_routes": admission.get("unavailable_routes") or [],
    "unavailable_route_details": admission.get("unavailable_route_details") or [],
    "invalid_routes": admission.get("invalid_routes") or [],
    "invalid_route_details": admission.get("invalid_route_details") or [],
    "admitted_route_count": admitted_route_count,
    "numerically_admitted_routes": admitted_routes,
    "numerically_admitted_route_count": admitted_route_count,
    "composition_enabled": compose_enabled,
    "demo_count": admitted_route_count if compose_enabled else 0,
    "scene_admitted": admitted_route_count > 0 if compose_enabled else False,
    "status": "failed" if errors else "ok",
    "errors": errors,
    "warnings": warnings,
}
output_json.parent.mkdir(parents=True, exist_ok=True)
text = json.dumps(report, indent=2, ensure_ascii=False)
output_json.write_text(text + "\n", encoding="utf-8")
print(text)
raise SystemExit(1 if errors else 0)
PY
if (( matrix_audit_rc != 0 )); then
  echo "[full12] strict matrix admission audit failed: $matrix_audit_json" >&2
  exit "$matrix_audit_rc"
fi

cat <<EOF
[full12] done
source_run: experiments/$source_run
matrix_run: experiments/$matrix_run_name
videos:     experiments/$matrix_run_name/videos/
manifest:   experiments/$matrix_run_name/reuse_12_demo_manifest.json
audit:      experiments/$matrix_run_name/audit_retarget_run.json
EOF
