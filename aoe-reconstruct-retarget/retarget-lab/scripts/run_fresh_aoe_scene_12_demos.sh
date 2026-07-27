#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_ego="${EGOINFINITY_PYTHON:-python3}"
python_ret="${RETARGETING_PYTHON:-python3}"
spider_python="${SPIDER_PYTHON:-python3}"
sam3_python="${SAM3_PYTHON:-python3}"
sam3_repo="${SAM3_REPO:-${repo_root}/third_party/sam3}"
sam3d_python="${SAM3D_PYTHON:-python3}"
sam3d_repo="${SAM3D_REPO:-${repo_root}/third_party/sam-3d-objects}"
dinov2_local_repo="${DINOV2_LOCAL_REPO:-${repo_root}/third_party/dinov2}"
egoinfinity_sam3_version="${EGOINFINITY_SAM3_VERSION:-sam3}"

dataset_root="${AOE_DATA_ROOT:-}"
scene=""
segment=""
annotation_id=""
object_name=""
ego_objects=""
task=""
hand_type=""
anchor_hand=""
source_run=""
matrix_run=""
duration_sec="0"
start_offset_sec="0"
extra_sec_after="0"
ref_frame_fraction="-1"
ref_source_frame="-1"
scale_width="1280"
main_cuda="${MAIN_CUDA:-5}"
dai_cuda_visible_devices="${DAI_CUDA_VISIBLE_DEVICES:-$main_cuda}"
sam3_cuda="${SAM3_WORKER_CUDA:-5}"
sam3d_cuda="${SAM3D_WORKER_CUDA:-6}"
sam3d_min_free_mb="${SAM3D_MIN_FREE_MB:-19000}"
sam3d_wait_for_gpu="${SAM3D_WAIT_FOR_GPU:-1}"
sam3d_wait_timeout_sec="${SAM3D_WAIT_TIMEOUT_SEC:-0}"
sam3d_wait_poll_sec="${SAM3D_WAIT_POLL_SEC:-60}"
spider_cuda_visible_devices="${SPIDER_CUDA_VISIBLE_DEVICES:-6}"
spider_device="${SPIDER_DEVICE:-cuda:0}"
spider_max_sim_steps="${SPIDER_MAX_SIM_STEPS:--1}"
spider_num_samples="${SPIDER_NUM_SAMPLES:-1024}"
spider_max_num_iterations="${SPIDER_MAX_NUM_ITERATIONS:-16}"
run_spider="${FULL12_RUN_SPIDER:-all}"
sam3_preflight_only=0
compose=1

usage() {
  cat <<'EOF'
Usage:
  scripts/run_fresh_aoe_scene_12_demos.sh --scene NAME --segment SEG --annotation-id ID \
    --object-name TEXT --task TASK --hand-type left|right|bimanual --anchor-hand left|right|bimanual

This creates a clean source run from AoE raw RGB, runs fresh Do-as-I-Do
reconstruction, then runs the two full pipelines and expands 12 triptych demos.
No old source-run or old pipeline output is accepted as input.

Use --sam3-preflight-only to stop after fresh text+bbox SAM3 masks and prompt
gate validation. The preflight does not create a matrix run or start SAM3D.
Use --no-compose to keep only exact-route assets and plain aligned robot
renders; no triptych demo videos will be generated.
EOF
}

need_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || { echo "error: $1 requires a value" >&2; exit 2; }
}

original_args=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) need_value "$@"; scene="$2"; shift 2 ;;
    --dataset-root) need_value "$@"; dataset_root="$2"; shift 2 ;;
    --segment) need_value "$@"; segment="$2"; shift 2 ;;
    --annotation-id) need_value "$@"; annotation_id="$2"; shift 2 ;;
    --object-name) need_value "$@"; object_name="$2"; shift 2 ;;
    --ego-objects) need_value "$@"; ego_objects="$2"; shift 2 ;;
    --task) need_value "$@"; task="$2"; shift 2 ;;
    --hand-type) need_value "$@"; hand_type="$2"; shift 2 ;;
    --anchor-hand) need_value "$@"; anchor_hand="$2"; shift 2 ;;
    --source-run) need_value "$@"; source_run="$2"; shift 2 ;;
    --matrix-run) need_value "$@"; matrix_run="$2"; shift 2 ;;
    --duration-sec) need_value "$@"; duration_sec="$2"; shift 2 ;;
    --start-offset-sec) need_value "$@"; start_offset_sec="$2"; shift 2 ;;
    --extra-sec-after) need_value "$@"; extra_sec_after="$2"; shift 2 ;;
    --ref-frame-fraction) need_value "$@"; ref_frame_fraction="$2"; shift 2 ;;
    --ref-source-frame) need_value "$@"; ref_source_frame="$2"; shift 2 ;;
    --scale-width) need_value "$@"; scale_width="$2"; shift 2 ;;
    --main-cuda) need_value "$@"; main_cuda="$2"; shift 2 ;;
    --dai-cuda-visible-devices) need_value "$@"; dai_cuda_visible_devices="$2"; shift 2 ;;
    --sam3-cuda) need_value "$@"; sam3_cuda="$2"; shift 2 ;;
    --sam3d-cuda) need_value "$@"; sam3d_cuda="$2"; shift 2 ;;
    --sam3d-min-free-mb) need_value "$@"; sam3d_min_free_mb="$2"; shift 2 ;;
    --sam3d-wait-for-gpu) sam3d_wait_for_gpu="1"; shift ;;
    --no-sam3d-wait-for-gpu) sam3d_wait_for_gpu="0"; shift ;;
    --sam3d-wait-timeout-sec) need_value "$@"; sam3d_wait_timeout_sec="$2"; shift 2 ;;
    --sam3d-wait-poll-sec) need_value "$@"; sam3d_wait_poll_sec="$2"; shift 2 ;;
    --spider-cuda-visible-devices) need_value "$@"; spider_cuda_visible_devices="$2"; shift 2 ;;
    --spider-device) need_value "$@"; spider_device="$2"; shift 2 ;;
    --spider-max-sim-steps) need_value "$@"; spider_max_sim_steps="$2"; shift 2 ;;
    --spider-num-samples) need_value "$@"; spider_num_samples="$2"; shift 2 ;;
    --spider-max-num-iterations) need_value "$@"; spider_max_num_iterations="$2"; shift 2 ;;
    --run-spider) need_value "$@"; run_spider="$2"; shift 2 ;;
    --no-compose) compose=0; shift ;;
    --compose) compose=1; shift ;;
    --sam3-preflight-only) sam3_preflight_only=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$dataset_root" ]]; then
  echo "error: set AOE_DATA_ROOT or pass --dataset-root" >&2
  exit 2
fi

[[ -n "$scene" ]] || { echo "missing --scene" >&2; exit 2; }
[[ -n "$segment" ]] || { echo "missing --segment" >&2; exit 2; }
[[ -n "$annotation_id" ]] || { echo "missing --annotation-id" >&2; exit 2; }
[[ -n "$object_name" ]] || { echo "missing --object-name" >&2; exit 2; }
[[ -n "$task" ]] || { echo "missing --task" >&2; exit 2; }
[[ -n "$hand_type" ]] || { echo "missing --hand-type" >&2; exit 2; }
[[ -n "$anchor_hand" ]] || { echo "missing --anchor-hand" >&2; exit 2; }
if [[ -z "$ego_objects" ]]; then
  ego_objects="$object_name"
fi

date_tag="$(date +%Y%m%d)"
if [[ -z "$source_run" ]]; then
  source_run="${scene}_full_source_${date_tag}_fresh01"
fi
if [[ -z "$matrix_run" ]]; then
  matrix_run="${scene}_matrix12_${date_tag}_fresh01"
fi

exp="$repo_root/experiments/$source_run"
matrix_exp="$repo_root/experiments/$matrix_run"
dai_clip="$exp/intermediates/trajectory_6dof/do_as_i_do/reconstruction/clip"
raw_video="$dataset_root/$segment/ego_process/ego_undistorted_video/raw_video_undistorted.mp4"
ego_clip_id="${segment}__a${annotation_id}__$(echo "$task" | tr ' ' '_')"
plan="$repo_root/experiments/${source_run}_fresh_plan.txt"
run_command="$(printf '%q ' "$0" "${original_args[@]}")"
git_commit="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)"
git_status_short="$(git -C "$repo_root" status --short 2>/dev/null || true)"
git_dirty_diff_sha256="$(git -C "$repo_root" diff --no-ext-diff --binary 2>/dev/null | sha256sum | awk '{print $1}' || true)"

if [[ -e "$exp" ]]; then
  echo "Refusing to reuse existing source run: $exp" >&2
  exit 3
fi
if [[ -e "$matrix_exp" ]]; then
  echo "Refusing to reuse existing matrix run: $matrix_exp" >&2
  exit 3
fi

mkdir -p "$exp/logs"
if [[ "$sam3_preflight_only" != "1" ]]; then
  mkdir -p "$matrix_exp/logs"
fi
run_outcome="$exp/fresh_run_outcome.json"
current_stage="initializing"
record_run_outcome() {
  local rc=$?
  local status="failed"
  trap - EXIT HUP INT TERM
  if [[ "$rc" -eq 0 ]]; then
    status="ok"
  elif [[ "$rc" -eq 129 || "$rc" -eq 130 || "$rc" -eq 143 ]]; then
    status="aborted"
  fi
  "$python_ret" - "$run_outcome" "$status" "$rc" "$current_stage" "$source_run" "$matrix_run" <<'PY' || true
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, status, return_code, stage, source_run, matrix_run = sys.argv[1:]
payload = {
    "status": status,
    "return_code": int(return_code),
    "stage": stage,
    "source_run": source_run,
    "matrix_run": matrix_run,
    "finished_at": datetime.now(timezone.utc).isoformat(),
}
termination_signals = {129: "SIGHUP", 130: "SIGINT", 143: "SIGTERM"}
if int(return_code) in termination_signals:
    payload["termination_signal"] = termination_signals[int(return_code)]
Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  exit "$rc"
}
handle_signal() {
  local rc="$1"
  trap - HUP INT TERM
  exit "$rc"
}
trap record_run_outcome EXIT
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
{
  echo "run_command=$run_command"
  echo "git_commit=$git_commit"
  echo "git_dirty=$([[ -n "$git_status_short" ]] && echo 1 || echo 0)"
  echo "git_dirty_diff_sha256=$git_dirty_diff_sha256"
  echo "git_status_short_begin"
  printf '%s\n' "$git_status_short"
  echo "git_status_short_end"
  echo "scene=$scene"
  echo "source_run=$source_run"
  echo "matrix_run=$matrix_run"
  echo "dataset_root=$dataset_root"
  echo "segment=$segment"
  echo "annotation_id=$annotation_id"
  echo "raw_rgb=$raw_video"
  echo "dai_fresh_clip=$dai_clip"
  echo "egoinfinity_output=$exp/intermediates/egoinfinity/clip"
  echo "do_as_i_do_reconstruction_output=$exp/intermediates/trajectory_6dof/do_as_i_do/reconstruction"
  echo "do_as_i_do_retargeting_output=$exp/intermediates/retargeting/do_as_i_do"
  echo "spider_output=$matrix_exp/intermediates/retargeting/spider"
  echo "object_name=$object_name"
  echo "ego_objects=$ego_objects"
  echo "task=$task"
  echo "hand_type=$hand_type"
  echo "anchor_hand=$anchor_hand"
  echo "start_offset_sec=$start_offset_sec"
  echo "main_cuda=$main_cuda"
  echo "dai_cuda_visible_devices=$dai_cuda_visible_devices"
  echo "sam3_cuda=$sam3_cuda"
  echo "sam3d_cuda=$sam3d_cuda"
  echo "sam3d_min_free_mb=$sam3d_min_free_mb"
  echo "sam3d_wait_for_gpu=$sam3d_wait_for_gpu"
  echo "sam3d_wait_timeout_sec=$sam3d_wait_timeout_sec"
  echo "sam3d_wait_poll_sec=$sam3d_wait_poll_sec"
  echo "spider_cuda_visible_devices=$spider_cuda_visible_devices"
  echo "run_spider=$run_spider"
  echo "sam3_preflight_only=$sam3_preflight_only"
  echo "sam3_python=$sam3_python"
  echo "sam3_repo=$sam3_repo"
  echo "sam3d_python=$sam3d_python"
  echo "sam3d_repo=$sam3d_repo"
  echo "dinov2_local_repo=$dinov2_local_repo"
  echo "egoinfinity_sam3_version=$egoinfinity_sam3_version"
} | tee "$plan"

export AOE_FRESH_RUN_COMMAND="$run_command"
export AOE_FRESH_GIT_COMMIT="$git_commit"
export AOE_FRESH_GIT_STATUS_SHORT="$git_status_short"
export AOE_FRESH_GIT_DIRTY_DIFF_SHA256="$git_dirty_diff_sha256"
export AOE_FRESH_SCENE="$scene"
export AOE_FRESH_SOURCE_RUN="$source_run"
export AOE_FRESH_MATRIX_RUN="$matrix_run"
export AOE_FRESH_DATASET_ROOT="$dataset_root"
export AOE_FRESH_SEGMENT="$segment"
export AOE_FRESH_ANNOTATION_ID="$annotation_id"
export AOE_FRESH_RAW_RGB="$raw_video"
export AOE_FRESH_OBJECT_NAME="$object_name"
export AOE_FRESH_EGO_OBJECTS="$ego_objects"
export AOE_FRESH_TASK="$task"
export AOE_FRESH_HAND_TYPE="$hand_type"
export AOE_FRESH_ANCHOR_HAND="$anchor_hand"
export AOE_FRESH_MAIN_CUDA="$main_cuda"
export AOE_FRESH_DAI_CUDA_VISIBLE_DEVICES="$dai_cuda_visible_devices"
export AOE_FRESH_SAM3_CUDA="$sam3_cuda"
export AOE_FRESH_SAM3D_CUDA="$sam3d_cuda"
export AOE_FRESH_SPIDER_CUDA_VISIBLE_DEVICES="$spider_cuda_visible_devices"
export AOE_FRESH_SPIDER_DEVICE="$spider_device"

"$python_ret" - "$exp/fresh_run_metadata.json" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "run_command": os.environ.get("AOE_FRESH_RUN_COMMAND", ""),
    "git": {
        "commit": os.environ.get("AOE_FRESH_GIT_COMMIT", ""),
        "dirty": bool(os.environ.get("AOE_FRESH_GIT_STATUS_SHORT", "")),
        "dirty_diff_sha256": os.environ.get("AOE_FRESH_GIT_DIRTY_DIFF_SHA256", ""),
        "status_short": os.environ.get("AOE_FRESH_GIT_STATUS_SHORT", ""),
    },
    "runs": {
        "scene": os.environ.get("AOE_FRESH_SCENE", ""),
        "source_run": os.environ.get("AOE_FRESH_SOURCE_RUN", ""),
        "matrix_run": os.environ.get("AOE_FRESH_MATRIX_RUN", ""),
    },
    "dataset": {
        "dataset_root": os.environ.get("AOE_FRESH_DATASET_ROOT", ""),
        "segment": os.environ.get("AOE_FRESH_SEGMENT", ""),
        "annotation_id": os.environ.get("AOE_FRESH_ANNOTATION_ID", ""),
        "raw_rgb": os.environ.get("AOE_FRESH_RAW_RGB", ""),
    },
    "task": {
        "object_name": os.environ.get("AOE_FRESH_OBJECT_NAME", ""),
        "ego_objects": os.environ.get("AOE_FRESH_EGO_OBJECTS", ""),
        "task": os.environ.get("AOE_FRESH_TASK", ""),
        "hand_type": os.environ.get("AOE_FRESH_HAND_TYPE", ""),
        "anchor_hand": os.environ.get("AOE_FRESH_ANCHOR_HAND", ""),
    },
    "gpu_mapping": {
        "main_cuda": os.environ.get("AOE_FRESH_MAIN_CUDA", ""),
        "dai_cuda_visible_devices": os.environ.get("AOE_FRESH_DAI_CUDA_VISIBLE_DEVICES", ""),
        "sam3_cuda": os.environ.get("AOE_FRESH_SAM3_CUDA", ""),
        "sam3d_cuda": os.environ.get("AOE_FRESH_SAM3D_CUDA", ""),
        "spider_cuda_visible_devices": os.environ.get("AOE_FRESH_SPIDER_CUDA_VISIBLE_DEVICES", ""),
        "spider_device": os.environ.get("AOE_FRESH_SPIDER_DEVICE", ""),
    },
    "environment": {
        key: os.environ.get(key)
        for key in [
            "MAIN_CUDA",
            "DAI_CUDA_VISIBLE_DEVICES",
            "SAM3_WORKER_CUDA",
            "SAM3D_WORKER_CUDA",
            "SPIDER_CUDA_VISIBLE_DEVICES",
            "SPIDER_DEVICE",
            "FULL12_RUN_SPIDER",
        ]
    },
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"fresh_run_metadata": str(path)}, indent=2))
PY

current_stage="prepare_fresh_aoe_clip"
echo "=== Preflight old-output guard ===" | tee "$exp/logs/softlink_check_pre.log"
find "$exp" -type l -ls 2>/dev/null | tee -a "$exp/logs/softlink_check_pre.log"

"$python_ret" "$repo_root/scripts/prepare_fresh_aoe_clip.py" \
  --dataset-root "$dataset_root" \
  --segment "$segment" \
  --annotation-id "$annotation_id" \
  --run-name "$source_run" \
  --task "$task" \
  --object-name "$object_name" \
  --anchor-hand "$anchor_hand" \
  --start-offset-sec "$start_offset_sec" \
  --duration-sec "$duration_sec" \
  --extra-sec-after "$extra_sec_after" \
  --ego-fps 15 \
  --ref-frame-fraction "$ref_frame_fraction" \
  --ref-source-frame "$ref_source_frame" \
  --scale-width "$scale_width" \
  2>&1 | tee "$exp/logs/prepare_fresh_aoe_clip.log"

ref_frame="$("$python_ret" - <<PY
import json
from pathlib import Path
print(json.loads(Path("$dai_clip/metadata.json").read_text())["ref_frame_in_clip"])
PY
)"
ego_start="$("$python_ret" - <<PY
import json
from pathlib import Path
meta=json.loads(Path("$dai_clip/metadata.json").read_text())
print(meta["start_sec"])
PY
)"
ego_end="$("$python_ret" - <<PY
import json
from pathlib import Path
meta=json.loads(Path("$dai_clip/metadata.json").read_text())
print(meta["start_sec"] + meta["duration"])
PY
)"
sam3_bbox="$("$python_ret" - <<PY
import json
from pathlib import Path
meta=json.loads(Path("$dai_clip/metadata.json").read_text())
bbox=(meta.get("action") or {}).get("bbox") or []
print(",".join(str(int(x)) for x in bbox))
PY
)"
sam3_point="$("$python_ret" - <<PY
import json
from pathlib import Path
meta=json.loads(Path("$dai_clip/metadata.json").read_text())
bbox=(meta.get("action") or {}).get("bbox") or []
if len(bbox) == 4:
    print(f"{(float(bbox[0]) + float(bbox[2])) * 0.5:.1f},{(float(bbox[1]) + float(bbox[3])) * 0.5:.1f}")
else:
    print("")
PY
)"
if [[ -n "${SAM3_PROMPT_POINTS:-}" ]]; then
  echo "error: production fresh runs require semantic SAM3 text prompts; SAM3_PROMPT_POINTS is diagnostic-only" >&2
  exit 2
fi
sam3_prompt_points=""
ego_source_point="$("$python_ret" - <<PY
import json
from pathlib import Path
meta=json.loads(Path("$dai_clip/metadata.json").read_text())
transform=meta.get("bbox_coordinate_transform") or {}
bbox=transform.get("original_bbox") or []
source_size=transform.get("source_video_size") or []
annotation_size=transform.get("annotation_size") or []
if len(bbox) == 4 and len(source_size) == 2 and len(annotation_size) == 2:
    x=(float(bbox[0]) + float(bbox[2])) * 0.5 * float(source_size[0]) / max(float(annotation_size[0]), 1.0)
    y=(float(bbox[1]) + float(bbox[3])) * 0.5 * float(source_size[1]) / max(float(annotation_size[1]), 1.0)
    print(f"{x:.1f},{y:.1f}")
else:
    print("")
PY
)"
egoinfinity_target_point="${EGOINFINITY_TARGET_POINT:-$ego_source_point}"
egoinfinity_selection_mode="${EGOINFINITY_OBJECT_SELECTION_MODE:-}"
if [[ -z "$egoinfinity_selection_mode" ]]; then
  if [[ -n "$egoinfinity_target_point" ]]; then
    egoinfinity_selection_mode="best"
  else
    egoinfinity_selection_mode="all"
  fi
fi
echo "sam3_target_bbox=${sam3_bbox:-<none>}" | tee -a "$plan"
echo "sam3_bbox_center=${sam3_point:-<none>}" | tee -a "$plan"
echo "sam3_prompt_points=${sam3_prompt_points:-<text-prompt>}" | tee -a "$plan"
echo "ego_source_target_point=${ego_source_point:-<none>}" | tee -a "$plan"
echo "egoinfinity_target_point=${egoinfinity_target_point:-<none>}" | tee -a "$plan"
echo "egoinfinity_object_selection_mode=$egoinfinity_selection_mode" | tee -a "$plan"

set +e
current_stage="do_as_i_do_fresh_reconstruction"
SAM3_PROMPT_POINTS="$sam3_prompt_points" \
SAM3_PROMPT_POINT_LABELS="1" \
SAM3_TARGET_BBOX="$sam3_bbox" \
SAM3_REQUIRE_TEXT_PROMPT="1" \
DAI_RECON_STOP_AFTER_SAM3="$sam3_preflight_only" \
SAM3_CUDA="$sam3_cuda" \
SAM3D_CUDA="$sam3d_cuda" \
SAM3D_MIN_FREE_MB="$sam3d_min_free_mb" \
SAM3D_WAIT_FOR_GPU="$sam3d_wait_for_gpu" \
SAM3D_WAIT_TIMEOUT_SEC="$sam3d_wait_timeout_sec" \
SAM3D_WAIT_POLL_SEC="$sam3d_wait_poll_sec" \
CUDA_VISIBLE_DEVICES="$main_cuda" \
  "$repo_root/scripts/run_do_as_i_do_reconstruction_fresh.sh" \
    "$dai_clip/raw.mp4" "$ref_frame" "$object_name" "$anchor_hand" \
  2>&1 | tee "$exp/logs/do_as_i_do_fresh_reconstruction.log"
reconstruction_rc="${PIPESTATUS[0]}"
set -e
if [[ "$reconstruction_rc" -ne 0 ]]; then
  echo "Fresh Do-as-I-Do reconstruction failed: rc=$reconstruction_rc" >&2
  exit "$reconstruction_rc"
fi

current_stage="fresh_raw_video_attestation"
"$python_ret" "$repo_root/scripts/materialize_reused_raw_video.py" \
  --source-dir "$dai_clip" \
  --attest-existing \
  2>&1 | tee "$exp/logs/fresh_raw_video_attestation.log"

if [[ "$sam3_preflight_only" == "1" ]]; then
  current_stage="sam3_preflight_postflight"
  "$python_ret" - "$exp" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
clip = root / "intermediates/trajectory_6dof/do_as_i_do/reconstruction/clip"
payload = {
    "status": "ok",
    "mode": "sam3_preflight_only",
    "fresh_run_metadata": str(root / "fresh_run_metadata.json"),
    "clip_metadata": str(clip / "metadata.json"),
    "prompt_gate": str(clip / "video_segmentation/prompt_gate.json"),
    "mask_qc": str(clip / "video_segmentation/mask_qc_summary.json"),
    "tracked_video": None,
}
prompt_gate_path = Path(payload["prompt_gate"])
mask_qc_path = Path(payload["mask_qc"])
tracked = sorted((clip / "video_segmentation").glob("tracked_*.mp4"))
if tracked:
    payload["tracked_video"] = str(tracked[0])
for key in ("fresh_run_metadata", "clip_metadata", "prompt_gate", "mask_qc", "tracked_video"):
    value = payload.get(key)
    if value and not Path(value).is_file():
        raise SystemExit(f"SAM3 preflight missing {key}: {value}")
prompt_gate = json.loads(prompt_gate_path.read_text(encoding="utf-8"))
mask_qc = json.loads(mask_qc_path.read_text(encoding="utf-8"))
gate_errors = prompt_gate.get("errors") or []
if prompt_gate.get("status") != "ok" or gate_errors:
    raise SystemExit(
        "SAM3 preflight prompt gate failed: "
        f"status={prompt_gate.get('status')} errors={gate_errors}"
    )
if mask_qc.get("qa_pass") is not True:
    raise SystemExit(
        "SAM3 preflight mask QC failed: "
        f"reasons={mask_qc.get('fail_reasons') or []}"
    )
(root / "sam3_preflight_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
  echo "SAM3 preflight complete: $exp/sam3_preflight_manifest.json"
  exit 0
fi

current_stage="full_12_demos"
full12_compose_args=()
if [[ "$compose" -eq 0 ]]; then
  full12_compose_args+=(--no-compose)
fi
FULL12_PYTHON="$python_ret" \
RETARGETING_PYTHON="$python_ret" \
EGOINFINITY_PYTHON="$python_ego" \
SAM3_PYTHON="$sam3_python" \
SAM3_REPO="$sam3_repo" \
SAM3D_PYTHON="$sam3d_python" \
SAM3D_REPO="$sam3d_repo" \
DINOV2_LOCAL_REPO="$dinov2_local_repo" \
DINOV2_REPO_DIR="$dinov2_local_repo" \
EGOINFINITY_SAM3_VERSION="$egoinfinity_sam3_version" \
MAIN_CUDA="$main_cuda" \
DAI_CUDA_VISIBLE_DEVICES="$dai_cuda_visible_devices" \
SAM3_WORKER_CUDA="$sam3_cuda" \
SAM3D_WORKER_CUDA="$sam3d_cuda" \
SPIDER_PYTHON="$spider_python" \
SPIDER_CUDA_VISIBLE_DEVICES="$spider_cuda_visible_devices" \
SPIDER_DEVICE="$spider_device" \
SPIDER_MAX_SIM_STEPS="$spider_max_sim_steps" \
SPIDER_NUM_SAMPLES="$spider_num_samples" \
SPIDER_MAX_NUM_ITERATIONS="$spider_max_num_iterations" \
V4_DAI_RETARGET_HAND_TYPE="$hand_type" \
EGOINFINITY_TARGET_POINT="$egoinfinity_target_point" \
EGOINFINITY_OBJECT_SELECTION_MODE="$egoinfinity_selection_mode" \
  "$repo_root/scripts/run_full_12_demos.sh" \
    --run-name "$source_run" \
    --matrix-run-name "$matrix_run" \
    --task "$task" \
    --hand-type "$hand_type" \
    --ego-video "$raw_video" \
    --ego-clip-id "$ego_clip_id" \
    --ego-objects "$ego_objects" \
    --ego-start "$ego_start" \
    --ego-end "$ego_end" \
    --ego-fps "15" \
    --dai-raw-dir "$dai_clip" \
    --dai-clip-dir "$dai_clip" \
    --dai-cuda-visible-devices "$dai_cuda_visible_devices" \
    --run-spider "$run_spider" \
    "${full12_compose_args[@]}" \
    --no-spider-fallback \
    --force-spider \
    --duration 0 \
  2>&1 | tee "$exp/logs/run_full_12_demos_fresh.log"

{
  echo "=== readlink -f EgoInfinity clip ==="
  readlink -f "$exp/intermediates/egoinfinity/clip" || true
  echo "=== source run symlinks ==="
  find "$exp" -type l -ls || true
  echo "=== banned-link scan ==="
  if find "$exp" -type l -print0 | xargs -0 -r readlink -f | grep -E "experiments/.*20260629|openaoe-method-pilots/egoinfinity_native_aoe"; then
    echo "BANNED_SOURCE_LINK_FOUND"
    exit 9
  fi
  echo "OK: no source symlink resolves to old 20260629 outputs or openaoe-method-pilots EgoInfinity outputs"
} | tee "$exp/logs/softlink_check_post.log"

current_stage="strict_postflight_audit"
echo "=== Strict 12-cell postflight audit ==="
postflight_audit_args=(
  "$python_ret" "$repo_root/scripts/audit_retarget_run.py"
  --experiments-root "$repo_root/experiments"
  --matrix-run "$matrix_run"
  --source-run "$source_run"
  --output-json "$matrix_exp/audit_retarget_run.json"
)
if [[ "$compose" -eq 0 ]]; then
  postflight_audit_args+=(--no-require-triptych)
fi
"${postflight_audit_args[@]}" \
  2>&1 | tee "$matrix_exp/logs/audit_retarget_run.log"

echo "source_run=$exp"
echo "matrix_run=$matrix_exp"
echo "videos=$matrix_exp/videos"
echo "manifest=$matrix_exp/reuse_12_demo_manifest.json"
current_stage="complete"
