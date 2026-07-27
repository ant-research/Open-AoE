#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${EGOINFINITY_PYTHON:-python3}"
sam3_python="${SAM3_PYTHON:-python3}"
sam3_repo="${SAM3_REPO:-${repo_root}/third_party/sam3}"
sam3d_python="${SAM3D_PYTHON:-python3}"
sam3d_repo="${SAM3D_REPO:-${repo_root}/third_party/sam-3d-objects}"
dinov2_local_repo="${DINOV2_LOCAL_REPO:-${repo_root}/third_party/dinov2}"
hf_home="${HF_HOME:-${HOME}/.cache/huggingface}"
hf_hub_cache="${HUGGINGFACE_HUB_CACHE:-$hf_home/hub}"
transformers_cache="${TRANSFORMERS_CACHE:-$hf_home/hub}"
hf_hub_offline="${HF_HUB_OFFLINE:-1}"
transformers_offline="${TRANSFORMERS_OFFLINE:-1}"
run_name="foundation_jar_visual_demo_v1"
clip_dir=""
clip_id="poc_raw_video_20260201_193100_part003__a1__grasp__foundation_jar"
objects="foundation jar"
start_sec="0.0"
end_sec=""
fps="15"
robot="g1"
force_arg=()
set_args=("--set" "phase1.args.with_sam3d_worker=true")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-name) run_name="$2"; shift 2 ;;
    --clip-dir) clip_dir="$2"; shift 2 ;;
    --clip-id) clip_id="$2"; shift 2 ;;
    --objects) objects="$2"; shift 2 ;;
    --start) start_sec="$2"; shift 2 ;;
    --end) end_sec="$2"; shift 2 ;;
    --fps) fps="$2"; shift 2 ;;
    --robot) robot="$2"; shift 2 ;;
    --force) force_arg=(--force "export_retarget_samples,retarget" --cascade); shift ;;
    --no-sam3d-worker) set_args=("--set" "phase1.args.with_sam3d_worker=false"); shift ;;
    -h|--help)
      echo "Usage: $0 --clip-dir PATH [--run-name NAME] [--robot g1] [--force]"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$clip_dir" ]]; then
  echo "Missing --clip-dir" >&2
  exit 2
fi

exp="$repo_root/experiments/$run_name"
work_clip="$exp/intermediates/egoinfinity/clip"
traj_clip="$exp/intermediates/trajectory_6dof/egoinfinity/clip"
source_link="$exp/intermediates/trajectory_6dof/egoinfinity/source"
mkdir -p "$(dirname "$work_clip")" "$(dirname "$traj_clip")" "$exp/logs"
if [[ "${EGOINFINITY_RESET_OUTPUT:-0}" == "1" ]]; then
  rm -rf "$work_clip"
fi
rm -rf "$traj_clip" "$source_link"

if [[ -f "$clip_dir" ]]; then
  ln -s "$clip_dir" "$source_link"
  process_target="$clip_dir"
  output_args=(--output "$work_clip" --clip-id "$clip_id" --objects "$objects" --start "$start_sec" --fps "$fps")
  if [[ -n "$end_sec" ]]; then output_args+=(--end "$end_sec"); fi
elif [[ -d "$clip_dir" ]]; then
  echo "EgoInfinity retargeting needs a raw RGB video path so outputs can be created inside experiments/." >&2
  echo "Got existing artifact dir: $clip_dir" >&2
  echo "Pass --clip-dir /path/to/raw_video.mp4 instead." >&2
  exit 2
else
  echo "Missing EgoInfinity input: $clip_dir" >&2
  exit 2
fi

valid_object_mesh_result() {
  local result_path="$1"
  [[ -f "$result_path" ]] || return 1
  "$python_bin" - "$result_path" <<'PY'
import gzip
import pickle
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
try:
    with gzip.open(result_path, "rb") as handle:
        result = pickle.load(handle)
except Exception:
    raise SystemExit(1)

mesh_info = result.get("sam3_mesh_info") or {}
pose_info_all = result.get("pose_track_info") or {}
for obj_id, info in mesh_info.items():
    if not isinstance(info, dict):
        continue
    pose_info = pose_info_all.get(obj_id) or pose_info_all.get(str(obj_id))
    if not isinstance(pose_info, dict):
        continue
    ply_path = Path(str(info.get("ply_path", "")))
    transforms = pose_info.get("T_seq")
    if ply_path.exists() and transforms is not None and len(transforms) > 0:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

if [[ "${EGOINFINITY_KEEP_BROKEN_CACHE:-0}" != "1" && -f "$work_clip/pipeline_result.pkl.gz" ]]; then
  if ! valid_object_mesh_result "$work_clip/pipeline_result.pkl.gz"; then
    echo "Existing EgoInfinity result has no reconstructed object mesh + pose; resetting $work_clip" >&2
    rm -rf "$work_clip"
  fi
fi
ln -s "$work_clip" "$traj_clip"

cd "$repo_root/third_party/EgoInfinity"
PYTHONPATH="$repo_root/scripts/python_compat:$repo_root/third_party/EgoInfinity:$repo_root/third_party/EgoInfinity/third_party/sam2:$repo_root/third_party/EgoInfinity/third_party" \
HF_HOME="$hf_home" \
HUGGINGFACE_HUB_CACHE="$hf_hub_cache" \
TRANSFORMERS_CACHE="$transformers_cache" \
HF_HUB_OFFLINE="$hf_hub_offline" \
TRANSFORMERS_OFFLINE="$transformers_offline" \
TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}" \
SAM3_PYTHON="$sam3_python" \
SAM3_REPO="$sam3_repo" \
SAM3D_PYTHON="$sam3d_python" \
SAM3D_REPO="$sam3d_repo" \
DINOV2_LOCAL_REPO="$dinov2_local_repo" \
DINOV2_REPO_DIR="$dinov2_local_repo" \
EGOINFINITY_SKIP_DEPTH_STABILIZE="${EGOINFINITY_SKIP_DEPTH_STABILIZE:-1}" \
EGOINFINITY_SKIP_BAKE_FP="${EGOINFINITY_SKIP_BAKE_FP:-1}" \
SAM3D_SPAWN_TIMEOUT="${SAM3D_SPAWN_TIMEOUT:-900}" \
MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
  "$python_bin" -m egoinfinity process "$process_target" "${output_args[@]}" --robot "$robot" "${set_args[@]}" --fail-fast "${force_arg[@]}" \
  2>&1 | tee "$exp/logs/retarget_egoinfinity_${robot}.log"

if [[ -f "$work_clip/pipeline_result.pkl.gz" ]]; then
  overlay_select_best_args=()
  if [[ -n "$objects" && "${EGOINFINITY_OVERLAY_SELECT_BEST_OBJECT:-1}" != "0" ]]; then
    overlay_select_best_args+=(--select-best-object)
  fi

  if [[ "${EGOINFINITY_POST_SCALE_SANITY:-1}" != "0" ]]; then
    scale_tmp="$(mktemp -d /tmp/egoinfinity_scale_sanity.XXXXXX)"
    mkdir -p "$scale_tmp/favorites"
    ln -s "$work_clip" "$scale_tmp/favorites/clip"
    PYTHONPATH="$repo_root/scripts/python_compat:$repo_root/third_party/EgoInfinity:$repo_root/third_party/EgoInfinity/third_party/sam2:$repo_root/third_party/EgoInfinity/third_party" \
    ACTION100M_CACHE="$scale_tmp" \
      "$python_bin" -m egoinfinity.pipeline.post_tracking.scale_sanity \
        --only=clip \
        --threshold "${EGOINFINITY_SCALE_SANITY_THRESHOLD:-1.35}" \
      2>&1 | tee "$exp/logs/egoinfinity_scale_sanity.log"
    rm -rf "$scale_tmp"
  fi

  if ! "$python_bin" "$repo_root/scripts/review/render_egoinfinity_mask_diagnostics.py" \
    --pipeline-result "$work_clip/pipeline_result.pkl.gz" \
    --output "$work_clip/mask_overlay.mp4" \
    --fps "$fps" \
    --target-prompt "$objects" \
    "${overlay_select_best_args[@]}" \
    2>&1 | tee "$exp/logs/egoinfinity_mask_overlay.log"; then
    echo "[warn] EgoInfinity mask diagnostics failed; continuing with mesh overlays and retargeting outputs." \
      | tee -a "$exp/logs/egoinfinity_mask_overlay.log" >&2
  fi

  "$python_bin" "$repo_root/scripts/review/render_egoinfinity_rgb_overlay.py" \
    --pipeline-result "$work_clip/pipeline_result.pkl.gz" \
    --output "$work_clip/rgb_mesh_overlay.mp4" \
    --fps "$fps" \
    --target-prompt "$objects" \
    --max-centroid-jump-px "${EGOINFINITY_MAX_CENTROID_JUMP_PX:-320}" \
    "${overlay_select_best_args[@]}" \
    2>&1 | tee "$exp/logs/egoinfinity_rgb_overlay.log"

  "$python_bin" "$repo_root/scripts/review/render_egoinfinity_rgb_overlay.py" \
    --pipeline-result "$work_clip/pipeline_result.pkl.gz" \
    --output "$work_clip/mesh_pure_camera.mp4" \
    --fps "$fps" \
    --target-prompt "$objects" \
    --max-centroid-jump-px "${EGOINFINITY_MAX_CENTROID_JUMP_PX:-320}" \
    "${overlay_select_best_args[@]}" \
    --background black \
    2>&1 | tee "$exp/logs/egoinfinity_mesh_pure_camera.log"
else
  echo "Missing EgoInfinity pipeline_result.pkl.gz; cannot render RGB overlay." >&2
  exit 1
fi
