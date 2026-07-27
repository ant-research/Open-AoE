#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
recon_root="${DAI_RECON_ROOT:-$repo_root/third_party/do-as-i-do/reconstruction}"
script_dir="${DAI_RECON_SCRIPTS_DIR:-$recon_root/scripts}"
integration_script_dir="$repo_root/scripts/compatibility/reconstruction"
module_root="${DAI_RECON_MODULE_ROOT:-$recon_root/modules}"
if [[ ! -d "$module_root/Fast-SAM3D" ]]; then
  module_root="${DAI_RECON_MODULE_ROOT_FALLBACK:-$module_root}"
fi

sam3d_env="${SAM3D_CONDA_PREFIX:-${CONDA_PREFIX:-}}"
py="${SAM3D_PYTHON:-python3}"
sam3_py="${SAM3_PYTHON:-python3}"
default_cuda="${CUDA_VISIBLE_DEVICES:-0}"
sam3_cuda="${SAM3_CUDA:-${SAM3_WORKER_CUDA:-$default_cuda}}"
sam3d_cuda="${SAM3D_CUDA:-${SAM3D_WORKER_CUDA:-$default_cuda}}"
sam3_min_free_mb="${SAM3_MIN_FREE_MB:-12000}"
sam3_wait_for_gpu="${SAM3_WAIT_FOR_GPU:-0}"
sam3_wait_timeout_sec="${SAM3_WAIT_TIMEOUT_SEC:-0}"
sam3_wait_poll_sec="${SAM3_WAIT_POLL_SEC:-60}"
sam3d_min_free_mb="${SAM3D_MIN_FREE_MB:-0}"
sam3d_wait_for_gpu="${SAM3D_WAIT_FOR_GPU:-0}"
sam3d_wait_timeout_sec="${SAM3D_WAIT_TIMEOUT_SEC:-0}"
sam3d_wait_poll_sec="${SAM3D_WAIT_POLL_SEC:-60}"
fastsam3d_latent_opt_steps="${DAI_FASTSAM3D_LATENT_OPT_STEPS:-250}"
fastsam3d_latent_opt_lr="${DAI_FASTSAM3D_LATENT_OPT_LR:-0.05}"
sam3_require_text_prompt="${SAM3_REQUIRE_TEXT_PROMPT:-0}"
stop_after_sam3="${DAI_RECON_STOP_AFTER_SAM3:-0}"
geocalib_dir="${GEOCALIB_DIR:-$repo_root/third_party/GeoCalib}"
geocalib_max_frames="${DAI_GEOCALIB_MAX_FRAMES:-16}"
geocalib_max_aggregate_p95_deg="${DAI_GEOCALIB_MAX_AGGREGATE_P95_DEG:-25.0}"
require_geocalib="${DAI_REQUIRE_GEOCALIB:-1}"

video_path="${1:?usage: run_do_as_i_do_reconstruction_fresh.sh VIDEO_PATH FRAME_N OBJECT ANCHOR_HAND}"
frame_n="${2:?usage: run_do_as_i_do_reconstruction_fresh.sh VIDEO_PATH FRAME_N OBJECT ANCHOR_HAND}"
object_name="${3:?usage: run_do_as_i_do_reconstruction_fresh.sh VIDEO_PATH FRAME_N OBJECT ANCHOR_HAND}"
anchor_hand="${4:?usage: run_do_as_i_do_reconstruction_fresh.sh VIDEO_PATH FRAME_N OBJECT ANCHOR_HAND}"
object_id="${object_name// /_}"

video_path="$(realpath "$video_path")"
video_dir="$(dirname "$video_path")"
video_name="$(basename "$video_path")"
video_name="${video_name%%.*}"
frame_path="$video_dir/$(printf "%04d.png" "$frame_n")"
pointmap_path="$video_dir/$(printf "%04d_pointmap.npy" "$frame_n")"
masks_dir="$video_dir/video_segmentation/masks/frame_$(printf "%06d" "$frame_n")_masks"
video_masks_dir="$video_dir/video_segmentation/masks"
hand_meshes_path="$video_dir/$video_name/all_hand_meshes.npz"

export RECON_ROOT="$recon_root"
export SAM3D_DIR="${SAM3D_DIR:-$module_root/sam-3d-objects}"
external_sam3d_dir="${SAM3D_REPO:-}"
export FASTSAM3D_DIR="${FASTSAM3D_DIR:-$module_root/Fast-SAM3D}"
export HAWOR_DIR="${HAWOR_DIR:-$module_root/HaWoR}"
export TAPNET_DIR="${TAPNET_DIR:-$module_root/tapnet}"
export SAM3_PKG_DIR="${SAM3_PKG_DIR:-$module_root/sam3}"
export SAM3_VERSION="${SAM3_VERSION:-sam3}"
export SCRIPTS_DIR="$script_dir"
export SAM3D_REPO_ROOT="$FASTSAM3D_DIR"
export TAPNET_CKPT="${TAPNET_CKPT:-$recon_root/weights/tapnet/bootstapir_checkpoint_v2.pt}"
export CUDA_VISIBLE_DEVICES="$sam3d_cuda"
if [[ -n "$sam3d_env" ]]; then
  export CONDA_PREFIX="$sam3d_env"
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="$geocalib_dir:$script_dir/shims:$FASTSAM3D_DIR:$SAM3D_DIR:$HAWOR_DIR:$TAPNET_DIR:$SAM3_PKG_DIR:$script_dir"

run_sam3_py() {
  CUDA_VISIBLE_DEVICES="$sam3_cuda" "$sam3_py" "$@"
}

run_sam3d_py() {
  if [[ "${SAM3D_GDB_DIAGNOSTIC:-0}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$sam3d_cuda" gdb -batch \
      -ex run \
      -ex "thread apply all bt 24" \
      --args "$py" "$@"
  else
    CUDA_VISIBLE_DEVICES="$sam3d_cuda" "$py" "$@"
  fi
}

write_tapir_status() {
  local status_json="$1"
  local status="$2"
  local return_code="$3"
  local message="$4"
  local motion_stats_json="$5"
  local timeout_sec="$6"
  local required="$7"
  mkdir -p "$(dirname "$status_json")"
  "$py" - "$status_json" "$status" "$return_code" "$message" "$motion_stats_json" "$timeout_sec" "$required" "$video_path" "$video_masks_dir" "$object_id" "$TAPNET_CKPT" <<'PY'
import json
import sys
from pathlib import Path

status_json, status, return_code, message, motion_stats_json, timeout_sec, required, video, mask_dir, object_id, checkpoint = sys.argv[1:12]
try:
    rc = int(return_code)
except ValueError:
    rc = -1
try:
    timeout = float(timeout_sec)
except ValueError:
    timeout = 0.0
payload = {
    "stage": "tapir_velocity_tracking",
    "status": status,
    "return_code": rc,
    "message": message,
    "required": required == "1",
    "timeout_sec": timeout,
    "motion_stats_json": motion_stats_json,
    "motion_stats_exists": Path(motion_stats_json).is_file(),
    "command": {
        "script": "tapir_velocity_tracking.py",
        "video": video,
        "mask_dir": mask_dir,
        "object": object_id,
        "checkpoint": checkpoint,
    },
}
Path(status_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

select_gpu_by_free_memory() {
  local label="$1" candidates="$2" min_free_mb="$3" wait_for_gpu="$4"
  local wait_timeout_sec="$5" wait_poll_sec="$6" skip_check="$7" result_var="$8"
  [[ "$min_free_mb" =~ ^[0-9]+$ ]] || {
    echo "error: ${label}_MIN_FREE_MB must be an integer, got '$min_free_mb'" >&2
    exit 2
  }
  if (( min_free_mb <= 0 )); then
    return 0
  fi
  if [[ "$skip_check" == "1" ]]; then
    echo "[gpu-check] $label free-memory check explicitly skipped"
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu-check] nvidia-smi is unavailable; continuing without $label free-memory check"
    return 0
  fi

  local start_ts now_ts elapsed best_gpu best_free
  start_ts="$(date +%s)"

  while true; do
    best_gpu=""
    best_free="-1"
    IFS=',' read -r -a gpu_list <<< "$candidates"
    for gpu in "${gpu_list[@]}"; do
      gpu="$(echo "$gpu" | tr -d '[:space:]')"
      [[ -n "$gpu" ]] || continue
      local free_mb
      free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
      [[ "$free_mb" =~ ^[0-9]+$ ]] || {
        echo "[gpu-check] GPU $gpu free memory unavailable"
        continue
      }
      echo "[gpu-check] $label GPU $gpu free=${free_mb}MiB required=${min_free_mb}MiB"
      if (( free_mb > best_free )); then
        best_free="$free_mb"
        best_gpu="$gpu"
      fi
      if (( free_mb >= min_free_mb )); then
        printf -v "$result_var" '%s' "$gpu"
        echo "[gpu-check] selected GPU $gpu for $label"
        return 0
      fi
    done

    if [[ "$wait_for_gpu" != "1" ]]; then
      echo "error: no $label candidate GPU has ${min_free_mb}MiB free; best GPU ${best_gpu:-<none>} has ${best_free}MiB." >&2
      exit 86
    fi

    if (( wait_timeout_sec > 0 )); then
      now_ts="$(date +%s)"
      elapsed=$((now_ts - start_ts))
      if (( elapsed >= wait_timeout_sec )); then
        echo "error: timed out after ${wait_timeout_sec}s waiting for $label GPU memory; best GPU ${best_gpu:-<none>} has ${best_free}MiB." >&2
        exit 86
      fi
    fi
    echo "[gpu-check] waiting ${wait_poll_sec}s for $label GPU memory; best GPU ${best_gpu:-<none>} has ${best_free}MiB"
    sleep "$wait_poll_sec"
  done
}

select_sam3_gpu_if_needed() {
  select_gpu_by_free_memory \
    "SAM3" "$sam3_cuda" "$sam3_min_free_mb" "$sam3_wait_for_gpu" \
    "$sam3_wait_timeout_sec" "$sam3_wait_poll_sec" \
    "${SAM3_SKIP_FREE_MEM_CHECK:-0}" sam3_cuda
}

select_sam3d_gpu_if_needed() {
  select_gpu_by_free_memory \
    "SAM3D" "$sam3d_cuda" "$sam3d_min_free_mb" "$sam3d_wait_for_gpu" \
    "$sam3d_wait_timeout_sec" "$sam3d_wait_poll_sec" \
    "${SAM3D_SKIP_FREE_MEM_CHECK:-0}" sam3d_cuda
  export CUDA_VISIBLE_DEVICES="$sam3d_cuda"
}

echo "=== Do-as-I-Do fresh reconstruction provenance ==="
echo "repo_root=$repo_root"
echo "video_path=$video_path"
echo "video_dir=$video_dir"
echo "recon_root=$recon_root"
echo "module_root=$module_root"
echo "scripts_dir=$script_dir"
echo "integration_scripts_dir=$integration_script_dir"
echo "object_name=$object_name"
echo "object_id=$object_id"
echo "frame_n=$frame_n"
echo "anchor_hand=$anchor_hand"
echo "default_cuda=$default_cuda"
echo "sam3_cuda=$sam3_cuda"
echo "sam3d_cuda=$sam3d_cuda"
echo "SAM3_MIN_FREE_MB=$sam3_min_free_mb"
echo "SAM3_WAIT_FOR_GPU=$sam3_wait_for_gpu"
echo "SAM3_WAIT_TIMEOUT_SEC=$sam3_wait_timeout_sec"
echo "SAM3_WAIT_POLL_SEC=$sam3_wait_poll_sec"
echo "SAM3D_MIN_FREE_MB=$sam3d_min_free_mb"
echo "SAM3D_WAIT_FOR_GPU=$sam3d_wait_for_gpu"
echo "SAM3D_WAIT_TIMEOUT_SEC=$sam3d_wait_timeout_sec"
echo "SAM3D_WAIT_POLL_SEC=$sam3d_wait_poll_sec"
echo "DAI_FASTSAM3D_LATENT_OPT_STEPS=$fastsam3d_latent_opt_steps"
echo "DAI_FASTSAM3D_LATENT_OPT_LR=$fastsam3d_latent_opt_lr"
echo "SAM3_REQUIRE_TEXT_PROMPT=$sam3_require_text_prompt"
echo "DAI_RECON_STOP_AFTER_SAM3=$stop_after_sam3"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

echo "=== [0] Extracting frames ==="
mkdir -p "$video_dir/all_frames"
ffmpeg -hide_banner -loglevel error -y -i "$video_path" -vsync 0 -start_number 0 "$video_dir/all_frames/%06d.png"
ffmpeg -hide_banner -loglevel error -y -i "$video_path" -vf "select=eq(n\\,$frame_n)" -vsync 0 -vframes 1 "$frame_path"

echo "=== [1] SAM3 video masks before SAM3D ==="
select_sam3_gpu_if_needed
echo "SAM3_SELECTED_CUDA=$sam3_cuda"
sam3_args=(
  --video "$video_dir/all_frames"
  --obj_id "$object_id"
  --frame_idx "$frame_n"
  --output_dir "$video_dir/video_segmentation"
  --version "$SAM3_VERSION"
  --min-valid-ratio "${SAM3_MIN_VALID_RATIO:-0.70}"
  --max-centroid-jump-px "${SAM3_MAX_CENTROID_JUMP_PX:-320}"
)
if [[ -n "${SAM3_PROMPT_POINTS:-}" ]]; then
  sam3_args+=(--points "$SAM3_PROMPT_POINTS" --point_labels "${SAM3_PROMPT_POINT_LABELS:-1}")
else
  sam3_args+=(--text "$object_name")
fi
if [[ -n "${SAM3_TARGET_POINT:-}" ]]; then
  sam3_args+=(--target-point "$SAM3_TARGET_POINT" --select-mode point)
fi
if [[ -n "${SAM3_TARGET_BBOX:-}" ]]; then
  sam3_args+=(--target-bbox "$SAM3_TARGET_BBOX" --select-mode bbox)
fi
# Keep the pinned reconstruction checkout pristine. Retarget Lab owns the
# prompt-instance selection, temporal QA, and output packaging compatibility
# layer while still calling the upstream SAM3 model and predictor unchanged.
run_sam3_py "$repo_root/scripts/compatibility/run_sam3_video.py" "${sam3_args[@]}"

if [[ "$sam3_require_text_prompt" == "1" ]]; then
  sam3_qc_json="$video_dir/video_segmentation/mask_qc_summary.json"
  sam3_prompt_gate_json="$video_dir/video_segmentation/prompt_gate.json"
  "$sam3_py" "$repo_root/scripts/validate_sam3_prompt_gate.py" \
    --qc "$sam3_qc_json" \
    --output "$sam3_prompt_gate_json" \
    --expected-text "$object_name" \
    --expected-bbox "${SAM3_TARGET_BBOX:-}"
fi

if [[ "$stop_after_sam3" == "1" ]]; then
  echo "=== SAM3 preflight complete; stopping before SAM3D by request ==="
  exit 0
fi

echo "=== [2a] SAM3D masks -> object mesh ==="
select_sam3d_gpu_if_needed
echo "SAM3D_SELECTED_CUDA=$sam3d_cuda"
sam3d_config="${SAM3D_CONFIG:-}"
if [[ -z "$sam3d_config" ]]; then
  sam3d_config_roots=()
  if [[ -n "$external_sam3d_dir" ]]; then
    sam3d_config_roots+=("$external_sam3d_dir")
  fi
  sam3d_config_roots+=("$SAM3D_DIR")
  sam3d_config_suffixes=(
    "checkpoints/hf_local_moge_v1_vitl/pipeline.yaml"
    "checkpoints/hf_local_moge_v1_vitl/pipeline_hjd_meshres48.yaml"
    "checkpoints/hf/pipeline.yaml"
    "checkpoints/hf/pipeline_mesh_only.yaml"
  )
  for candidate in \
    "${sam3d_config_roots[@]}"; do
    for suffix in "${sam3d_config_suffixes[@]}"; do
      if [[ -f "$candidate/$suffix" ]]; then
        sam3d_config="$candidate/$suffix"
        break 2
      fi
    done
  done
fi
if [[ ! -f "$sam3d_config" ]]; then
  echo "error: missing SAM3D config under $SAM3D_DIR/checkpoints" >&2
  exit 2
fi
echo "SAM3D_CONFIG=$sam3d_config"
pointmap_config="${SAM3D_POINTMAP_CONFIG:-$sam3d_config}"
pointmap_config_root="$video_dir/.sam3d_pointmap_config_root"
mkdir -p "$pointmap_config_root/checkpoints/hf"
ln -sfn "$pointmap_config" "$pointmap_config_root/checkpoints/hf/pipeline.yaml"
fastsam3d_base_config="${FASTSAM3D_CONFIG:-$FASTSAM3D_DIR/checkpoints/hf/pipeline.yaml}"
fastsam3d_config_root="$video_dir/.fastsam3d_config_root"
fastsam3d_config="$fastsam3d_config_root/checkpoints/hf/pipeline.yaml"
local_moge_v1_model="${MOGE_V1_MODEL:-${SAM3D_MOGE_V1_MODEL:-Ruicheng/moge-vitl}}"
if [[ ! -f "$fastsam3d_base_config" ]]; then
  echo "error: missing Fast-SAM3D config: $fastsam3d_base_config" >&2
  exit 2
fi
mkdir -p "$fastsam3d_config_root/checkpoints/hf"
find "$FASTSAM3D_DIR/checkpoints/hf" -maxdepth 1 -mindepth 1 -exec ln -sfn {} "$fastsam3d_config_root/checkpoints/hf/" \;
rm -f "$fastsam3d_config"
"$py" - "$fastsam3d_base_config" "$fastsam3d_config" "$local_moge_v1_model" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
model = sys.argv[3]
text = src.read_text()
old = "pretrained_model_name_or_path: Ruicheng/moge-vitl"
new = f"pretrained_model_name_or_path: {model}"
if old not in text:
    raise SystemExit(f"missing MoGe model key in {src}")
dst.write_text(text.replace(old, new))
PY
echo "FASTSAM3D_CONFIG=$fastsam3d_config"
cd "$SAM3D_DIR"
run_sam3d_py "$repo_root/scripts/compatibility/run_sam3d_mesh.py" \
  --sam3d-root "$SAM3D_DIR" \
  --image_path "$frame_path" \
  --masks_dir "$masks_dir" \
  --config "$sam3d_config" \
  --with_texture_baking False \
  --use_vertex_color True

echo "=== [2b] MoGe pointmap for reference frame ==="
cd "$script_dir"
SAM3D_REPO_ROOT="$pointmap_config_root" run_sam3d_py get_pointmap_dir.py --image "$frame_path" --output "$pointmap_path"

echo "=== [2c] Export AoE MANO hands as Do-as-I-Do hand meshes ==="
run_sam3d_py "$integration_script_dir/export_aoe_hands_to_dai_npz.py" \
  --clip-dir "$video_dir" \
  --metadata "$video_dir/metadata.json" \
  --hawor-dir "$HAWOR_DIR" \
  --output "$hand_meshes_path" \
  --pose-space camera \
  --device cpu

echo "=== [2d] MoGe pointmaps for all frames ==="
SAM3D_REPO_ROOT="$pointmap_config_root" run_sam3d_py get_pointmap_dir.py --image_dir "$video_dir/all_frames"

echo "=== [2d.1] Render AoE hand masks for scale optimization ==="
frame_count="$(find "$video_dir/all_frames" -maxdepth 1 -name '[0-9][0-9][0-9][0-9][0-9][0-9].png' | wc -l | tr -d '[:space:]')"
run_sam3d_py "$integration_script_dir/render_aoe_hand_masks_for_dai.py" \
  --clip-dir "$video_dir" \
  --hand-meshes "$hand_meshes_path" \
  --frames "0:$frame_count" \
  --dilate "${DAI_HAND_MASK_DILATE:-3}"

echo "=== [2e] Gravity estimation ==="
geocalib_commit="unavailable"
if [[ -d "$geocalib_dir/.git" ]]; then
  geocalib_commit="$(git -c safe.directory="$geocalib_dir" -C "$geocalib_dir" rev-parse HEAD 2>/dev/null || echo unknown)"
fi
geocalib_weights="$TORCH_HOME/hub/checkpoints/pinhole.tar"
geocalib_weights_sha256="unavailable"
if [[ -s "$geocalib_weights" ]]; then
  geocalib_weights_sha256="$(sha256sum "$geocalib_weights" | awk '{print $1}')"
fi
if run_sam3d_py predict_video_gravity.py "$video_dir/all_frames" \
  --max_frames "$geocalib_max_frames" \
  --output_path "$video_dir/gravity.json"; then
  "$py" - "$video_dir/gravity.json" "$geocalib_commit" "$geocalib_weights_sha256" "$geocalib_max_frames" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["estimation_status"] = "estimated"
payload["estimation_source"] = "geocalib"
payload["is_fallback"] = False
payload["geocalib_commit"] = sys.argv[2]
payload["weights_sha256"] = sys.argv[3]
payload["max_frames"] = int(sys.argv[4])
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
  "$py" "$repo_root/scripts/select_geocalib_gravity.py" \
    --gravity-json "$video_dir/gravity.json" \
    --reference-frame "$frame_n" \
    --max-aggregate-p95-deg "$geocalib_max_aggregate_p95_deg"
else
  cat > "$video_dir/gravity.json" <<'JSON'
{
  "vec3d": [0.0, -1.0, 0.0],
  "roll_deg": 0.0,
  "pitch_deg": 0.0,
  "frame": "camera_frame_x_right_y_down_z_fwd",
  "semantics": "world_up_direction_in_camera_frame",
  "estimation_status": "fallback_assumption",
  "estimation_source": "upright_camera_assumption",
  "is_fallback": true,
  "required_for_production": true,
  "fallback_reason": "GeoCalib is unavailable or gravity prediction failed",
  "note": "upright camera-frame fallback because GeoCalib is unavailable; Do-as-I-Do aligns this vector to MuJoCo +Z"
}
JSON
  if [[ "$require_geocalib" == "1" ]]; then
    echo "GeoCalib gravity estimation is required for production fresh runs; refusing upright fallback" >&2
    exit 1
  fi
fi

echo "=== [2.5] TAPIR velocity tracking ==="
tapir_tracking_dir="$video_dir/perframe_tracking_$object_id"
tapir_motion_stats_json="$tapir_tracking_dir/motion_stats.json"
tapir_status_json="$tapir_tracking_dir/tapir_status.json"
tapir_required="${DAI_TAPIR_REQUIRED:-0}"
tapir_timeout_sec="${DAI_TAPIR_TIMEOUT_SEC:-0}"
tapir_rc=0
tapir_status="success"
tapir_message="motion stats available"
mkdir -p "$tapir_tracking_dir"
if [[ "${DAI_SKIP_TAPIR:-0}" == "1" ]]; then
  tapir_status="skipped"
  tapir_message="skipped by DAI_SKIP_TAPIR=1"
elif [[ "$tapir_timeout_sec" =~ ^[0-9]+$ && "$tapir_timeout_sec" -gt 0 && "$(command -v timeout || true)" != "" ]]; then
  CUDA_VISIBLE_DEVICES="$sam3d_cuda" timeout "${tapir_timeout_sec}s" "$py" tapir_velocity_tracking.py \
    --video "$video_path" \
    --mask-dir "$video_masks_dir" \
    --object "$object_id" \
    --checkpoint "$TAPNET_CKPT" || tapir_rc=$?
else
  run_sam3d_py tapir_velocity_tracking.py \
    --video "$video_path" \
    --mask-dir "$video_masks_dir" \
    --object "$object_id" \
    --checkpoint "$TAPNET_CKPT" || tapir_rc=$?
fi
if (( tapir_rc != 0 )); then
  tapir_status="failed"
  tapir_message="tapir_velocity_tracking.py exited non-zero; continuing without adaptive rotation velocity prior"
  if (( tapir_rc == 124 )); then
    tapir_status="timeout"
    tapir_message="tapir_velocity_tracking.py timed out; continuing without adaptive rotation velocity prior"
  elif (( tapir_rc == 137 || tapir_rc == 143 )); then
    tapir_status="terminated"
    tapir_message="tapir_velocity_tracking.py was terminated; continuing without adaptive rotation velocity prior"
  fi
fi
if [[ "$tapir_status" == "success" && ! -s "$tapir_motion_stats_json" ]]; then
  tapir_status="missing_output"
  tapir_rc=1
  tapir_message="tapir_velocity_tracking.py finished but motion_stats.json is missing; continuing without adaptive rotation velocity prior"
fi
write_tapir_status "$tapir_status_json" "$tapir_status" "$tapir_rc" "$tapir_message" "$tapir_motion_stats_json" "$tapir_timeout_sec" "$tapir_required"
if [[ "$tapir_status" != "success" ]]; then
  echo "[warn] TAPIR velocity prior unavailable: $tapir_status ($tapir_message)" >&2
  if [[ "$tapir_required" == "1" ]]; then
    echo "error: DAI_TAPIR_REQUIRED=1 and TAPIR did not succeed; see $tapir_status_json" >&2
    exit 1
  fi
fi

echo "=== [3] Fast-SAM3D guided pose tracking ==="
cd "$FASTSAM3D_DIR"
track_object_args=(
  --config "$fastsam3d_config" \
  --vid_dir "$video_dir" \
  --masks_root "$video_masks_dir" \
  --object_name "$object_id" \
  --init_frame "$frame_n" \
  --output_dir "$video_dir/obj_tracking_out/$object_id" \
  --guidance_strength 1 \
  --save_layout \
  --fix_scale_to_init_frame \
  --pose_guidance_strength 0.5 \
  --num_pose_samples 8 \
  --scoring_metric render_iou \
  --pose_selection cluster \
  --cluster_dist_thresh 0.3 \
  --cluster_min_size 3 \
  --cluster_w_rot 1.5 \
  --chain_poses \
  --post_optimize \
  --no-enable_shape_icp \
  --chain_on_diffusion \
  --enable_ss_cache \
  --euler_steps 8 \
  --latent_opt_steps "$fastsam3d_latent_opt_steps" \
  --latent_opt_lr "$fastsam3d_latent_opt_lr"
)
if [[ -s "$tapir_motion_stats_json" ]]; then
  track_object_args+=(--rotvel_json "$tapir_motion_stats_json")
else
  echo "[warn] Running Fast-SAM3D without --rotvel_json; TAPIR status is recorded in $tapir_status_json" >&2
fi
run_sam3d_py "$repo_root/scripts/compatibility/run_fast_sam3d_tracker.py" \
  --fast-sam3d-root "$FASTSAM3D_DIR" \
  "${track_object_args[@]}"

cd "$script_dir"
echo "=== [3b] Project mesh ==="
run_sam3d_py run_project_mesh_combined.py \
  --video "$video_path" \
  --mesh "$masks_dir/$object_id/$object_id.obj" \
  --json "$video_dir/obj_tracking_out/$object_id/combined_visualization/layout.json" \
  --output-base "$video_dir/obj_tracking_out/$object_id/combined_visualization/projected" \
  --object-color "0.6,0.9,0.6"

echo "=== [3c] Convert layout to camera frame ==="
run_sam3d_py convert_layout_to_camera_frame.py \
  --input "$video_dir/obj_tracking_out/$object_id/combined_visualization/layout.json" \
  --output "$video_dir/obj_tracking_out/$object_id/combined_visualization/layout_camera_frame.json"

echo "=== [4] Optimize translation/scale ==="
layout_dir="$video_dir/obj_tracking_out/$object_id/combined_visualization"
layout_camera="$layout_dir/layout_camera_frame.json"
layout_optimized="$layout_dir/layout_camera_frame_optimized.json"
scale_anchor_report="$video_dir/scale_anchor_hand_selection.json"
run_sam3d_py "$repo_root/scripts/select_dai_scale_anchor_hand.py" \
  --hand-npz "$hand_meshes_path" \
  --frames-dir "$video_dir/all_frames" \
  --masks-dir "$video_masks_dir" \
  --object-id "$object_id" \
  --requested "$anchor_hand" \
  --output "$scale_anchor_report"
optimizer_anchor_hand="$("$py" -c 'import json, sys; print(json.load(open(sys.argv[1]))["selected_anchor_hand"])' "$scale_anchor_report")"
case "$optimizer_anchor_hand" in
  left|right) ;;
  *) echo "error: scale anchor resolver returned invalid hand: $optimizer_anchor_hand" >&2; exit 1 ;;
esac
echo "DAI scale anchor: requested=$anchor_hand selected=$optimizer_anchor_hand report=$scale_anchor_report"
"$py" - "$video_dir/config.json" "$anchor_hand" "$optimizer_anchor_hand" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["requested_anchor_hand"] = sys.argv[2]
payload["scale_anchor_hand"] = sys.argv[3]
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
if ! run_sam3d_py optimize_translation_scale.py \
  --video-dir "$video_dir" \
  --layout-json "$layout_camera" \
  --hand-meshes "$hand_meshes_path" \
  --anchor-hand "$optimizer_anchor_hand" \
  --ref-frame "$frame_n"; then
  if [[ "${DAI_SCALE_OPT_STRICT:-0}" == "1" ]]; then
    echo "error: optimize_translation_scale.py failed and DAI_SCALE_OPT_STRICT=1" >&2
    exit 1
  fi
  echo "[warn] optimize_translation_scale.py failed; using unoptimized camera-frame layout"
  "$py" -c 'import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)
data["translation_scale_optimization"] = {
    "status": "fallback_unoptimized",
    "reason": "optimize_translation_scale_failed",
    "mesh_scale": 1.0,
}
with open(dst, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
' "$layout_camera" "$layout_optimized"
fi

cat <<EOF
=== Pipeline complete ===
layout: $video_dir/obj_tracking_out/$object_id/combined_visualization/layout_camera_frame_optimized.json
mesh:   $masks_dir/$object_id/$object_id.obj
hands:  $hand_meshes_path
EOF
