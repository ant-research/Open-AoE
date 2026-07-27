#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_name="${V4_RUN_NAME:-foundation_jar_v4_two_full_$(date +%Y%m%d_%H%M%S)}"
exp="$repo_root/experiments/$run_name"
python_ego="${EGOINFINITY_PYTHON:-python}"
python_ret="${RETARGETING_PYTHON:-python}"

task="${V4_TASK:-foundation_jar_bimanual_leftscale}"
hand_type="${V4_HAND_TYPE:-bimanual}"
dai_retarget_hand_type="${V4_DAI_RETARGET_HAND_TYPE:-auto}"
dai_raw_dir="${V4_DAI_RAW_DIR:?Set V4_DAI_RAW_DIR to a Do-as-I-Do reconstruction raw directory.}"
dai_native_aligned_raw_dir="${V4_DAI_NATIVE_ALIGNED_RAW_DIR:-$exp/intermediates/trajectory_6dof/do_as_i_do/dai_native_diagnostic_raw_dir}"
dai_clip_dir="${V4_DAI_CLIP_DIR:?Set V4_DAI_CLIP_DIR to the Do-as-I-Do clip directory used for visualization.}"
ego_dai_raw_dir="${V4_EGO_DAI_RAW_DIR:-$exp/intermediates/trajectory_6dof/egoinfinity/do_as_i_do_raw_dir}"
ego_dai_aoe_raw_dir="${V4_EGO_DAI_AOE_RAW_DIR:-$exp/intermediates/trajectory_6dof/egoinfinity/do_as_i_do_raw_dir_hand_aoe}"
dai_estimated_raw_dir="${V4_DAI_ESTIMATED_RAW_DIR:-$exp/intermediates/trajectory_6dof/do_as_i_do/do_as_i_do_raw_dir_hand_estimated}"
ego_video="${V4_EGO_VIDEO:?Set V4_EGO_VIDEO to an AoE raw RGB video path.}"
ego_clip_id="${V4_EGO_CLIP_ID:?Set V4_EGO_CLIP_ID to a stable clip id.}"
ego_objects="${V4_EGO_OBJECTS:?Set V4_EGO_OBJECTS to the EgoInfinity object prompt.}"
ego_start="${V4_EGO_START:-0.0}"
ego_end="${V4_EGO_END:-3.5}"
ego_fps="${V4_EGO_FPS:-15}"
ego_target_point="${EGOINFINITY_TARGET_POINT:-}"
dai_max_sim_steps="${DAI_MAX_SIM_STEPS:--1}"

main_cuda="${MAIN_CUDA:-0}"
dai_cuda_visible_devices="${DAI_CUDA_VISIBLE_DEVICES:-$main_cuda}"
sam3_cuda="${SAM3_WORKER_CUDA:-1}"
sam3d_cuda="${SAM3D_WORKER_CUDA:-7}"
sam3_python="${SAM3_PYTHON:?Set SAM3_PYTHON to the SAM3/SAM3.1 worker Python.}"
sam3_repo="${SAM3_REPO:?Set SAM3_REPO to the SAM3 source checkout.}"
sam3_bpe_path="${SAM3_BPE_PATH:-$sam3_repo/assets/bpe_simple_vocab_16e6.txt.gz}"
sam3_checkpoint="${SAM3_CHECKPOINT:-}"
sam3d_python="${SAM3D_PYTHON:?Set SAM3D_PYTHON to the SAM3D worker Python.}"
sam3d_repo="${SAM3D_REPO:?Set SAM3D_REPO to the SAM 3D Objects source checkout.}"
sam3d_conda_prefix="${SAM3D_CONDA_PREFIX:-$(dirname "$(dirname "$sam3d_python")")}"
dinov2_local_repo="${DINOV2_LOCAL_REPO:-$repo_root/third_party/dinov2}"
sam3d_config="${SAM3D_CONFIG:-}"
if [[ -z "$sam3d_config" ]]; then
  if [[ -f "$sam3d_repo/checkpoints/hf_local_moge_v1_vitl/pipeline_hjd_meshres48.yaml" ]]; then
    sam3d_config="checkpoints/hf_local_moge_v1_vitl/pipeline_hjd_meshres48.yaml"
  elif [[ -f "$sam3d_repo/checkpoints/hf_local_moge_v1_vitl/pipeline.yaml" ]]; then
    sam3d_config="checkpoints/hf_local_moge_v1_vitl/pipeline.yaml"
  else
    sam3d_config="checkpoints/hf/pipeline.yaml"
  fi
fi
hf_home="${HF_HOME:-}"
if [[ -z "$hf_home" ]]; then
  hf_home="$HOME/.cache/huggingface"
fi
hf_hub_offline="${HF_HUB_OFFLINE:-1}"
transformers_offline="${TRANSFORMERS_OFFLINE:-1}"
torch_home="${TORCH_HOME:-}"
if [[ -z "$torch_home" ]]; then
  torch_home="$HOME/.cache/torch"
fi
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
dai_raw_dir="$(resolve_repo_path "$dai_raw_dir")"
dai_native_source_raw_dir="$dai_raw_dir"
dai_clip_dir="$(resolve_repo_path "$dai_clip_dir")"
ego_video="$(resolve_repo_path "$ego_video")"
worker_socket_id="$(printf '%s' "$run_name" | sha256sum | awk '{print substr($1, 1, 16)}')"
sam3_socket="${SAM3_WORKER_SOCKET:-/tmp/eis3_${USER:-user}_${worker_socket_id}.sock}"
sam3d_socket="${SAM3D_WORKER_SOCKET:-/tmp/eis3d_${USER:-user}_${worker_socket_id}.sock}"
for worker_socket in "$sam3_socket" "$sam3d_socket"; do
  if (( ${#worker_socket} > 99 )); then
    echo "Worker socket path exceeds the conservative AF_UNIX limit (99 chars): $worker_socket" >&2
    exit 2
  fi
done
sam3_pid=""
sam3d_pid=""
dai_native_retarget_object_source="dai_native"
dai_native_input_rc=4
ego_aoe_preflight_rc=4
ego_estimated_preflight_rc=4
dai_native_post_retarget_rc=null
dai_native_post_retarget_status="not_run_input_unavailable"
dai_native_route_available=false
dai_native_estimated_post_retarget_rc=null
dai_native_estimated_post_retarget_status="disabled_policy"
dai_native_estimated_route_available=false
ego_aoe_post_retarget_rc=null
ego_aoe_post_retarget_status="not_run_input_unavailable"
ego_aoe_route_available=false
ego_estimated_post_retarget_rc=null
ego_estimated_post_retarget_status="not_run_input_unavailable"
ego_estimated_route_available=false
last_stage_rc=0
dai_native_preflight_evidence_rel="logs/route_preflight__traj_do_as_i_do__hand_aoe__retarget_do_as_i_do.json"
dai_native_estimated_preflight_evidence_rel="logs/route_preflight__traj_do_as_i_do__hand_estimated__retarget_do_as_i_do.json"
ego_aoe_preflight_evidence_rel="logs/route_preflight__traj_egoinfinity__hand_aoe__retarget_do_as_i_do.json"
ego_estimated_preflight_evidence_rel="logs/route_preflight__traj_egoinfinity__hand_estimated__retarget_do_as_i_do.json"
dai_native_preflight_evidence_sha256=""
dai_native_estimated_preflight_evidence_sha256=""
ego_aoe_preflight_evidence_sha256=""
ego_estimated_preflight_evidence_sha256=""

mkdir -p "$exp/logs/workers" "$exp/videos" "$exp/reuse"
stage_failures_file="$exp/logs/stage_failures.tsv"
: > "$stage_failures_file"
review_failures_file="$exp/logs/review_failures.tsv"
: > "$review_failures_file"

run_stage() {
  local stage_name="$1"
  shift
  if (set -euo pipefail; "$@"); then
    last_stage_rc=0
    printf '%s\t%s\t0\n' "$(date -Is)" "$stage_name" >> "$stage_failures_file"
  else
    local status=$?
    last_stage_rc=$status
    printf '%s\t%s\t%s\n' "$(date -Is)" "$stage_name" "$status" >> "$stage_failures_file"
    echo "stage failed but matrix execution will continue: $stage_name (exit=$status)" >&2
  fi
  return 0
}

# Review/index/composition runs after the pristine backend. Its status controls
# review availability but never rewrites the native backend return code.
run_review_stage() {
  local stage_name="$1"
  shift
  if (set -euo pipefail; "$@"); then
    last_review_rc=0
    printf '%s\t%s\t0\n' "$(date -Is)" "$stage_name" >> "$review_failures_file"
  else
    local status=$?
    last_review_rc=$status
    printf '%s\t%s\t%s\n' "$(date -Is)" "$stage_name" "$status" >> "$review_failures_file"
    echo "review packaging failed after native backend completion: $stage_name (exit=$status)" >&2
  fi
  return "$last_review_rc"
}

resolve_native_dai_hand() {
  local output_root="$1"
  local expected_task="$2"
  "$python_ret" - "$output_root" "$expected_task" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
expected_task = sys.argv[2]
matches = []
for path in sorted(root.rglob("task_info.json")):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if (
        payload.get("task") == expected_task
        and payload.get("dataset_name") == "do_as_i_do"
        and int(payload.get("data_id", -1)) == 0
        and payload.get("embodiment_type") in {"left", "right", "bimanual"}
    ):
        matches.append(
            (
                payload["embodiment_type"],
                payload["task"],
                payload["dataset_name"],
                int(payload["data_id"]),
                path,
            )
        )
identities = {match[:4] for match in matches}
hands = {identity[0] for identity in identities}
if len(identities) != 1 or len(hands) != 1:
    detail = ", ".join(
        f"{hand}:{task}:{dataset}:{data_id}:{path}"
        for hand, task, dataset, data_id, path in matches
    ) or "none"
    raise SystemExit(
        f"expected one exact native task identity for {expected_task!r}; found {detail}"
    )
print(next(iter(hands)))
PY
}

clear_route_demo_artifacts() {
  local cell="$1"
  local triptych="${2:-}"
  # The official DAI wrapper performs its own index attempt before returning
  # hard-QC exit 7.  Remove that partial/stale admission state so a failed
  # launcher can remain diagnostic without masquerading as a source demo.
  rm -rf "$exp/cells/$cell" "$exp/assets/cells/$cell"
  rm -f "$exp/videos/${cell}__triptych.mp4"
  if [[ -n "$triptych" ]]; then
    rm -f "$triptych"
  fi
}

write_route_preflight_evidence() {
  local output="$1"
  local route="$2"
  local trajectory_6dof="$3"
  local hand_source="$4"
  local input_rc="$5"
  local status="$6"
  local adapter_manifest="${7:-}"
  "$python_ret" - \
    "$output" "$route" "$trajectory_6dof" "$hand_source" \
    "$input_rc" "$status" "$adapter_manifest" "$exp" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
route = sys.argv[2]
trajectory_6dof = sys.argv[3]
hand_source = sys.argv[4]
input_rc = int(sys.argv[5])
status = sys.argv[6]
adapter_arg = sys.argv[7]
experiment_root = Path(sys.argv[8]).resolve()

adapter_path = Path(adapter_arg).resolve() if adapter_arg else None
adapter = None
adapter_sha256 = None
adapter_errors = []
if adapter_path is not None and adapter_path.is_file():
    try:
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        adapter_sha256 = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    except (OSError, json.JSONDecodeError) as exc:
        adapter_errors.append(f"adapter_manifest_unreadable:{exc}")
elif input_rc == 0:
    adapter_errors.append("adapter_manifest_missing_for_available_input")

if input_rc == 0 and isinstance(adapter, dict):
    if adapter.get("hand_source") != hand_source:
        adapter_errors.append(
            f"adapter_hand_source={adapter.get('hand_source')},expected={hand_source}"
        )
    if (adapter.get("retarget_input_qc") or {}).get("status") != "ok":
        adapter_errors.append("adapter_retarget_input_qc_not_ok")
    if (adapter.get("adapter_rigid_invariance") or {}).get("status") != "ok":
        adapter_errors.append("adapter_rigid_invariance_not_ok")

def display_path(path):
    if path is None:
        return None
    try:
        return os.path.relpath(path, experiment_root)
    except ValueError:
        return str(path)

payload = {
    "schema_version": 1,
    "route": route,
    "trajectory_6dof": trajectory_6dof,
    "hand_source": hand_source,
    "retargeting": "do_as_i_do",
    "status": status,
    "input_rc": input_rc,
    "underlying_adapter_manifest": {
        "path": display_path(adapter_path),
        "path_base": "experiment_root",
        "sha256": adapter_sha256,
        "exists": bool(adapter_path is not None and adapter_path.is_file()),
        "adapter_schema_version": adapter.get("adapter_schema_version") if isinstance(adapter, dict) else None,
        "hand_source": adapter.get("hand_source") if isinstance(adapter, dict) else None,
        "retarget_input_qc_status": (
            (adapter.get("retarget_input_qc") or {}).get("status")
            if isinstance(adapter, dict)
            else None
        ),
        "adapter_rigid_invariance_status": (
            (adapter.get("adapter_rigid_invariance") or {}).get("status")
            if isinstance(adapter, dict)
            else None
        ),
    },
    "evidence_validation": {
        "status": "invalid" if adapter_errors else "ok",
        "errors": adapter_errors,
    },
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(5 if input_rc == 0 and adapter_errors else 0)
PY
}

file_sha256() {
  "$python_ret" -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$1"
}

stop_workers() {
  if [[ -n "$sam3_pid" ]]; then
    kill "$sam3_pid" 2>/dev/null || true
    wait "$sam3_pid" 2>/dev/null || true
    sam3_pid=""
  fi
  if [[ -n "$sam3d_pid" ]]; then
    kill "$sam3d_pid" 2>/dev/null || true
    wait "$sam3d_pid" 2>/dev/null || true
    sam3d_pid=""
  fi
}

cleanup() {
  stop_workers
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

wait_ready() {
  local pid="$1"
  local ready="$2"
  local name="$3"
  local log="$4"
  local timeout="${5:-900}"
  for _ in $(seq 1 "$timeout"); do
    if [[ -f "$ready" ]]; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "$name exited early; see $log" >&2
      return 1
    fi
    sleep 1
  done
  echo "$name did not become ready; see $log" >&2
  return 1
}

start_workers() {
  local sam3_log="$exp/logs/workers/sam3_worker.log"
  local sam3d_log="$exp/logs/workers/sam3d_worker.log"

  rm -f "$sam3_socket" "$sam3_socket.ready"
  (
    cd "$repo_root/third_party/EgoInfinity"
    local sam3_compat_args=(
      --worker "$repo_root/third_party/EgoInfinity/scripts/sam3_worker.py"
    )
    if [[ -n "$sam3_checkpoint" ]]; then
      sam3_compat_args+=(--checkpoint "$sam3_checkpoint")
    fi
    CUDA_VISIBLE_DEVICES="$sam3_cuda" \
    HF_HOME="$hf_home" \
    HF_HUB_OFFLINE="$hf_hub_offline" \
    TRANSFORMERS_OFFLINE="$transformers_offline" \
    PYTHONPATH="$sam3_repo${PYTHONPATH:+:$PYTHONPATH}" \
      exec "$sam3_python" "$repo_root/scripts/compatibility/run_sam3_worker.py" \
        "${sam3_compat_args[@]}" \
        --socket "$sam3_socket" \
        --version "${EGOINFINITY_SAM3_VERSION:-sam3}" \
        --bpe_path "$sam3_bpe_path"
  ) > "$sam3_log" 2>&1 &
  sam3_pid=$!

  rm -f "$sam3d_socket" "$sam3d_socket.ready"
  (
    cd "$repo_root/third_party/EgoInfinity"
    CUDA_VISIBLE_DEVICES="$sam3d_cuda" \
    CONDA_PREFIX="$sam3d_conda_prefix" \
    HF_HOME="$hf_home" \
    HF_HUB_OFFLINE="$hf_hub_offline" \
    TRANSFORMERS_OFFLINE="$transformers_offline" \
    TORCH_HOME="$torch_home" \
    SAM3D_REPO="$sam3d_repo" \
    DINOV2_LOCAL_REPO="$dinov2_local_repo" \
    DINOV2_REPO_DIR="${DINOV2_REPO_DIR:-$dinov2_local_repo}" \
      exec "$sam3d_python" "$repo_root/scripts/compatibility/run_egoinfinity_sam3d_worker.py" \
        --worker "$repo_root/third_party/EgoInfinity/scripts/sam3d_worker.py" \
        --socket "$sam3d_socket" \
        --repo "$sam3d_repo" \
        --config "$sam3d_config" \
        --quality_default "${SAM3D_QUALITY_DEFAULT:-tier1}"
  ) > "$sam3d_log" 2>&1 &
  sam3d_pid=$!

  wait_ready "$sam3_pid" "$sam3_socket.ready" "SAM3 worker" "$sam3_log" "${SAM3_WORKER_READY_TIMEOUT:-420}"
  wait_ready "$sam3d_pid" "$sam3d_socket.ready" "SAM3D worker" "$sam3d_log" "${SAM3D_WORKER_READY_TIMEOUT:-900}"
}

compose_cell() {
  local cell="$1"
  local output="$2"
  "$python_ego" "$repo_root/scripts/review/compose_triptych.py" \
    --overlay "$exp/cells/$cell/overlay.mp4" \
    --depth "$exp/cells/$cell/depth.mp4" \
    --robot "$exp/cells/$cell/robot.mp4" \
    --output "$output" \
    --duration 0
}

run_egoinfinity_full() {
  local log="$exp/logs/v4_egoinfinity_full.log"
  {
    echo "=== egoinfinity_full start $(date -Is) ==="
    cd "$repo_root"
    export CUDA_VISIBLE_DEVICES="$main_cuda"
    export SAM3_WORKER_SOCKET="$sam3_socket"
    export SAM3D_WORKER_SOCKET="$sam3d_socket"
    export EGOINFINITY_OBJECT_SELECTION_MODE="${EGOINFINITY_OBJECT_SELECTION_MODE:-best}"
    export EGOINFINITY_TARGET_PROMPT="${EGOINFINITY_TARGET_PROMPT:-$ego_objects}"
    scripts/run_egoinfinity_retarget.sh \
      --clip-dir "$ego_video" \
      --clip-id "$ego_clip_id" \
      --objects "$ego_objects" \
      --start "$ego_start" \
      --end "$ego_end" \
      --fps "$ego_fps" \
      --run-name "$run_name" \
      --robot g1 \
      --no-sam3d-worker
    "$python_ego" "$repo_root/scripts/index_cell_assets.py" \
      --run-name "$run_name" \
      --trajectory-6dof egoinfinity \
      --hand-source estimated \
      --retargeting egoinfinity \
      --task "$task" \
      --hand-type "$hand_type" \
      --robot g1
    compose_cell "traj_egoinfinity__hand_estimated__retarget_egoinfinity" "$exp/videos/v4_egoinfinity_full__triptych.mp4"
    echo "=== egoinfinity_full end $(date -Is) ==="
  } 2>&1 | tee "$log"
}

select_egoinfinity_object() {
  local raw_result="$exp/intermediates/egoinfinity/clip/pipeline_result.pkl.gz"
  local selected_result="$exp/intermediates/egoinfinity/clip/pipeline_result_selected.pkl.gz"
  local report="$exp/intermediates/egoinfinity/clip/object_selection_report.json"
  local selected_overlay="$exp/intermediates/egoinfinity/clip/rgb_mesh_overlay_selected.mp4"
  [[ -f "$raw_result" ]] || { echo "missing EgoInfinity result: $raw_result" >&2; return 4; }
  local filter_args=(
    "$python_ego"
    "$repo_root/scripts/filter_egoinfinity_objects.py"
    --input "$raw_result"
    --output "$selected_result"
    --target-prompt "$ego_objects"
    --selection-mode best
    --report "$report"
  )
  if [[ -n "$ego_target_point" ]]; then
    filter_args+=(--target-point "$ego_target_point")
  fi
  "${filter_args[@]}"
  "$python_ego" "$repo_root/scripts/review/render_egoinfinity_rgb_overlay.py" \
    --pipeline-result "$selected_result" \
    --output "$selected_overlay" \
    --fps "$ego_fps" \
    --target-prompt "$ego_objects" \
    --select-best-object
}

prepare_dai_native_retarget_input() {
  local anchor_hand="${V4_DAI_SCALE_ANCHOR_HAND:-}"
  if [[ "$anchor_hand" != "left" && "$anchor_hand" != "right" ]]; then
    anchor_hand="$hand_type"
  fi
  if [[ "$anchor_hand" != "left" && "$anchor_hand" != "right" && -s "$dai_native_source_raw_dir/scale_anchor_hand_selection.json" ]]; then
    anchor_hand="$("$python_ret" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("selected_anchor_hand") or "")' "$dai_native_source_raw_dir/scale_anchor_hand_selection.json")"
  fi
  if [[ "$anchor_hand" != "left" && "$anchor_hand" != "right" ]]; then
    anchor_hand="$("$python_ret" -c 'import json, sys; c=json.load(open(sys.argv[1])); print(c.get("scale_anchor_hand") or c.get("anchor_hand") or "")' "$dai_native_source_raw_dir/config.json")"
  fi
  case "$anchor_hand" in
    left|right) ;;
    *) echo "DAI-native contact alignment requires a left/right anchor hand" >&2; return 4 ;;
  esac
  "$python_ret" "$repo_root/scripts/prepare_do_as_i_do_scene_adapter.py" \
    --source-dir "$dai_native_source_raw_dir" \
    --output-dir "$dai_native_aligned_raw_dir" \
    --task "$task" \
    --anchor-hand "$anchor_hand" \
    --hand-source aoe \
    --no-optimize-scale \
    --python-bin "$python_ret" \
    --hoi-contact-alignment diagnostic \
    --force || return $?
  [[ -s "$dai_native_aligned_raw_dir/adapter_manifest.json" ]] || return 4
  dai_native_retarget_object_source="$("$python_ret" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("retarget_object_source") or "dai_native")' "$dai_native_aligned_raw_dir/adapter_manifest.json")" || return $?
  dai_raw_dir="$dai_native_aligned_raw_dir"
}

prepare_ego_retarget_adapter() {
  local hand_source="$1"
  local output_dir="$2"
  local hand_geometry_source="$3"
  local anchor_args=()
  local retarget_hand_request="auto"
  if [[ "$hand_type" == "left" || "$hand_type" == "right" ]]; then
    anchor_args=(--anchor-hand "$hand_type")
    retarget_hand_request="$hand_type"
  elif [[ "$hand_type" == "bimanual" ]]; then
    retarget_hand_request="bimanual"
  fi
  local pipeline_result="$exp/intermediates/egoinfinity/clip/pipeline_result_selected.pkl.gz"
  if [[ ! -f "$pipeline_result" ]]; then
    echo "missing EgoInfinity pipeline result: $pipeline_result" >&2
    return 4
  fi
  "$python_ret" "$repo_root/scripts/prepare_egoinfinity_do_as_i_do_raw_dir.py" \
    --source-dir "$dai_native_source_raw_dir" \
    --pipeline-result "$pipeline_result" \
    --output-dir "$output_dir" \
    --task "$task" \
    --target-prompt "$ego_objects" \
    --retarget-hand "$retarget_hand_request" \
    "${anchor_args[@]}" \
    --hand-selection-source hybrid \
    --hand-geometry-source "$hand_geometry_source" \
    --hand-source "$hand_source" \
    --object-geometry-source ego \
    --source-layout-alignment none \
    --no-fit-object-scale-to-ego-mask \
    --hoi-contact-alignment diagnostic \
    --require-retarget-input-qc \
    --python-bin "$python_ret" \
    --force || return $?
  echo "Ego retarget adapter passed input QC: hand_source=$hand_source output=$output_dir"
}

preflight_ego_retarget_inputs() {
  ego_aoe_preflight_rc=0
  ego_estimated_preflight_rc=0
  prepare_ego_retarget_adapter aoe "$ego_dai_aoe_raw_dir" source || ego_aoe_preflight_rc=$?
  prepare_ego_retarget_adapter estimated "$ego_dai_raw_dir" ego || ego_estimated_preflight_rc=$?
  if (( ego_aoe_preflight_rc != 0 || ego_estimated_preflight_rc != 0 )); then
    echo "Ego retarget input preflight failed: aoe_rc=$ego_aoe_preflight_rc estimated_rc=$ego_estimated_preflight_rc" >&2
    return 4
  fi
}

ego_adapter_is_reusable() {
  local output_dir="$1"
  local expected_hand_source="$2"
  [[ -s "$output_dir/adapter_manifest.json" ]] || return 1
  "$python_ret" -c '
import json, sys
m = json.load(open(sys.argv[1]))
ok = (
    m.get("adapter_schema_version") == 3
    and m.get("production_hoi_policy") == "preserve_source_relative_transform"
    and m.get("hand_source") == sys.argv[2]
    and (m.get("retarget_input_qc") or {}).get("status") == "ok"
    and (m.get("adapter_rigid_invariance") or {}).get("status") == "ok"
    and (m.get("adapter_rigid_invariance") or {}).get("true_pre_post_hand_comparison") is True
    and not (m.get("hoi_contact_alignment") or {}).get("applied")
    and not (m.get("hoi_refinement") or {}).get("hand_contact_translation_applied")
)
raise SystemExit(0 if ok else 1)
' "$output_dir/adapter_manifest.json" "$expected_hand_source"
}

materialize_and_validate_pristine_dai() {
  local output_root="$1"
  local raw_dir="$2"
  local trajectory_6dof="$3"
  local hand_source="$4"
  local resolved_hand_type="$5"
  local run_dir="$output_root/sharpa/$resolved_hand_type/$task/0"
  local mano_dir="$output_root/mano/$resolved_hand_type/$task/0"

  "$python_ret" "$repo_root/scripts/materialize_pristine_dai_review.py" \
    --repo-root "$repo_root" \
    --output-root "$output_root" \
    --raw-dir "$raw_dir" \
    --task "$task" \
    --hand-type "$resolved_hand_type" \
    --trajectory-6dof "$trajectory_6dof" \
    --hand-source "$hand_source" \
    --python-bin "$python_ret"

  "$python_ret" "$repo_root/scripts/diagnostics/validate_dai_raw_to_processed_hoi.py" \
    --adapter-manifest "$raw_dir/adapter_manifest.json" \
    --processed-keypoints "$mano_dir/trajectory_keypoints.npz" \
    --trajectory-6dof "$trajectory_6dof" \
    --hand-type "$resolved_hand_type" \
    --output "$mano_dir/raw_to_processed_hoi_invariance.json"

  "$python_ret" "$repo_root/scripts/diagnostics/evaluate_mjwp_object_tracking.py" \
    --scene "$run_dir/scene_act.xml" \
    --mjwp "$run_dir/trajectory_mjwp_act_aligned.npz" \
    --reference "$run_dir/trajectory_kinematic.npz" \
    --output "$run_dir/mjwp_object_tracking_quality.json" \
    --alignment-manifest "$run_dir/mjwp_alignment_manifest.json" \
    --pos-median-threshold 0.05 \
    --pos-p95-threshold 0.10 \
    --rot-median-threshold 0.75 \
    --rot-p95-threshold 2.0 \
    --lost-pos-threshold 0.10 \
    --lost-fraction-threshold 0.10 \
    --hoi-reference-contact-threshold 0.05 \
    --hoi-executed-contact-threshold 0.08 \
    --hoi-distance-median-threshold 0.03 \
    --hoi-distance-p95-threshold 0.08 \
    --hoi-min-reference-contact-frames 3 \
    --hoi-min-contact-retention 0.50
}

run_do_as_i_do_full() {
  local log="$exp/logs/v4_do_as_i_do_full.log"
  local cell="traj_do_as_i_do__hand_aoe__retarget_do_as_i_do"
  local triptych="$exp/videos/v4_do_as_i_do_full__triptych.mp4"
  local output_root="$exp/intermediates/retargeting/do_as_i_do/$cell/retargeting_outputs"
  local review_ready_marker="$exp/logs/review_ready__${cell}.json"
  {
    echo "=== do_as_i_do_full start $(date -Is) ==="
    cd "$repo_root"
    clear_route_demo_artifacts "$cell" "$triptych"
    rm -f "$review_ready_marker"
    export CUDA_VISIBLE_DEVICES="$main_cuda"
    scripts/prepare_do_as_i_do_trajectory_6dof.sh \
      --run-name "$run_name" \
      --task "$task" \
      --raw-dir "$dai_raw_dir"
    "$python_ret" "$repo_root/scripts/review/materialize_do_as_i_do_visuals.py" \
      --run-name "$run_name" \
      --clip-dir "$dai_clip_dir" \
      --task "$task" \
      --fps "$ego_fps"
    local launcher_rc=0
    scripts/run_do_as_i_do_official_retarget.sh \
      --raw-dir "$dai_raw_dir" \
      --task "$task" \
      --output-root-dir "$output_root" \
      --cuda-visible-devices "$dai_cuda_visible_devices" \
      --egl-device-id "${DAI_EGL_DEVICE_ID:-0}" \
      --headless \
      --no-wait || launcher_rc=$?
    if (( launcher_rc != 0 )); then
      clear_route_demo_artifacts "$cell" "$triptych"
      return "$launcher_rc"
    fi
    local review_ready=0
    local dai_resolved_hand_type=""
    if dai_resolved_hand_type="$(run_review_stage \
      "${cell}_resolve_native_hand" resolve_native_dai_hand "$output_root" "$task")"; then
      echo "do_as_i_do_requested_hand=$hand_type"
      echo "do_as_i_do_resolved_hand=$dai_resolved_hand_type"
      if run_review_stage "${cell}_materialize_and_qc" \
        materialize_and_validate_pristine_dai \
          "$output_root" "$dai_raw_dir" do_as_i_do aoe "$dai_resolved_hand_type" &&
        run_review_stage "${cell}_index" \
        "$python_ret" "$repo_root/scripts/index_cell_assets.py" \
          --run-name "$run_name" \
          --trajectory-6dof do_as_i_do \
          --hand-source aoe \
          --retargeting do_as_i_do \
          --task "$task" \
          --hand-type "$dai_resolved_hand_type" \
          --robot sharpa; then
        if (( compose )); then
          if run_review_stage "${cell}_triptych" compose_cell "$cell" "$triptych"; then
            review_ready=1
          fi
        else
          review_ready=1
        fi
      fi
    fi
    if (( review_ready )); then
      printf '{"native_backend_rc":0,"review_ready":true,"resolved_hand":"%s"}\n' \
        "$dai_resolved_hand_type" > "$review_ready_marker"
    else
      clear_route_demo_artifacts "$cell" "$triptych"
      echo "native DAI completed successfully; review/index/triptych remains unavailable" >&2
    fi
    echo "=== do_as_i_do_full end $(date -Is) ==="
  } 2>&1 | tee "$log"
}

run_adapted_do_as_i_do_retarget() {
  local trajectory_6dof="$1"
  local hand_source="$2"
  local output_dir="$3"
  local object_geometry_source="$4"
  local hand_geometry_source="$5"
  local anchor_args=()
  local retarget_hand_request="auto"
  local source_layout_mode="none"
  local adapter_source_dir="$dai_native_source_raw_dir"
  local cell="traj_${trajectory_6dof}__hand_${hand_source}__retarget_do_as_i_do"
  local retarget_output_root="$exp/intermediates/retargeting/do_as_i_do/$cell/retargeting_outputs"
  local review_ready_marker="$exp/logs/review_ready__${cell}.json"
  if [[ "$object_geometry_source" != "ego" ]]; then
    echo "Refusing production hand-only fusion: a retained DAI object cannot be canonicalized with an independently transformed Ego hand" >&2
    return 4
  fi
  if [[ "$hand_type" == "left" || "$hand_type" == "right" ]]; then
    anchor_args=(--anchor-hand "$hand_type")
    retarget_hand_request="$hand_type"
  elif [[ "$hand_type" == "bimanual" ]]; then
    retarget_hand_request="bimanual"
  fi
  local log="$exp/logs/v4_${trajectory_6dof}_${hand_source}_do_as_i_do_retarget.log"
  {
    echo "=== ${trajectory_6dof}_${hand_source}_do_as_i_do_retarget start $(date -Is) ==="
    cd "$repo_root"
    clear_route_demo_artifacts "$cell"
    rm -f "$review_ready_marker"
    if [[ "$object_geometry_source" == "ego" ]] && ego_adapter_is_reusable "$output_dir" "$hand_source"; then
      echo "reusing preflighted Ego adapter: $output_dir"
    else
      local pipeline_result="$exp/intermediates/egoinfinity/clip/pipeline_result_selected.pkl.gz"
      if [[ ! -f "$pipeline_result" ]]; then
        echo "missing EgoInfinity pipeline result: $pipeline_result" >&2
        exit 4
      fi
      "$python_ret" "$repo_root/scripts/prepare_egoinfinity_do_as_i_do_raw_dir.py" \
        --source-dir "$adapter_source_dir" \
        --pipeline-result "$pipeline_result" \
        --output-dir "$output_dir" \
        --task "$task" \
        --target-prompt "$ego_objects" \
        --retarget-hand "$retarget_hand_request" \
        "${anchor_args[@]}" \
        --hand-selection-source hybrid \
        --hand-geometry-source "$hand_geometry_source" \
        --hand-source "$hand_source" \
        --object-geometry-source "$object_geometry_source" \
        --source-layout-alignment "$source_layout_mode" \
        --no-fit-object-scale-to-ego-mask \
        --hoi-contact-alignment "$([[ "$object_geometry_source" == "ego" ]] && echo diagnostic || echo none)" \
        $([[ "$object_geometry_source" == "ego" ]] && echo --require-retarget-input-qc) \
        --python-bin "$python_ret" \
        --force
    fi

    local adapter_hand_type
    adapter_hand_type="$("$python_ret" -c 'import json, sys; m=json.load(open(sys.argv[1])); print(m.get("selected_hand") or sys.argv[2])' "$output_dir/adapter_manifest.json" "$hand_type")"
    if [[ "$adapter_hand_type" != "left" && "$adapter_hand_type" != "right" && "$adapter_hand_type" != "bimanual" ]]; then
      adapter_hand_type="$hand_type"
    fi
    echo "${trajectory_6dof}_${hand_source}_adapter_hand=$adapter_hand_type"

    if [[ "$object_geometry_source" == "source" && "$hand_geometry_source" == "ego" ]]; then
      "$python_ret" "$repo_root/scripts/diagnostics/validate_adapter_geometry_contact.py" \
        --adapter-manifest "$output_dir/adapter_manifest.json"
    fi

    local launcher_rc=0
    scripts/run_do_as_i_do_official_retarget.sh \
      --raw-dir "$output_dir" \
      --task "$task" \
      --output-root-dir "$retarget_output_root" \
      --cuda-visible-devices "$dai_cuda_visible_devices" \
      --egl-device-id "${DAI_EGL_DEVICE_ID:-0}" \
      --headless \
      --no-wait || launcher_rc=$?
    if (( launcher_rc != 0 )); then
      clear_route_demo_artifacts "$cell"
      return "$launcher_rc"
    fi
    local review_ready=0
    local resolved_hand_type=""
    if resolved_hand_type="$(run_review_stage \
      "${cell}_resolve_native_hand" resolve_native_dai_hand "$retarget_output_root" "$task")"; then
      if run_review_stage "${cell}_hand_binding" \
        test "$resolved_hand_type" = "$adapter_hand_type"; then
        if run_review_stage "${cell}_materialize_and_qc" \
          materialize_and_validate_pristine_dai \
            "$retarget_output_root" "$output_dir" "$trajectory_6dof" \
            "$hand_source" "$resolved_hand_type" &&
          run_review_stage "${cell}_index" \
          "$python_ret" "$repo_root/scripts/index_cell_assets.py" \
            --run-name "$run_name" \
            --trajectory-6dof "$trajectory_6dof" \
            --hand-source "$hand_source" \
            --retargeting do_as_i_do \
            --task "$task" \
            --hand-type "$resolved_hand_type" \
            --robot sharpa; then
          review_ready=1
        fi
      fi
    fi
    if (( review_ready )); then
      printf '{"native_backend_rc":0,"review_ready":true,"resolved_hand":"%s"}\n' \
        "$resolved_hand_type" > "$review_ready_marker"
      echo "${trajectory_6dof}_${hand_source}_resolved_hand=$resolved_hand_type"
    else
      clear_route_demo_artifacts "$cell"
      echo "native DAI completed successfully; adapted-route review/index remains unavailable" >&2
    fi
    echo "=== ${trajectory_6dof}_${hand_source}_do_as_i_do_retarget end $(date -Is) ==="
  } 2>&1 | tee "$log"
}

{
  echo "run_name=$run_name"
  echo "task=$task"
  echo "hand_type=$hand_type"
  echo "dai_retarget_hand_type=$dai_retarget_hand_type"
  echo "dai_raw_dir=$dai_raw_dir"
  echo "dai_native_source_raw_dir=$dai_native_source_raw_dir"
  echo "dai_native_aligned_raw_dir=$dai_native_aligned_raw_dir"
  echo "DAI_NATIVE_HOI_CONTACT_ALIGNMENT=diagnostic"
  echo "dai_clip_dir=$dai_clip_dir"
  echo "ego_dai_raw_dir=$ego_dai_raw_dir"
  echo "ego_dai_aoe_raw_dir=$ego_dai_aoe_raw_dir"
  echo "dai_estimated_raw_dir=$dai_estimated_raw_dir"
  echo "ego_video=$ego_video"
  echo "ego_clip_id=$ego_clip_id"
  echo "ego_objects=$ego_objects"
  echo "ego_start=$ego_start"
  echo "ego_end=$ego_end"
  echo "ego_fps=$ego_fps"
  echo "EGOINFINITY_TARGET_POINT=${ego_target_point:-}"
  echo "DAI_MAX_SIM_STEPS=$dai_max_sim_steps"
  echo "MAIN_CUDA=$main_cuda"
  echo "DAI_CUDA_VISIBLE_DEVICES=$dai_cuda_visible_devices"
  echo "SAM3_WORKER_CUDA=$sam3_cuda"
  echo "SAM3D_WORKER_CUDA=$sam3d_cuda"
  echo "SAM3D_CONFIG=$sam3d_config"
  echo "SAM3_WORKER_SOCKET=$sam3_socket"
  echo "SAM3D_WORKER_SOCKET=$sam3d_socket"
  echo "EGOINFINITY_WORKER_LIFECYCLE=stop_after_egoinfinity_object_selection_before_dai_spider"
  echo "HF_HOME=$hf_home"
  echo "HF_HUB_OFFLINE=$hf_hub_offline"
  echo "TRANSFORMERS_OFFLINE=$transformers_offline"
  echo "TORCH_HOME=$torch_home"
  echo "DO_AS_I_DO_FAST_DECOMP=${DO_AS_I_DO_FAST_DECOMP:-1}"
  echo "CELL_TIMEOUT_SEC=disabled"
} > "$exp/run_env.txt"

start_workers
run_stage egoinfinity_full run_egoinfinity_full
run_stage egoinfinity_object_selection select_egoinfinity_object
echo "=== releasing EgoInfinity workers before DAI/SPIDER retarget $(date -Is) ===" \
  | tee -a "$exp/logs/workers/lifecycle.log"
stop_workers
dai_native_input_rc=0
dai_native_object_pose_rc=4
if prepare_dai_native_retarget_input; then
  printf '%s\t%s\t0\n' "$(date -Is)" "dai_native_retarget_input_preflight" >> "$stage_failures_file"
else
  dai_native_input_rc=$?
  printf '%s\t%s\t%s\n' "$(date -Is)" "dai_native_retarget_input_preflight" "$dai_native_input_rc" >> "$stage_failures_file"
  echo "pure DAI route disabled by DAI-native input QC; independent routes will continue" >&2
fi
echo "DAI-object/Ego-hand estimated route disabled: production hand-only canonical fusion is forbidden" >&2
printf '%s\t%s\t%s\n' "$(date -Is)" "dai_native_object_pose_preflight" "$dai_native_object_pose_rc" >> "$stage_failures_file"
if preflight_ego_retarget_inputs; then
  printf '%s\t%s\t0\n' "$(date -Is)" "egoinfinity_retarget_input_preflight" >> "$stage_failures_file"
else
  preflight_rc=$?
  printf '%s\t%s\t%s\n' "$(date -Is)" "egoinfinity_retarget_input_preflight" "$preflight_rc" >> "$stage_failures_file"
  echo "Ego-dependent routes selectively disabled by input QC; pure DAI routes will continue" >&2
fi

dai_native_preflight_status="ok"
if (( dai_native_input_rc != 0 )); then
  dai_native_preflight_status="failed_input_preflight"
fi
if ! write_route_preflight_evidence \
  "$exp/$dai_native_preflight_evidence_rel" \
  "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do" \
  do_as_i_do aoe "$dai_native_input_rc" "$dai_native_preflight_status" \
  "$dai_native_aligned_raw_dir/adapter_manifest.json"; then
  dai_native_input_rc=5
  dai_native_preflight_status="invalid_input"
  printf '%s\t%s\t5\n' "$(date -Is)" "dai_native_route_preflight_evidence" >> "$stage_failures_file"
  write_route_preflight_evidence \
    "$exp/$dai_native_preflight_evidence_rel" \
    "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do" \
    do_as_i_do aoe "$dai_native_input_rc" "$dai_native_preflight_status" \
    "$dai_native_aligned_raw_dir/adapter_manifest.json"
fi
dai_native_preflight_evidence_sha256="$(file_sha256 "$exp/$dai_native_preflight_evidence_rel")"

write_route_preflight_evidence \
  "$exp/$dai_native_estimated_preflight_evidence_rel" \
  "traj_do_as_i_do__hand_estimated__retarget_do_as_i_do" \
  do_as_i_do estimated "$dai_native_object_pose_rc" disabled_policy ""
dai_native_estimated_preflight_evidence_sha256="$(file_sha256 "$exp/$dai_native_estimated_preflight_evidence_rel")"

ego_aoe_preflight_status="ok"
if (( ego_aoe_preflight_rc != 0 )); then
  ego_aoe_preflight_status="failed_input_preflight"
fi
if ! write_route_preflight_evidence \
  "$exp/$ego_aoe_preflight_evidence_rel" \
  "traj_egoinfinity__hand_aoe__retarget_do_as_i_do" \
  egoinfinity aoe "$ego_aoe_preflight_rc" "$ego_aoe_preflight_status" \
  "$ego_dai_aoe_raw_dir/adapter_manifest.json"; then
  ego_aoe_preflight_rc=5
  ego_aoe_preflight_status="invalid_input"
  printf '%s\t%s\t5\n' "$(date -Is)" "egoinfinity_aoe_route_preflight_evidence" >> "$stage_failures_file"
  write_route_preflight_evidence \
    "$exp/$ego_aoe_preflight_evidence_rel" \
    "traj_egoinfinity__hand_aoe__retarget_do_as_i_do" \
    egoinfinity aoe "$ego_aoe_preflight_rc" "$ego_aoe_preflight_status" \
    "$ego_dai_aoe_raw_dir/adapter_manifest.json"
fi
ego_aoe_preflight_evidence_sha256="$(file_sha256 "$exp/$ego_aoe_preflight_evidence_rel")"

ego_estimated_preflight_status="ok"
if (( ego_estimated_preflight_rc != 0 )); then
  ego_estimated_preflight_status="failed_input_preflight"
fi
if ! write_route_preflight_evidence \
  "$exp/$ego_estimated_preflight_evidence_rel" \
  "traj_egoinfinity__hand_estimated__retarget_do_as_i_do" \
  egoinfinity estimated "$ego_estimated_preflight_rc" "$ego_estimated_preflight_status" \
  "$ego_dai_raw_dir/adapter_manifest.json"; then
  ego_estimated_preflight_rc=5
  ego_estimated_preflight_status="invalid_input"
  printf '%s\t%s\t5\n' "$(date -Is)" "egoinfinity_estimated_route_preflight_evidence" >> "$stage_failures_file"
  write_route_preflight_evidence \
    "$exp/$ego_estimated_preflight_evidence_rel" \
    "traj_egoinfinity__hand_estimated__retarget_do_as_i_do" \
    egoinfinity estimated "$ego_estimated_preflight_rc" "$ego_estimated_preflight_status" \
    "$ego_dai_raw_dir/adapter_manifest.json"
fi
ego_estimated_preflight_evidence_sha256="$(file_sha256 "$exp/$ego_estimated_preflight_evidence_rel")"

# Produce the recommended Ego-object routes first. A single valid plain route
# is enough for demo admission, so slow DAI-native baselines must not delay it.
if (( ego_aoe_preflight_rc == 0 )); then
  run_stage egoinfinity_aoe_retarget \
    run_adapted_do_as_i_do_retarget egoinfinity aoe "$ego_dai_aoe_raw_dir" ego source
  ego_aoe_post_retarget_rc=$last_stage_rc
  if (( ego_aoe_post_retarget_rc == 0 )); then
    ego_aoe_post_retarget_status="ok"
    ego_aoe_route_available=true
  else
    ego_aoe_post_retarget_status="failed"
    clear_route_demo_artifacts "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
  fi
else
  printf '%s\t%s\t%s\n' "$(date -Is)" "egoinfinity_aoe_retarget" "$ego_aoe_preflight_rc" >> "$stage_failures_file"
  clear_route_demo_artifacts "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
fi
if (( ego_estimated_preflight_rc == 0 )); then
  run_stage egoinfinity_estimated_retarget \
    run_adapted_do_as_i_do_retarget egoinfinity estimated "$ego_dai_raw_dir" ego ego
  ego_estimated_post_retarget_rc=$last_stage_rc
  if (( ego_estimated_post_retarget_rc == 0 )); then
    ego_estimated_post_retarget_status="ok"
    ego_estimated_route_available=true
  else
    ego_estimated_post_retarget_status="failed"
    clear_route_demo_artifacts "traj_egoinfinity__hand_estimated__retarget_do_as_i_do"
  fi
else
  printf '%s\t%s\t%s\n' "$(date -Is)" "egoinfinity_estimated_retarget" "$ego_estimated_preflight_rc" >> "$stage_failures_file"
  clear_route_demo_artifacts "traj_egoinfinity__hand_estimated__retarget_do_as_i_do"
fi
if (( dai_native_input_rc == 0 )); then
  run_stage do_as_i_do_aoe_retarget \
    run_do_as_i_do_full
  dai_native_post_retarget_rc=$last_stage_rc
  if (( dai_native_post_retarget_rc == 0 )); then
    dai_native_post_retarget_status="ok"
    dai_native_route_available=true
  else
    dai_native_post_retarget_status="failed"
    clear_route_demo_artifacts \
      "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do" \
      "$exp/videos/v4_do_as_i_do_full__triptych.mp4"
  fi
else
  printf '%s\t%s\t%s\n' "$(date -Is)" "do_as_i_do_aoe_retarget" "$dai_native_input_rc" >> "$stage_failures_file"
  clear_route_demo_artifacts \
    "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do" \
    "$exp/videos/v4_do_as_i_do_full__triptych.mp4"
fi
printf '%s\t%s\t%s\n' "$(date -Is)" "do_as_i_do_estimated_retarget" "$dai_native_object_pose_rc" >> "$stage_failures_file"
clear_route_demo_artifacts "traj_do_as_i_do__hand_estimated__retarget_do_as_i_do"

cat > "$exp/reuse/reuse_manifest.json" <<EOF
{
  "run_name": "$run_name",
  "policy": "v4 computes EgoInfinity once and Do-as-I-Do once. Downstream demos should reuse these outputs instead of invoking the original pipelines per matrix cell.",
  "stage_failures_tsv": "logs/stage_failures.tsv",
  "route_availability": {
    "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do": {
      "input_rc": $dai_native_input_rc,
      "post_retarget_rc": $dai_native_post_retarget_rc,
      "status": "$dai_native_post_retarget_status",
      "available": $dai_native_route_available,
      "preflight_evidence": {"path": "$dai_native_preflight_evidence_rel", "path_base": "experiment_root", "sha256": "$dai_native_preflight_evidence_sha256", "trajectory_6dof": "do_as_i_do", "hand_source": "aoe", "status": "$dai_native_preflight_status", "input_rc": $dai_native_input_rc}
    },
    "traj_do_as_i_do__hand_estimated__retarget_do_as_i_do": {
      "input_rc": $dai_native_object_pose_rc,
      "post_retarget_rc": $dai_native_estimated_post_retarget_rc,
      "status": "$dai_native_estimated_post_retarget_status",
      "available": $dai_native_estimated_route_available,
      "preflight_evidence": {"path": "$dai_native_estimated_preflight_evidence_rel", "path_base": "experiment_root", "sha256": "$dai_native_estimated_preflight_evidence_sha256", "trajectory_6dof": "do_as_i_do", "hand_source": "estimated", "status": "disabled_policy", "input_rc": $dai_native_object_pose_rc}
    },
    "traj_egoinfinity__hand_aoe__retarget_do_as_i_do": {
      "input_rc": $ego_aoe_preflight_rc,
      "post_retarget_rc": $ego_aoe_post_retarget_rc,
      "status": "$ego_aoe_post_retarget_status",
      "available": $ego_aoe_route_available,
      "preflight_evidence": {"path": "$ego_aoe_preflight_evidence_rel", "path_base": "experiment_root", "sha256": "$ego_aoe_preflight_evidence_sha256", "trajectory_6dof": "egoinfinity", "hand_source": "aoe", "status": "$ego_aoe_preflight_status", "input_rc": $ego_aoe_preflight_rc}
    },
    "traj_egoinfinity__hand_estimated__retarget_do_as_i_do": {
      "input_rc": $ego_estimated_preflight_rc,
      "post_retarget_rc": $ego_estimated_post_retarget_rc,
      "status": "$ego_estimated_post_retarget_status",
      "available": $ego_estimated_route_available,
      "preflight_evidence": {"path": "$ego_estimated_preflight_evidence_rel", "path_base": "experiment_root", "sha256": "$ego_estimated_preflight_evidence_sha256", "trajectory_6dof": "egoinfinity", "hand_source": "estimated", "status": "$ego_estimated_preflight_status", "input_rc": $ego_estimated_preflight_rc}
    }
  },
  "worker_lifecycle": "stop_after_egoinfinity_object_selection_before_dai_spider",
  "egoinfinity_full": {
    "cell": "traj_egoinfinity__hand_estimated__retarget_egoinfinity",
    "pipeline_result": "intermediates/egoinfinity/clip/pipeline_result_selected.pkl.gz",
    "raw_pipeline_result": "intermediates/egoinfinity/clip/pipeline_result.pkl.gz",
    "object_selection_report": "intermediates/egoinfinity/clip/object_selection_report.json",
    "target_point": "$ego_target_point",
    "triptych": "videos/v4_egoinfinity_full__triptych.mp4"
  },
  "do_as_i_do_full": {
    "cell": "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do",
    "retargeting_outputs": "intermediates/retargeting/do_as_i_do/traj_do_as_i_do__hand_aoe__retarget_do_as_i_do/retargeting_outputs",
    "triptych": "videos/v4_do_as_i_do_full__triptych.mp4"
  },
  "do_as_i_do_hand_combinations": [
    {
      "cell": "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do",
      "adapter_raw_dir": "intermediates/trajectory_6dof/do_as_i_do/dai_native_diagnostic_raw_dir",
      "object_track_source": "dai_native",
      "object_mesh_source": "dai_native",
      "retarget_object_source": "$dai_native_retarget_object_source",
      "hand_geometry_source": "aoe_hawor"
    },
    {
      "cell": "traj_do_as_i_do__hand_estimated__retarget_do_as_i_do",
      "disabled": true,
      "disabled_reason": "hand_only_canonical_fusion_forbidden",
      "adapter_raw_dir": null,
      "object_track_source": "dai_native",
      "object_mesh_source": "dai_native",
      "retarget_object_source": "$dai_native_retarget_object_source",
      "hand_geometry_source": "egoinfinity_estimated"
    },
    {
      "cell": "traj_egoinfinity__hand_aoe__retarget_do_as_i_do",
      "adapter_raw_dir": "intermediates/trajectory_6dof/egoinfinity/do_as_i_do_raw_dir_hand_aoe",
      "object_track_source": "egoinfinity",
      "hand_geometry_source": "aoe_hawor"
    },
    {
      "cell": "traj_egoinfinity__hand_estimated__retarget_do_as_i_do",
      "adapter_raw_dir": "intermediates/trajectory_6dof/egoinfinity/do_as_i_do_raw_dir",
      "object_track_source": "egoinfinity",
      "hand_geometry_source": "egoinfinity_estimated"
    }
  ]
}
EOF

echo "v4_done=$exp"
