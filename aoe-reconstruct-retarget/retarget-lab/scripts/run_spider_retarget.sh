#!/usr/bin/env bash
set -euo pipefail

readonly SPIDER_PINNED_COMMIT="2e54f19ee6ab8e0690c0f585beb1e9f8f53a6898"

die() {
  echo "run_spider_retarget: $*" >&2
  exit "${2:-2}"
}

require_value() {
  (( $# >= 2 )) || die "$1 requires a value"
}

spider_root=""
python_bin=""
input_root=""
output_dir=""
task=""
robot_type=""
embodiment_type=""
data_id=""
ref_dt=""
cuda_visible_devices=""
egl_device_id=""
check_only=0

while (( $# > 0 )); do
  case "$1" in
    --spider-root) require_value "$@"; spider_root="$2"; shift 2 ;;
    --python) require_value "$@"; python_bin="$2"; shift 2 ;;
    --input-root) require_value "$@"; input_root="$2"; shift 2 ;;
    --output-dir) require_value "$@"; output_dir="$2"; shift 2 ;;
    --task) require_value "$@"; task="$2"; shift 2 ;;
    --robot-type) require_value "$@"; robot_type="$2"; shift 2 ;;
    --embodiment-type) require_value "$@"; embodiment_type="$2"; shift 2 ;;
    --data-id) require_value "$@"; data_id="$2"; shift 2 ;;
    --ref-dt) require_value "$@"; ref_dt="$2"; shift 2 ;;
    --cuda-visible-devices) require_value "$@"; cuda_visible_devices="$2"; shift 2 ;;
    --egl-device-id) require_value "$@"; egl_device_id="$2"; shift 2 ;;
    --check-only) check_only=1; shift ;;
    --num-samples|--max-num-iterations|--max-sim-steps|--sim-dt|--horizon|\
    --joint-noise-scale|--pos-noise-scale|--rot-noise-scale|--contact-rew-scale|\
    --contact-guidance|--object-floor-collision|--object-object-collision|\
    --release-step|--quality-threshold)
      die "backend-tuning argument is forbidden: $1"
      ;;
    -h|--help)
      echo "Run the pinned original SPIDER pipeline on a read-only exact DAI output."
      exit 0
      ;;
    *) die "Unknown argument: $1" ;;
  esac
done

for pair in \
  "spider-root:$spider_root" "python:$python_bin" "input-root:$input_root" \
  "output-dir:$output_dir" "task:$task" "robot-type:$robot_type" \
  "embodiment-type:$embodiment_type" "data-id:$data_id" "ref-dt:$ref_dt"; do
  name="${pair%%:*}"
  value="${pair#*:}"
  [[ -n "$value" ]] || die "missing --$name"
done

[[ -x "$python_bin" ]] || die "python is not executable: $python_bin"

manifest_python="$(command -v python3)"
resolved_text="$(
  "$manifest_python" - "$spider_root" "$input_root" "$output_dir" <<'PY'
import os
import sys
from pathlib import Path

spider = Path(sys.argv[1]).expanduser().resolve(strict=True)
source = Path(sys.argv[2]).expanduser().resolve(strict=True)
output = Path(sys.argv[3]).expanduser().resolve(strict=False)
if not spider.is_dir() or not source.is_dir():
    raise SystemExit("SPIDER root and input root must be directories")
common = Path(os.path.commonpath((str(source), str(output))))
if common == source or common == output:
    raise SystemExit("output-dir must not overlap the read-only input root")
print(spider)
print(source)
print(output)
PY
)" || die "output-dir must not overlap the read-only input root" 4
resolved=()
while IFS= read -r line; do
  resolved+=("$line")
done <<<"$resolved_text"
spider_root="${resolved[0]}"
input_root="${resolved[1]}"
output_dir="${resolved[2]}"

[[ ! -e "$output_dir" ]] || die "output-dir must be a new path: $output_dir" 4

actual_commit="$(git -C "$spider_root" rev-parse HEAD)" || die "cannot read SPIDER commit" 4
actual_tree="$(git -C "$spider_root" rev-parse 'HEAD^{tree}')" || die "cannot read SPIDER tree" 4
[[ "$actual_commit" == "$SPIDER_PINNED_COMMIT" ]] || \
  die "SPIDER commit $actual_commit does not match pinned $SPIDER_PINNED_COMMIT" 4
[[ -z "$(git -C "$spider_root" status --porcelain=v1 --untracked-files=all)" ]] || \
  die "SPIDER checkout is dirty" 4

for relative in \
  spider/preprocess/generate_xml.py \
  spider/preprocess/ik_fast.py \
  examples/run_mjwp.py; do
  [[ -f "$spider_root/$relative" ]] || die "missing original SPIDER entrypoint: $relative" 4
done

validation_json="$(
  "$manifest_python" - \
    "$input_root" "$task" "$robot_type" "$embodiment_type" "$data_id" "$ref_dt" <<'PY'
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
task, robot, hand = sys.argv[2:5]
data_id = int(sys.argv[5])
ref_dt = float(sys.argv[6])
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", task):
    raise SystemExit("task is not a safe path component")
if hand not in {"left", "right", "bimanual"}:
    raise SystemExit("invalid embodiment-type")
if data_id < 0 or not math.isfinite(ref_dt) or ref_dt <= 0:
    raise SystemExit("invalid data-id or ref-dt")

# Preserve the exact source frame grid while satisfying the pinned SPIDER
# native divisibility invariants. Choose the largest substep no coarser than
# the upstream 0.01 s default that exactly divides ref/ctrl/knot/horizon time.
sim_dt = None
for substeps in range(1, 1001):
    candidate = ref_dt / substeps
    if candidate > 0.01 + 1e-12:
        continue
    if all(
        math.isclose(value / candidate, round(value / candidate), abs_tol=1e-9)
        for value in (ref_dt, 0.4, 1.6)
    ):
        sim_dt = candidate
        break
if sim_dt is None:
    raise SystemExit("cannot derive an exact native SPIDER simulation substep")

for path in root.rglob("*"):
    if path.is_symlink():
        raise SystemExit(f"symlink is forbidden in read-only input tree: {path}")

matches = []
for path in root.rglob("task_info.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("task") == task
        and payload.get("dataset_name") == "do_as_i_do"
        and payload.get("embodiment_type") == hand
        and int(payload.get("data_id", -1)) == data_id
        and (path.parent / str(data_id) / "trajectory_keypoints.npz").is_file()
    ):
        matches.append((path, payload))
if len(matches) != 1:
    raise SystemExit(f"expected one exact task_info, found {len(matches)}")
task_info_path, task_info = matches[0]
existing_ref_dt = task_info.get("ref_dt")
if existing_ref_dt is not None and abs(float(existing_ref_dt) - ref_dt) > 1e-12:
    raise SystemExit(f"ref_dt conflicts: task_info={existing_ref_dt}, requested={ref_dt}")

keypoints = task_info_path.parent / str(data_id) / "trajectory_keypoints.npz"
if not keypoints.is_file():
    raise SystemExit(f"missing trajectory keypoints: {keypoints}")
robot_assets = root / "assets" / "robots" / robot
if not robot_assets.is_dir():
    raise SystemExit(f"missing robot assets: {robot_assets}")

object_dirs = []
for side in ("left", "right"):
    for suffix in ("object_mesh_dir", "object_convex_dir"):
        value = task_info.get(f"{side}_{suffix}")
        if value:
            path = root / value
            if not path.exists():
                raise SystemExit(f"missing object asset: {path}")
            object_dirs.append(path)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def files(path):
    return [
        {"relative": str(p.relative_to(root)), "sha256": digest(p)}
        for p in sorted(path.rglob("*"))
        if p.is_file()
    ]

print(json.dumps({
    "task_info": str(task_info_path),
    "task_info_payload": task_info,
    "keypoints": str(keypoints),
    "keypoints_sha256": digest(keypoints),
    "robot_assets": str(robot_assets),
    "robot_files": files(robot_assets),
    "object_dirs": [str(path) for path in object_dirs],
    "object_files": [item for path in object_dirs for item in files(path)],
    "ref_dt": ref_dt,
    "sim_dt": sim_dt,
    "sim_dt_policy": "largest_exact_substep_not_coarser_than_upstream_0.01s",
}))
PY
)" || exit 4

sim_dt="$(
  "$manifest_python" - "$validation_json" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["sim_dt"])
PY
)"

if (( check_only == 1 )); then
  echo "$validation_json"
  exit 0
fi

mkdir -p "$(dirname "$output_dir")"
staging="$(mktemp -d "$(dirname "$output_dir")/.spider-run.XXXXXXXX")"
cleanup() {
  if [[ -d "$staging" ]]; then
    rm -rf "$staging"
  fi
}
trap cleanup EXIT

dataset_root="$staging/dataset"
processed_root="$dataset_root/processed/do_as_i_do"
mkdir -p "$processed_root"

"$manifest_python" - "$validation_json" "$input_root" "$processed_root" \
  "$task" "$embodiment_type" "$data_id" "$ref_dt" <<'PY'
import json
import shutil
import sys
from pathlib import Path

binding = json.loads(sys.argv[1])
source = Path(sys.argv[2])
target = Path(sys.argv[3])
task, hand = sys.argv[4:6]
data_id = int(sys.argv[6])
ref_dt = float(sys.argv[7])

shutil.copytree(source / "assets", target / "assets", symlinks=False)
source_task = Path(binding["task_info"])
target_task_dir = target / "mano" / hand / task
target_task_dir.mkdir(parents=True)
payload = dict(binding["task_info_payload"])
payload["ref_dt"] = ref_dt
for side in ("left", "right"):
    for suffix in ("object_mesh_dir", "object_convex_dir"):
        key = f"{side}_{suffix}"
        value = payload.get(key)
        if value:
            payload[key] = f"processed/do_as_i_do/{value}"
(target_task_dir / "task_info.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
source_keypoints = Path(binding["keypoints"])
target_keypoints = target_task_dir / str(data_id) / "trajectory_keypoints.npz"
target_keypoints.parent.mkdir()
shutil.copy2(source_keypoints, target_keypoints)
PY

backend_path="$(dirname "$python_bin"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
backend_env=(
  "HOME=${HOME:-/tmp}"
  "USER=${USER:-unknown}"
  "LOGNAME=${LOGNAME:-${USER:-unknown}}"
  "PATH=$backend_path"
  "LANG=C.UTF-8"
  "LC_ALL=C.UTF-8"
  "PYTHONUNBUFFERED=1"
  "PYTHONDONTWRITEBYTECODE=1"
  "PYTHONPATH=$spider_root"
)
[[ -z "$cuda_visible_devices" ]] || \
  backend_env+=("CUDA_VISIBLE_DEVICES=$cuda_visible_devices")
[[ -z "$egl_device_id" ]] || backend_env+=(
  "MUJOCO_GL=egl"
  "PYOPENGL_PLATFORM=egl"
  "EGL_DEVICE_ID=$egl_device_id"
)

inherited_removed="$(
  "$manifest_python" - <<'PY'
import json
import os
keep = {"HOME", "USER", "LOGNAME", "PATH", "LANG", "LC_ALL"}
print(json.dumps(sorted(key for key in os.environ if key not in keep)))
PY
)"

write_outcome() {
  local stage="$1"
  local rc="$2"
  local clean_after=true
  if [[ -n "$(git -C "$spider_root" status --porcelain=v1 --untracked-files=all)" ]]; then
    clean_after=false
  fi
  "$manifest_python" - "$staging/vanilla_spider_outcome.json" \
    "$stage" "$rc" "$clean_after" "$actual_commit" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "scope": "pristine_official_spider_native_returncode",
    "stage": sys.argv[2],
    "returncode": int(sys.argv[3]),
    "backend_clean_after": sys.argv[4].lower() == "true",
    "backend_commit_before": sys.argv[5],
    "backend_commit_after": sys.argv[5],
    "backend_patches_applied": [],
    "backend_behavior_overrides": [],
}, indent=2) + "\n", encoding="utf-8")
PY
}

run_backend_stage() {
  local stage="$1"
  shift
  set +e
  env -i "${backend_env[@]}" "$python_bin" "$@"
  local rc=$?
  set -e
  if (( rc != 0 )); then
    write_outcome "$stage" "$rc"
    mv "$staging" "$output_dir"
    trap - EXIT
    return "$rc"
  fi
  return 0
}

run_backend_stage generate_xml \
  "$spider_root/spider/preprocess/generate_xml.py" \
  --dataset-dir "$dataset_root" --dataset-name do_as_i_do \
  --robot-type "$robot_type" --embodiment-type "$embodiment_type" \
  --task "$task" --data-id "$data_id" --no-show-viewer || exit $?

run_backend_stage ik_fast \
  "$spider_root/spider/preprocess/ik_fast.py" \
  --dataset-dir "$dataset_root" --dataset-name do_as_i_do \
  --robot-type "$robot_type" --embodiment-type "$embodiment_type" \
  --task "$task" --data-id "$data_id" --ref-dt "$ref_dt" --no-save-video || exit $?

run_backend_stage run_mjwp \
  "$spider_root/examples/run_mjwp.py" \
  "dataset_dir=$dataset_root" "dataset_name=do_as_i_do" \
  "robot_type=$robot_type" "embodiment_type=$embodiment_type" \
  "task=$task" "data_id=$data_id" "ref_dt=$ref_dt" \
  "sim_dt=$sim_dt" \
  "render_dt=$ref_dt" "trace_dt=$ref_dt" \
  "show_viewer=false" "save_video=true" || exit $?

write_outcome run_mjwp 0

"$manifest_python" - "$staging/vanilla_spider_run_manifest.json" \
  "$validation_json" "$actual_commit" "$actual_tree" "$spider_root" \
  "$inherited_removed" "$processed_root" "$task" "$embodiment_type" \
  "$data_id" "$ref_dt" "$robot_type" "$sim_dt" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
binding = json.loads(sys.argv[2])
commit, tree, spider_root = sys.argv[3:6]
removed = json.loads(sys.argv[6])
processed = Path(sys.argv[7])
task, hand = sys.argv[8:10]
data_id = int(sys.argv[10])
ref_dt = float(sys.argv[11])
robot_type = sys.argv[12]
sim_dt = float(sys.argv[13])

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

staged_keypoints = (
    processed / "mano" / hand / task / str(data_id) / "trajectory_keypoints.npz"
)
robot_files = [
    path for path in (processed / "assets" / "robots").rglob("*") if path.is_file()
]
object_files = [
    path for path in (processed / "assets" / "objects").rglob("*") if path.is_file()
]
manifest = {
    "schema_version": 1,
    "scope": "pristine_official_spider_input_staging_only",
    "backend": {
        "root": spider_root,
        "commit": commit,
        "tree": tree,
        "patches_applied": [],
        "behavior_overrides": [],
    },
    "backend_environment": {
        "clear_environment": True,
        "PYTHONPATH": spider_root,
        "inherited_variables_removed": removed,
    },
    "input_timebase_binding": "exact_dai_keypoint_frame_grid",
    "input_root": str(Path(binding["task_info"]).parents[4]),
    "route": {
        "dataset_name": "do_as_i_do",
        "task": task,
        "robot_type": robot_type,
        "embodiment_type": hand,
        "data_id": data_id,
        "ref_dt": ref_dt,
    },
    "ref_dt": ref_dt,
    "sim_dt": sim_dt,
    "sim_dt_policy": "largest_exact_substep_not_coarser_than_upstream_0.01s",
    "trajectory_keypoints": {
        "source": binding["keypoints"],
        "source_sha256": binding["keypoints_sha256"],
        "staged": str(staged_keypoints),
        "staged_sha256": digest(staged_keypoints),
        "byte_identical": digest(staged_keypoints) == binding["keypoints_sha256"],
        "modified": False,
    },
    "robot_assets": {
        "byte_identical": all(
            any(path.name == Path(item["relative"]).name and digest(path) == item["sha256"]
                for path in robot_files)
            for item in binding["robot_files"]
        ),
    },
    "object_assets": [
        {
            "source": item["relative"],
            "byte_identical": any(
                path.name == Path(item["relative"]).name and digest(path) == item["sha256"]
                for path in object_files
            ),
        }
        for item in binding["object_files"]
    ],
}
out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

[[ -z "$(git -C "$spider_root" status --porcelain=v1 --untracked-files=all)" ]] || \
  die "original SPIDER checkout became dirty" 4
mv "$staging" "$output_dir"
trap - EXIT
