#!/usr/bin/env bash
set -euo pipefail

readonly DAI_PINNED_COMMIT="b5a617060a970ba2d9af7b2e216903c643cda185"
readonly DAI_OFFICIAL_REMOTE="https://github.com/malik-group/do-as-i-do.git"

usage() {
  cat <<'EOF'
Run the pristine, pinned Do-as-I-Do pipeline without changing backend behavior.

Required:
  --raw-dir PATH                 Audited Do-as-I-Do-format input directory.
  --task NAME                    Native Do-as-I-Do task basename.
  --output-root-dir PATH         New archive path for native outputs.

Runtime-only options:
  --cuda-visible-devices VALUE   Value exposed as CUDA_VISIBLE_DEVICES.
  --egl-device-id ID             Logical EGL device (requires --headless).
  --headless                     Pass native --no-show-viewer and use EGL.
  --no-wait                      Pass native --no-wait-on-finish.
  --check-only                   Validate input/runtime/staging without launch.

Environment used only to locate, not alter, the backend:
  RETARGETING_PYTHON             Python executable for the pristine backend.
  DAI_PRISTINE_SOURCE            Clean checkout, bundle, or Git URL to clone.
  DAI_PRISTINE_RUNTIME           Cache for the pinned clean source checkout.
  DAI_PRISTINE_REMOTE            Clone URL if no local source exists.

Safe legacy runtime aliases:
  DAI_CUDA_VISIBLE_DEVICES       Alias for --cuda-visible-devices.
  DAI_EGL_DEVICE_ID              Alias for --egl-device-id.

The pinned launch.py does not consistently propagate --output-root-dir to all
native stages.  This wrapper therefore runs launch.py unmodified in a fresh,
disposable checkout and lets every stage use its native retargeting/outputs
default.  The pinned upstream commit tracks example files under that output
directory, so the disposable checkout uses Git sparse-checkout to omit only
that tracked seed-output subtree before launch.  Code, configs, and runtime
assets remain byte-identical to the pinned commit.  The complete newly created
native output directory is atomically renamed to the requested archive path
after the backend exits.  Native files are never rewritten, and the launch.py
return code remains authoritative.
If publishing the untouched native output directory fails, the wrapper exits
nonzero while preserving the native return code in publication_failure.json.
EOF
}

die() {
  echo "run_do_as_i_do_official_retarget: $*" >&2
  exit 2
}

require_value() {
  local option="$1"
  local remaining="$2"
  if (( remaining < 2 )); then
    die "$option requires a value"
  fi
}

# Fail closed when an old launcher environment would silently change the
# upstream optimizer, preprocessing, renderer, or acceptance behavior.  The
# five DAI_* names below are wrapper-only source/device locators; every other
# DAI_* variable is intentionally rejected instead of relying only on env -i.
while IFS= read -r variable_name; do
  case "$variable_name" in
    DAI_CUDA_VISIBLE_DEVICES|DAI_EGL_DEVICE_ID|DAI_PRISTINE_SOURCE|DAI_PRISTINE_RUNTIME|DAI_PRISTINE_REMOTE)
      ;;
    DAI_*|MJWP_*|DO_AS_I_DO_*|ROBOT_RENDER_*)
      die "deprecated backend-tuning environment variable is forbidden: $variable_name"
      ;;
  esac
done < <(compgen -e)

raw_dir=""
task=""
output_root_dir=""
cuda_visible_devices=""
egl_device_id=""
headless=0
no_wait=0
check_only=0
cuda_from_cli=0
egl_from_cli=0

while (( $# > 0 )); do
  case "$1" in
    --raw-dir)
      require_value "$1" "$#"
      raw_dir="$2"
      shift 2
      ;;
    --task)
      require_value "$1" "$#"
      task="$2"
      shift 2
      ;;
    --output-root-dir)
      require_value "$1" "$#"
      output_root_dir="$2"
      shift 2
      ;;
    --cuda-visible-devices)
      require_value "$1" "$#"
      cuda_visible_devices="$2"
      cuda_from_cli=1
      shift 2
      ;;
    --egl-device-id)
      require_value "$1" "$#"
      egl_device_id="$2"
      egl_from_cli=1
      shift 2
      ;;
    --headless)
      headless=1
      shift
      ;;
    --no-wait)
      no_wait=1
      shift
      ;;
    --check-only)
      check_only=1
      shift
      ;;
    --run-name|--trajectory-6dof|--hand-type|--hand-source|--robot-type|--max-sim-steps|--force|--no-force|--copy)
      die "deprecated option $1 is forbidden; provide only input, task, output, GPU/EGL, --headless, --no-wait, and --check-only"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ -n "$raw_dir" ]] || die "missing --raw-dir"
[[ -n "$task" ]] || die "missing --task"
[[ -n "$output_root_dir" ]] || die "missing --output-root-dir"

if (( cuda_from_cli == 0 )); then
  if [[ -n "${DAI_CUDA_VISIBLE_DEVICES:-}" ]]; then
    cuda_visible_devices="$DAI_CUDA_VISIBLE_DEVICES"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    cuda_visible_devices="$CUDA_VISIBLE_DEVICES"
  fi
elif [[ -n "${DAI_CUDA_VISIBLE_DEVICES:-}" && "$DAI_CUDA_VISIBLE_DEVICES" != "$cuda_visible_devices" ]]; then
  die "conflicting --cuda-visible-devices and DAI_CUDA_VISIBLE_DEVICES"
fi

if (( egl_from_cli == 0 )); then
  if [[ -n "${DAI_EGL_DEVICE_ID:-}" ]]; then
    egl_device_id="$DAI_EGL_DEVICE_ID"
  elif [[ -n "${EGL_DEVICE_ID:-}" ]]; then
    egl_device_id="$EGL_DEVICE_ID"
  fi
elif [[ -n "${DAI_EGL_DEVICE_ID:-}" && "$DAI_EGL_DEVICE_ID" != "$egl_device_id" ]]; then
  die "conflicting --egl-device-id and DAI_EGL_DEVICE_ID"
fi

if (( headless == 0 )) && [[ -n "$egl_device_id" ]]; then
  die "--egl-device-id requires --headless"
fi
if (( headless == 1 )) && [[ -z "$egl_device_id" ]]; then
  egl_device_id="0"
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
manifest_python="$(command -v python3 || true)"
[[ -n "$manifest_python" ]] || die "python3 is required for path and provenance handling"

python_bin="${RETARGETING_PYTHON:-$(command -v python3 || true)}"
[[ -n "$python_bin" && -x "$python_bin" ]] || die "RETARGETING_PYTHON is not executable: $python_bin"

if ! "$manifest_python" - "$task" <<'PY'
import re
import sys

task = sys.argv[1]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", task):
    raise SystemExit(1)
if task in {".", ".."}:
    raise SystemExit(1)
PY
then
  die "--task must be a strict safe basename containing only letters, digits, '.', '_', or '-': $task"
fi

raw_dir="$($manifest_python - "$raw_dir" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1]).expanduser().resolve(strict=True)
if not path.is_dir():
    raise SystemExit(f"raw-dir is not a directory: {path}")
print(path)
PY
)" || die "cannot resolve --raw-dir"

output_root_dir="$($manifest_python - "$output_root_dir" <<'PY'
import os
import sys
from pathlib import Path

raw = Path(sys.argv[1]).expanduser()
if os.path.lexists(raw):
    raise SystemExit(f"output archive path already exists; a new path is required: {raw}")
print(raw.resolve(strict=False))
PY
)" || die "--output-root-dir must name a brand-new path"

if ! "$manifest_python" - "$raw_dir" "$output_root_dir" <<'PY'
import os
import sys
from pathlib import Path

raw = Path(sys.argv[1]).resolve(strict=True)
output = Path(sys.argv[2]).resolve(strict=False)
common = Path(os.path.commonpath((str(raw), str(output))))
if common == raw or common == output:
    raise SystemExit(f"raw-dir and output-root-dir must not overlap: {raw} vs {output}")
PY
then
  die "raw-dir and output-root-dir must be disjoint (neither may contain the other)"
fi

output_parent="$(dirname "$output_root_dir")"
mkdir -p "$output_parent"
output_parent="$(cd "$output_parent" && pwd -P)"
output_root_dir="$output_parent/$(basename "$output_root_dir")"

default_local_source="$repo_root/third_party/do-as-i-do"
if git -C "$default_local_source" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  pristine_source="${DAI_PRISTINE_SOURCE:-$default_local_source}"
else
  pristine_source="${DAI_PRISTINE_SOURCE:-${DAI_PRISTINE_REMOTE:-$DAI_OFFICIAL_REMOTE}}"
fi
pristine_runtime="${DAI_PRISTINE_RUNTIME:-${XDG_CACHE_HOME:-$HOME/.cache}/aoe-retarget/pristine-do-as-i-do-$DAI_PINNED_COMMIT}"
pristine_runtime="$($manifest_python - "$pristine_runtime" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
)" || die "cannot resolve DAI_PRISTINE_RUNTIME"

checkout_head() {
  git -C "$1" rev-parse --verify HEAD 2>/dev/null
}

checkout_tree() {
  git -C "$1" rev-parse --verify 'HEAD^{tree}' 2>/dev/null
}

checkout_status() {
  git -C "$1" status --porcelain=v1 --untracked-files=all 2>/dev/null
}

verify_clean_checkout() {
  local checkout="$1"
  local label="$2"
  local actual_head
  local actual_status
  git -C "$checkout" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
    die "$label is not a Git checkout: $checkout"
  actual_head="$(checkout_head "$checkout")" || die "cannot read $label commit: $checkout"
  [[ "$actual_head" == "$DAI_PINNED_COMMIT" ]] || \
    die "$label commit is $actual_head, expected $DAI_PINNED_COMMIT"
  actual_status="$(checkout_status "$checkout")" || die "cannot inspect $label status: $checkout"
  [[ -z "$actual_status" ]] || die "$label is dirty; refusing to run or modify it: $checkout"
}

# A local worktree source must itself be the exact clean pinned checkout. A Git
# bundle or URL is immutable input to clone and is verified after checkout.
if git -C "$pristine_source" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  verify_clean_checkout "$pristine_source" "DAI_PRISTINE_SOURCE"
fi

if [[ ! -e "$pristine_runtime" ]]; then
  runtime_parent="$(dirname "$pristine_runtime")"
  mkdir -p "$runtime_parent"
  staging_runtime="$(mktemp -d "$runtime_parent/.pristine-do-as-i-do.XXXXXXXX")"
  if ! git clone --quiet --no-checkout --no-hardlinks "$pristine_source" "$staging_runtime"; then
    echo "incomplete clone preserved for diagnosis: $staging_runtime" >&2
    exit 3
  fi
  if ! git -C "$staging_runtime" checkout --quiet --detach "$DAI_PINNED_COMMIT"; then
    echo "clone does not contain pinned commit; staging preserved: $staging_runtime" >&2
    exit 3
  fi
  verify_clean_checkout "$staging_runtime" "new pristine DAI source checkout"
  if ! mv "$staging_runtime" "$pristine_runtime"; then
    echo "could not install pristine checkout; staging preserved: $staging_runtime" >&2
    exit 3
  fi
fi
verify_clean_checkout "$pristine_runtime" "DAI_PRISTINE_RUNTIME"

if ! "$manifest_python" - "$pristine_runtime" "$raw_dir" "$output_root_dir" <<'PY'
import os
import sys
from pathlib import Path

runtime, raw, output = (Path(value).resolve(strict=False) for value in sys.argv[1:])
for left_name, left, right_name, right in (
    ("pristine runtime", runtime, "raw-dir", raw),
    ("pristine runtime", runtime, "output-root-dir", output),
):
    common = Path(os.path.commonpath((str(left), str(right))))
    if common == left or common == right:
        raise SystemExit(f"{left_name} and {right_name} must not overlap: {left} vs {right}")
PY
then
  die "input/output placement overlaps the pristine source checkout"
fi

base_head_before="$(checkout_head "$pristine_runtime")"
base_tree_before="$(checkout_tree "$pristine_runtime")"
base_status_before="$(checkout_status "$pristine_runtime")"
backend_origin="$(git -C "$pristine_runtime" remote get-url origin 2>/dev/null || true)"

run_workspace="$(mktemp -d "$output_parent/.aoe-pristine-dai-run.XXXXXXXX")"
disposable_runtime="$run_workspace/backend"
keep_workspace=0
cleanup_workspace() {
  if [[ -n "${run_workspace:-}" && -d "$run_workspace" && "$keep_workspace" -eq 0 ]]; then
    rm -rf "$run_workspace"
  fi
}
trap cleanup_workspace EXIT

if ! git clone --quiet --no-checkout --no-hardlinks "$pristine_runtime" "$disposable_runtime"; then
  keep_workspace=1
  echo "could not create disposable pristine checkout; preserved: $run_workspace" >&2
  exit 3
fi
if ! git -C "$disposable_runtime" checkout --quiet --detach "$DAI_PINNED_COMMIT"; then
  keep_workspace=1
  echo "could not checkout pinned commit in disposable runtime; preserved: $run_workspace" >&2
  exit 3
fi
verify_clean_checkout "$disposable_runtime" "disposable pristine DAI checkout"

launch_py="$disposable_runtime/retargeting/launch.py"
[[ -f "$launch_py" ]] || die "pinned checkout lacks retargeting/launch.py: $launch_py"
native_output_root="$disposable_runtime/retargeting/outputs"
native_seed_outputs_excluded=false
if [[ -e "$native_output_root" || -L "$native_output_root" ]]; then
  [[ -d "$native_output_root" && ! -L "$native_output_root" ]] || \
    die "pinned checkout retargeting/outputs is not a real directory"
  # b5a617 tracks example outputs. Exclude exactly that seed-output subtree in
  # the disposable worktree, while leaving HEAD, the index, all backend code,
  # configs, and runtime assets untouched. Git status must remain clean.
  git -C "$disposable_runtime" sparse-checkout init --no-cone || \
    die "cannot initialize sparse output isolation in disposable checkout"
  git -C "$disposable_runtime" sparse-checkout set \
    '/*' '!/retargeting/outputs/' || \
    die "cannot exclude tracked upstream seed outputs"
  [[ ! -e "$native_output_root" && ! -L "$native_output_root" ]] || \
    die "tracked upstream seed outputs remain after sparse isolation"
  verify_clean_checkout "$disposable_runtime" \
    "sparse disposable pristine DAI checkout"
  native_seed_outputs_excluded=true
fi

fingerprint_tree() {
  local root="$1"
  local output="$2"
  "$manifest_python" - "$root" "$output" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
output = Path(sys.argv[2])
if not root.is_dir():
    raise SystemExit(f"fingerprint root is not a directory: {root}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_target(path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return f"EXTERNAL:{path}"
    return f"ROOT:{relative.as_posix()}"


def collect(path: Path, logical: str, ancestors: frozenset[tuple[int, int]]) -> list[dict]:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        target_text = os.readlink(path)
        resolved = path.resolve(strict=True)
        target_records = collect(resolved, f"{logical}/@resolved", ancestors)
        target_digest = hashlib.sha256(
            json.dumps(target_records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return [
            {
                "type": "symlink",
                "path": logical,
                "link_text": target_text,
                "resolved_target": canonical_target(resolved),
                "resolved_target_tree_sha256": target_digest,
            },
            *target_records,
        ]
    if stat.S_ISREG(info.st_mode):
        return [
            {
                "type": "file",
                "path": logical,
                "size": info.st_size,
                "sha256": file_sha256(path),
            }
        ]
    if stat.S_ISDIR(info.st_mode):
        inode = (info.st_dev, info.st_ino)
        if inode in ancestors:
            raise RuntimeError(f"symlink directory cycle at {logical}: {path}")
        records = [{"type": "directory", "path": logical}]
        next_ancestors = ancestors | {inode}
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            child_logical = child.name if logical == "." else f"{logical}/{child.name}"
            records.extend(collect(child, child_logical, next_ancestors))
        return records
    raise RuntimeError(f"unsupported special file in fingerprint at {logical}: {path}")


records = collect(root, ".", frozenset())
encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
payload = {
    "root": str(root),
    "tree_sha256": hashlib.sha256(encoded).hexdigest(),
    "record_count": len(records),
    "regular_file_count": sum(record["type"] == "file" for record in records),
    "symlink_count": sum(record["type"] == "symlink" for record in records),
    "regular_file_bytes": sum(
        int(record.get("size", 0)) for record in records if record["type"] == "file"
    ),
    "resolved_symlink_targets": [
        {
            "path": record["path"],
            "link_text": record["link_text"],
            "resolved_target": record["resolved_target"],
            "resolved_target_tree_sha256": record["resolved_target_tree_sha256"],
        }
        for record in records
        if record["type"] == "symlink"
    ],
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

validate_input_schema() {
  local root="$1"
  local expected_task="$2"
  local output="$3"
  "$manifest_python" - "$root" "$expected_task" "$output" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
task = sys.argv[2]
output = Path(sys.argv[3])


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    return path.resolve(strict=True)


config_path = require_file(root / "config.json", "config.json")
config = json.loads(config_path.read_text(encoding="utf-8"))
object_names = config.get("object_names")
if not isinstance(object_names, list) or not object_names or not isinstance(object_names[0], str):
    raise RuntimeError("config.json object_names must be a non-empty string list")
object_name = object_names[0]
if not object_name or Path(object_name).name != object_name or object_name in {".", ".."} or "\\" in object_name:
    raise RuntimeError(f"unsafe object name in config.json: {object_name!r}")
anchor_hand = config.get("anchor_hand", "bimanual")
if anchor_hand not in {"left", "right", "bimanual"}:
    raise RuntimeError(f"invalid config.json anchor_hand: {anchor_hand!r}")

exact_hand_npz = root / task / "all_hand_meshes.npz"
require_file(exact_hand_npz, "route-specific all_hand_meshes.npz")
hand_candidates = [exact_hand_npz]

tracking_root = root / "obj_tracking_out" / object_name
layout_path = require_file(
    tracking_root / "combined_visualization" / "layout_camera_frame_optimized.json",
    "optimized object layout",
)
layout = json.loads(layout_path.read_text(encoding="utf-8"))
if not isinstance(layout.get("objects"), list):
    raise RuntimeError("optimized layout must contain an objects list")
scale = (layout.get("translation_scale_optimization") or {}).get("mesh_scale")
if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) <= 0:
    raise RuntimeError("optimized layout mesh_scale must be finite and positive")

gravity_path = require_file(root / "gravity.json", "gravity.json")
gravity = json.loads(gravity_path.read_text(encoding="utf-8"))
vec3d = gravity.get("vec3d")
if (
    not isinstance(vec3d, list)
    or len(vec3d) != 3
    or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in vec3d)
    or sum(float(value) ** 2 for value in vec3d) <= 0
):
    raise RuntimeError("gravity.json vec3d must be a finite nonzero 3-vector")

mesh_candidates = sorted(
    path
    for path in (root / "video_segmentation" / "masks").glob(
        f"frame_*_masks/{object_name}/{object_name}.obj"
    )
    if path.is_file()
)
if not mesh_candidates:
    raise RuntimeError(f"missing canonical object mesh for {object_name!r}")

payload = {
    "schema_version": 1,
    "status": "pass",
    "task": task,
    "config": str(config_path),
    "object_name": object_name,
    "resolved_anchor_hand": anchor_hand,
    "hand_mesh_candidates": [str(path.resolve(strict=True)) for path in hand_candidates],
    "selected_hand_mesh_npz": str(hand_candidates[0].resolve(strict=True)),
    "ambiguous_hand_mesh_fallback": False,
    "object_layout": str(layout_path),
    "gravity": str(gravity_path),
    "object_mesh_candidates": [str(path.resolve(strict=True)) for path in mesh_candidates],
    "selected_object_mesh": str(mesh_candidates[0].resolve(strict=True)),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

input_fingerprint_before_path="$run_workspace/input_fingerprint_before.json"
input_schema_path="$run_workspace/input_schema.json"
fingerprint_tree "$raw_dir" "$input_fingerprint_before_path" || \
  die "cannot fingerprint raw input (broken/cyclic symlink or unsupported file)"
validate_input_schema "$raw_dir" "$task" "$input_schema_path" || \
  die "raw input does not satisfy the pinned native DAI filesystem schema"

if (( check_only == 1 )); then
  input_tree_sha="$($manifest_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["tree_sha256"])' "$input_fingerprint_before_path")"
  resolved_anchor_hand="$($manifest_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["resolved_anchor_hand"])' "$input_schema_path")"
  base_status_after="$(checkout_status "$pristine_runtime")"
  [[ -z "$base_status_after" ]] || die "pristine source checkout changed during check-only"
  printf 'DAI_CHECK_ONLY_OK commit=%s task=%s resolved_anchor_hand=%s input_tree_sha256=%s native_output_root=%s tracked_seed_outputs_excluded=%s\n' \
    "$DAI_PINNED_COMMIT" "$task" "$resolved_anchor_hand" "$input_tree_sha" \
    "retargeting/outputs" "$native_seed_outputs_excluded"
  exit 0
fi

backend_head_before="$(checkout_head "$disposable_runtime")"
backend_tree_before="$(checkout_tree "$disposable_runtime")"
backend_status_before="$(checkout_status "$disposable_runtime")"

backend_python_dir="$(cd "$(dirname "$python_bin")" && pwd -P)"
backend_path="$backend_python_dir:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
backend_env=(
  "HOME=${HOME:-/tmp}"
  "USER=${USER:-unknown}"
  "LOGNAME=${LOGNAME:-${USER:-unknown}}"
  "PATH=$backend_path"
  "LANG=C.UTF-8"
  "LC_ALL=C.UTF-8"
  "PYTHONUNBUFFERED=1"
  "PYTHONDONTWRITEBYTECODE=1"
)
if [[ -n "$cuda_visible_devices" ]]; then
  backend_env+=("CUDA_VISIBLE_DEVICES=$cuda_visible_devices")
fi
if (( headless == 1 )); then
  backend_env+=(
    "MUJOCO_GL=egl"
    "PYOPENGL_PLATFORM=egl"
    "EGL_DEVICE_ID=$egl_device_id"
  )
else
  [[ -z "${DISPLAY:-}" ]] || backend_env+=("DISPLAY=$DISPLAY")
  [[ -z "${XAUTHORITY:-}" ]] || backend_env+=("XAUTHORITY=$XAUTHORITY")
  [[ -z "${WAYLAND_DISPLAY:-}" ]] || backend_env+=("WAYLAND_DISPLAY=$WAYLAND_DISPLAY")
  [[ -z "${XDG_RUNTIME_DIR:-}" ]] || backend_env+=("XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR")
fi

# Deliberately omit --output-root-dir.  At pinned b5a617 launch.py forwards that
# option only to some stages.  The native default is the sole consistent root.
backend_argv=(
  "launch.py"
  "--raw-dir" "$raw_dir"
  "--task" "$task"
)
if (( headless == 1 )); then
  backend_argv+=("--no-show-viewer")
fi
if (( no_wait == 1 )); then
  backend_argv+=("--no-wait-on-finish")
fi

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "DAI pristine commit: $DAI_PINNED_COMMIT"
echo "DAI pristine source runtime: $pristine_runtime"
echo "DAI disposable runtime: $disposable_runtime"
echo "DAI raw input: $raw_dir"
echo "DAI native default output root: $native_output_root"
echo "DAI native archive destination: $output_root_dir"
printf 'DAI native argv:'
printf ' %q' "$python_bin" "${backend_argv[@]}"
printf '\n'

previous_directory="$PWD"
cd "$disposable_runtime/retargeting"
keep_workspace=1
set +e
env -i "${backend_env[@]}" "$python_bin" "${backend_argv[@]}"
backend_rc=$?
set -e
cd "$previous_directory"
ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

publication_failure() {
  local reason="$1"
  local wrapper_rc=74
  keep_workspace=1
  "$manifest_python" - \
    "$run_workspace/publication_failure.json" "$backend_rc" "$wrapper_rc" \
    "$reason" "$native_output_root" "$output_root_dir" <<'PY' || true
import json
import os
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "scope": "pristine_dai_wrapper_publication_failure",
    "status": "publication_failed",
    "native_backend_rc": int(sys.argv[2]),
    "native_backend_rc_authoritative": True,
    "wrapper_rc": int(sys.argv[3]),
    "reason": sys.argv[4],
    "native_output_root": sys.argv[5],
    "requested_archive": sys.argv[6],
}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    "w", encoding="utf-8", dir=path.parent, delete=False
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    temporary = Path(handle.name)
os.replace(temporary, path)
PY
  echo "$reason; native backend rc=$backend_rc; wrapper rc=$wrapper_rc; workspace preserved: $run_workspace" >&2
  exit "$wrapper_rc"
}

input_fingerprint_after_path="$run_workspace/input_fingerprint_after.json"
if ! fingerprint_tree "$raw_dir" "$input_fingerprint_after_path"; then
  printf '{"error":"input fingerprint failed after native run"}\n' > "$input_fingerprint_after_path"
fi

if [[ -L "$native_output_root" || ( -e "$native_output_root" && ! -d "$native_output_root" ) ]]; then
  publication_failure "native output root is not a real directory"
fi
if ! mkdir -p "$native_output_root"; then
  publication_failure "cannot materialize native output root"
fi

native_fingerprint_before_path="$run_workspace/native_fingerprint_before.json"
native_fingerprint_after_path="$run_workspace/native_fingerprint_after.json"
if ! fingerprint_tree "$native_output_root" "$native_fingerprint_before_path"; then
  printf '{"error":"native output fingerprint failed before archive"}\n' \
    > "$native_fingerprint_before_path"
fi

# run_workspace lives beside output_root_dir, so os.rename is a same-filesystem
# atomic publication.  It also fails rather than nesting under a raced output.
if ! "$manifest_python" - "$native_output_root" "$output_root_dir" <<'PY'
import os
import sys

source, target = sys.argv[1:]
if os.path.lexists(target):
    raise SystemExit(f"archive destination appeared during run: {target}")
os.rename(source, target)
PY
then
  publication_failure "native output archive failed"
fi

if ! fingerprint_tree "$output_root_dir" "$native_fingerprint_after_path"; then
  printf '{"error":"native output fingerprint failed after archive"}\n' \
    > "$native_fingerprint_after_path"
fi

resolved_route_path="$run_workspace/resolved_native_route.json"
if ! "$manifest_python" - "$output_root_dir" "$task" "$resolved_route_path" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
expected_task = sys.argv[2]
output = Path(sys.argv[3])
candidates = sorted(root.glob(f"mano/*/{expected_task}/task_info.json"))
errors = []
resolved = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if len(candidates) != 1:
    errors.append(f"expected exactly one native task_info.json, found {len(candidates)}")
else:
    task_info_path = candidates[0]
    try:
        payload = json.loads(task_info_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"cannot parse native task_info.json: {exc}")
    else:
        hand = payload.get("embodiment_type")
        expected = {
            "task": expected_task,
            "dataset_name": "do_as_i_do",
            "robot_type": "mano",
            "data_id": 0,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                errors.append(f"native task_info {key}={payload.get(key)!r}, expected {value!r}")
        if hand not in {"left", "right", "bimanual"}:
            errors.append(f"native task_info embodiment_type is invalid: {hand!r}")
        elif task_info_path.parts[-3] != hand:
            errors.append(
                f"native task_info path hand {task_info_path.parts[-3]!r} != payload {hand!r}"
            )
        keypoints = root / "mano" / str(hand) / expected_task / "0" / "trajectory_keypoints.npz"
        if not keypoints.is_file():
            errors.append(f"native trajectory_keypoints.npz missing: {keypoints}")
        robot_output_root = root / "sharpa" / str(hand) / expected_task / "0"
        robot_trajectory_candidates = sorted(
            path
            for path in robot_output_root.glob("trajectory*.npz")
            if path.is_file()
        )
        route_native_candidates = sorted(
            {
                *(
                    path
                    for path in robot_output_root.rglob("*")
                    if path.is_file()
                ),
                *(
                    path
                    for path in robot_output_root.parent.glob("scene*.xml")
                    if path.is_file()
                ),
            },
            key=lambda path: path.relative_to(root).as_posix(),
        )

        def artifact_record(path: Path) -> dict:
            return {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "size": path.stat().st_size,
            }

        resolved = {
            "dataset_name": "do_as_i_do",
            "task": expected_task,
            "data_id": 0,
            "retarget_robot_type": "sharpa",
            "embodiment_type": hand,
            "hand_type": hand,
            "task_info_relative_path": task_info_path.relative_to(root).as_posix(),
            "task_info_sha256": sha256(task_info_path),
            "trajectory_keypoints_relative_path": keypoints.relative_to(root).as_posix(),
            "trajectory_keypoints_sha256": sha256(keypoints) if keypoints.is_file() else None,
            "robot_output_relative_path": robot_output_root.relative_to(root).as_posix(),
            "robot_trajectory_artifacts": [
                artifact_record(path)
                for path in robot_trajectory_candidates
            ],
            "route_native_artifacts": [
                artifact_record(path) for path in route_native_candidates
            ],
        }

result = {
    "schema_version": 1,
    "status": "ok" if not errors else ("missing" if not candidates else "invalid"),
    "errors": errors,
    "resolved": resolved,
    "candidate_task_info_relative_paths": [
        path.relative_to(root).as_posix() for path in candidates
    ],
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
then
  printf '{"status":"error","errors":["native route provenance resolver failed"],"resolved":null}\n' \
    > "$resolved_route_path"
fi

# Do not chmod, rewrite, normalize, or otherwise mutate native payload files.
# Fresh-path publication plus exact fingerprints provide immutability evidence
# without changing upstream file metadata.
payload_read_only=false

backend_head_after="$(checkout_head "$disposable_runtime" || true)"
backend_tree_after="$(checkout_tree "$disposable_runtime" || true)"
backend_status_after="$(checkout_status "$disposable_runtime" || true)"
base_head_after="$(checkout_head "$pristine_runtime" || true)"
base_tree_after="$(checkout_tree "$pristine_runtime" || true)"
base_status_after="$(checkout_status "$pristine_runtime" || true)"

workspace_preserved=true
if [[ "$backend_head_after" == "$backend_head_before" &&
      "$backend_tree_after" == "$backend_tree_before" &&
      -z "$backend_status_after" ]]; then
  workspace_preserved=false
fi

manifest_path="$output_root_dir/aoe_pristine_dai_run_manifest.json"
write_manifest() {
  "$manifest_python" - \
    "$manifest_path" "$backend_rc" "$DAI_PINNED_COMMIT" \
    "$pristine_runtime" "$disposable_runtime" "$backend_origin" \
    "$raw_dir" "$task" "$output_root_dir" "$python_bin" \
    "$started_at" "$ended_at" \
    "$base_head_before" "$base_tree_before" "$base_status_before" \
    "$base_head_after" "$base_tree_after" "$base_status_after" \
    "$backend_head_before" "$backend_tree_before" "$backend_status_before" \
    "$backend_head_after" "$backend_tree_after" "$backend_status_after" \
    "$input_fingerprint_before_path" "$input_fingerprint_after_path" \
    "$input_schema_path" "$native_fingerprint_before_path" \
    "$native_fingerprint_after_path" "$resolved_route_path" \
    "$payload_read_only" "$workspace_preserved" \
    "$native_seed_outputs_excluded" \
    --env "${backend_env[@]}" --argv "$python_bin" "${backend_argv[@]}" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path

(
    manifest_raw,
    backend_rc_raw,
    pinned_commit,
    source_runtime_raw,
    disposable_runtime_raw,
    origin,
    raw_dir_raw,
    task,
    output_root_raw,
    python_bin,
    started_at,
    ended_at,
    base_head_before,
    base_tree_before,
    base_status_before,
    base_head_after,
    base_tree_after,
    base_status_after,
    run_head_before,
    run_tree_before,
    run_status_before,
    run_head_after,
    run_tree_after,
    run_status_after,
    input_before_raw,
    input_after_raw,
    input_schema_raw,
    native_before_raw,
    native_after_raw,
    resolved_route_raw,
    payload_read_only_raw,
    workspace_preserved_raw,
    native_seed_outputs_excluded_raw,
    *tail,
) = sys.argv[1:]

env_index = tail.index("--env")
argv_index = tail.index("--argv")
environment_entries = tail[env_index + 1 : argv_index]
backend_argv = tail[argv_index + 1 :]


def load(path_raw: str) -> dict:
    with Path(path_raw).open("r", encoding="utf-8") as handle:
        return json.load(handle)


input_before = load(input_before_raw)
input_after = load(input_after_raw)
input_schema = load(input_schema_raw)
native_before = load(native_before_raw)
native_after = load(native_after_raw)
resolved_route = load(resolved_route_raw)
resolved = resolved_route.get("resolved") or {}

payload = {
    "schema_version": 2,
    "scope": "pristine_official_dai_no_backend_modification",
    "status": "backend_exited",
    "backend_rc": int(backend_rc_raw),
    "backend_rc_authoritative": True,
    "started_at_utc": started_at,
    "ended_at_utc": ended_at,
    "backend": {
        "name": "do-as-i-do",
        "pinned_commit": pinned_commit,
        "source_runtime_root": str(Path(source_runtime_raw)),
        "disposable_runtime_root": str(Path(disposable_runtime_raw)),
        "origin": origin,
        "entrypoint": "retargeting/launch.py",
        "default_output_root_used": True,
        "default_output_root_relative_path": "retargeting/outputs",
        "tracked_seed_outputs_excluded_via_sparse_checkout": (
            native_seed_outputs_excluded_raw == "true"
        ),
        "commit_before": run_head_before,
        "tree_before": run_tree_before,
        "status_before": run_status_before,
        "commit_after": run_head_after,
        "tree_after": run_tree_after,
        "status_after": run_status_after,
        "tree_unchanged": (
            run_head_before == run_head_after
            and run_tree_before == run_tree_after
            and run_status_before == ""
            and run_status_after == ""
        ),
        "source_checkout": {
            "commit_before": base_head_before,
            "tree_before": base_tree_before,
            "status_before": base_status_before,
            "commit_after": base_head_after,
            "tree_after": base_tree_after,
            "status_after": base_status_after,
            "tree_unchanged": (
                base_head_before == base_head_after
                and base_tree_before == base_tree_after
                and base_status_before == ""
                and base_status_after == ""
            ),
        },
        "disposable_workspace_preserved_for_diagnosis": workspace_preserved_raw == "true",
        "patches_applied": [],
        "behavior_overrides": [],
    },
    "input": {
        "raw_dir": str(Path(raw_dir_raw)),
        "schema": input_schema,
        "fingerprint_before": input_before,
        "fingerprint_after": input_after,
        "unchanged_during_native_run": (
            input_before.get("tree_sha256") is not None
            and input_before.get("tree_sha256") == input_after.get("tree_sha256")
        ),
        "tree_sha256": input_before.get("tree_sha256"),
        "resolved_symlink_targets": input_before.get("resolved_symlink_targets", []),
    },
    "task": task,
    "resolved_route": resolved_route,
    "resolved_embodiment_type": resolved.get("embodiment_type"),
    "resolved_hand_type": resolved.get("hand_type"),
    "native_output_layout": {
        "upstream_runtime_relative_root": "retargeting/outputs",
        "archive_root": str(Path(output_root_raw)),
        "mano_relative_root": "mano",
        "assets_relative_root": "assets",
        "retarget_robot_relative_root": "sharpa",
        "tracked_upstream_seed_outputs_present": (
            native_seed_outputs_excluded_raw == "true"
        ),
        "tracked_upstream_seed_outputs_excluded_via_sparse_checkout": (
            native_seed_outputs_excluded_raw == "true"
        ),
        "task_info_relative_path": resolved.get("task_info_relative_path"),
        "trajectory_keypoints_relative_path": resolved.get(
            "trajectory_keypoints_relative_path"
        ),
        "robot_output_relative_path": resolved.get("robot_output_relative_path"),
    },
    "output_root_dir": str(Path(output_root_raw)),
    "native_archive": {
        "publication": "same_filesystem_atomic_directory_rename",
        "native_payload_fingerprint_before": native_before,
        "archived_payload_fingerprint": native_after,
        "payload_byte_tree_identical": (
            native_before.get("tree_sha256") is not None
            and native_before.get("tree_sha256") == native_after.get("tree_sha256")
        ),
        "payload_files_made_read_only": payload_read_only_raw == "true",
        "payload_file_metadata_mutated": False,
        "wrapper_manifest_relative_path": "aoe_pristine_dai_run_manifest.json",
    },
    "python": python_bin,
    "argv": backend_argv,
    "environment": dict(entry.split("=", 1) for entry in environment_entries),
    "native_outputs_deleted_or_rewritten": False,
    "native_outputs_archived": True,
    "custom_acceptance_gate": False,
    "wrapper_safety_checks": [
        "new_output_archive_path",
        "raw_output_nonoverlap",
        "strict_task_basename",
        "recursive_symlink_target_fingerprint",
        "native_input_filesystem_schema",
        "clean_pinned_source_checkout",
        "clean_fresh_disposable_checkout",
        "tracked_seed_output_sparse_isolation",
        "post_archive_checkout_status",
    ],
}

manifest = Path(manifest_raw)
manifest.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(
    mode="w",
    encoding="utf-8",
    dir=manifest.parent,
    prefix=f".{manifest.name}.",
    delete=False,
) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    temporary = Path(handle.name)
os.replace(temporary, manifest)
PY
}

if ! write_manifest; then
  echo "warning: could not write final native archive manifest: $manifest_path" >&2
else
  chmod a-w "$manifest_path" || true
fi

if [[ "$workspace_preserved" == "false" ]]; then
  keep_workspace=0
else
  echo "warning: disposable backend checkout is dirty after moving native outputs; preserved: $run_workspace" >&2
fi
if [[ -n "$base_status_after" || "$base_head_after" != "$base_head_before" || "$base_tree_after" != "$base_tree_before" ]]; then
  echo "warning: pristine source checkout changed; recorded in $manifest_path" >&2
fi

# The original launch.py return code is authoritative for the backend outcome.
# Wrapper input/publication failures remain separate nonzero wrapper outcomes;
# post-run diagnostics never reinterpret backend success or failure.
exit "$backend_rc"
