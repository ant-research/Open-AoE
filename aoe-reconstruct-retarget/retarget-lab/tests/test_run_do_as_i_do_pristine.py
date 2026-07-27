from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_do_as_i_do_official_retarget.sh"
PINNED_COMMIT = "b5a617060a970ba2d9af7b2e216903c643cda185"
FAKE_TREE = "1" * 40


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            digest.update(relative.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _sandbox(
    tmp_path: Path,
    *,
    backend_rc: int = 0,
    dirty: bool = False,
    backend_head: str = PINNED_COMMIT,
    native_output_as_file: bool = False,
    tracked_seed_outputs: bool = False,
) -> dict:
    runtime = tmp_path / "pristine-dai"
    (runtime / ".git").mkdir(parents=True)
    (runtime / "retargeting").mkdir()
    (runtime / "retargeting" / "launch.py").write_text(
        "raise AssertionError('the fake python executable handles this test')\n",
        encoding="utf-8",
    )
    if tracked_seed_outputs:
        seed = runtime / "retargeting" / "outputs" / "sample" / "seed.txt"
        seed.parent.mkdir(parents=True)
        seed.write_text("tracked upstream sample output\n", encoding="utf-8")

    raw_dir = tmp_path / "raw"
    (raw_dir / "vinegar_bottle_left").mkdir(parents=True)
    (raw_dir / "vinegar_bottle_left" / "all_hand_meshes.npz").write_bytes(
        b"audited-hand-input"
    )
    (raw_dir / "config.json").write_text(
        json.dumps({"object_names": ["vinegar_bottle"], "anchor_hand": "left"}),
        encoding="utf-8",
    )
    layout_dir = (
        raw_dir
        / "obj_tracking_out"
        / "vinegar_bottle"
        / "combined_visualization"
    )
    layout_dir.mkdir(parents=True)
    (layout_dir / "layout_camera_frame_optimized.json").write_text(
        json.dumps(
            {
                "objects": [],
                "translation_scale_optimization": {"mesh_scale": 1.0},
            }
        ),
        encoding="utf-8",
    )
    (raw_dir / "gravity.json").write_text(
        json.dumps({"vec3d": [0.0, 0.0, 1.0], "roll_deg": 0.0, "pitch_deg": 0.0}),
        encoding="utf-8",
    )
    mesh_dir = (
        raw_dir
        / "video_segmentation"
        / "masks"
        / "frame_000000_masks"
        / "vinegar_bottle"
    )
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "vinegar_bottle.obj").write_text("v 0 0 0\n", encoding="utf-8")
    output_root = tmp_path / "outputs"
    capture = tmp_path / "backend-capture.txt"

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    git_status = " M retargeting/launch.py" if dirty else ""
    _write_executable(
        fake_bin / "git",
        f"""#!/usr/bin/env bash
set -euo pipefail
repo=""
if [[ "${{1:-}}" == "-C" ]]; then
  repo="$2"
  shift 2
fi
case "${{1:-}}" in
  rev-parse)
    if [[ "${{*: -1}}" == "--is-inside-work-tree" ]]; then
      printf '%s\\n' true
    elif [[ "${{*: -1}}" == "HEAD^{{tree}}" ]]; then
      printf '%s\\n' {shlex.quote(FAKE_TREE)}
    else
      printf '%s\\n' {shlex.quote(backend_head)}
    fi
    ;;
  status)
    printf '%s\\n' {shlex.quote(git_status)}
    ;;
  remote)
    printf '%s\\n' fake://pristine-dai
    ;;
  clone)
    args=("$@")
    count="${{#args[@]}}"
    source="${{args[$((count - 2))]}}"
    target="${{args[$((count - 1))]}}"
    mkdir -p "$target"
    cp -R "$source/." "$target/"
    ;;
  checkout)
    ;;
  sparse-checkout)
    if [[ "${{2:-}}" == "set" ]]; then
      rm -rf "$repo/retargeting/outputs"
    fi
    ;;
  *)
    printf 'unexpected fake git invocation: %s\\n' "$*" >&2
    exit 91
    ;;
esac
""",
    )

    fake_python = fake_bin / "retargeting-python"
    malformed_native_output = ""
    if native_output_as_file:
        malformed_native_output = "rm -rf outputs\nprintf '%s' malformed-native-output > outputs"
    _write_executable(
        fake_python,
        f"""#!/usr/bin/env bash
set -euo pipefail
{{
  printf '%s\\n' '---argv---'
  printf '%s\\n' "$@"
  printf '%s\\n' '---env---'
  env | LC_ALL=C sort
}} > {shlex.quote(str(capture))}
task=""
while (( $# > 0 )); do
  if [[ "$1" == "--task" ]]; then
    task="$2"
    shift 2
  else
    shift
  fi
done
mkdir -p "outputs/mano/left/$task/0" "outputs/sharpa/left/$task/0" "outputs/assets"
printf '%s' native-keypoints > "outputs/mano/left/$task/0/trajectory_keypoints.npz"
printf '%s\\n' '{{"task":"vinegar_bottle_left","dataset_name":"do_as_i_do","robot_type":"mano","embodiment_type":"left","data_id":0}}' \\
  > "outputs/mano/left/$task/task_info.json"
printf '%s' native-result > "outputs/sharpa/left/$task/0/trajectory_mjwp.npz"
{malformed_native_output}
exit {backend_rc}
""",
    )

    env = os.environ.copy()
    for name in tuple(env):
        if (
            name.startswith("DAI_MJWP_")
            or name.startswith("DAI_RUNTIME_")
            or name.startswith("DO_AS_I_DO_")
            or name.startswith("ROBOT_RENDER_")
            or name == "DAI_MAX_SIM_STEPS"
        ):
            env.pop(name)
    env.update(
        PATH=f"{fake_bin}{os.pathsep}{env['PATH']}",
        HOME=str(tmp_path / "home"),
        USER="runner-test",
        LOGNAME="runner-test",
        RETARGETING_PYTHON=str(fake_python),
        DAI_PRISTINE_SOURCE=str(runtime),
        DAI_PRISTINE_RUNTIME=str(runtime),
    )
    (tmp_path / "home").mkdir()
    command = [
        str(RUNNER),
        "--raw-dir",
        str(raw_dir),
        "--task",
        "vinegar_bottle_left",
        "--output-root-dir",
        str(output_root),
        "--cuda-visible-devices",
        "5",
        "--egl-device-id",
        "0",
        "--headless",
        "--no-wait",
    ]
    return {
        "runtime": runtime,
        "raw_dir": raw_dir,
        "output_root": output_root,
        "capture": capture,
        "fake_python": fake_python,
        "env": env,
        "command": command,
    }


def _run(case: dict) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        case["command"],
        cwd=ROOT,
        env=case["env"],
        text=True,
        capture_output=True,
        check=False,
    )


def test_pristine_runner_uses_only_native_allowed_argv_and_runtime_env(tmp_path: Path) -> None:
    case = _sandbox(tmp_path)
    before = _tree_digest(case["runtime"])

    completed = _run(case)

    assert completed.returncode == 0, completed.stderr
    assert _tree_digest(case["runtime"]) == before
    capture = case["capture"].read_text(encoding="utf-8")
    argv_text, env_text = capture.split("---env---\n", 1)
    argv = argv_text.splitlines()[1:]
    assert argv == [
        "launch.py",
        "--raw-dir",
        str(case["raw_dir"].resolve()),
        "--task",
        "vinegar_bottle_left",
        "--no-show-viewer",
        "--no-wait-on-finish",
    ]
    environment = dict(
        line.split("=", 1)
        for line in env_text.splitlines()
        if "=" in line
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "5"
    assert environment["EGL_DEVICE_ID"] == "0"
    assert environment["MUJOCO_GL"] == "egl"
    assert environment["PYOPENGL_PLATFORM"] == "egl"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert not any(name.startswith("DAI_") for name in environment)
    assert not any(name.startswith("DO_AS_I_DO_") for name in environment)

    manifest = json.loads(
        (case["output_root"] / "aoe_pristine_dai_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["scope"] == "pristine_official_dai_no_backend_modification"
    assert manifest["backend_rc"] == 0
    assert manifest["backend_rc_authoritative"] is True
    assert manifest["backend"]["pinned_commit"] == PINNED_COMMIT
    assert manifest["backend"]["commit_before"] == PINNED_COMMIT
    assert manifest["backend"]["commit_after"] == PINNED_COMMIT
    assert manifest["backend"]["tree_unchanged"] is True
    assert manifest["backend"]["behavior_overrides"] == []
    assert manifest["custom_acceptance_gate"] is False
    assert manifest["native_outputs_deleted_or_rewritten"] is False
    assert manifest["native_output_layout"]["upstream_runtime_relative_root"] == (
        "retargeting/outputs"
    )
    assert manifest["native_archive"]["publication"] == (
        "same_filesystem_atomic_directory_rename"
    )
    assert manifest["native_archive"]["payload_byte_tree_identical"] is True
    assert manifest["native_archive"]["payload_file_metadata_mutated"] is False
    assert manifest["input"]["unchanged_during_native_run"] is True
    assert manifest["resolved_route"]["status"] == "ok"
    resolved = manifest["resolved_route"]["resolved"]
    assert resolved["embodiment_type"] == "left"
    assert resolved["hand_type"] == "left"
    assert resolved["task_info_sha256"] == hashlib.sha256(
        (case["output_root"] / resolved["task_info_relative_path"]).read_bytes()
    ).hexdigest()
    assert resolved["trajectory_keypoints_sha256"] == hashlib.sha256(
        (case["output_root"] / resolved["trajectory_keypoints_relative_path"]).read_bytes()
    ).hexdigest()
    assert resolved["robot_trajectory_artifacts"] == [
        {
            "relative_path": (
                "sharpa/left/vinegar_bottle_left/0/trajectory_mjwp.npz"
            ),
            "sha256": hashlib.sha256(b"native-result").hexdigest(),
            "size": len(b"native-result"),
        }
    ]
    assert resolved["route_native_artifacts"] == [
        {
            "relative_path": (
                "sharpa/left/vinegar_bottle_left/0/trajectory_mjwp.npz"
            ),
            "sha256": hashlib.sha256(b"native-result").hexdigest(),
            "size": len(b"native-result"),
        }
    ]
    assert (
        case["output_root"]
        / "mano/left/vinegar_bottle_left/0/trajectory_keypoints.npz"
    ).read_bytes() == b"native-keypoints"
    assert manifest["argv"] == [str(case["fake_python"]), *argv]


def test_pristine_runner_returns_native_backend_rc_unchanged(tmp_path: Path) -> None:
    case = _sandbox(tmp_path, backend_rc=37)

    completed = _run(case)

    assert completed.returncode == 37
    manifest = json.loads(
        (case["output_root"] / "aoe_pristine_dai_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "backend_exited"
    assert manifest["backend_rc"] == 37
    assert manifest["backend_rc_authoritative"] is True


def test_pristine_runner_isolates_tracked_upstream_seed_outputs(tmp_path: Path) -> None:
    case = _sandbox(tmp_path, tracked_seed_outputs=True)

    completed = _run(case)

    assert completed.returncode == 0, completed.stderr
    assert not (case["output_root"] / "sample" / "seed.txt").exists()
    manifest = json.loads(
        (case["output_root"] / "aoe_pristine_dai_run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["backend"][
        "tracked_seed_outputs_excluded_via_sparse_checkout"
    ] is True
    assert manifest["native_output_layout"][
        "tracked_upstream_seed_outputs_excluded_via_sparse_checkout"
    ] is True
    assert manifest["backend"]["tree_unchanged"] is True


def test_pristine_runner_reports_publication_failure_separately_from_native_rc(
    tmp_path: Path,
) -> None:
    case = _sandbox(tmp_path, backend_rc=0, native_output_as_file=True)

    completed = _run(case)

    assert completed.returncode == 74
    assert "native backend rc=0; wrapper rc=74" in completed.stderr
    assert not case["output_root"].exists()
    workspaces = list(tmp_path.glob(".aoe-pristine-dai-run.*"))
    assert len(workspaces) == 1
    failure = json.loads(
        (workspaces[0] / "publication_failure.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "publication_failed"
    assert failure["native_backend_rc"] == 0
    assert failure["native_backend_rc_authoritative"] is True
    assert failure["wrapper_rc"] == 74
    assert failure["reason"] == "native output root is not a real directory"


@pytest.mark.parametrize(
    "deprecated",
    [
        ["--max-sim-steps", "540"],
        ["--force"],
        ["--hand-type", "left"],
        ["--robot-type", "sharpa"],
    ],
)
def test_pristine_runner_rejects_deprecated_backend_options(
    tmp_path: Path, deprecated: list[str]
) -> None:
    case = _sandbox(tmp_path)
    case["command"].extend(deprecated)

    completed = _run(case)

    assert completed.returncode == 2
    assert "deprecated option" in completed.stderr
    assert not case["capture"].exists()


def test_pristine_runner_check_only_validates_without_launch_or_output(
    tmp_path: Path,
) -> None:
    case = _sandbox(tmp_path)
    case["command"].append("--check-only")

    completed = _run(case)

    assert completed.returncode == 0, completed.stderr
    assert "DAI_CHECK_ONLY_OK" in completed.stdout
    assert f"commit={PINNED_COMMIT}" in completed.stdout
    assert "resolved_anchor_hand=left" in completed.stdout
    assert "native_output_root=retargeting/outputs" in completed.stdout
    assert not case["capture"].exists()
    assert not case["output_root"].exists()


def test_pristine_runner_rejects_backend_tuning_environment(tmp_path: Path) -> None:
    case = _sandbox(tmp_path)
    case["env"]["DAI_MJWP_NUM_SAMPLES"] = "2048"

    completed = _run(case)

    assert completed.returncode == 2
    assert "backend-tuning environment variable is forbidden" in completed.stderr
    assert "DAI_MJWP_NUM_SAMPLES" in completed.stderr
    assert not case["capture"].exists()


@pytest.mark.parametrize(
    "name",
    ["DAI_COLLISION_TARGET", "MJWP_NUM_SAMPLES", "DO_AS_I_DO_RELEASE_STEP"],
)
def test_pristine_runner_rejects_any_ambient_backend_tuning_prefix(
    tmp_path: Path, name: str
) -> None:
    case = _sandbox(tmp_path)
    case["env"][name] = "1"

    completed = _run(case)

    assert completed.returncode == 2
    assert name in completed.stderr
    assert not case["capture"].exists()


def test_pristine_runner_rejects_stale_output_path(tmp_path: Path) -> None:
    case = _sandbox(tmp_path)
    case["output_root"].mkdir()

    completed = _run(case)

    assert completed.returncode == 2
    assert "brand-new path" in completed.stderr
    assert not case["capture"].exists()


def test_pristine_runner_rejects_raw_output_overlap(tmp_path: Path) -> None:
    case = _sandbox(tmp_path)
    overlap = case["raw_dir"] / "native-output"
    output_index = case["command"].index("--output-root-dir") + 1
    case["command"][output_index] = str(overlap)

    completed = _run(case)

    assert completed.returncode == 2
    assert "must not overlap" in completed.stderr
    assert not case["capture"].exists()


def test_pristine_runner_never_falls_back_to_another_task_hand_input(
    tmp_path: Path,
) -> None:
    case = _sandbox(tmp_path)
    exact_task_dir = case["raw_dir"] / "vinegar_bottle_left"
    exact_task_dir.rename(case["raw_dir"] / "unrelated_right")

    completed = _run(case)

    assert completed.returncode == 2
    assert "route-specific all_hand_meshes.npz" in completed.stderr
    assert not case["capture"].exists()


@pytest.mark.parametrize("unsafe_task", ["../escape", "bad/task", ".", "two words"])
def test_pristine_runner_rejects_unsafe_task(
    tmp_path: Path, unsafe_task: str
) -> None:
    case = _sandbox(tmp_path)
    task_index = case["command"].index("--task") + 1
    case["command"][task_index] = unsafe_task

    completed = _run(case)

    assert completed.returncode == 2
    assert "strict safe basename" in completed.stderr
    assert not case["capture"].exists()


def test_pristine_runner_refuses_dirty_backend_before_launch(tmp_path: Path) -> None:
    case = _sandbox(tmp_path, dirty=True)

    completed = _run(case)

    assert completed.returncode == 2
    assert "is dirty" in completed.stderr
    assert not case["capture"].exists()


def test_pristine_runner_refuses_non_pinned_backend_before_launch(
    tmp_path: Path,
) -> None:
    case = _sandbox(tmp_path, backend_head="f" * 40)

    completed = _run(case)

    assert completed.returncode == 2
    assert f"expected {PINNED_COMMIT}" in completed.stderr
    assert not case["capture"].exists()


def test_pristine_runner_contains_no_backend_patch_or_stage_entrypoints() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    forbidden = (
        "patch_do_as_i_do_hoi_pipeline.py",
        "patch_dai_sampling_nonfinite_rewards.py",
        "materialize_dai_runtime.py",
        "write_dai_runtime_provenance.py",
        "pipeline/optimize_physics.py",
        "pipeline/solve_ik.py",
        "render_mujoco_trajectory.py",
    )
    for token in forbidden:
        assert token not in source
    assert '"launch.py"' in source
    assert "exit \"$backend_rc\"" in source
