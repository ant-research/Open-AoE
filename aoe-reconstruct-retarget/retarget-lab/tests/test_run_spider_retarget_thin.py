from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PINNED_COMMIT = "2e54f19ee6ab8e0690c0f585beb1e9f8f53a6898"
PINNED_TREE = "0123456789abcdef0123456789abcdef01234567"
ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_spider_retarget.sh"


def test_public_spider_runner_is_executable() -> None:
    assert RUNNER.is_file()
    assert os.access(RUNNER, os.X_OK)


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _fake_checkout(tmp_path: Path, *, failing_stage: str | None = None) -> tuple[Path, Path]:
    spider_root = tmp_path / "SPIDER"
    (spider_root / ".git").mkdir(parents=True)
    scripts = {
        "spider/preprocess/generate_xml.py": "generate_xml",
        "spider/preprocess/ik_fast.py": "ik_fast",
        "examples/run_mjwp.py": "run_mjwp",
    }
    for relative, stage in scripts.items():
        path = spider_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        exit_code = 19 if stage == failing_stage else 0
        record_path = tmp_path / "backend_calls.json"
        path.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            f"record = Path({str(record_path)!r})\n"
            "rows = json.loads(record.read_text()) if record.exists() else []\n"
            f"rows.append({{'stage': {stage!r}, 'argv': sys.argv[1:], "
            "'environment': {key: os.environ.get(key) for key in "
            "('PYTHONPATH', 'AOE_SENTINEL', 'SPIDER_MAX_SIM_STEPS', "
            "'CUDA_VISIBLE_DEVICES', 'EGL_DEVICE_ID')}})\n"
            "record.write_text(json.dumps(rows))\n"
            f"raise SystemExit({exit_code})\n",
            encoding="utf-8",
        )

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "git",
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "case \" $* \" in\n"
        f"  *\" rev-parse HEAD^{{tree}} \"*) echo {PINNED_TREE} ;;\n"
        f"  *\" rev-parse HEAD \"*) echo {PINNED_COMMIT} ;;\n"
        "  *\" status \"*) exit 0 ;;\n"
        "  *) echo \"unexpected fake git invocation: $*\" >&2; exit 91 ;;\n"
        "esac\n",
    )
    return spider_root, fake_bin


def _fake_dai_inputs(
    tmp_path: Path, *, embodiment_type: str = "left"
) -> Path:
    root = tmp_path / "retargeting_outputs"
    mano = root / "mano" / embodiment_type / "test_task"
    (mano / "0").mkdir(parents=True)
    (mano / "0" / "trajectory_keypoints.npz").write_bytes(b"exact-keypoints")
    (mano / "task_info.json").write_text(
        json.dumps(
            {
                "task": "test_task",
                "dataset_name": "do_as_i_do",
                "robot_type": "mano",
                "embodiment_type": embodiment_type,
                "data_id": 0,
                "right_object_mesh_dir": (
                    "assets/objects/test_task"
                    if embodiment_type in {"right", "bimanual"}
                    else None
                ),
                "right_object_convex_dir": (
                    "assets/objects/test_task/convex"
                    if embodiment_type in {"right", "bimanual"}
                    else None
                ),
                "left_object_mesh_dir": (
                    "assets/objects/test_task" if embodiment_type == "left" else None
                ),
                "left_object_convex_dir": (
                    "assets/objects/test_task/convex"
                    if embodiment_type == "left"
                    else None
                ),
            }
        ),
        encoding="utf-8",
    )
    objects = root / "assets" / "objects" / "test_task"
    (objects / "convex").mkdir(parents=True)
    (objects / "visual.obj").write_bytes(b"visual-mesh")
    (objects / "convex" / "0.obj").write_bytes(b"convex-mesh")
    robots = root / "assets" / "robots" / "sharpa"
    (robots / "meshes").mkdir(parents=True)
    (robots / f"{embodiment_type}.xml").write_bytes(b"<mujoco/>")
    (robots / "meshes" / "hand.stl").write_bytes(b"robot-mesh")
    return root


def _run(
    tmp_path: Path,
    *,
    failing_stage: str | None = None,
    embodiment_type: str = "left",
    task_info_updates: dict | None = None,
    input_mutator=None,
    extra_args: list[str] | None = None,
    output_dir_override: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    spider_root, fake_bin = _fake_checkout(tmp_path, failing_stage=failing_stage)
    input_root = _fake_dai_inputs(
        tmp_path, embodiment_type=embodiment_type
    )
    if task_info_updates:
        task_info_path = (
            input_root
            / "mano"
            / embodiment_type
            / "test_task"
            / "task_info.json"
        )
        task_info = json.loads(task_info_path.read_text(encoding="utf-8"))
        task_info.update(task_info_updates)
        task_info_path.write_text(json.dumps(task_info), encoding="utf-8")
    if input_mutator is not None:
        input_mutator(input_root)
    output_dir = output_dir_override or (tmp_path / "run")
    record = tmp_path / "backend_calls.json"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "PYTHONPATH": "/tmp/aoe-should-not-reach-backend",
            "AOE_SENTINEL": "must_be_removed",
            "SPIDER_MAX_SIM_STEPS": "999",
        }
    )
    args = [
        "bash",
        str(RUNNER),
        "--spider-root",
        str(spider_root),
        "--python",
        sys.executable,
        "--input-root",
        str(input_root),
        "--output-dir",
        str(output_dir),
        "--task",
        "test_task",
        "--robot-type",
        "sharpa",
        "--embodiment-type",
        embodiment_type,
        "--data-id",
        "0",
        "--ref-dt",
        "0.03333333333333333",
        "--cuda-visible-devices",
        "7",
        "--egl-device-id",
        "0",
    ]
    args.extend(extra_args or [])
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    return result, output_dir, record, input_root


def test_thin_runner_stages_exact_inputs_and_calls_only_original_pipeline(tmp_path):
    result, output_dir, record, input_root = _run(tmp_path)
    assert result.returncode == 0, result.stderr

    calls = json.loads(record.read_text(encoding="utf-8"))
    assert [row["stage"] for row in calls] == [
        "generate_xml",
        "ik_fast",
        "run_mjwp",
    ]
    assert calls[0]["argv"][-1] == "--no-show-viewer"
    assert "--dataset-dir" in calls[0]["argv"]
    assert "--ref-dt" in calls[1]["argv"]
    assert "--dataset-dir" in calls[1]["argv"]
    assert "--no-save-video" in calls[1]["argv"]
    assert "ref_dt=0.03333333333333333" in calls[2]["argv"]
    assert "sim_dt=0.008333333333333333" in calls[2]["argv"]
    assert "render_dt=0.03333333333333333" in calls[2]["argv"]
    assert "trace_dt=0.03333333333333333" in calls[2]["argv"]
    assert any(
        item.startswith("dataset_dir=") for item in calls[2]["argv"]
    )
    assert "show_viewer=false" in calls[2]["argv"]
    assert "save_video=true" in calls[2]["argv"]
    for call in calls:
        backend_env = call["environment"]
        assert Path(backend_env["PYTHONPATH"]).resolve() == (
            output_dir.parent / "SPIDER"
        ).resolve()
        assert backend_env["AOE_SENTINEL"] is None
        assert backend_env["SPIDER_MAX_SIM_STEPS"] is None
        assert backend_env["CUDA_VISIBLE_DEVICES"] == "7"
        assert backend_env["EGL_DEVICE_ID"] == "0"

    forbidden = (
        "num_samples",
        "max_num_iterations",
        "horizon",
        "noise_scale",
        "rew_scale",
        "contact_guidance",
        "collision",
        "release",
        "threshold",
    )
    all_backend_args = "\n".join(arg for call in calls for arg in call["argv"])
    assert not any(token in all_backend_args for token in forbidden)

    source_keypoints = (
        input_root
        / "mano"
        / "left"
        / "test_task"
        / "0"
        / "trajectory_keypoints.npz"
    )
    staged_keypoints = (
        output_dir
        / "dataset"
        / "processed"
        / "do_as_i_do"
        / "mano"
        / "left"
        / "test_task"
        / "0"
        / "trajectory_keypoints.npz"
    )
    assert staged_keypoints.read_bytes() == source_keypoints.read_bytes()
    assert (
        output_dir
        / "dataset"
        / "processed"
        / "do_as_i_do"
        / "assets"
        / "objects"
        / "test_task"
        / "visual.obj"
    ).read_bytes() == b"visual-mesh"

    manifest = json.loads(
        (output_dir / "vanilla_spider_run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["backend"]["commit"] == PINNED_COMMIT
    assert manifest["backend"]["tree"] == PINNED_TREE
    assert manifest["backend"]["patches_applied"] == []
    assert manifest["backend"]["behavior_overrides"] == []
    assert manifest["sim_dt"] == pytest.approx(1.0 / 120.0)
    assert (
        manifest["sim_dt_policy"]
        == "largest_exact_substep_not_coarser_than_upstream_0.01s"
    )
    assert manifest["backend_environment"]["clear_environment"] is True
    assert Path(manifest["backend_environment"]["PYTHONPATH"]).resolve() == (
        output_dir.parent / "SPIDER"
    ).resolve()
    assert "PYTHONPATH" in manifest["backend_environment"][
        "inherited_variables_removed"
    ]
    assert "AOE_SENTINEL" in manifest["backend_environment"][
        "inherited_variables_removed"
    ]
    assert "SPIDER_MAX_SIM_STEPS" in manifest["backend_environment"][
        "inherited_variables_removed"
    ]
    assert manifest["trajectory_keypoints"]["byte_identical"] is True
    assert manifest["trajectory_keypoints"]["modified"] is False
    assert manifest["input_timebase_binding"] == "exact_dai_keypoint_frame_grid"
    assert manifest["robot_assets"]["byte_identical"] is True
    assert all(item["byte_identical"] for item in manifest["object_assets"])

    task_info = json.loads(
        (
            output_dir
            / "dataset"
            / "processed"
            / "do_as_i_do"
            / "mano"
            / "left"
            / "test_task"
            / "task_info.json"
        ).read_text(encoding="utf-8")
    )
    assert task_info["ref_dt"] == 0.03333333333333333
    assert task_info["left_object_mesh_dir"] == (
        "processed/do_as_i_do/assets/objects/test_task"
    )

    outcome = json.loads(
        (output_dir / "vanilla_spider_outcome.json").read_text(encoding="utf-8")
    )
    assert outcome["stage"] == "run_mjwp"
    assert outcome["returncode"] == 0
    assert outcome["backend_clean_after"] is True
    assert outcome["backend_patches_applied"] == []


def test_bimanual_shared_object_schema_with_only_right_binding_is_preserved(
    tmp_path,
):
    result, output_dir, record, _ = _run(
        tmp_path, embodiment_type="bimanual"
    )
    assert result.returncode == 0, result.stderr
    calls = json.loads(record.read_text(encoding="utf-8"))
    assert [row["stage"] for row in calls] == [
        "generate_xml",
        "ik_fast",
        "run_mjwp",
    ]
    task_info = json.loads(
        (
            output_dir
            / "dataset"
            / "processed"
            / "do_as_i_do"
            / "mano"
            / "bimanual"
            / "test_task"
            / "task_info.json"
        ).read_text(encoding="utf-8")
    )
    assert task_info["right_object_mesh_dir"].endswith(
        "assets/objects/test_task"
    )
    assert task_info["left_object_mesh_dir"] is None


def test_check_only_rejects_conflicting_task_info_before_writing_output(
    tmp_path,
):
    result, output_dir, record, _ = _run(
        tmp_path,
        task_info_updates={"ref_dt": 0.04},
        extra_args=["--check-only"],
    )
    assert result.returncode != 0
    assert "ref_dt conflicts" in result.stderr
    assert not output_dir.exists()
    assert not record.exists()


def test_runner_rejects_output_nested_in_read_only_input(tmp_path):
    nested_output = tmp_path / "retargeting_outputs" / "native_run"
    result, _, record, input_root = _run(
        tmp_path, output_dir_override=nested_output
    )
    assert result.returncode == 4
    assert "must not overlap the read-only input root" in result.stderr
    assert not nested_output.exists()
    assert not record.exists()
    assert input_root.is_dir()


def test_runner_rejects_symlinked_asset_tree_before_backend(tmp_path):
    def add_external_robot_mesh_symlink(input_root: Path) -> None:
        meshes = input_root / "assets" / "robots" / "sharpa" / "meshes"
        shutil.rmtree(meshes)
        external = tmp_path / "external-robot-meshes"
        external.mkdir()
        (external / "hand.stl").write_bytes(b"mutable-external-robot-mesh")
        meshes.symlink_to(external, target_is_directory=True)

    result, output_dir, record, _ = _run(
        tmp_path,
        input_mutator=add_external_robot_mesh_symlink,
        extra_args=["--check-only"],
    )

    assert result.returncode != 0
    assert "symlink" in result.stderr
    assert not output_dir.exists()
    assert not record.exists()


def test_thin_runner_preserves_native_failure_returncode(tmp_path):
    result, output_dir, record, _ = _run(tmp_path, failing_stage="generate_xml")
    assert result.returncode == 19
    calls = json.loads(record.read_text(encoding="utf-8"))
    assert [row["stage"] for row in calls] == ["generate_xml"]
    outcome = json.loads(
        (output_dir / "vanilla_spider_outcome.json").read_text(encoding="utf-8")
    )
    assert outcome["stage"] == "generate_xml"
    assert outcome["returncode"] == 19


def test_thin_runner_rejects_backend_tuning_arguments(tmp_path):
    result, _, record, _ = _run(tmp_path, extra_args=["--num-samples", "32"])
    assert result.returncode == 2
    assert "backend-tuning argument" in result.stderr
    assert not record.exists()


def test_runner_source_contains_no_backend_patch_or_custom_gate_cluster():
    source = RUNNER.read_text(encoding="utf-8")
    forbidden_components = (
        "materialize_spider_runtime.py",
        "patch_spider_",
        "stabilize_spider_ik_artifacts.py",
        "validate_spider_pre_mjwp.py",
        "validate_spider_passive_release.py",
        "write_spider_runtime_provenance.py",
        "write_spider_run_outcome.py",
    )
    assert not any(component in source for component in forbidden_components)
    assert "2e54f19ee6ab8e0690c0f585beb1e9f8f53a6898" in source
    assert "spider/preprocess/generate_xml.py" in source
    assert "spider/preprocess/ik_fast.py" in source
    assert "examples/run_mjwp.py" in source
