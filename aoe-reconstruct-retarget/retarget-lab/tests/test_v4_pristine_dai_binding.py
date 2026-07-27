import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "scripts" / "run_v4_two_full_pipelines.sh"


def _resolver_program() -> str:
    source = PIPELINE.read_text(encoding="utf-8")
    function = source.split("resolve_native_dai_hand() {", 1)[1].split(
        "\n}\n\nclear_route_demo_artifacts()", 1
    )[0]
    return function.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]


def _write_task_info(
    output_root: Path,
    *,
    robot: str,
    hand: str,
    task: str,
) -> None:
    path = output_root / robot / hand / task / "task_info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "embodiment_type": hand,
                "task": task,
                "dataset_name": "do_as_i_do",
                "data_id": 0,
            }
        ),
        encoding="utf-8",
    )


def _run_resolver(output_root: Path, task: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-", str(output_root), task],
        input=_resolver_program(),
        text=True,
        capture_output=True,
        check=False,
    )


def test_v4_uses_pristine_dai_cli_and_route_local_outputs() -> None:
    source = PIPELINE.read_text(encoding="utf-8")

    assert source.count("scripts/run_do_as_i_do_official_retarget.sh") == 2
    assert source.count('--output-root-dir "$output_root"') == 1
    assert source.count('--output-root-dir "$retarget_output_root"') == 1
    assert source.count('--cuda-visible-devices "$dai_cuda_visible_devices"') == 2
    assert source.count("--headless") >= 2
    assert source.count("--no-wait") >= 2

    for deprecated in (
        '--robot-type sharpa',
        '--max-sim-steps "$dai_max_sim_steps"',
    ):
        assert deprecated not in source


def test_v4_binds_review_to_exact_native_task_info() -> None:
    source = PIPELINE.read_text(encoding="utf-8")

    assert "resolve_native_dai_hand" in source
    assert 'payload.get("task") == expected_task' in source
    assert 'payload.get("dataset_name") == "do_as_i_do"' in source
    assert 'int(payload.get("data_id", -1)) == 0' in source
    assert "identities = {match[:4] for match in matches}" in source
    assert "expected one exact native task identity" in source
    assert 'test "$resolved_hand_type" = "$adapter_hand_type"' in source


def test_v4_native_hand_resolver_accepts_identical_mirrored_manifests(
    tmp_path: Path,
) -> None:
    task = "test_object_bimanual"
    _write_task_info(tmp_path, robot="mano", hand="bimanual", task=task)
    _write_task_info(tmp_path, robot="sharpa", hand="bimanual", task=task)

    result = _run_resolver(tmp_path, task)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bimanual"


def test_v4_native_hand_resolver_rejects_conflicting_identities(
    tmp_path: Path,
) -> None:
    task = "test_object_bimanual"
    _write_task_info(tmp_path, robot="mano", hand="bimanual", task=task)
    _write_task_info(tmp_path, robot="sharpa", hand="right", task=task)

    result = _run_resolver(tmp_path, task)

    assert result.returncode != 0
    assert "expected one exact native task identity" in result.stderr


def test_v4_materializes_and_gates_pristine_dai_before_indexing() -> None:
    source = PIPELINE.read_text(encoding="utf-8")

    assert "materialize_and_validate_pristine_dai()" in source
    assert "scripts/materialize_pristine_dai_review.py" in source
    assert "scripts/diagnostics/validate_dai_raw_to_processed_hoi.py" in source
    assert "scripts/diagnostics/evaluate_mjwp_object_tracking.py" in source
    assert source.count('"${cell}_materialize_and_qc"') == 2
    assert source.index('"${cell}_materialize_and_qc"') < source.index('"${cell}_index"')
