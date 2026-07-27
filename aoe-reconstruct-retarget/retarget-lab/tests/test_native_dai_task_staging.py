import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
MODULE_PATH = ROOT / "scripts" / "prepare_do_as_i_do_scene_adapter.py"
SPEC = importlib.util.spec_from_file_location("prepare_adapter_task_staging", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_stages_exact_hand_mesh_for_native_task(tmp_path: Path) -> None:
    hand_mesh = tmp_path / "raw" / "all_hand_meshes.npz"
    hand_mesh.parent.mkdir()
    hand_mesh.write_bytes(b"exact-hand-payload")

    staged = MODULE.stage_native_task_hand_mesh(tmp_path, "bottle_right", hand_mesh)

    assert staged == tmp_path / "bottle_right" / "all_hand_meshes.npz"
    assert staged.resolve() == hand_mesh.resolve()
    assert staged.read_bytes() == b"exact-hand-payload"


@pytest.mark.parametrize("task", ["", ".", "..", "../escape", "nested/task", "bad\\task"])
def test_rejects_unsafe_native_task_names(tmp_path: Path, task: str) -> None:
    hand_mesh = tmp_path / "all_hand_meshes.npz"
    hand_mesh.write_bytes(b"x")
    with pytest.raises(ValueError):
        MODULE.stage_native_task_hand_mesh(tmp_path, task, hand_mesh)


def test_ego_adapter_and_runner_stage_exact_native_task() -> None:
    adapter = (
        ROOT / "scripts" / "prepare_egoinfinity_do_as_i_do_raw_dir.py"
    ).read_text(encoding="utf-8")
    runner = (ROOT / "scripts" / "run_v4_two_full_pipelines.sh").read_text(
        encoding="utf-8"
    )

    assert 'parser.add_argument(\n        "--task",' in adapter
    assert (
        "stage_native_task_hand_mesh(output_dir, native_task, adapted_hand_path)"
        in adapter
    )
    assert runner.count('--task "$task"') >= 2
