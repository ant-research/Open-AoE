import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnostics" / "inspect_hand_provenance.py"
SPEC = importlib.util.spec_from_file_location("inspect_hand_provenance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_hand_npz(raw_dir: Path, **metadata: str) -> Path:
    path = raw_dir / "raw" / "all_hand_meshes.npz"
    path.parent.mkdir(parents=True)
    payload = {key: np.asarray(value) for key, value in metadata.items()}
    payload["source_frame_ids"] = np.arange(3)
    np.savez(path, **payload)
    return path


def test_aoe_hawor_input_matches_aoe_label(tmp_path):
    write_hand_npz(
        tmp_path,
        source_hands_npz="/dataset/ego_hands_reconstruction/hands.npz",
        pose_space="camera",
    )
    report = MODULE.inspect(tmp_path, "aoe")
    assert report["status"] == "ok"
    assert report["actual_hand_source"] == "aoe_hawor"


def test_egoinfinity_geometry_overrides_original_aoe_container(tmp_path):
    write_hand_npz(
        tmp_path,
        source_hands_npz="/dataset/ego_hands_reconstruction/hands.npz",
        hand_geometry_source="EgoInfinity frame_data vertices_3d/joints_3d_pred",
    )
    report = MODULE.inspect(tmp_path, "estimated")
    assert report["status"] == "ok"
    assert report["actual_hand_source"] == "egoinfinity_estimated"


def test_mislabeled_aoe_geometry_is_rejected_as_estimated(tmp_path):
    write_hand_npz(
        tmp_path,
        source_hands_npz="/dataset/ego_hands_reconstruction/hands.npz",
    )
    report = MODULE.inspect(tmp_path, "estimated")
    assert report["status"] == "invalid"
    assert "requested hand_source=estimated" in report["errors"][0]
