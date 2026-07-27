from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "compatibility" / "run_sam3_video.py"
WRAPPER = ROOT / "scripts" / "run_do_as_i_do_reconstruction_fresh.sh"


def _load_runner():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    spec = importlib.util.spec_from_file_location("aoe_sam3_video", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outputs(masks: list[np.ndarray], scores: list[float], ids: list[int]):
    return {
        "out_obj_ids": np.asarray(ids),
        "out_probs": np.asarray(scores),
        "out_binary_masks": np.asarray(masks),
    }


def test_wrapper_uses_lab_compatibility_without_mutating_backend() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert 'scripts/compatibility/run_sam3_video.py' in text
    assert 'patch_sam3_prompt_frame.py' not in text
    assert 'run_sam3_py "$script_dir/run_sam3_video.py"' not in text


def test_bbox_selection_prefers_overlapping_instance() -> None:
    module = _load_runner()
    outside = np.zeros((20, 20), dtype=np.uint8)
    outside[1:5, 1:5] = 1
    inside = np.zeros((20, 20), dtype=np.uint8)
    inside[8:14, 9:15] = 1
    selected, candidates = module.select_prompt_object(
        _outputs([outside, inside], [0.99, 0.60], [3, 7]),
        target_bbox=[7, 7, 17, 17],
        select_mode="bbox",
    )
    assert selected == 7
    assert len(candidates) == 2
    assert next(row for row in candidates if row["sam3_obj_id"] == 7)[
        "mask_pixels"
    ] == 36


def test_frame_selection_keeps_preferred_sam3_identity() -> None:
    module = _load_runner()
    wrong = np.zeros((20, 20), dtype=np.uint8)
    wrong[1:8, 1:8] = 1
    target = np.zeros((20, 20), dtype=np.uint8)
    target[10:15, 10:15] = 1
    mask, info = module.choose_frame_mask(
        _outputs([wrong, target], [0.95, 0.55], [2, 9]),
        preferred_obj_id=9,
        target_bbox=[8, 8, 17, 17],
    )
    assert mask is not None
    assert int(mask.sum()) == 25
    assert info["sam3_obj_id"] == 9


def test_prompt_response_is_explicit_anchor_fallback() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert '(prompt_out, "add_prompt")' in text
    assert "anchor_source=anchor_source" in text
