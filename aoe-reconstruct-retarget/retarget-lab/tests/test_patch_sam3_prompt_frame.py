from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "maintenance" / "patch_sam3_prompt_frame.py"
SPEC = importlib.util.spec_from_file_location("patch_sam3_prompt_frame", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_patch_falls_back_to_prompt_response_without_relaxing_selection() -> None:
    source = '''    anchor_outputs = outputs_per_frame.get(anchor)
    if anchor_outputs is not None:
        mask, info = choose_frame_mask(
            anchor_outputs,
            preferred_obj_id=selected_obj_id,
            target_bbox=target_bbox,
            target_point=target_point,
        )
        if mask is not None:
            chosen_masks[anchor] = mask
            chosen_infos[anchor] = info
'''

    patched, changed = MODULE.patch_text(source)

    assert changed
    assert MODULE.SENTINEL in patched
    assert '(prompt_out, "add_prompt")' in patched
    assert "preferred_obj_id=selected_obj_id" in patched
    assert "target_bbox=target_bbox" in patched
    assert "anchor_source=anchor_source" in patched

    patched_again, changed_again = MODULE.patch_text(patched)
    assert not changed_again
    assert patched_again == patched
