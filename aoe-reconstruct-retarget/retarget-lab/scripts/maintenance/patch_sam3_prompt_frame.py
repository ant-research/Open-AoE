#!/usr/bin/env python3
"""Preserve SAM3's add-prompt mask when propagation omits the anchor frame."""

from __future__ import annotations

import argparse
from pathlib import Path


SENTINEL = "AOE_PRESERVE_PROMPT_FRAME_MASK"

OLD_BLOCK = '''    anchor_outputs = outputs_per_frame.get(anchor)
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

NEW_BLOCK = '''    # AOE_PRESERVE_PROMPT_FRAME_MASK: SAM3 can return a valid mask from
    # add_prompt while omitting that same anchor from propagation output. Keep
    # the propagated result when available, then fall back to the official
    # prompt response instead of writing an all-zero mask at the anchor.
    anchor_sources = (
        (outputs_per_frame.get(anchor), "propagation"),
        (prompt_out, "add_prompt"),
    )
    for anchor_outputs, anchor_source in anchor_sources:
        if anchor_outputs is None:
            continue
        mask, info = choose_frame_mask(
            anchor_outputs,
            preferred_obj_id=selected_obj_id,
            target_bbox=target_bbox,
            target_point=target_point,
        )
        if mask is not None:
            chosen_masks[anchor] = mask
            chosen_infos[anchor] = dict(info or {}, anchor_source=anchor_source)
            break
'''


def patch_text(text: str) -> tuple[str, bool]:
    if SENTINEL in text:
        return text, False
    if OLD_BLOCK not in text:
        raise ValueError("expected SAM3 anchor mask block not found")
    return text.replace(OLD_BLOCK, NEW_BLOCK, 1), True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-sam3-py", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    original = args.run_sam3_py.read_text(encoding="utf-8")
    patched, changed = patch_text(original)
    if args.dry_run:
        print(f"[sam3-prompt-frame] {'would patch' if changed else 'already current'}: {args.run_sam3_py}")
        return 0
    if not changed:
        print(f"[sam3-prompt-frame] already current: {args.run_sam3_py}")
        return 0

    backup = args.run_sam3_py.with_suffix(args.run_sam3_py.suffix + ".aoe_prompt_frame.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    args.run_sam3_py.write_text(patched, encoding="utf-8")
    print(f"[sam3-prompt-frame] installed: {args.run_sam3_py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
