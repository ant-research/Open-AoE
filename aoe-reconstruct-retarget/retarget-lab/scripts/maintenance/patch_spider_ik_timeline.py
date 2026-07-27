#!/usr/bin/env python3
"""Keep SPIDER IK output on the same timeline as its input keypoints."""

from __future__ import annotations

import argparse
from pathlib import Path


SENTINEL = "AOE_PRESERVE_IK_TIMELINE"

OLD_BLOCK = '''        # average filter
        def moving_average_filter(signal_data, window_size=5):
            return np.convolve(
                signal_data, np.ones(window_size) / window_size, mode="valid"
            )

        # Apply moving average filter
        filtered_qpos_list = np.zeros(
            (qpos_list.shape[0] - average_frame_size + 1, qpos_list.shape[1])
        )
        for i in range(qpos_list.shape[1]):
            filtered_qpos_list[:, i] = moving_average_filter(
                qpos_list[:, i], average_frame_size
            )
        qpos_list = filtered_qpos_list
'''

NEW_BLOCK = '''        # AOE_PRESERVE_IK_TIMELINE: upstream's ``valid`` convolution removes
        # average_frame_size - 1 source frames before MJWP. Pad at the temporal
        # boundaries so smoothing does not shorten or shift the RGB timeline.
        def moving_average_filter(signal_data, window_size=5):
            window_size = max(1, int(window_size))
            if window_size == 1:
                return signal_data.copy()
            pad_left = window_size // 2
            pad_right = window_size - 1 - pad_left
            padded = np.pad(signal_data, (pad_left, pad_right), mode="edge")
            return np.convolve(
                padded, np.ones(window_size) / window_size, mode="valid"
            )

        filtered_qpos_list = np.zeros_like(qpos_list)
        for i in range(qpos_list.shape[1]):
            filtered_qpos_list[:, i] = moving_average_filter(
                qpos_list[:, i], average_frame_size
            )
        qpos_list = filtered_qpos_list
'''

OLD_VELOCITY_BLOCK = '''        H = qpos_list.shape[0]
        # get qvel
        qvel_list = np.zeros((H - 1, mj_model_ik.nv))
        for i in range(1, H):
            mujoco.mj_differentiatePos(
                mj_model_ik,
                qvel_list[i - 1, :],
                ref_dt,
                qpos_list[i - 1, :],
                qpos_list[i, :],
            )
        qpos_list = qpos_list[1:]
        contact_pos_list = np.array(contact_pos_list)[1:]
        contact_list = np.array(contact_list)[1:]
'''

NEW_VELOCITY_BLOCK = '''        H = qpos_list.shape[0]
        # Preserve one IK state per source frame. The first velocity has no
        # preceding sample, so copy the first finite difference instead of
        # dropping the first qpos/contact frame.
        qvel_list = np.zeros((H, mj_model_ik.nv))
        for i in range(1, H):
            mujoco.mj_differentiatePos(
                mj_model_ik,
                qvel_list[i, :],
                ref_dt,
                qpos_list[i - 1, :],
                qpos_list[i, :],
            )
        if H > 1:
            qvel_list[0] = qvel_list[1]
        contact_pos_list = np.array(contact_pos_list)
        contact_list = np.array(contact_list)
'''


def patch_text(text: str) -> tuple[str, bool]:
    if SENTINEL in text:
        return text, False
    if OLD_BLOCK not in text:
        raise ValueError("expected SPIDER IK moving-average block not found")
    if OLD_VELOCITY_BLOCK not in text:
        raise ValueError("expected SPIDER IK velocity block not found")
    patched = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    patched = patched.replace(OLD_VELOCITY_BLOCK, NEW_VELOCITY_BLOCK, 1)
    return patched, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ik-py", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    original = args.ik_py.read_text(encoding="utf-8")
    patched, changed = patch_text(original)
    if args.dry_run:
        print(f"[spider-ik-timeline] {'would patch' if changed else 'already current'}: {args.ik_py}")
        return 0
    if not changed:
        print(f"[spider-ik-timeline] already current: {args.ik_py}")
        return 0

    backup = args.ik_py.with_suffix(args.ik_py.suffix + ".aoe_timeline.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    args.ik_py.write_text(patched, encoding="utf-8")
    print(f"[spider-ik-timeline] installed: {args.ik_py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
