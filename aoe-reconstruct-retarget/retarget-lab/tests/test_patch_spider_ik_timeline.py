from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "maintenance" / "patch_spider_ik_timeline.py"
SPEC = importlib.util.spec_from_file_location("patch_spider_ik_timeline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_patch_keeps_smoothed_qpos_velocity_and_contact_lengths_equal() -> None:
    source = '''        # average filter
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

        H = qpos_list.shape[0]
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

    patched, changed = MODULE.patch_text(source)

    assert changed
    assert MODULE.SENTINEL in patched
    assert "filtered_qpos_list = np.zeros_like(qpos_list)" in patched
    assert "qvel_list = np.zeros((H, mj_model_ik.nv))" in patched
    assert "qpos_list = qpos_list[1:]" not in patched
    assert "contact_list = np.array(contact_list)[1:]" not in patched

    patched_again, changed_again = MODULE.patch_text(patched)
    assert not changed_again
    assert patched_again == patched
