from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "maintenance" / "patch_spider_interp_timeline.py"
SPEC = importlib.util.spec_from_file_location("patch_spider_interp_timeline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_interp_patch_preserves_mpc_noise_length_and_exact_reference_opt_in() -> None:
    source = '''def interp(src: torch.Tensor, n: int, order: int = 1) -> torch.Tensor:
    N, H, D = src.shape
    if H <= 1:
        return src.repeat(1, n, 1)
    dst_len = H * n
    return dst_len
'''
    patched, changed = MODULE.patch_interp_text(source)
    assert changed
    assert MODULE.INTERP_SENTINEL in patched
    assert "preserve_endpoint_duration: bool = False" in patched
    assert (
        "dst_len = (H - 1) * n + 1 if preserve_endpoint_duration else H * n"
        in patched
    )
    assert 4 * 18 == 72
    assert (4 - 1) * 18 + 1 == 55
    patched_again, changed_again = MODULE.patch_interp_text(patched)
    assert not changed_again
    assert patched_again == patched


def test_io_patch_marks_only_five_reference_calls_exact() -> None:
    source = '''def load_data(config):
    # interpolate to match sim_dt
    a = interp(qpos, config.ref_steps)
    b = interp(qvel, config.ref_steps)
    c = interp(ctrl, config.ref_steps)
    d = interp(contact, config.ref_steps)
    e = interp(contact_pos, config.ref_steps)
    return a, b, c, d, e
'''
    patched, changed = MODULE.patch_io_text(source)
    assert changed
    assert MODULE.IO_SENTINEL in patched
    assert patched.count("preserve_endpoint_duration=True") == 5
    compile(patched, "io.py", "exec")


RUN_SOURCE = '''def main(config):
    ctrls = ctrl_ref[: config.horizon_steps]
    # buffers for saving info and trajectory
    info_list = []
    while True:
        sim_step = 0
        step_info = {"qpos": [], "qvel": [], "time": [], "ctrl": []}
        for i in range(config.ctrl_steps):
            pass
        for k in step_info:
            step_info[k] = np.stack(step_info[k], axis=0)
        sim_step = 1
        prev_ctrl = ctrls[config.ctrl_steps :]
        new_ctrl = ctrl_ref[
            sim_step + prev_ctrl.shape[0] : sim_step
            + prev_ctrl.shape[0]
            + config.ctrl_steps
        ]
        break

    if config.save_info and len(info_list) > 0:
        info_aggregated = {}
        for k in info_list[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in info_list], axis=0)

    errors = None
    if info_list:
        qpos_traj = np.concatenate([info["qpos"] for info in info_list], axis=0)
'''


def test_run_patch_saves_t0_and_partial_exact_final_chunk() -> None:
    patched, changed = MODULE.patch_run_mjwp_text(RUN_SOURCE)
    assert changed
    assert MODULE.RUN_SENTINEL in patched
    assert '"qpos": get_qpos(config, env)[0]' in patched
    assert "for i in range(execute_steps)" in patched
    assert "max(int(config.max_sim_steps) - sim_step, 0)" in patched
    assert "prev_ctrl = ctrls[execute_steps:]" in patched
    assert "np.concatenate([initial, *values], axis=0)" in patched
    assert '[initial_step_state["qpos"][None, ...]]' in patched
    compile(patched, "run_mjwp.py", "exec")
    patched_again, changed_again = MODULE.patch_run_mjwp_text(patched)
    assert not changed_again
    assert patched_again == patched
