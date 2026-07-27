#!/usr/bin/env python3
"""Install exact-reference timing fixes into the ignored SPIDER checkout."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


INTERP_SENTINEL = "AOE_REFERENCE_ONLY_ENDPOINT_DURATION"
LEGACY_INTERP_SENTINEL = "AOE_PRESERVE_REFERENCE_DURATION"
IO_SENTINEL = "AOE_EXACT_REFERENCE_CALLS"
RUN_SENTINEL = "AOE_EXACT_FINAL_CONTROL_CHUNK_AND_T0"


def patch_interp_text(text: str) -> tuple[str, bool]:
    """Keep MPC knot interpolation at H*n and make exact duration opt-in."""
    if INTERP_SENTINEL in text:
        return text, False
    old_signature = (
        "def interp(src: torch.Tensor, n: int, order: int = 1) -> torch.Tensor:\n"
    )
    new_signature = (
        "def interp(src: torch.Tensor, n: int, order: int = 1, *, "
        "preserve_endpoint_duration: bool = False) -> torch.Tensor:\n"
    )
    if text.count(old_signature) != 1:
        raise ValueError("expected unique SPIDER interp signature not found")
    if text.count("        return src.repeat(1, n, 1)\n") != 1:
        raise ValueError("expected SPIDER singleton interpolation behavior not found")
    if LEGACY_INTERP_SENTINEL in text:
        old_length = "    dst_len = (H - 1) * n + 1\n"
    else:
        old_length = "    dst_len = H * n\n"
    if text.count(old_length) != 1:
        raise ValueError("expected unique SPIDER interpolation length not found")

    patched = text.replace(old_signature, new_signature, 1)
    patched = patched.replace(
        "        return src.repeat(1, n, 1)\n",
        "        return src if preserve_endpoint_duration else src.repeat(1, n, 1)\n",
        1,
    )
    length_block = f'''    # {INTERP_SENTINEL}: reference poses span H-1 intervals,
    # while optimizer knot schedules intentionally require H*n samples.
    dst_len = (H - 1) * n + 1 if preserve_endpoint_duration else H * n
'''
    return patched.replace(old_length, length_block, 1), True


def patch_io_text(text: str) -> tuple[str, bool]:
    """Mark only reference-data interpolation calls as exact-duration calls."""
    if IO_SENTINEL in text:
        return text, False
    call_token = "config.ref_steps"
    if text.count(call_token) != 5:
        raise ValueError(
            "expected exactly five SPIDER reference interpolation calls; "
            f"found {text.count(call_token)}"
        )
    anchor = "    # interpolate to match sim_dt\n"
    if text.count(anchor) != 1:
        raise ValueError("expected unique SPIDER reference interpolation anchor")
    patched = text.replace(
        anchor,
        anchor
        + f"    # {IO_SENTINEL}: these five calls, unlike MPC knot noise, preserve endpoints.\n",
        1,
    )
    patched = patched.replace(
        call_token,
        "config.ref_steps, preserve_endpoint_duration=True",
    )
    return patched, True


def patch_run_mjwp_text(text: str) -> tuple[str, bool]:
    """Save t=0 and execute a partial final control chunk exactly."""
    if RUN_SENTINEL in text:
        return text, False
    info_anchor = '''    # buffers for saving info and trajectory
    info_list = []
'''
    info_replacement = f'''    # buffers for saving info and trajectory
    info_list = []
    # {RUN_SENTINEL}: max_sim_steps counts integration intervals.  Preserve
    # the initial t=0 state so K intervals yield K+1 timestamped samples.
    initial_step_state = {{
        "qpos": get_qpos(config, env)[0].detach().cpu().numpy().copy(),
        "qvel": get_qvel(config, env)[0].detach().cpu().numpy().copy(),
        "time": float(mj_data.time),
        "ctrl": mj_data.ctrl.copy(),
    }}
'''
    if text.count(info_anchor) != 1:
        raise ValueError("expected unique SPIDER info buffer anchor")
    patched = text.replace(info_anchor, info_replacement, 1)

    loop_matches = list(
        re.finditer(
            r"(?m)^(?P<indent>[ \t]+)for i in range\(config\.ctrl_steps\):\n",
            patched,
        )
    )
    if len(loop_matches) != 1:
        raise ValueError("expected unique SPIDER control execution loop")
    loop_anchor = loop_matches[0].group(0)
    indent = loop_matches[0].group("indent")
    loop_replacement = (
        f"{indent}execute_steps = int(config.ctrl_steps)\n"
        f"{indent}if int(config.max_sim_steps) > 0:\n"
        f"{indent}    execute_steps = min(\n"
        f"{indent}        execute_steps,\n"
        f"{indent}        max(int(config.max_sim_steps) - sim_step, 0),\n"
        f"{indent}    )\n"
        f"{indent}if execute_steps <= 0:\n"
        f"{indent}    break\n"
        f"{indent}for i in range(execute_steps):\n"
    )
    patched = patched.replace(loop_anchor, loop_replacement, 1)

    shift_anchor = (
        f"{indent}prev_ctrl = ctrls[config.ctrl_steps :]\n"
        f"{indent}new_ctrl = ctrl_ref[\n"
        f"{indent}    sim_step + prev_ctrl.shape[0] : sim_step\n"
        f"{indent}    + prev_ctrl.shape[0]\n"
        f"{indent}    + config.ctrl_steps\n"
        f"{indent}]\n"
    )
    shift_replacement = (
        f"{indent}prev_ctrl = ctrls[execute_steps:]\n"
        f"{indent}new_ctrl = ctrl_ref[\n"
        f"{indent}    sim_step + prev_ctrl.shape[0] : sim_step\n"
        f"{indent}    + prev_ctrl.shape[0]\n"
        f"{indent}    + execute_steps\n"
        f"{indent}]\n"
    )
    if patched.count(shift_anchor) != 1:
        raise ValueError("expected unique SPIDER receding-horizon shift")
    patched = patched.replace(shift_anchor, shift_replacement, 1)

    save_anchor = '''        for k in info_list[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in info_list], axis=0)
'''
    save_replacement = '''        for k in info_list[0].keys():
            values = [info[k] for info in info_list]
            if k in {"qpos", "qvel", "time", "ctrl"}:
                initial = np.asarray(initial_step_state[k])[None, ...]
                info_aggregated[k] = np.concatenate([initial, *values], axis=0)
            else:
                info_aggregated[k] = np.stack(values, axis=0)
'''
    if patched.count(save_anchor) != 1:
        raise ValueError("expected unique SPIDER info aggregation block")
    patched = patched.replace(save_anchor, save_replacement, 1)

    tracking_anchor = '''        qpos_traj = np.concatenate([info["qpos"] for info in info_list], axis=0)
'''
    tracking_replacement = '''        qpos_traj = np.concatenate(
            [initial_step_state["qpos"][None, ...]]
            + [info["qpos"] for info in info_list],
            axis=0,
        )
'''
    if patched.count(tracking_anchor) != 1:
        raise ValueError("expected unique SPIDER tracking aggregation block")
    return patched.replace(tracking_anchor, tracking_replacement, 1), True


def _backup_and_write(path: Path, original: str, patched: str) -> None:
    backup = path.with_suffix(path.suffix + ".aoe_timeline.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interp-py", type=Path, required=True)
    parser.add_argument("--io-py", type=Path, required=True)
    parser.add_argument("--run-mjwp-py", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = (
        ("interp", args.interp_py, patch_interp_text),
        ("io", args.io_py, patch_io_text),
        ("run_mjwp", args.run_mjwp_py, patch_run_mjwp_text),
    )
    prepared = []
    report = {"schema_version": 2, "dry_run": bool(args.dry_run), "targets": []}
    for name, path, patcher in targets:
        original = path.read_text(encoding="utf-8")
        patched, changed = patcher(original)
        prepared.append((path, original, patched, changed))
        report["targets"].append(
            {"name": name, "path": str(path.resolve()), "changed": changed}
        )
    if not args.dry_run:
        for path, original, patched, changed in prepared:
            if changed:
                _backup_and_write(path, original, patched)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
