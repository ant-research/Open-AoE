from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
import types

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "maintenance"
    / "patch_do_as_i_do_hoi_pipeline.py"
)
SPEC = importlib.util.spec_from_file_location("patch_do_as_i_do_hoi_pipeline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


PROCESS_SOURCE = '''import json
import os

import numpy as np
from scipy.spatial.transform import Rotation


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class _Loguru:
    logger = _Logger()


loguru = _Loguru()


def in_hand_at_endpoint(**kwargs):
    return False, None, None


def compute_in_hand_mask(*args):
    return np.array([False, True, False])


def find_freeze_indices(mask):
    return 1, 1


def _load_obj_verts(path: str) -> np.ndarray:
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(x) for x in parts[1:4]])
    return np.array(verts) if verts else np.zeros((0, 3))


def _scale_and_save_obj(src_path: str, scale: float, dst_path: str) -> None:
    pass


def _estimate_fingertip_contacts(
    fingertips_world: np.ndarray,
    qpos_obj: np.ndarray,
    obj_verts_local: np.ndarray,
    threshold: float,
):
    return None


def _copy_obj_texture(src_obj: str, dst_dir: str):
    return None


def main(out_data_dir):
    process_right = True
    process_left = False
    N = 3
    qpos_wrist_right = np.zeros((N, 7), dtype=np.float64)
    qpos_wrist_right[:, 3] = 1.0
    qpos_finger_right = np.zeros((N, 5, 7), dtype=np.float64)
    qpos_finger_right[:, :, 3] = 1.0
    qpos_wrist_left = qpos_wrist_right.copy()
    qpos_finger_left = qpos_finger_right.copy()
    qpos_obj_right = np.zeros((N, 7), dtype=np.float64)
    qpos_obj_right[:, 0] = np.array([0.0, 1.0, 2.0])
    qpos_obj_right[:, 3] = 1.0
    qpos_obj_left = qpos_obj_right.copy()
    mano_verts_right = np.ones((N, 1, 3), dtype=np.float64)
    mano_verts_left = np.empty((0, 1, 3), dtype=np.float64)
    freeze_verts = np.ones((1, 3), dtype=np.float64)
    os.makedirs(out_data_dir, exist_ok=True)
    mesh_src = os.path.join(out_data_dir, "mesh.obj")
    with open(mesh_src, "w") as f:
        f.write("v 0 0 0\\nv 1 0 0\\nv 0 1 0\\nf 1 2 3\\n")
    mesh_scale = 1.0
    verts = _load_obj_verts(mesh_src) * mesh_scale

    for side, hand_verts_side, qpos_obj_side in (
        ("right", mano_verts_right, qpos_obj_right),
        ("left",  mano_verts_left,  qpos_obj_left),
    ):
        if hand_verts_side.shape[0] == 0 or len(freeze_verts) == 0:
            continue
        needs = {}
        for ep_name, frame_idx in (("start", 0), ("end", -1)):
            in_hand, *_ = in_hand_at_endpoint(
                hand_verts_world=hand_verts_side,
                qpos_obj=qpos_obj_side,
                obj_verts=freeze_verts,
                frame=frame_idx,
            )
            needs[ep_name] = not in_hand
        if not (needs["start"] or needs["end"]):
            continue
        mask = compute_in_hand_mask(hand_verts_side, qpos_obj_side, freeze_verts)
        i_start, j_end = find_freeze_indices(mask)
        if i_start is None:
            loguru.logger.warning("no in-hand frame")
            continue
        if needs["start"] and i_start > 0:
            qpos_obj_side[:i_start] = qpos_obj_side[i_start]
            loguru.logger.info("froze start")
        if needs["end"] and j_end < qpos_obj_side.shape[0] - 1:
            qpos_obj_side[j_end + 1 :] = qpos_obj_side[j_end]
            loguru.logger.info("froze end")

    if False:
        _estimate_fingertip_contacts(
            qpos_finger_right[:, :, :3],
            qpos_obj_right,
            verts,
            contact_thresh,
        )
        _estimate_fingertip_contacts(
            qpos_finger_left[:, :, :3],
            qpos_obj_left,
            verts,
            contact_thresh,
        )
    contact_thresh = 0.065
    os.makedirs(out_data_dir, exist_ok=True)

    np.savez(
        f"{out_data_dir}/trajectory_keypoints.npz",
        qpos_obj_right=qpos_obj_right,
    )
    return qpos_obj_right
'''


def test_process_patch_defaults_to_rigid_hoi_and_writes_pass_report(
    tmp_path: Path, monkeypatch
) -> None:
    patched, changed = MODULE.patch_process_dataset_text(PROCESS_SOURCE)
    assert changed
    compile(patched, "process_dataset.py", "exec")
    namespace: dict[str, object] = {}
    exec(patched, namespace)

    monkeypatch.delenv(MODULE.FREEZE_ENV, raising=False)
    qpos_obj = namespace["main"](str(tmp_path))
    np.testing.assert_allclose(qpos_obj[:, 0], [0.0, 1.0, 2.0])
    report = json.loads((tmp_path / "hoi_preprocess_invariance.json").read_text())
    assert report["status"] == "pass"
    assert report["production_eligible"] is True
    assert report["object_only_freeze"] == {
        "environment_variable": MODULE.FREEZE_ENV,
        "requested": False,
        "applied": False,
        "diagnostic_only": True,
    }
    assert report["hoi_refinement"] is False
    assert report["sides"]["right"]["wrist_object_distance_error_m"]["max"] == 0.0
    assert report["sides"]["right"]["fingertip_object_distance_error_m"]["max"] == 0.0

    patched_again, changed_again = MODULE.patch_process_dataset_text(patched)
    assert not changed_again
    assert patched_again == patched


def test_process_patch_keeps_legacy_freeze_diagnostic_and_ineligible(
    tmp_path: Path, monkeypatch
) -> None:
    patched, _ = MODULE.patch_process_dataset_text(PROCESS_SOURCE)
    namespace: dict[str, object] = {}
    exec(patched, namespace)

    monkeypatch.setenv(MODULE.FREEZE_ENV, "1")
    qpos_obj = namespace["main"](str(tmp_path))
    np.testing.assert_allclose(qpos_obj[:, 0], [1.0, 1.0, 1.0])
    report = json.loads((tmp_path / "hoi_preprocess_invariance.json").read_text())
    assert report["status"] == "diagnostic"
    assert report["production_eligible"] is False
    assert report["object_only_freeze"]["requested"] is True
    assert report["object_only_freeze"]["applied"] is True
    assert report["hoi_refinement"] is True
    assert report["sides"]["right"]["passed"] is False


def test_process_patch_uses_triangle_surface_not_sparse_vertices(monkeypatch) -> None:
    patched, _ = MODULE.patch_process_dataset_text(PROCESS_SOURCE)
    namespace: dict[str, object] = {}
    exec(patched, namespace)

    class FakeMesh:
        def __init__(self, vertices, faces, **kwargs):
            self.vertices = np.asarray(vertices)
            self.faces = np.asarray(faces)

    def closest_point_naive(mesh, points):
        # Exact closest points for the test square in z=0.  The fixture keeps
        # trimesh optional while still proving that the patched estimator uses
        # returned triangle-surface points rather than sparse OBJ vertices.
        points = np.asarray(points)
        closest = points.copy()
        closest[:, 0] = np.clip(closest[:, 0], -0.5, 0.5)
        closest[:, 1] = np.clip(closest[:, 1], -0.5, 0.5)
        closest[:, 2] = 0.0
        distance = np.linalg.norm(points - closest, axis=1)
        return closest, distance, np.zeros(len(points), dtype=np.int64)

    fake_trimesh = types.SimpleNamespace(
        Trimesh=FakeMesh,
        proximity=types.SimpleNamespace(closest_point_naive=closest_point_naive),
    )
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)

    verts = np.array(
        [
            [-0.5, -0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
        ]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    fingertip = np.array([[[0.0, 0.0, 0.05]]])
    qpos_obj = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    nearest_vertex_distance = np.min(np.linalg.norm(verts - fingertip[0, 0], axis=1))
    assert nearest_vertex_distance > 0.065

    contact, contact_world, contact_site = namespace[
        "_estimate_fingertip_contacts"
    ](fingertip, qpos_obj, verts, faces, 0.065)
    np.testing.assert_array_equal(contact, [[1.0]])
    np.testing.assert_allclose(contact_world, [[[0.0, 0.0, 0.0]]])
    np.testing.assert_allclose(contact_site, [[0.0, 0.0, 0.0]])


def test_process_patch_upgrades_legacy_rigid_hoi_sentinel_to_surface_v2() -> None:
    patched, _ = MODULE.patch_process_dataset_text(PROCESS_SOURCE)
    original_loader = PROCESS_SOURCE[
        PROCESS_SOURCE.index(MODULE._OBJ_LOADER_START) : PROCESS_SOURCE.index(
            MODULE._OBJ_LOADER_END,
            PROCESS_SOURCE.index(MODULE._OBJ_LOADER_START),
        )
    ]
    original_estimator = PROCESS_SOURCE[
        PROCESS_SOURCE.index(MODULE._CONTACT_ESTIMATOR_START) : PROCESS_SOURCE.index(
            MODULE._CONTACT_ESTIMATOR_END,
            PROCESS_SOURCE.index(MODULE._CONTACT_ESTIMATOR_START),
        )
    ]
    legacy = patched.replace(MODULE._SURFACE_OBJ_LOADER, original_loader, 1)
    legacy = legacy.replace(MODULE._SURFACE_CONTACT_ESTIMATOR, original_estimator, 1)
    legacy = legacy.replace(
        "    verts, obj_faces = _load_obj_geometry(mesh_src)\n"
        "    verts = verts * mesh_scale\n",
        "    verts = _load_obj_verts(mesh_src) * mesh_scale\n",
        1,
    )
    legacy = re.sub(
        r"(?m)^(?P<indent>[ \t]+)verts,\n(?P=indent)obj_faces,\n(?P=indent)contact_thresh,\n",
        lambda match: (
            f"{match.group('indent')}verts,\n"
            f"{match.group('indent')}contact_thresh,\n"
        ),
        legacy,
    )
    assert MODULE.PROCESS_SENTINEL in legacy
    assert MODULE.SURFACE_CONTACT_SENTINEL not in legacy

    upgraded, changed = MODULE.patch_process_dataset_text(legacy)
    assert changed
    assert MODULE.SURFACE_CONTACT_SENTINEL in upgraded
    assert "trimesh.proximity.closest_point_naive" in upgraded
    assert len(
        re.findall(
            r"(?m)^(?P<indent>[ \t]+)verts,\n(?P=indent)obj_faces,\n",
            upgraded,
        )
    ) == 2
    compile(upgraded, "legacy_process_dataset.py", "exec")

    unchanged, changed_again = MODULE.patch_process_dataset_text(upgraded)
    assert not changed_again
    assert unchanged == upgraded


SOLVE_IK_SOURCE = '''import numpy as np


def main(file_path, start_idx=0, end_idx=-1, smoothing=True, average_frame_size=3):
    loaded_data = np.load(file_path)
    qpos_finger_right = loaded_data["a"][start_idx:end_idx]
    qpos_finger_left = loaded_data["b"][start_idx:end_idx]
    qpos_wrist_right = loaded_data["c"][start_idx:end_idx]
    qpos_wrist_left = loaded_data["d"][start_idx:end_idx]
    qpos_obj_right = loaded_data["e"][start_idx:end_idx]
    qpos_obj_left = loaded_data["f"][start_idx:end_idx]
    contact_right = loaded_data["contact"][start_idx:end_idx]
    contact_world_right = loaded_data["contact_pos"][start_idx:end_idx]

    qpos_list = qpos_finger_right.copy()
    contact = contact_right.copy()
    contact_pos = contact_world_right.copy()
    if smoothing:
        def moving_average_filter(signal_data, window_size=5):
            return np.convolve(
                signal_data, np.ones(window_size) / window_size, mode="valid"
            )

        filtered = np.zeros(
            (qpos_list.shape[0] - average_frame_size + 1, qpos_list.shape[1])
        )
        for i in range(qpos_list.shape[1]):
            filtered[:, i] = moving_average_filter(qpos_list[:, i], average_frame_size)
        qpos_list = filtered
        contact = contact[: qpos_list.shape[0]]
        contact_pos = contact_pos[: qpos_list.shape[0]]
    else:
        loguru.logger.info("Skipping IK smoothing.")

    # qvel via finite differences; first frame starts from rest
    return qpos_list, contact, contact_pos
'''


def test_solve_ik_patch_includes_final_frame_and_preserves_aligned_lengths(
    tmp_path: Path,
) -> None:
    patched, changed = MODULE.patch_solve_ik_text(SOLVE_IK_SOURCE)
    assert changed
    assert "slice_end = None if end_idx is None or int(end_idx) < 0" in patched
    assert "filtered = np.zeros_like(qpos_list)" in patched
    assert "contact = contact[: qpos_list.shape[0]]" not in patched
    compile(patched, "solve_ik.py", "exec")
    namespace: dict[str, object] = {}
    exec(patched, namespace)

    path = tmp_path / "input.npz"
    base = np.arange(10, dtype=np.float64).reshape(5, 2)
    contact = np.arange(5, dtype=np.float64)[:, None]
    contact_pos = np.arange(15, dtype=np.float64).reshape(5, 1, 3)
    np.savez(
        path,
        a=base,
        b=base,
        c=base,
        d=base,
        e=base,
        f=base,
        contact=contact,
        contact_pos=contact_pos,
    )
    qpos, actual_contact, actual_contact_pos = namespace["main"](path)
    assert qpos.shape[0] == 5
    np.testing.assert_array_equal(actual_contact, contact)
    np.testing.assert_array_equal(actual_contact_pos, contact_pos)

    patched_again, changed_again = MODULE.patch_solve_ik_text(patched)
    assert not changed_again
    assert patched_again == patched


INTERP_SOURCE = MODULE._INTERP_OLD + '''
    if order == 0:
        mode = "nearest"
    else:
        mode = "linear"
    src_permuted = src.permute(0, 2, 1)
    dst_len = H * n
    dst_permuted = F.interpolate(src_permuted, size=dst_len, mode=mode)
    return dst_permuted.permute(0, 2, 1)


def align_to_sim_dt(arr, ref_dt, sim_dt, order=1):
    if ref_dt >= sim_dt:
        ref_steps = max(1, int(round(ref_dt / sim_dt)))
        if ref_steps == 1:
            return arr
        return interp(arr.unsqueeze(0), ref_steps, order=order).squeeze(0)
    return arr
'''


def test_interp_patch_limits_exact_duration_to_reference_callers() -> None:
    patched, changed = MODULE.patch_interp_text(INTERP_SOURCE)
    assert changed
    assert MODULE.INTERP_SENTINEL in patched
    assert (
        "dst_len = (H - 1) * n + 1 if preserve_endpoint_duration else H * n"
        in patched
    )
    assert "return src if preserve_endpoint_duration else src.repeat(1, n, 1)" in patched
    assert "preserve_endpoint_duration=True" in patched
    assert (5 - 1) * 6 + 1 == 25
    assert 5 * 6 == 30

    patched_again, changed_again = MODULE.patch_interp_text(patched)
    assert not changed_again
    assert patched_again == patched


def test_interp_patch_upgrades_legacy_global_exact_duration_patch() -> None:
    legacy = INTERP_SOURCE.replace(MODULE._INTERP_OLD, MODULE._INTERP_LEGACY_EXACT)
    legacy = legacy.replace("    dst_len = H * n\n", "    dst_len = (H - 1) * n + 1\n")
    patched, changed = MODULE.patch_interp_text(legacy)
    assert changed
    assert MODULE.INTERP_SENTINEL in patched
    assert MODULE.LEGACY_INTERP_SENTINEL not in patched
    assert "preserve_endpoint_duration=True" in patched


LAUNCH_SOURCE = '''import os


def _apply_env_overrides(cfg_dict):
    env_map = {
        "DAI_MJWP_NUM_SAMPLES": ("num_samples", int),
    }


def run_pipeline(cfg):
    solve_ik(
        output_root_dir=cfg.output_root_dir,
        force=cfg.force,
        smoothing=cfg.smoothing,
    )

    # Load the MJWP Config (YAML defaults + dataset override) before Stage 4.5:
    config = load_mjwp_config()
    contact_guidance_enabled = bool(getattr(config, "contact_guidance", False))
'''


def test_launch_patch_applies_and_asserts_exact_timing() -> None:
    patched, changed = MODULE.patch_launch_text(LAUNCH_SOURCE)
    assert changed
    assert MODULE.LAUNCH_SENTINEL in patched
    assert '"DAI_MJWP_SIM_DT": ("sim_dt", float)' in patched
    assert 'sim_dt=float(os.environ.get("DAI_MJWP_SIM_DT", "0.005"))' in patched
    assert "AOE effective timing mismatch" in patched
    compile(patched, "launch.py", "exec")


OPTIMIZE_SOURCE = '''import os
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp


def main(config):
    qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos = load_data(
        config, config.data_path
    )
    ref_data = (qpos_ref, qvel_ref, ctrl_ref, contact, contact_pos)

    # For warmup, start the simulation with an open hand (finger joints = 0)
    # so the optimizer can close the fingers onto the object surface.
    # The reference stays as the closed grasp -- only the initial sim state changes.
    _warmup_init_enabled = False
    finger_qpos_indices: list[int] = []
    if config.warmup_steps > 0 and config.embodiment_type in [
        "bimanual", "right", "left"
    ]:
        _warmup_init_enabled = True
        nq_obj = config.nq_obj
        robot_nq = qpos_ref.shape[1] - nq_obj
        wrist_dof = 6
        if config.embodiment_type == "bimanual":
            half = robot_nq // 2
            finger_qpos_indices += list(range(wrist_dof, half))
            finger_qpos_indices += list(range(half + wrist_dof, robot_nq))
        else:
            finger_qpos_indices += list(range(wrist_dof, robot_nq))

        init_qpos = qpos_ref[0].clone()
        init_ctrl = ctrl_ref[0].clone()
        init_qpos[finger_qpos_indices] = 0.0
        init_ctrl[finger_qpos_indices] = 0.0
        init_ref_data = (
            torch.cat([init_qpos.unsqueeze(0), qpos_ref[1:]], dim=0),
            qvel_ref,
            torch.cat([init_ctrl.unsqueeze(0), ctrl_ref[1:]], dim=0),
            contact,
            contact_pos,
        )
    else:
        init_ref_data = ref_data

    env = setup_env(config, init_ref_data)
    precompute_hand_object_geom_mask(config, env.model_cpu)

    # GPU warmup: analytical init pose -- translate wrist along -palm_normal
    if _warmup_init_enabled:
        warmup_analytical_init(
            config, env, qpos_ref[0], ctrl_ref[0], finger_qpos_indices
        )

    base_interp_n = min(
        config.warmup_ref_base_interp_steps, config.warmup_steps
    ) if config.warmup_ref_base_interp_steps > 0 else 0
    finger_interp_n = min(
        config.warmup_ref_finger_interp_steps, config.warmup_steps
    ) if config.warmup_ref_finger_interp_steps > 0 else 0
    if (base_interp_n > 0 or finger_interp_n > 0) and config.warmup_steps > 0:
        pass

    mj_model = setup_mj_model(config)
    mj_data = mujoco.MjData(mj_model)
    warmup_enabled = True
    ctrls = ctrl_ref[: config.horizon_steps]
    info_list = []
    while True:
        sim_step = 0
        for i in range(config.ctrl_steps):
            ctrl_step = ctrls[i]
            abs_step = sim_step + i
            if warmup_enabled and abs_step == config.warmup_steps:
                set_weld_active(config, env, False)
                set_gravity(env, list(config.default_gravity))
            pass
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
'''


def test_optimize_patch_executes_partial_final_chunk_and_concatenates_steps() -> None:
    patched, changed = MODULE.patch_optimize_physics_text(OPTIMIZE_SOURCE)
    assert changed
    assert MODULE.OPTIMIZE_SENTINEL in patched
    assert MODULE.WARMUP_RELEASE_SENTINEL in patched
    assert "max(int(config.max_sim_steps) - sim_step, 0)" in patched
    assert "for i in range(execute_steps)" in patched
    assert "prev_ctrl = ctrls[execute_steps:]" in patched
    assert 'if k in {"qpos", "qvel", "time", "ctrl"}' in patched
    assert "import json" in patched
    assert "import mujoco_warp as mjwarp" in patched
    compile(patched, "optimize_physics.py", "exec")


def test_optimize_patch_preserves_closed_reference_grasp_by_default() -> None:
    patched, changed = MODULE.patch_optimize_physics_text(OPTIMIZE_SOURCE)
    assert changed
    assert (
        '_aoe_preserve_reference_grasp = _aoe_bool_env(\n'
        '        "DAI_MJWP_PRESERVE_REFERENCE_GRASP", True'
    ) in patched
    assert (
        '_aoe_diagnostic_open_hand = _aoe_bool_env(\n'
        '        "DAI_MJWP_DIAGNOSTIC_OPEN_HAND_ANALYTICAL_INIT", False'
    ) in patched
    assert "init_ref_data = ref_data" in patched
    assert "_aoe_restore_reference_grasp_state(" in patched
    assert "qpos_ref[0],\n            qvel_ref[0],\n            ctrl_ref[0]" in patched
    # Finger zeroing survives only inside the explicit diagnostic branch.
    diagnostic_branch = patched[
        patched.index("if _warmup_init_enabled:") : patched.index(
            "    else:\n        init_ref_data = ref_data"
        )
    ]
    assert "init_qpos[finger_qpos_indices] = 0.0" in diagnostic_branch
    assert "_aoe_diagnostic_open_hand" in patched[
        patched.index("_warmup_init_enabled = bool(") : patched.index(
            "if _warmup_init_enabled:"
        )
    ]
    assert "if _warmup_init_enabled and (" in patched


def test_optimize_patch_gates_release_before_weld_and_gravity_change() -> None:
    patched, _ = MODULE.patch_optimize_physics_text(OPTIMIZE_SOURCE)
    release = patched.index("_aoe_release_grasp_preflight(", patched.index("abs_step"))
    unweld = patched.index("set_weld_active(config, env, False)", release)
    gravity = patched.index("set_gravity(env, list(config.default_gravity))", release)
    assert release < unweld < gravity
    assert "contact[abs_step]" in patched[release:unweld]
    assert "mujoco.mj_forward(mj_model, mj_data)" in patched
    assert '"source": "outer_execution_world_0_cpu_mujoco_forward"' in patched
    assert "distance <= 0.0" in patched
    assert 'Path(config.output_dir) / "release_grasp_preflight.json"' in patched
    assert "production_eligible" in patched


def test_optimize_patch_upgrades_old_exact_chunk_sentinel() -> None:
    legacy, legacy_changed = MODULE._patch_exact_control_chunk_text(OPTIMIZE_SOURCE)
    assert legacy_changed
    assert MODULE.OPTIMIZE_SENTINEL in legacy
    assert MODULE.WARMUP_RELEASE_SENTINEL not in legacy

    upgraded, changed = MODULE.patch_optimize_physics_text(legacy)
    assert changed
    assert MODULE.OPTIMIZE_SENTINEL in upgraded
    assert MODULE.WARMUP_RELEASE_SENTINEL in upgraded
    compile(upgraded, "legacy_optimize_physics.py", "exec")

    unchanged, changed_again = MODULE.patch_optimize_physics_text(upgraded)
    assert not changed_again
    assert unchanged == upgraded


def test_pristine_production_wrapper_rejects_reference_grasp_runtime_patching() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_do_as_i_do_official_retarget.sh"
    ).read_text(encoding="utf-8")
    assert "DAI_*|MJWP_*|DO_AS_I_DO_*|ROBOT_RENDER_*" in wrapper
    assert "deprecated backend-tuning environment variable is forbidden" in wrapper
    assert "DAI_MJWP_PRESERVE_REFERENCE_GRASP=1" not in wrapper
    assert "DAI_MJWP_RELEASE_GRASP_GATE=1" not in wrapper
    assert 'env -i "${backend_env[@]}" "$python_bin" "${backend_argv[@]}"' in wrapper
