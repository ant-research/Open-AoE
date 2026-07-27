#!/usr/bin/env python3
"""Install AoE's rigid-HOI and exact-timeline fixes into Do-as-I-Do.

The Do-as-I-Do checkout is an ignored runtime dependency, so these fixes live in
the tracked retarget-lab tree and are installed immediately before a run.  The
patch is deliberately fail-closed: all target files are validated in memory
before any of them is changed, and already-patched files are left untouched.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROCESS_SENTINEL = "AOE_OBJECT_ONLY_HOI_FREEZE_DIAGNOSTIC"
SURFACE_CONTACT_SENTINEL = "AOE_TRIANGLE_SURFACE_CONTACT"
SOLVE_IK_SENTINEL = "AOE_PRESERVE_DAI_IK_TIMELINE"
INTERP_SENTINEL = "AOE_REFERENCE_ONLY_ENDPOINT_DURATION"
LEGACY_INTERP_SENTINEL = "AOE_PRESERVE_REFERENCE_ENDPOINT_DURATION"
LAUNCH_SENTINEL = "AOE_EXACT_TIMING_ENV_OVERRIDES"
OPTIMIZE_SENTINEL = "AOE_EXACT_FINAL_CONTROL_CHUNK"
WARMUP_RELEASE_SENTINEL = "AOE_PRESERVE_REFERENCE_GRASP_RELEASE_GATE"

FREEZE_ENV = "DO_AS_I_DO_DIAGNOSTIC_OBJECT_ONLY_FREEZE"
PRESERVE_REFERENCE_GRASP_ENV = "DAI_MJWP_PRESERVE_REFERENCE_GRASP"
DIAGNOSTIC_OPEN_HAND_ENV = "DAI_MJWP_DIAGNOSTIC_OPEN_HAND_ANALYTICAL_INIT"
RELEASE_GRASP_GATE_ENV = "DAI_MJWP_RELEASE_GRASP_GATE"
RELEASE_GRASP_REPORT_ENV = "DAI_MJWP_RELEASE_GRASP_PREFLIGHT_PATH"

_FREEZE_LOOP_START = '''    for side, hand_verts_side, qpos_obj_side in (
        ("right", mano_verts_right, qpos_obj_right),
        ("left",  mano_verts_left,  qpos_obj_left),
    ):
'''
_FREEZE_BLOCK_END = "\n    contact_thresh ="

_OBJ_LOADER_START = "def _load_obj_verts(path: str) -> np.ndarray:\n"
_OBJ_LOADER_END = "\n\ndef _scale_and_save_obj"
_CONTACT_ESTIMATOR_START = "def _estimate_fingertip_contacts(\n"
_CONTACT_ESTIMATOR_END = "\n\ndef _copy_obj_texture"

_SURFACE_OBJ_LOADER = '''def _load_obj_geometry(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load OBJ vertices and triangulated faces without changing geometry."""
    # AOE_TRIANGLE_SURFACE_CONTACT: retaining face topology is essential for
    # contact distance.  Sparse meshes (a box has only eight vertices) make a
    # nearest-vertex test reject fingertips that are close to a face centre.
    verts = []
    polygon_tokens = []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(x) for x in parts[1:4]])
            elif line.startswith("f "):
                polygon_tokens.append(line.strip().split()[1:])

    verts_array = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    faces = []
    for tokens in polygon_tokens:
        polygon = []
        for token in tokens:
            vertex_token = token.split("/", 1)[0]
            if not vertex_token:
                raise ValueError(f"OBJ face has no vertex index: {path}: {token}")
            raw_index = int(vertex_token)
            index = raw_index - 1 if raw_index > 0 else len(verts) + raw_index
            if index < 0 or index >= len(verts):
                raise ValueError(f"OBJ face index out of range: {path}: {raw_index}")
            polygon.append(index)
        for i in range(1, len(polygon) - 1):
            faces.append([polygon[0], polygon[i], polygon[i + 1]])
    faces_array = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    return verts_array, faces_array


def _load_obj_verts(path: str) -> np.ndarray:
    """Backward-compatible vertex-only helper for external callers."""
    return _load_obj_geometry(path)[0]
'''

_SURFACE_CONTACT_ESTIMATOR = '''def _estimate_fingertip_contacts(
    fingertips_world: np.ndarray,
    qpos_obj: np.ndarray,
    obj_verts_local: np.ndarray,
    obj_faces: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate contacts using closest points on object triangles.

    This changes contact diagnostics/guidance only.  It never modifies hand or
    object trajectories.  For face-less point-cloud OBJs, the explicit
    nearest-vertex fallback preserves upstream compatibility and is logged.
    """
    n, nf, _ = fingertips_world.shape
    contact = np.zeros((n, nf), dtype=np.float32)
    contact_world = np.zeros((n, nf, 3), dtype=np.float32)
    if obj_verts_local.size == 0 or n == 0:
        return contact, contact_world, np.zeros((nf, 3), dtype=np.float32)

    rotations = Rotation.from_quat(qpos_obj[:, 3:7][:, [1, 2, 3, 0]])
    query_local = np.empty_like(fingertips_world, dtype=np.float64)
    for t in range(n):
        query_local[t] = rotations[t].inv().apply(
            fingertips_world[t] - qpos_obj[t, :3]
        )

    flat_query = query_local.reshape(-1, 3)
    if len(obj_faces):
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=np.asarray(obj_verts_local, dtype=np.float64),
            faces=np.asarray(obj_faces, dtype=np.int64),
            process=False,
            validate=False,
        )
        closest_flat, distance_flat, _ = trimesh.proximity.closest_point_naive(
            mesh, flat_query
        )
        method = "triangle_surface_closest_point"
    else:
        loguru.logger.warning(
            "AOE_TRIANGLE_SURFACE_CONTACT: OBJ has no faces; using "
            "nearest-vertex diagnostic fallback"
        )
        delta = flat_query[:, None, :] - obj_verts_local[None, :, :]
        all_distance = np.linalg.norm(delta, axis=-1)
        nearest_index = np.argmin(all_distance, axis=1)
        closest_flat = obj_verts_local[nearest_index]
        distance_flat = all_distance[np.arange(len(flat_query)), nearest_index]
        method = "nearest_vertex_fallback"

    closest_local = np.asarray(closest_flat, dtype=np.float64).reshape(n, nf, 3)
    distances = np.asarray(distance_flat, dtype=np.float64).reshape(n, nf)
    contact[:] = (distances <= float(threshold)).astype(np.float32)
    for t in range(n):
        contact_world[t] = (
            rotations[t].apply(closest_local[t]) + qpos_obj[t, :3]
        ).astype(np.float32)

    contact_sites = np.zeros((nf, 3), dtype=np.float32)
    for f in range(nf):
        hits = closest_local[contact[:, f] > 0.5, f]
        if len(hits):
            contact_sites[f] = np.median(hits, axis=0).astype(np.float32)
        else:
            # Keep the closest surface point as a plausible inactive site.
            contact_sites[f] = closest_local[int(np.argmin(distances[:, f])), f]

    loguru.logger.info(
        "AOE_TRIANGLE_SURFACE_CONTACT: method={} faces={} active_frames={}/{}",
        method,
        int(len(obj_faces)),
        int(np.any(contact > 0.5, axis=1).sum()),
        n,
    )
    return contact, contact_world, contact_sites
'''

_REPORT_ANCHOR = '''    os.makedirs(out_data_dir, exist_ok=True)

    np.savez(
'''

_REPORT_BLOCK = '''    os.makedirs(out_data_dir, exist_ok=True)

    # AOE_OBJECT_ONLY_HOI_FREEZE_DIAGNOSTIC: record the actual hand/object
    # invariance of this preprocessing run next to trajectory_keypoints.npz.
    # A production run is rejected if any object-only mutation occurred.
    _aoe_thresholds = {
        "relative_distance_max_error_m": 1.0e-7,
        "object_position_max_error_m": 1.0e-7,
        "object_rotation_max_error_rad": 1.0e-7,
    }

    def _aoe_error_stats(values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return {"median": 0.0, "p95": 0.0, "max": 0.0}
        return {
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }

    def _aoe_side_invariance(before_obj, after_obj, wrist, fingertips):
        before_obj = np.asarray(before_obj, dtype=np.float64)
        after_obj = np.asarray(after_obj, dtype=np.float64)
        wrist_xyz = np.asarray(wrist[..., :3], dtype=np.float64)
        fingertip_xyz = np.asarray(fingertips[..., :3], dtype=np.float64)

        wrist_before = np.linalg.norm(wrist_xyz - before_obj[:, :3], axis=-1)
        wrist_after = np.linalg.norm(wrist_xyz - after_obj[:, :3], axis=-1)
        fingertip_before = np.linalg.norm(
            fingertip_xyz - before_obj[:, None, :3], axis=-1
        )
        fingertip_after = np.linalg.norm(
            fingertip_xyz - after_obj[:, None, :3], axis=-1
        )

        # Work on copies: normalising a view of ``after_obj`` here would itself
        # mutate the production trajectory this audit is meant to protect.
        before_quat = before_obj[:, 3:7].copy()
        after_quat = after_obj[:, 3:7].copy()
        before_quat /= np.maximum(
            np.linalg.norm(before_quat, axis=-1, keepdims=True), 1e-12
        )
        after_quat /= np.maximum(
            np.linalg.norm(after_quat, axis=-1, keepdims=True), 1e-12
        )
        quat_dot = np.abs(np.sum(before_quat * after_quat, axis=-1))
        rotation_error = 2.0 * np.arccos(np.clip(quat_dot, -1.0, 1.0))

        result = {
            "frame_count": int(before_obj.shape[0]),
            "wrist_object_distance_error_m": _aoe_error_stats(
                np.abs(wrist_after - wrist_before)
            ),
            "fingertip_object_distance_error_m": _aoe_error_stats(
                np.abs(fingertip_after - fingertip_before)
            ),
            "object_position_error_m": _aoe_error_stats(
                np.linalg.norm(after_obj[:, :3] - before_obj[:, :3], axis=-1)
            ),
            "object_rotation_error_rad": _aoe_error_stats(rotation_error),
        }
        result["passed"] = bool(
            result["wrist_object_distance_error_m"]["max"]
            <= _aoe_thresholds["relative_distance_max_error_m"]
            and result["fingertip_object_distance_error_m"]["max"]
            <= _aoe_thresholds["relative_distance_max_error_m"]
            and result["object_position_error_m"]["max"]
            <= _aoe_thresholds["object_position_max_error_m"]
            and result["object_rotation_error_rad"]["max"]
            <= _aoe_thresholds["object_rotation_max_error_rad"]
        )
        return result

    _aoe_sides = {}
    if process_right:
        _aoe_sides["right"] = _aoe_side_invariance(
            _aoe_qpos_obj_pre_freeze["right"],
            qpos_obj_right,
            qpos_wrist_right,
            qpos_finger_right,
        )
    if process_left:
        _aoe_sides["left"] = _aoe_side_invariance(
            _aoe_qpos_obj_pre_freeze["left"],
            qpos_obj_left,
            qpos_wrist_left,
            qpos_finger_left,
        )
    _aoe_invariance_passed = bool(
        _aoe_sides and all(side["passed"] for side in _aoe_sides.values())
    )
    _aoe_production_eligible = bool(
        not _aoe_object_only_freeze_requested
        and not _aoe_object_only_freeze_applied
        and _aoe_invariance_passed
        and len(obj_faces) > 0
    )
    _aoe_invariance_report = {
        "schema_version": 1,
        "status": (
            "pass"
            if _aoe_production_eligible
            else ("diagnostic" if _aoe_object_only_freeze_requested else "invalid")
        ),
        "production_eligible": _aoe_production_eligible,
        "object_only_freeze": {
            "environment_variable": "DO_AS_I_DO_DIAGNOSTIC_OBJECT_ONLY_FREEZE",
            "requested": bool(_aoe_object_only_freeze_requested),
            "applied": bool(_aoe_object_only_freeze_applied),
            "diagnostic_only": True,
        },
        "hoi_refinement": bool(_aoe_object_only_freeze_applied),
        "contact_estimator": {
            "method": (
                "triangle_surface_closest_point"
                if len(obj_faces)
                else "nearest_vertex_fallback"
            ),
            "object_face_count": int(len(obj_faces)),
            "production_eligible": bool(len(obj_faces) > 0),
        },
        "thresholds": _aoe_thresholds,
        "sides": _aoe_sides,
    }
    _aoe_report_path = os.path.join(out_data_dir, "hoi_preprocess_invariance.json")
    with open(_aoe_report_path, "w", encoding="utf-8") as f:
        json.dump(_aoe_invariance_report, f, indent=2, sort_keys=True)
    loguru.logger.info(
        "AOE HOI preprocessing invariance: status={} report={}",
        _aoe_invariance_report["status"],
        _aoe_report_path,
    )
    if not _aoe_object_only_freeze_requested and not _aoe_production_eligible:
        raise RuntimeError(
            "AOE production HOI preprocessing invariance failed; "
            f"see {_aoe_report_path}"
        )

    np.savez(
'''


def _indent_block(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "".join(prefix + line if line.strip() else line for line in text.splitlines(True))


def _patch_surface_contact_text(text: str) -> tuple[str, bool]:
    """Install or validate the version-2 triangle-surface contact upgrade."""

    if SURFACE_CONTACT_SENTINEL in text:
        required = (
            "def _load_obj_geometry(",
            "trimesh.proximity.closest_point_naive(",
            "verts, obj_faces = _load_obj_geometry(mesh_src)",
        )
        missing = [token for token in required if token not in text]
        surface_calls = re.findall(
            r"(?m)^(?P<indent>[ \t]+)verts,\n(?P=indent)obj_faces,\n",
            text,
        )
        if missing or len(surface_calls) != 2:
            raise ValueError(
                "corrupt/incomplete triangle-surface contact patch: "
                + ",".join(missing or ["right/left obj_faces calls"])
            )
        return text, False

    loader_start = text.find(_OBJ_LOADER_START)
    loader_end = text.find(_OBJ_LOADER_END, loader_start)
    if loader_start < 0 or loader_end < 0:
        raise ValueError("expected Do-as-I-Do OBJ vertex loader not found")
    estimator_start = text.find(_CONTACT_ESTIMATOR_START)
    estimator_end = text.find(_CONTACT_ESTIMATOR_END, estimator_start)
    if estimator_start < 0 or estimator_end < 0:
        raise ValueError("expected Do-as-I-Do fingertip contact estimator not found")
    mesh_load = "    verts = _load_obj_verts(mesh_src) * mesh_scale\n"
    if text.count(mesh_load) != 1:
        raise ValueError("expected unique scaled OBJ vertex load not found")
    contact_call_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]+)verts,\n(?P=indent)contact_thresh,\n"
    )
    if len(contact_call_pattern.findall(text)) != 2:
        raise ValueError("expected right/left fingertip contact calls not found")

    text = text[:loader_start] + _SURFACE_OBJ_LOADER + text[loader_end:]
    estimator_start = text.find(_CONTACT_ESTIMATOR_START)
    estimator_end = text.find(_CONTACT_ESTIMATOR_END, estimator_start)
    text = text[:estimator_start] + _SURFACE_CONTACT_ESTIMATOR + text[estimator_end:]
    text = text.replace(
        mesh_load,
        "    verts, obj_faces = _load_obj_geometry(mesh_src)\n"
        "    verts = verts * mesh_scale\n",
        1,
    )
    text = contact_call_pattern.sub(
        lambda match: (
            f"{match.group('indent')}verts,\n"
            f"{match.group('indent')}obj_faces,\n"
            f"{match.group('indent')}contact_thresh,\n"
        ),
        text,
    )
    return text, True


def patch_process_dataset_text(text: str) -> tuple[str, bool]:
    """Gate the upstream object-only freeze and add per-run HOI invariance QC."""
    # PROCESS_SENTINEL predates triangle-surface contact estimation.  Returning
    # solely on that legacy marker would silently leave a previously patched
    # runtime on nearest-vertex contact.  Upgrade it in place, or validate that
    # the complete v2 surface patch is already present.
    text, surface_changed = _patch_surface_contact_text(text)
    if PROCESS_SENTINEL in text:
        if text.count(PROCESS_SENTINEL) < 2 or "production_eligible" not in text:
            raise ValueError("corrupt/incomplete rigid-HOI process_dataset patch")
        return text, surface_changed

    start = text.find(_FREEZE_LOOP_START)
    if start < 0:
        raise ValueError("expected Do-as-I-Do object-freeze loop not found")
    end = text.find(_FREEZE_BLOCK_END, start)
    if end < 0:
        raise ValueError("expected contact-estimation anchor after object-freeze loop not found")
    if text.count(_REPORT_ANCHOR) != 1:
        raise ValueError("expected unique trajectory_keypoints output anchor not found")

    freeze_loop = text[start:end]
    start_assignment = "            qpos_obj_side[:i_start] = qpos_obj_side[i_start]\n"
    end_assignment = "            qpos_obj_side[j_end + 1 :] = qpos_obj_side[j_end]\n"
    if freeze_loop.count(start_assignment) != 1 or freeze_loop.count(end_assignment) != 1:
        raise ValueError("expected object-freeze assignments not found")
    freeze_loop = freeze_loop.replace(
        start_assignment,
        start_assignment + "            _aoe_object_only_freeze_applied = True\n",
        1,
    )
    freeze_loop = freeze_loop.replace(
        end_assignment,
        end_assignment + "            _aoe_object_only_freeze_applied = True\n",
        1,
    )
    indented_freeze_loop = _indent_block(freeze_loop, 4)
    if not indented_freeze_loop.endswith("\n"):
        indented_freeze_loop += "\n"

    guarded_loop = f'''    # {PROCESS_SENTINEL}: production preprocessing must preserve the
    # original hand/object relationship.  The legacy object-only freeze is
    # available solely as an explicitly invalid diagnostic run.
    _aoe_qpos_obj_pre_freeze = {{
        "right": qpos_obj_right.copy(),
        "left": qpos_obj_left.copy(),
    }}
    _aoe_object_only_freeze_requested = os.environ.get(
        "{FREEZE_ENV}", "0"
    ).strip().lower() in {{"1", "true", "yes", "on"}}
    _aoe_object_only_freeze_applied = False
    if _aoe_object_only_freeze_requested:
        loguru.logger.warning(
            "{PROCESS_SENTINEL} requested=true; "
            "this object-only mutation is diagnostic-only and production-ineligible"
        )
{indented_freeze_loop}        loguru.logger.warning(
            "{PROCESS_SENTINEL} applied={{}} hoi_refinement={{}}",
            _aoe_object_only_freeze_applied,
            _aoe_object_only_freeze_applied,
        )
    else:
        loguru.logger.info(
            "{PROCESS_SENTINEL} disabled: hoi_refinement=false; "
            "production object trajectories remain unchanged"
        )
'''
    patched = text[:start] + guarded_loop + text[end:]
    patched = patched.replace(_REPORT_ANCHOR, _REPORT_BLOCK, 1)
    return patched, True


_SMOOTHING_START = "    if smoothing:\n"
_SMOOTHING_END = "\n    # qvel via finite differences; first frame starts from rest\n"

_SAME_LENGTH_SMOOTHING = '''    if smoothing:
        # AOE_PRESERVE_DAI_IK_TIMELINE: edge padding keeps one centered IK
        # state per source frame, including matching contact/contact_pos rows.
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

        filtered = np.zeros_like(qpos_list)
        for i in range(qpos_list.shape[1]):
            filtered[:, i] = moving_average_filter(qpos_list[:, i], average_frame_size)
        qpos_list = filtered
    else:
        loguru.logger.info("Skipping IK smoothing.")

    if not (qpos_list.shape[0] == contact.shape[0] == contact_pos.shape[0]):
        raise RuntimeError(
            "AOE_PRESERVE_DAI_IK_TIMELINE failed: qpos/contact/contact_pos "
            f"lengths are {qpos_list.shape[0]}/{contact.shape[0]}/{contact_pos.shape[0]}"
        )
'''


def patch_solve_ik_text(text: str) -> tuple[str, bool]:
    """Preserve the final source frame and keep smoothed arrays same-length."""
    if SOLVE_IK_SENTINEL in text:
        return text, False
    load_anchor = "    loaded_data = np.load(file_path)\n"
    if text.count(load_anchor) != 1:
        raise ValueError("expected unique solve_ik keypoint-load anchor not found")
    slice_count = text.count("[start_idx:end_idx]")
    if slice_count < 8:
        raise ValueError(
            f"expected solve_ik start/end slices not found (found {slice_count})"
        )
    smooth_start = text.find(_SMOOTHING_START)
    smooth_end = text.find(_SMOOTHING_END, smooth_start)
    if smooth_start < 0 or smooth_end < 0:
        raise ValueError("expected solve_ik smoothing block not found")
    old_smoothing = text[smooth_start:smooth_end]
    if 'mode="valid"' not in old_smoothing or "contact = contact[: qpos_list.shape[0]]" not in old_smoothing:
        raise ValueError("solve_ik smoothing block does not match expected shrinking behavior")

    slice_setup = f'''{load_anchor}    # {SOLVE_IK_SENTINEL}: negative end_idx means through EOF, not
    # Python's "drop the final frame" slice semantics.
    slice_end = None if end_idx is None or int(end_idx) < 0 else int(end_idx)
'''
    patched = text.replace(load_anchor, slice_setup, 1)
    patched = patched.replace("[start_idx:end_idx]", "[start_idx:slice_end]")
    patched_smooth_start = patched.find(_SMOOTHING_START)
    patched_smooth_end = patched.find(_SMOOTHING_END, patched_smooth_start)
    if patched_smooth_start < 0 or patched_smooth_end < 0:
        raise AssertionError("solve_ik smoothing anchors disappeared during patching")
    patched = (
        patched[:patched_smooth_start]
        + _SAME_LENGTH_SMOOTHING
        + patched[patched_smooth_end:]
    )
    return patched, True


_INTERP_OLD = '''def interp(src: torch.Tensor, n: int, order: int = 1) -> torch.Tensor:
    """Upsample (N, H, D) by factor n with order 0/1/2 hold -> (N, H*n, D)."""
    if order not in [0, 1, 2]:
        raise ValueError("Order must be an integer: 0, 1, or 2.")

    N, H, D = src.shape

    if H <= 1:
        return src.repeat(1, n, 1)
'''

_INTERP_LEGACY_EXACT = '''def interp(src: torch.Tensor, n: int, order: int = 1) -> torch.Tensor:
    """Upsample while preserving both endpoints and exact reference duration."""
    # AOE_PRESERVE_REFERENCE_ENDPOINT_DURATION: H reference samples span H-1
    # intervals.  Splitting every interval into n simulation intervals yields
    # (H-1)*n+1 samples; H*n silently adds almost one extra reference interval.
    if order not in [0, 1, 2]:
        raise ValueError("Order must be an integer: 0, 1, or 2.")
    if int(n) != n or n < 1:
        raise ValueError(f"n must be a positive integer; got {n}")
    n = int(n)

    N, H, D = src.shape

    if H <= 1:
        return src
'''

_INTERP_REFERENCE_ONLY = '''def interp(
    src: torch.Tensor,
    n: int,
    order: int = 1,
    *,
    preserve_endpoint_duration: bool = False,
) -> torch.Tensor:
    """Upsample a signal; reference timelines may request exact duration."""
    # AOE_REFERENCE_ONLY_ENDPOINT_DURATION: optimizer knot schedules require
    # H*n samples, while H reference poses span only H-1 time intervals.  Keep
    # those semantics explicit instead of changing the shared helper globally.
    if order not in [0, 1, 2]:
        raise ValueError("Order must be an integer: 0, 1, or 2.")
    if int(n) != n or n < 1:
        raise ValueError(f"n must be a positive integer; got {n}")
    n = int(n)

    N, H, D = src.shape

    if H <= 1:
        return src if preserve_endpoint_duration else src.repeat(1, n, 1)
'''

_ALIGN_REFERENCE_OLD = (
    "        return interp(arr.unsqueeze(0), ref_steps, order=order).squeeze(0)\n"
)
_ALIGN_REFERENCE_NEW = '''        return interp(
            arr.unsqueeze(0),
            ref_steps,
            order=order,
            preserve_endpoint_duration=True,
        ).squeeze(0)
'''


def patch_interp_text(text: str) -> tuple[str, bool]:
    """Use exact endpoint duration for references without resizing MPC noise."""
    if INTERP_SENTINEL in text:
        return text, False
    if LEGACY_INTERP_SENTINEL in text:
        if text.count(_INTERP_LEGACY_EXACT) != 1:
            raise ValueError("legacy exact-duration interp header does not match")
        if text.count("    dst_len = (H - 1) * n + 1\n") != 1:
            raise ValueError("legacy exact-duration output length does not match")
        patched = text.replace(_INTERP_LEGACY_EXACT, _INTERP_REFERENCE_ONLY, 1)
        patched = patched.replace(
            "    dst_len = (H - 1) * n + 1\n",
            "    dst_len = (H - 1) * n + 1 if preserve_endpoint_duration else H * n\n",
            1,
        )
    else:
        if text.count(_INTERP_OLD) != 1:
            raise ValueError("expected unique Do-as-I-Do interp function header not found")
        if text.count("    dst_len = H * n\n") != 1:
            raise ValueError("expected Do-as-I-Do interpolation output-length expression not found")
        patched = text.replace(_INTERP_OLD, _INTERP_REFERENCE_ONLY, 1)
        patched = patched.replace(
            "    dst_len = H * n\n",
            "    dst_len = (H - 1) * n + 1 if preserve_endpoint_duration else H * n\n",
            1,
        )
    if patched.count(_ALIGN_REFERENCE_OLD) != 1:
        raise ValueError("expected unique Do-as-I-Do reference alignment call not found")
    patched = patched.replace(_ALIGN_REFERENCE_OLD, _ALIGN_REFERENCE_NEW, 1)
    return patched, True


def patch_launch_text(text: str) -> tuple[str, bool]:
    """Make the official launcher consume the wrapper's exact timing values."""
    if LAUNCH_SENTINEL in text:
        return text, False
    anchor = "    env_map = {\n"
    if text.count(anchor) != 1:
        raise ValueError("expected unique Do-as-I-Do launch env_map not found")
    replacement = f'''    env_map = {{
        # {LAUNCH_SENTINEL}: production timing is derived from the source FPS
        # by the AoE wrapper and must reach Config rather than remain log-only.
        "DAI_MJWP_SIM_DT": ("sim_dt", float),
        "DAI_MJWP_REF_DT": ("ref_dt", float),
        "DAI_MJWP_RENDER_DT": ("render_dt", float),
        "DAI_MJWP_TRACE_DT": ("trace_dt", float),
'''
    patched = text.replace(anchor, replacement, 1)

    solve_anchor = '''        force=cfg.force,
        smoothing=cfg.smoothing,
    )

    # Load the MJWP Config'''
    solve_replacement = '''        force=cfg.force,
        smoothing=cfg.smoothing,
        sim_dt=float(os.environ.get("DAI_MJWP_SIM_DT", "0.005")),
        ref_dt=float(os.environ.get("DAI_MJWP_REF_DT", "0.02")),
    )

    # Load the MJWP Config'''
    if patched.count(solve_anchor) != 1:
        raise ValueError("expected unique Do-as-I-Do solve_ik timing call not found")
    patched = patched.replace(solve_anchor, solve_replacement, 1)

    config_anchor = '''    contact_guidance_enabled = bool(getattr(config, "contact_guidance", False))
'''
    config_replacement = '''    # Fail closed if wrapper timing was exported but did not reach Config.
    for _env_name, _field_name in (
        ("DAI_MJWP_SIM_DT", "sim_dt"),
        ("DAI_MJWP_REF_DT", "ref_dt"),
        ("DAI_MJWP_RENDER_DT", "render_dt"),
        ("DAI_MJWP_TRACE_DT", "trace_dt"),
    ):
        _raw_timing = os.environ.get(_env_name)
        if not _raw_timing:
            continue
        _expected_timing = float(_raw_timing)
        _actual_timing = float(getattr(config, _field_name))
        if abs(_actual_timing - _expected_timing) > max(
            1.0e-12, abs(_expected_timing) * 1.0e-9
        ):
            raise RuntimeError(
                f"AOE effective timing mismatch: {_field_name}="
                f"{_actual_timing} expected={_expected_timing} from {_env_name}"
            )
    contact_guidance_enabled = bool(getattr(config, "contact_guidance", False))
'''
    if patched.count(config_anchor) != 1:
        raise ValueError("expected unique Do-as-I-Do effective Config anchor not found")
    patched = patched.replace(config_anchor, config_replacement, 1)
    return patched, True


_WARMUP_RUNTIME_HELPERS = '''# AOE_PRESERVE_REFERENCE_GRASP_RELEASE_GATE: production warmup starts from
# the exact closed reference grasp and may release a weld only after a real
# selected-world hand/object touching contact has been verified.
def _aoe_bool_env(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return bool(default)
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag; got {raw!r}")


def _aoe_restore_reference_grasp_state(config, env, qpos_0, qvel_0, ctrl_0):
    """Undo setup_env's seed mj_step and restore frame-0 state in every world."""
    num_worlds = int(env.num_worlds)
    qpos = qpos_0.to(config.device).float().unsqueeze(0).expand(
        num_worlds, -1
    ).contiguous()
    qvel = qvel_0.to(config.device).float().unsqueeze(0).expand(
        num_worlds, -1
    ).contiguous()
    ctrl = ctrl_0.to(config.device).float().unsqueeze(0).expand(
        num_worlds, -1
    ).contiguous()
    wp.copy(env.data_wp.qpos, wp.from_torch(qpos))
    wp.copy(env.data_wp.qvel, wp.from_torch(qvel))
    wp.copy(env.data_wp.ctrl, wp.from_torch(ctrl))
    zero_time = torch.zeros(num_worlds, device=config.device, dtype=torch.float32)
    wp.copy(env.data_wp.time, wp.from_torch(zero_time))
    if hasattr(env.data_wp, "qacc"):
        zero_qacc = torch.zeros(
            (num_worlds, env.model_cpu.nv),
            device=config.device,
            dtype=torch.float32,
        )
        wp.copy(env.data_wp.qacc, wp.from_torch(zero_qacc))
    if hasattr(env.data_wp, "qacc_warmstart"):
        zero_warmstart = torch.zeros(
            (num_worlds, env.model_cpu.nv),
            device=config.device,
            dtype=torch.float32,
        )
        wp.copy(
            env.data_wp.qacc_warmstart,
            wp.from_torch(zero_warmstart),
        )
    with wp.ScopedDevice(env.device):
        mjwarp.kinematics(env.model_wp, env.data_wp)
    wp.synchronize()
    loguru.logger.info(
        "AOE production warmup restored exact qpos_ref[0] closed grasp "
        "in {} worlds.",
        num_worlds,
    )


def _aoe_release_grasp_preflight(
    config,
    mj_model,
    mj_data,
    exact_contact_mask,
    release_step,
):
    """Fail before release when frame-0 expects a grasp but world 0 has none."""
    gate_enabled = _aoe_bool_env("DAI_MJWP_RELEASE_GRASP_GATE", True)
    mask_values = (
        exact_contact_mask.detach().cpu().numpy().astype(np.float64).reshape(-1)
    )
    active_indices = np.flatnonzero(mask_values > 0.5).astype(int).tolist()
    reference_in_hand = bool(active_indices)

    # mj_data is the selected outer execution world (world 0), not an MPC
    # sample.  Recompute its contacts so an optimizer rollout cannot leave a
    # stale MJWarp contact buffer that accidentally passes the release gate.
    mujoco.mj_forward(mj_model, mj_data)
    hand_mask_t = getattr(config, "_is_hand_geom", None)
    object_mask_t = getattr(config, "_is_object_geom", None)
    classification_error = None
    if hand_mask_t is None or object_mask_t is None:
        hand_mask = object_mask = None
        classification_error = "hand/object geom masks are unavailable"
    else:
        hand_mask = hand_mask_t.detach().cpu().numpy().astype(bool)
        object_mask = object_mask_t.detach().cpu().numpy().astype(bool)

    candidates = []
    touching = []
    if hand_mask is not None and object_mask is not None:
        for contact_index in range(int(mj_data.ncon)):
            contact_item = mj_data.contact[contact_index]
            geom1 = int(contact_item.geom1)
            geom2 = int(contact_item.geom2)
            is_hand_object = bool(
                (hand_mask[geom1] and object_mask[geom2])
                or (object_mask[geom1] and hand_mask[geom2])
            )
            if not is_hand_object:
                continue
            distance = float(contact_item.dist)
            record = {
                "contact_index": contact_index,
                "geom1": geom1,
                "geom2": geom2,
                "geom1_name": mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom1
                ),
                "geom2_name": mujoco.mj_id2name(
                    mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom2
                ),
                "distance_m": distance,
            }
            candidates.append(record)
            if np.isfinite(distance) and distance <= 0.0:
                touching.append(record)

    hand_object_touching = bool(touching)
    if not gate_enabled:
        status = "diagnostic_gate_disabled"
    elif not reference_in_hand:
        status = "not_required"
    elif classification_error is not None or not hand_object_touching:
        status = "invalid"
    else:
        status = "pass"
    production_eligible = bool(
        gate_enabled and status in {"pass", "not_required"}
    )
    finite_candidate_distances = [
        item["distance_m"]
        for item in candidates
        if np.isfinite(item["distance_m"])
    ]
    report = {
        "schema_version": 1,
        "sentinel": "AOE_PRESERVE_REFERENCE_GRASP_RELEASE_GATE",
        "status": status,
        "production_eligible": production_eligible,
        "release_grasp_gate": {
            "environment_variable": "DAI_MJWP_RELEASE_GRASP_GATE",
            "enabled": bool(gate_enabled),
        },
        "release_step": int(release_step),
        "exact_reference_contact_mask": {
            "values": mask_values.tolist(),
            "active_indices": active_indices,
            "in_hand": reference_in_hand,
        },
        "selected_world": {
            "index": 0,
            "source": "outer_execution_world_0_cpu_mujoco_forward",
            "classification_error": classification_error,
            "hand_object_candidate_contact_count": len(candidates),
            "hand_object_touching_contact_count": len(touching),
            "hand_object_touching": hand_object_touching,
            "minimum_candidate_distance_m": (
                min(finite_candidate_distances)
                if finite_candidate_distances
                else None
            ),
            "touching_contacts": touching[:64],
        },
    }
    report_path_raw = os.environ.get(
        "DAI_MJWP_RELEASE_GRASP_PREFLIGHT_PATH", ""
    ).strip()
    report_path = (
        Path(report_path_raw)
        if report_path_raw
        else Path(config.output_dir) / "release_grasp_preflight.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\\n",
        encoding="utf-8",
    )
    loguru.logger.info(
        "AOE release grasp preflight: status={} reference_in_hand={} "
        "selected_world_touching={} report={}",
        status,
        reference_in_hand,
        hand_object_touching,
        report_path,
    )
    if gate_enabled and reference_in_hand and not production_eligible:
        raise RuntimeError(
            "AOE release grasp preflight failed: exact reference contact mask "
            "requires an in-hand grasp but selected execution world 0 has no "
            f"real hand-object touching contact; see {report_path}"
        )
    return report
'''

_WARMUP_INIT_START = '''    # For warmup, start the simulation with an open hand (finger joints = 0)
'''
_WARMUP_INIT_END = '''    env = setup_env(config, init_ref_data)
'''

_PRESERVE_REFERENCE_WARMUP = '''    # AOE_PRESERVE_REFERENCE_GRASP_RELEASE_GATE: production preserves the
    # exact closed hand and wrist from qpos_ref[0].  The legacy open-hand /
    # analytical back-off path is diagnostic-only and requires two explicit,
    # mutually consistent environment flags.
    _aoe_preserve_reference_grasp = _aoe_bool_env(
        "DAI_MJWP_PRESERVE_REFERENCE_GRASP", True
    )
    _aoe_diagnostic_open_hand = _aoe_bool_env(
        "DAI_MJWP_DIAGNOSTIC_OPEN_HAND_ANALYTICAL_INIT", False
    )
    if _aoe_diagnostic_open_hand == _aoe_preserve_reference_grasp:
        raise RuntimeError(
            "Exactly one warmup initialization policy must be selected: "
            "production uses DAI_MJWP_PRESERVE_REFERENCE_GRASP=1 and "
            "DAI_MJWP_DIAGNOSTIC_OPEN_HAND_ANALYTICAL_INIT=0; the legacy "
            "diagnostic requires the inverse values."
        )

    finger_qpos_indices: list[int] = []
    robot_nq = qpos_ref.shape[1] - config.nq_obj
    wrist_dof = 6
    if config.embodiment_type == "bimanual":
        half = robot_nq // 2
        finger_qpos_indices += list(range(wrist_dof, half))
        finger_qpos_indices += list(range(half + wrist_dof, robot_nq))
    elif config.embodiment_type in ("right", "left"):
        finger_qpos_indices += list(range(wrist_dof, robot_nq))

    _warmup_init_enabled = bool(
        _aoe_diagnostic_open_hand
        and config.warmup_steps > 0
        and config.embodiment_type in ("bimanual", "right", "left")
    )
    if _warmup_init_enabled:
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
        loguru.logger.warning(
            "Legacy open-hand analytical warmup explicitly enabled; this run "
            "is diagnostic-only and production-ineligible ({} finger joints "
            "zeroed).",
            len(finger_qpos_indices),
        )
    else:
        init_ref_data = ref_data
        loguru.logger.info(
            "Production warmup preserves qpos_ref[0] closed fingers and wrist; "
            "legacy analytical back-off is disabled."
        )

    env = setup_env(config, init_ref_data)
'''


def _patch_exact_control_chunk_text(text: str) -> tuple[str, bool]:
    """Stop on the exact requested step, including a partial final chunk."""
    if OPTIMIZE_SENTINEL in text:
        required = (
            "execute_steps = int(config.ctrl_steps)",
            "for i in range(execute_steps):",
            "prev_ctrl = ctrls[execute_steps:]",
            'if k in {"qpos", "qvel", "time", "ctrl"}:',
        )
        missing = [token for token in required if token not in text]
        if missing:
            raise ValueError(
                "corrupt/incomplete exact-control-chunk patch: "
                + ",".join(missing)
            )
        return text, False
    loop_matches = list(
        re.finditer(
            r"(?m)^(?P<indent>[ \t]+)for i in range\(config\.ctrl_steps\):\n",
            text,
        )
    )
    if len(loop_matches) != 1:
        raise ValueError("expected unique Do-as-I-Do control execution loop not found")
    loop_anchor = loop_matches[0].group(0)
    indent = loop_matches[0].group("indent")
    loop_replacement = (
        f"{indent}# {OPTIMIZE_SENTINEL}: max_sim_steps is an exact sample\n"
        f"{indent}# boundary, not a lower bound rounded up to the next ctrl_dt chunk.\n"
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
    patched = text.replace(loop_anchor, loop_replacement, 1)
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
        raise ValueError("expected unique Do-as-I-Do receding-horizon shift not found")
    patched = patched.replace(shift_anchor, shift_replacement, 1)

    save_anchor = '''        for k in info_list[0].keys():
            info_aggregated[k] = np.stack([info[k] for info in info_list], axis=0)
'''
    save_replacement = '''        for k in info_list[0].keys():
            values = [info[k] for info in info_list]
            if k in {"qpos", "qvel", "time", "ctrl"}:
                # The exact final chunk can be shorter than ctrl_steps.
                info_aggregated[k] = np.concatenate(values, axis=0)
            else:
                info_aggregated[k] = np.stack(values, axis=0)
'''
    if patched.count(save_anchor) != 1:
        raise ValueError("expected unique Do-as-I-Do info aggregation block not found")
    return patched.replace(save_anchor, save_replacement, 1), True


def _patch_warmup_release_text(text: str) -> tuple[str, bool]:
    """Preserve the reference grasp and gate the physical weld release."""
    if WARMUP_RELEASE_SENTINEL in text:
        required = (
            "def _aoe_restore_reference_grasp_state(",
            "def _aoe_release_grasp_preflight(",
            '"DAI_MJWP_PRESERVE_REFERENCE_GRASP", True',
            '"DAI_MJWP_DIAGNOSTIC_OPEN_HAND_ANALYTICAL_INIT", False',
            '"DAI_MJWP_RELEASE_GRASP_GATE", True',
            "_aoe_release_grasp_preflight(\n",
            "_aoe_restore_reference_grasp_state(\n",
            "if _warmup_init_enabled and (",
        )
        missing = [token for token in required if token not in text]
        if missing:
            raise ValueError(
                "corrupt/incomplete reference-grasp release patch: "
                + ",".join(missing)
            )
        return text, False

    if "import json\n" not in text:
        import_anchor = "import os\n"
        if text.count(import_anchor) != 1:
            raise ValueError("expected unique optimize_physics os import not found")
        text = text.replace(import_anchor, "import json\nimport os\n", 1)
    if "import mujoco_warp as mjwarp\n" not in text:
        import_anchor = "import mujoco\n"
        if text.count(import_anchor) != 1:
            raise ValueError("expected unique optimize_physics mujoco import not found")
        text = text.replace(
            import_anchor,
            import_anchor + "import mujoco_warp as mjwarp\n",
            1,
        )

    main_matches = list(
        re.finditer(r"(?m)^def main\(config(?:: Config)?\):\n", text)
    )
    if len(main_matches) != 1:
        raise ValueError("expected unique optimize_physics main function not found")
    main_start = main_matches[0].start()
    text = text[:main_start] + _WARMUP_RUNTIME_HELPERS + "\n\n" + text[main_start:]

    warmup_start = text.find(_WARMUP_INIT_START)
    warmup_end = text.find(_WARMUP_INIT_END, warmup_start)
    if warmup_start < 0 or warmup_end < 0:
        raise ValueError("expected legacy open-hand warmup initialization block not found")
    warmup_end += len(_WARMUP_INIT_END)
    text = text[:warmup_start] + _PRESERVE_REFERENCE_WARMUP + text[warmup_end:]

    env_setup_anchor = '''    precompute_hand_object_geom_mask(config, env.model_cpu)

    # GPU warmup: analytical init pose'''
    env_setup_replacement = '''    precompute_hand_object_geom_mask(config, env.model_cpu)
    if _aoe_preserve_reference_grasp and config.warmup_steps > 0:
        _aoe_restore_reference_grasp_state(
            config,
            env,
            qpos_ref[0],
            qvel_ref[0],
            ctrl_ref[0],
        )

    # GPU warmup: analytical init pose'''
    if text.count(env_setup_anchor) != 1:
        raise ValueError("expected unique warmup env-setup anchor not found")
    text = text.replace(env_setup_anchor, env_setup_replacement, 1)

    interp_anchor = (
        "    if (base_interp_n > 0 or finger_interp_n > 0) "
        "and config.warmup_steps > 0:\n"
    )
    interp_replacement = (
        "    if _warmup_init_enabled and (\n"
        "        base_interp_n > 0 or finger_interp_n > 0\n"
        "    ) and config.warmup_steps > 0:\n"
    )
    if text.count(interp_anchor) != 1:
        raise ValueError("expected unique diagnostic warmup interpolation anchor not found")
    text = text.replace(interp_anchor, interp_replacement, 1)

    release_pattern = re.compile(
        r"(?m)^(?P<indent>[ \t]+)if warmup_enabled and abs_step == "
        r"config\.warmup_steps:\n"
        r"(?P=indent)    set_weld_active\(config, env, False\)\n"
        r"(?P=indent)    set_gravity\(env, list\(config\.default_gravity\)\)\n"
    )
    release_matches = list(release_pattern.finditer(text))
    if len(release_matches) != 1:
        raise ValueError("expected unique warmup weld-release anchor not found")
    release_indent = release_matches[0].group("indent")
    release_replacement = (
        f"{release_indent}if warmup_enabled and abs_step == config.warmup_steps:\n"
        f"{release_indent}    _aoe_release_grasp_preflight(\n"
        f"{release_indent}        config,\n"
        f"{release_indent}        mj_model,\n"
        f"{release_indent}        mj_data,\n"
        f"{release_indent}        contact[abs_step],\n"
        f"{release_indent}        abs_step,\n"
        f"{release_indent}    )\n"
        f"{release_indent}    set_weld_active(config, env, False)\n"
        f"{release_indent}    set_gravity(env, list(config.default_gravity))\n"
    )
    text = release_pattern.sub(release_replacement, text, count=1)
    return text, True


def patch_optimize_physics_text(text: str) -> tuple[str, bool]:
    """Install exact-timeline plus production-safe warmup/release behavior."""
    text, exact_changed = _patch_exact_control_chunk_text(text)
    text, warmup_changed = _patch_warmup_release_text(text)
    return text, bool(exact_changed or warmup_changed)


def _backup_and_write(path: Path, original: str, patched: str) -> None:
    backup = path.with_suffix(path.suffix + ".aoe_hoi_pipeline.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-dataset-py", type=Path, required=True)
    parser.add_argument("--solve-ik-py", type=Path, required=True)
    parser.add_argument("--interp-py", type=Path, required=True)
    parser.add_argument("--launch-py", type=Path, required=True)
    parser.add_argument("--optimize-physics-py", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = (
        ("process_dataset", args.process_dataset_py, patch_process_dataset_text),
        ("solve_ik", args.solve_ik_py, patch_solve_ik_text),
        ("interp", args.interp_py, patch_interp_text),
        ("launch", args.launch_py, patch_launch_text),
        ("optimize_physics", args.optimize_physics_py, patch_optimize_physics_text),
    )
    results = []
    prepared = []
    # Validate every source before writing any source.
    for name, path, patcher in targets:
        original = path.read_text(encoding="utf-8")
        patched, changed = patcher(original)
        prepared.append((path, original, patched, changed))
        results.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "changed": changed,
                "sentinel": {
                    "process_dataset": PROCESS_SENTINEL,
                    "solve_ik": SOLVE_IK_SENTINEL,
                    "interp": INTERP_SENTINEL,
                    "launch": LAUNCH_SENTINEL,
                    "optimize_physics": OPTIMIZE_SENTINEL,
                }[name],
            }
        )

    if not args.dry_run:
        for path, original, patched, changed in prepared:
            if changed:
                _backup_and_write(path, original, patched)

    report = {
        "schema_version": 3,
        "dry_run": bool(args.dry_run),
        "diagnostic_object_only_freeze_env": FREEZE_ENV,
        "process_dataset_contact_patch_version": 2,
        "process_dataset_contact_patch_sentinel": SURFACE_CONTACT_SENTINEL,
        "optimize_warmup_release_patch_version": 1,
        "optimize_warmup_release_patch_sentinel": WARMUP_RELEASE_SENTINEL,
        "preserve_reference_grasp_env": PRESERVE_REFERENCE_GRASP_ENV,
        "diagnostic_open_hand_env": DIAGNOSTIC_OPEN_HAND_ENV,
        "release_grasp_gate_env": RELEASE_GRASP_GATE_ENV,
        "release_grasp_report_env": RELEASE_GRASP_REPORT_ENV,
        "targets": results,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
