#!/usr/bin/env python3
"""Validate exact-route raw adapter HOI against DAI processed keypoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FINGERTIP_INDICES = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(flat.size),
        "median": float(np.median(flat)) if flat.size else 0.0,
        "p95": float(np.percentile(flat, 95)) if flat.size else 0.0,
        "max": float(np.max(flat)) if flat.size else 0.0,
    }


def _quat_wxyz_to_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    """Convert finite, unit WXYZ quaternions to rotation matrices."""

    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape[-1:] != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError(f"{name} must contain finite WXYZ quaternions")
    norms = np.linalg.norm(quaternion, axis=-1)
    if np.any(norms <= 1.0e-12) or not np.allclose(
        norms, 1.0, atol=1.0e-6, rtol=1.0e-6
    ):
        raise ValueError(f"{name} contains a non-unit WXYZ quaternion")
    q = quaternion / norms[..., None]
    w, x, y, z = np.moveaxis(q, -1, 0)
    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def _matrix_to_quat_wxyz(value: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to unit WXYZ quaternions."""

    matrices = np.asarray(value, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3) or not np.all(np.isfinite(matrices)):
        raise ValueError("rotation matrices must be finite and end in (3, 3)")
    flat = matrices.reshape(-1, 3, 3)
    output = np.empty((len(flat), 4), dtype=np.float64)
    for index, matrix in enumerate(flat):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(trace + 1.0)
            quat = np.array(
                [
                    0.25 * scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
        else:
            axis = int(np.argmax(np.diag(matrix)))
            if axis == 0:
                scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
                quat = np.array(
                    [
                        (matrix[2, 1] - matrix[1, 2]) / scale,
                        0.25 * scale,
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                    ]
                )
            elif axis == 1:
                scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
                quat = np.array(
                    [
                        (matrix[0, 2] - matrix[2, 0]) / scale,
                        (matrix[0, 1] + matrix[1, 0]) / scale,
                        0.25 * scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                    ]
                )
            else:
                scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
                quat = np.array(
                    [
                        (matrix[1, 0] - matrix[0, 1]) / scale,
                        (matrix[0, 2] + matrix[2, 0]) / scale,
                        (matrix[1, 2] + matrix[2, 1]) / scale,
                        0.25 * scale,
                    ]
                )
        output[index] = quat / np.linalg.norm(quat)
    return output.reshape(matrices.shape[:-2] + (4,))


def _rotvec_to_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    rotvec = np.asarray(value, dtype=np.float64)
    if rotvec.shape[-1:] != (3,) or not np.all(np.isfinite(rotvec)):
        raise ValueError(f"{name} must contain finite rotation vectors")
    flat = rotvec.reshape(-1, 3)
    result = np.empty((len(flat), 3, 3), dtype=np.float64)
    for index, vector in enumerate(flat):
        theta = float(np.linalg.norm(vector))
        skew = np.array(
            [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]]
        )
        if theta < 1.0e-8:
            a = 1.0 - theta * theta / 6.0
            b = 0.5 - theta * theta / 24.0
        else:
            a = np.sin(theta) / theta
            b = (1.0 - np.cos(theta)) / (theta * theta)
        result[index] = np.eye(3) + a * skew + b * (skew @ skew)
    return result.reshape(rotvec.shape[:-1] + (3, 3))


def _rotation_error_rad(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(np.asarray(expected), -1, -2) @ np.asarray(actual)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def _wrist_rotations(joints: np.ndarray, rotvec: np.ndarray, side: str) -> np.ndarray:
    """Reproduce DAI's MANO-to-MCP wrist-frame convention before world alignment."""

    joints = np.asarray(joints, dtype=np.float64)

    def normalized(vector: np.ndarray, label: str) -> np.ndarray:
        norms = np.linalg.norm(vector, axis=-1, keepdims=True)
        if np.any(norms <= 1.0e-10) or not np.all(np.isfinite(norms)):
            raise ValueError(f"degenerate {side} wrist frame vector: {label}")
        return vector / norms

    z_axis = normalized(joints[:, 9] - joints[:, 0], "middle_mcp")
    y_source = joints[:, 5] - joints[:, 13] if side == "right" else joints[:, 13] - joints[:, 5]
    y_aux = normalized(y_source, "index_ring_mcp")
    x_axis = normalized(np.cross(y_aux, z_axis), "palm_normal")
    y_axis = normalized(np.cross(z_axis, x_axis), "orthogonal_y")
    geometric = np.stack([x_axis, y_axis, z_axis], axis=-1)
    mano = _rotvec_to_matrix(rotvec, name=f"raw {side}_rot")
    offsets = np.swapaxes(mano, -1, -2) @ geometric
    offset_quat = _matrix_to_quat_wxyz(offsets)
    signs = np.where(offset_quat @ offset_quat[0] < 0.0, -1.0, 1.0)
    aligned = offset_quat * signs[:, None]
    _, eigenvectors = np.linalg.eigh(aligned.T @ aligned)
    mean_offset = _quat_wxyz_to_matrix(
        eigenvectors[:, -1], name=f"raw {side} fixed wrist offset"
    )
    return mano @ mean_offset


def _layout_poses(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"layout has no ordered objects list: {path}")
    positions = []
    quaternions = []
    for index, entry in enumerate(entries):
        local = entry.get("local_to_scene") if isinstance(entry, dict) else None
        if not isinstance(local, dict):
            raise ValueError(f"layout object {index} has no local_to_scene")
        value = local.get("translation_camera_frame")
        if value is None:
            value = local.get("translation")
        position = np.asarray(value, dtype=np.float64).reshape(-1)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"invalid layout object position at frame {index}")
        quaternion = local.get("quat_wxyz_camera_frame")
        if quaternion is None and local.get("translation_camera_frame") is None:
            quaternion = local.get("new_quat", local.get("quat_wxyz"))
        if quaternion is None:
            raise ValueError(f"layout object {index} has no matching WXYZ quaternion")
        positions.append(position)
        quaternions.append(np.asarray(quaternion, dtype=np.float64))
    quaternion_array = np.asarray(quaternions, dtype=np.float64)
    return np.asarray(positions, dtype=np.float64), _quat_wxyz_to_matrix(
        quaternion_array, name="raw layout"
    )


def _resolve_layout(adapter: dict, raw_dir: Path) -> Path:
    raw_root = raw_dir.expanduser().resolve()
    config_candidate = None
    config_path = raw_root / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            object_names = config.get("object_names") if isinstance(config, dict) else None
            if isinstance(object_names, list) and object_names:
                config_candidate = (
                    raw_root
                    / "obj_tracking_out"
                    / str(object_names[0])
                    / "combined_visualization"
                    / "layout_camera_frame_optimized.json"
                )
        except (OSError, UnicodeError, json.JSONDecodeError):
            config_candidate = None
    candidates = (
        [config_candidate]
        if config_candidate is not None and config_candidate.is_file()
        else sorted(
            raw_root.glob(
                "obj_tracking_out/*/combined_visualization/"
                "layout_camera_frame_optimized.json"
            )
        )
    )
    recorded = adapter.get("layout")
    if recorded:
        path = Path(str(recorded)).expanduser()
        if not path.is_absolute():
            path = raw_root / path
        resolved = path.resolve()
        try:
            resolved.relative_to(raw_root)
        except ValueError as exc:
            raise ValueError(
                "adapter layout is outside the exact-route raw directory: "
                f"layout={resolved} raw_dir={raw_root}"
            ) from exc
        if not resolved.is_file():
            raise ValueError(f"adapter layout does not exist: {resolved}")
        candidate_resolved = {candidate.resolve() for candidate in candidates}
        if candidate_resolved and resolved not in candidate_resolved:
            raise ValueError(
                "adapter layout is not the exact config-selected raw-dir layout: "
                f"layout={resolved} candidates={sorted(str(item) for item in candidate_resolved)}"
            )
        return resolved
    if len(candidates) != 1:
        raise ValueError(
            f"expected one exact-route optimized layout under {raw_root}; found {len(candidates)}"
        )
    return candidates[0].resolve()


def _fit_common_se3(before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.asarray(before, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(after, dtype=np.float64).reshape(-1, 3)
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError(f"cannot fit common SE(3): before={source.shape} after={target.shape}")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_mean - rotation @ source_mean
    predicted = source @ rotation.T + translation
    return rotation, translation, np.linalg.norm(predicted - target, axis=-1)


def _file_binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def evaluate(
    *,
    adapter_manifest: Path,
    processed_keypoints: Path,
    trajectory_6dof: str,
    hand_type: str,
    relative_max_threshold_m: float = 1.0e-7,
    common_se3_max_threshold_m: float = 1.0e-5,
    rotation_max_threshold_rad: float = 1.0e-6,
    object_track_max_threshold: float = 1.0e-7,
) -> dict:
    adapter = json.loads(adapter_manifest.read_text(encoding="utf-8"))
    raw_dir = adapter_manifest.parent
    raw_hand = raw_dir / "raw" / "all_hand_meshes.npz"
    if not raw_hand.is_file():
        raw_hand = raw_dir / "all_hand_meshes.npz"
    if not raw_hand.is_file():
        raise FileNotFoundError(f"missing exact-route raw hand NPZ below {raw_dir}")
    layout = _resolve_layout(adapter, raw_dir)
    raw_object, raw_object_rotation = _layout_poses(layout)

    requested_sides = (
        ("left", "right") if hand_type == "bimanual" else (hand_type,)
    )
    if any(side not in {"left", "right"} for side in requested_sides):
        raise ValueError(f"hand_type must be left, right, or bimanual; got {hand_type}")

    side_reports = {}
    side_points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    processed_object_tracks: dict[str, np.ndarray] = {}
    processed_object_rotations: dict[str, np.ndarray] = {}
    raw_wrist_rotations: dict[str, np.ndarray] = {}
    processed_wrist_rotations: dict[str, np.ndarray] = {}
    aggregate_relative = []
    aggregate_relative_translation = []
    aggregate_relative_rotation = []
    with np.load(raw_hand, allow_pickle=False) as raw, np.load(
        processed_keypoints, allow_pickle=False
    ) as processed:
        for side in requested_sides:
            raw_key = f"{side}_joints"
            raw_rot_key = f"{side}_rot"
            wrist_key = f"qpos_wrist_{side}"
            finger_key = f"qpos_finger_{side}"
            object_key = f"qpos_obj_{side}"
            for key, container in (
                (raw_key, raw),
                (raw_rot_key, raw),
                (wrist_key, processed),
                (finger_key, processed),
                (object_key, processed),
            ):
                if key not in container:
                    raise KeyError(f"missing {key} for exact hand source comparison")
            joints = np.asarray(raw[raw_key], dtype=np.float64)
            raw_rotvec = np.asarray(raw[raw_rot_key], dtype=np.float64)
            wrist_track = np.asarray(processed[wrist_key], dtype=np.float64)
            finger_track = np.asarray(processed[finger_key], dtype=np.float64)
            wrist_after = wrist_track[..., :3]
            fingers_after = finger_track[..., :3]
            object_track = np.asarray(processed[object_key], dtype=np.float64)
            object_after = object_track[..., :3]
            frames = raw_object.shape[0]
            if (
                joints.shape != (frames, 21, 3)
                or raw_rotvec.shape != (frames, 3)
                or wrist_track.shape != (frames, 7)
                or wrist_after.shape != (frames, 3)
                or finger_track.shape != (frames, 5, 7)
                or fingers_after.shape != (frames, 5, 3)
                or object_track.ndim != 2
                or object_track.shape[0] != frames
                or object_track.shape[1] < 7
                or object_after.shape != (frames, 3)
            ):
                raise ValueError(
                    f"frame/shape mismatch for {side}: raw_obj={raw_object.shape}, "
                    f"raw_joints={joints.shape}, wrist={wrist_after.shape}, "
                    f"fingers={fingers_after.shape}, object={object_after.shape}"
                )
            wrist_before = joints[:, 0]
            fingers_before = joints[:, FINGERTIP_INDICES]
            raw_wrist_rotation = _wrist_rotations(joints, raw_rotvec, side)
            wrist_rotation_after = _quat_wxyz_to_matrix(
                wrist_track[:, 3:7], name=f"processed {wrist_key}"
            )
            object_rotation_after = _quat_wxyz_to_matrix(
                object_track[:, 3:7], name=f"processed {object_key}"
            )
            # Fingertip qpos rotations are marker orientations rather than a
            # physical hand frame, but they still must be valid WXYZ qpos.
            _quat_wxyz_to_matrix(
                finger_track[..., 3:7], name=f"processed {finger_key}"
            )
            wrist_error = np.abs(
                np.linalg.norm(wrist_after - object_after, axis=-1)
                - np.linalg.norm(wrist_before - raw_object, axis=-1)
            )
            finger_error = np.abs(
                np.linalg.norm(fingers_after - object_after[:, None], axis=-1)
                - np.linalg.norm(fingers_before - raw_object[:, None], axis=-1)
            )
            before_points = np.concatenate(
                [raw_object[:, None], wrist_before[:, None], fingers_before], axis=1
            )
            after_points = np.concatenate(
                [object_after[:, None], wrist_after[:, None], fingers_after], axis=1
            )
            relative = np.concatenate([wrist_error.reshape(-1), finger_error.reshape(-1)])
            aggregate_relative.append(relative)
            before_relative_translation = np.einsum(
                "nij,nj->ni",
                np.swapaxes(raw_wrist_rotation, -1, -2),
                raw_object - wrist_before,
            )
            after_relative_translation = np.einsum(
                "nij,nj->ni",
                np.swapaxes(wrist_rotation_after, -1, -2),
                object_after - wrist_after,
            )
            relative_translation_error = np.linalg.norm(
                after_relative_translation - before_relative_translation, axis=-1
            )
            before_relative_rotation = (
                np.swapaxes(raw_wrist_rotation, -1, -2) @ raw_object_rotation
            )
            after_relative_rotation = (
                np.swapaxes(wrist_rotation_after, -1, -2) @ object_rotation_after
            )
            relative_rotation_error = _rotation_error_rad(
                before_relative_rotation, after_relative_rotation
            )
            aggregate_relative_translation.append(relative_translation_error)
            aggregate_relative_rotation.append(relative_rotation_error)
            side_points[side] = (before_points, after_points)
            processed_object_tracks[side] = object_track
            processed_object_rotations[side] = object_rotation_after
            raw_wrist_rotations[side] = raw_wrist_rotation
            processed_wrist_rotations[side] = wrist_rotation_after
            side_reports[side] = {
                "frames": frames,
                "raw_hand_key": raw_key,
                "processed_wrist_key": wrist_key,
                "processed_finger_key": finger_key,
                "processed_object_key": object_key,
                "wrist_object_distance_error_m": _stats(wrist_error),
                "fingertip_object_distance_error_m": _stats(finger_error),
                "wrist_object_relative_translation_error_m": _stats(
                    relative_translation_error
                ),
                "wrist_object_relative_rotation_error_rad": _stats(
                    relative_rotation_error
                ),
            }

    # There is one physical object.  A bimanual DAI keypoint file must not
    # encode two divergent copies of its trajectory.  Compare the complete
    # qpos representation (translation and quaternion), not only xyz.
    object_track_consistency: dict[str, object] = {
        "required": hand_type == "bimanual",
        "sides": list(requested_sides),
        "threshold": object_track_max_threshold,
    }
    object_track_error = np.zeros(0, dtype=np.float64)
    object_track_rotation_error = np.zeros(0, dtype=np.float64)
    if hand_type == "bimanual":
        left_track = processed_object_tracks["left"]
        right_track = processed_object_tracks["right"]
        if left_track.shape != right_track.shape:
            raise ValueError(
                "processed bimanual object track shape mismatch: "
                f"left={left_track.shape} right={right_track.shape}"
            )
        # WXYZ quaternions have a q == -q sign ambiguity.  Sign-align the
        # right-hand copy before component diagnostics so two identical physical
        # object tracks cannot be rejected solely because of serialization.
        left_quaternion = left_track[:, 3:7]
        right_quaternion = right_track[:, 3:7]
        quaternion_sign = np.where(
            np.sum(left_quaternion * right_quaternion, axis=-1) < 0.0,
            -1.0,
            1.0,
        )
        right_equivalent = right_track[:, :7].copy()
        right_equivalent[:, 3:7] *= quaternion_sign[:, None]
        object_track_error = np.abs(left_track[:, :7] - right_equivalent)
        object_track_consistency["absolute_component_error"] = _stats(
            object_track_error
        )
        object_track_consistency["position_error_m"] = _stats(
            np.linalg.norm(left_track[:, :3] - right_track[:, :3], axis=-1)
        )
        object_track_rotation_error = _rotation_error_rad(
            processed_object_rotations["left"],
            processed_object_rotations["right"],
        )
        object_track_consistency["rotation_error_rad"] = _stats(
            object_track_rotation_error
        )
    else:
        object_track_consistency["absolute_component_error"] = _stats(
            object_track_error
        )
        object_track_consistency["position_error_m"] = _stats(object_track_error)
        object_track_consistency["rotation_error_rad"] = _stats(
            object_track_rotation_error
        )

    # Fit exactly one transform to the physical object plus every requested
    # hand.  In bimanual mode the object is included once and both hands share
    # the same fit; separate per-side fits would miss independent hand shifts.
    reference_side = requested_sides[0]
    joint_before = [raw_object[:, None]]
    joint_after = [processed_object_tracks[reference_side][:, None, :3]]
    for side in requested_sides:
        before_points, after_points = side_points[side]
        joint_before.append(before_points[:, 1:])
        joint_after.append(after_points[:, 1:])
    rotation, translation, common_residual = _fit_common_se3(
        np.concatenate(joint_before, axis=1),
        np.concatenate(joint_after, axis=1),
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    for side in requested_sides:
        before_points, after_points = side_points[side]
        predicted = before_points @ rotation.T + translation
        side_reports[side]["joint_common_se3_position_residual_m"] = _stats(
            np.linalg.norm(predicted - after_points, axis=-1)
        )
        expected_object_rotation = rotation @ raw_object_rotation
        object_rotation_residual = _rotation_error_rad(
            expected_object_rotation, processed_object_rotations[side]
        )
        expected_wrist_rotation = rotation @ raw_wrist_rotations[side]
        wrist_rotation_residual = _rotation_error_rad(
            expected_wrist_rotation, processed_wrist_rotations[side]
        )
        side_reports[side]["object_common_se3_rotation_residual_rad"] = _stats(
            object_rotation_residual
        )
        side_reports[side]["wrist_common_se3_rotation_residual_rad"] = _stats(
            wrist_rotation_residual
        )

    relative_stats = _stats(np.concatenate(aggregate_relative))
    relative_translation_stats = _stats(
        np.concatenate(aggregate_relative_translation)
    )
    relative_rotation_stats = _stats(np.concatenate(aggregate_relative_rotation))
    common_stats = _stats(common_residual)
    common_rotation_values = np.concatenate(
        [
            np.asarray(side_reports[side][field]["max"], dtype=np.float64).reshape(1)
            for side in requested_sides
            for field in (
                "object_common_se3_rotation_residual_rad",
                "wrist_common_se3_rotation_residual_rad",
            )
        ]
    )
    common_rotation_stats = _stats(common_rotation_values)
    errors = []
    adapter_hand_source = adapter.get("hand_source")
    if not isinstance(trajectory_6dof, str) or not trajectory_6dof.strip():
        errors.append("missing_trajectory_6dof_route_binding")
    if not isinstance(adapter_hand_source, str) or not adapter_hand_source.strip():
        errors.append("missing_adapter_hand_source_route_binding")
    if relative_stats["max"] > relative_max_threshold_m:
        errors.append("raw_to_processed_relative_distance_max_exceeds_threshold")
    if relative_translation_stats["max"] > relative_max_threshold_m:
        errors.append("raw_to_processed_relative_pose_translation_exceeds_threshold")
    if relative_rotation_stats["max"] > rotation_max_threshold_rad:
        errors.append("raw_to_processed_relative_pose_rotation_exceeds_threshold")
    if common_stats["max"] > common_se3_max_threshold_m:
        errors.append("raw_to_processed_is_not_one_common_se3")
    if common_rotation_stats["max"] > rotation_max_threshold_rad:
        errors.append("raw_to_processed_rotations_are_not_one_common_se3")
    if object_track_error.size and float(np.max(object_track_error)) > object_track_max_threshold:
        errors.append("processed_bimanual_object_tracks_are_inconsistent")
    if (
        object_track_rotation_error.size
        and float(np.max(object_track_rotation_error)) > rotation_max_threshold_rad
    ):
        errors.append("processed_bimanual_object_rotations_are_inconsistent")

    adapter_binding = _file_binding(adapter_manifest)
    raw_hand_binding = _file_binding(raw_hand)
    raw_layout_binding = _file_binding(layout)
    processed_binding = _file_binding(processed_keypoints)
    route_binding = {
        "trajectory_6dof": trajectory_6dof,
        "hand_source": adapter_hand_source,
        "hand_type": hand_type,
        "requested_sides": list(requested_sides),
        "adapter_manifest": adapter_binding,
        "raw_inputs": {
            "hand_npz": raw_hand_binding,
            "object_layout": raw_layout_binding,
        },
        "processed_keypoints": processed_binding,
    }
    return {
        "schema_version": 3,
        "status": "ok" if not errors else "invalid",
        "production_eligible": not errors,
        "errors": errors,
        "comparison_semantics": "exact_route_exact_hand_raw_adapter_to_dai_processed",
        "full_relative_se3_checked": True,
        "cross_hand_source_comparison": False,
        "trajectory_6dof": trajectory_6dof,
        "hand_source": adapter_hand_source,
        "hand_type": hand_type,
        "route_binding": route_binding,
        "object_track_source": adapter.get("object_track_source"),
        "object_mesh_source": adapter.get("object_mesh_source"),
        "retarget_object_source": adapter.get("retarget_object_source"),
        "canonical_transform": adapter.get("canonical_transform"),
        "hoi_refinement": adapter.get("hoi_refinement"),
        "hoi_contact_alignment": adapter.get("hoi_contact_alignment"),
        "adapter_manifest": adapter_binding["path"],
        "adapter_manifest_sha256": adapter_binding["sha256"],
        "raw_hand_npz": str(raw_hand.resolve()),
        "raw_hand_sha256": raw_hand_binding["sha256"],
        "raw_object_layout": str(layout.resolve()),
        "raw_object_layout_sha256": raw_layout_binding["sha256"],
        "processed_keypoints": str(processed_keypoints.resolve()),
        "processed_keypoints_sha256": processed_binding["sha256"],
        "thresholds_m": {
            "relative_distance_max": relative_max_threshold_m,
            "relative_pose_translation_max": relative_max_threshold_m,
            "common_se3_position_residual_max": common_se3_max_threshold_m,
        },
        "thresholds_rad": {
            "relative_pose_rotation_max": rotation_max_threshold_rad,
            "common_se3_rotation_residual_max": rotation_max_threshold_rad,
        },
        "quaternion_convention": {
            "layout": "wxyz_camera_frame",
            "processed_qpos": "xyz_wxyz",
            "validation": "finite_unit_quaternion",
        },
        "processed_object_track_component_threshold": object_track_max_threshold,
        "relative_distance_error_m": relative_stats,
        "wrist_object_relative_translation_error_m": relative_translation_stats,
        "wrist_object_relative_rotation_error_rad": relative_rotation_stats,
        "common_se3_position_residual_m": common_stats,
        "common_se3_rotation_residual_rad": common_rotation_stats,
        "fitted_joint_common_se3": transform.tolist(),
        "joint_fit_components": ["object", *requested_sides],
        "processed_object_track_consistency": object_track_consistency,
        "sides": side_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--processed-keypoints", type=Path, required=True)
    parser.add_argument("--trajectory-6dof", required=True)
    parser.add_argument("--hand-type", choices=("left", "right", "bimanual"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-max-threshold-m", type=float, default=1.0e-7)
    parser.add_argument("--common-se3-max-threshold-m", type=float, default=1.0e-5)
    parser.add_argument("--rotation-max-threshold-rad", type=float, default=1.0e-6)
    args = parser.parse_args()
    report = evaluate(
        adapter_manifest=args.adapter_manifest,
        processed_keypoints=args.processed_keypoints,
        trajectory_6dof=args.trajectory_6dof,
        hand_type=args.hand_type,
        relative_max_threshold_m=args.relative_max_threshold_m,
        common_se3_max_threshold_m=args.common_se3_max_threshold_m,
        rotation_max_threshold_rad=args.rotation_max_threshold_rad,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 7


if __name__ == "__main__":
    raise SystemExit(main())
