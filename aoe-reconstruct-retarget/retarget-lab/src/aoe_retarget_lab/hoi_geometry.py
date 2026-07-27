"""Rigid-transform utilities and numerical invariants for HOI adapters.

The helpers in this module deliberately treat a hand-object interaction (HOI) as
one geometric unit.  A canonical transform is a change of world coordinates, so
it must act on the hand, object, contacts, and any world-space mesh vertices in
exactly the same way.  Moving only the object is an HOI refinement, not a change
of coordinates, and is detected by :func:`summarize_adapter_rigid_invariance`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULT_FINGERTIP_INDICES = (4, 8, 12, 16, 20)


def validate_rigid_transform(
    transform: Any,
    *,
    atol: float = 1e-6,
) -> np.ndarray:
    """Return *transform* as float64 after validating that it is in SE(3).

    Validation rejects non-finite values, projective bottom rows, scaling,
    shear, and reflections.  A copy is returned so later caller mutations do
    not alter the matrix that was validated.
    """

    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be a positive finite number")
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"rigid transform must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("rigid transform contains non-finite values")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=atol, rtol=0.0):
        raise ValueError("rigid transform bottom row must be [0, 0, 0, 1]")

    rotation = matrix[:3, :3]
    gram = rotation.T @ rotation
    if not np.allclose(gram, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError("rigid transform rotation must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=atol, rtol=0.0):
        raise ValueError(
            "rigid transform rotation must have determinant +1 "
            f"(got {determinant:.9g})"
        )
    return matrix.copy()


def _validate_pose_array(poses: Any, *, name: str, atol: float = 1e-6) -> np.ndarray:
    array = np.asarray(poses, dtype=np.float64)
    if array.shape[-2:] != (4, 4):
        raise ValueError(f"{name} must end in shape (4, 4), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(
        array[..., 3, :],
        np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(f"{name} contains a non-rigid homogeneous bottom row")

    rotations = array[..., :3, :3]
    gram = np.swapaxes(rotations, -1, -2) @ rotations
    if not np.allclose(gram, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError(f"{name} contains a non-orthonormal rotation")
    determinants = np.linalg.det(rotations)
    if not np.allclose(determinants, 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"{name} contains a rotation whose determinant is not +1")
    return array


def _transform_points(points: Any, transform: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must end in xyz coordinates, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.einsum("ij,...j->...i", transform[:3, :3], array) + transform[:3, 3]


def _transform_point_collection(
    value: Any,
    transform: np.ndarray,
    *,
    name: str,
) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {
            key: _transform_points(points, transform, name=f"{name}[{key!r}]")
            for key, points in value.items()
        }
    return _transform_points(value, transform, name=name)


def _transform_poses(poses: Any, transform: np.ndarray, *, name: str) -> np.ndarray:
    array = _validate_pose_array(poses, name=name)
    return np.matmul(transform, array)


def apply_common_se3(
    canonical_transform: Any,
    *,
    hand_joints: Any,
    hand_vertices: Any,
    object_poses: Any,
    contact_points: Any,
    mesh_vertices: Any,
) -> dict[str, Any]:
    """Apply one canonical SE(3) to all world-space HOI geometry.

    ``hand_joints`` and ``object_poses`` are mandatory because their relative
    geometry is the invariant this helper protects.  The other inputs may be
    ``None`` when that representation is absent.  Point inputs may be arrays
    ending in ``(..., 3)`` or mappings (for example ``{"left": ..., "right":
    ...}``).  Object poses must end in ``(..., 4, 4)`` and are left-multiplied.

    ``mesh_vertices`` means *world-space* vertices.  Object-local mesh vertices
    should remain local; transforming the object pose already moves that mesh.
    """

    transform = validate_rigid_transform(canonical_transform)
    if hand_joints is None:
        raise ValueError("hand_joints are required for a common HOI transform")
    if object_poses is None:
        raise ValueError("object_poses are required for a common HOI transform")

    return {
        "hand_joints": _transform_point_collection(
            hand_joints, transform, name="hand_joints"
        ),
        "hand_vertices": _transform_point_collection(
            hand_vertices, transform, name="hand_vertices"
        ),
        "object_poses": _transform_poses(
            object_poses, transform, name="object_poses"
        ),
        "contact_points": _transform_point_collection(
            contact_points, transform, name="contact_points"
        ),
        "mesh_vertices": _transform_point_collection(
            mesh_vertices, transform, name="mesh_vertices"
        ),
    }


def _load_hand_npz(hand_npz: Any) -> dict[str, np.ndarray]:
    if isinstance(hand_npz, (str, bytes, os.PathLike)):
        with np.load(hand_npz, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    if isinstance(hand_npz, np.lib.npyio.NpzFile):
        return {key: np.asarray(hand_npz[key]) for key in hand_npz.files}
    if isinstance(hand_npz, Mapping):
        return {str(key): np.asarray(value) for key, value in hand_npz.items()}
    raise ValueError("hand_npz must be a path, an open NpzFile, or a mapping")


def _pose_sequence(poses: Any, *, name: str) -> np.ndarray:
    array = _validate_pose_array(poses, name=name)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (frames, 4, 4), got {array.shape}")
    return array


def _distance_stats(values: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return {"count": 0, "median": None, "p95": None, "max": None}
    return {
        "count": int(flat.size),
        "median": float(np.median(flat)),
        "p95": float(np.percentile(flat, 95)),
        "max": float(np.max(flat)),
    }


def _empty_invariance_summary(
    *,
    errors: list[str],
    tolerance_m: float | None,
    canonical_transform: np.ndarray | None,
) -> dict[str, Any]:
    thresholds = (
        _invariance_thresholds(tolerance_m)
        if tolerance_m is not None
        else {"median": None, "p95": None, "max": None}
    )
    return {
        "status": "invalid",
        "errors": errors,
        "median": None,
        "p95": None,
        "max": None,
        "thresholds_m": thresholds,
        "tolerance_m": tolerance_m,
        "hand_coordinate_frame": "explicit_adapter_boundary_frames",
        "canonical_transform_applied": canonical_transform is not None,
        "canonical_transform": (
            canonical_transform.tolist() if canonical_transform is not None else None
        ),
        "true_pre_post_hand_comparison": True,
        "frames": 0,
        "samples": 0,
        "compared_sides": [],
        "wrist_object": _distance_stats(np.asarray([], dtype=np.float64)),
        "fingertips_object": _distance_stats(np.asarray([], dtype=np.float64)),
        "hand_canonical_position_error_m": _distance_stats(
            np.asarray([], dtype=np.float64)
        ),
        "hand_vertices_canonical_position_error_m": _distance_stats(
            np.asarray([], dtype=np.float64)
        ),
        "object_pose": {
            "expected_relation": "canonical_transform @ before_object_pose",
            "translation_error_m": _distance_stats(np.asarray([], dtype=np.float64)),
            "rotation_error_rad": _distance_stats(np.asarray([], dtype=np.float64)),
        },
        "sides": {},
    }


def _invariance_thresholds(tolerance_m: float) -> dict[str, float]:
    # At the default 1e-6 maximum tolerance, these are 1e-8 / 1e-7 / 1e-6.
    median = min(tolerance_m, max(1e-8, tolerance_m * 1e-2))
    p95 = min(tolerance_m, max(1e-8, tolerance_m * 1e-1))
    return {
        "median": float(median),
        "p95": float(p95),
        "max": float(tolerance_m),
    }


def summarize_adapter_rigid_invariance(
    before_hand_npz: Any,
    after_hand_npz: Any,
    before_object_poses: Any,
    after_object_poses: Any,
    canonical_transform: Any | None = None,
    tolerance_m: float = 1e-6,
) -> dict[str, Any]:
    """Summarize wrist/object and fingertip/object distance invariance.

    ``before_hand_npz`` and ``after_hand_npz`` are the actual hand geometry at
    the two adapter boundaries.  Both must contain matching ``left_joints``
    and/or ``right_joints`` in OpenPose-21 order, plus optional matching valid
    masks.  A supplied canonical transform must map the *actual* before hand
    and object pose to the actual after values.  No side is reconstructed by
    inverting the other side: that is essential for detecting hand-only
    translations/rotations and semantic hand-slot swaps as well as object-only
    mutations.

    The result contains only Python scalars, lists, and dictionaries and is
    therefore directly JSON serializable.  At the default tolerance, median,
    P95, and maximum absolute distance-error gates are 1e-8, 1e-7, and 1e-6 m.
    """

    try:
        tolerance_value = float(tolerance_m)
    except (TypeError, ValueError):
        tolerance_value = None
    if (
        tolerance_value is None
        or not np.isfinite(tolerance_value)
        or tolerance_value <= 0
    ):
        return _empty_invariance_summary(
            errors=["tolerance_m_must_be_positive_and_finite"],
            tolerance_m=None,
            canonical_transform=None,
        )

    transform: np.ndarray | None = None
    try:
        if canonical_transform is not None:
            transform = validate_rigid_transform(canonical_transform)
        before_hands = _load_hand_npz(before_hand_npz)
        after_hands = _load_hand_npz(after_hand_npz)
        before_poses = _pose_sequence(
            before_object_poses, name="before_object_poses"
        )
        after_poses = _pose_sequence(after_object_poses, name="after_object_poses")
    except (OSError, ValueError, TypeError) as exc:
        return _empty_invariance_summary(
            errors=[f"invalid_input:{exc}"],
            tolerance_m=tolerance_value,
            canonical_transform=transform,
        )

    errors: list[str] = []
    if before_poses.shape[0] != after_poses.shape[0]:
        errors.append(
            "object_pose_frame_count_mismatch:"
            f"{before_poses.shape[0]}!={after_poses.shape[0]}"
        )
    object_frames = int(before_poses.shape[0])
    if errors:
        return _empty_invariance_summary(
            errors=errors,
            tolerance_m=tolerance_value,
            canonical_transform=transform,
        )

    effective_transform = (
        transform if transform is not None else np.eye(4, dtype=np.float64)
    )
    expected_after_poses = np.matmul(
        effective_transform,
        before_poses,
    )
    object_translation_error = np.linalg.norm(
        after_poses[:, :3, 3] - expected_after_poses[:, :3, 3], axis=-1
    )
    # Source reconstructions can contain tiny orthonormality drift within the
    # accepted SE(3) tolerance.  ``R.T @ R`` would then report a nonzero angle
    # even when the before/after matrices are byte-identical.  Solving the
    # actual relative transform makes identity adapters exactly identity while
    # still exposing a real object-only rotation.
    rotation_delta = np.linalg.solve(
        expected_after_poses[:, :3, :3], after_poses[:, :3, :3]
    )
    rotation_cosine = np.clip(
        (np.trace(rotation_delta, axis1=-2, axis2=-1) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    object_rotation_error = np.arccos(rotation_cosine)
    if float(np.max(object_translation_error, initial=0.0)) > tolerance_value:
        errors.append("object_pose_translation_error_exceeds_tolerance")
    if float(np.max(object_rotation_error, initial=0.0)) > tolerance_value:
        errors.append("object_pose_rotation_error_exceeds_tolerance")
    before_centers = before_poses[:, :3, 3]
    after_centers = after_poses[:, :3, 3]
    wrist_errors: list[np.ndarray] = []
    fingertip_errors: list[np.ndarray] = []
    hand_canonical_errors: list[np.ndarray] = []
    vertex_canonical_errors: list[np.ndarray] = []
    side_summaries: dict[str, Any] = {}
    compared_sides: list[str] = []
    valid_frame_union = np.zeros(object_frames, dtype=bool)

    before_joint_sides = {
        side for side in ("left", "right") if f"{side}_joints" in before_hands
    }
    after_joint_sides = {
        side for side in ("left", "right") if f"{side}_joints" in after_hands
    }
    if before_joint_sides != after_joint_sides:
        errors.append(
            "hand_joint_side_set_mismatch:"
            f"before={sorted(before_joint_sides)}:after={sorted(after_joint_sides)}"
        )
    joint_entries = [
        (side, f"{side}_joints")
        for side in ("left", "right")
        if side in before_joint_sides & after_joint_sides
    ]
    if not joint_entries and "joints" in before_hands and "joints" in after_hands:
        joint_entries = [("hand", "joints")]
    if not joint_entries:
        errors.append("missing_hand_joints")

    for side, joints_key in joint_entries:
        before_joints = np.asarray(before_hands[joints_key], dtype=np.float64)
        after_joints = np.asarray(after_hands[joints_key], dtype=np.float64)
        if before_joints.ndim == 2:
            before_joints = before_joints[None, ...]
        if after_joints.ndim == 2:
            after_joints = after_joints[None, ...]
        if (
            before_joints.ndim != 3
            or before_joints.shape[-1] != 3
            or after_joints.ndim != 3
            or after_joints.shape[-1] != 3
        ):
            errors.append(
                f"invalid_{joints_key}_shape:"
                f"before={before_joints.shape}:after={after_joints.shape}"
            )
            continue
        if before_joints.shape != after_joints.shape:
            errors.append(
                f"{side}_hand_shape_mismatch:"
                f"{before_joints.shape}!={after_joints.shape}"
            )
            continue
        if after_joints.shape[0] != object_frames:
            errors.append(
                f"{side}_hand_object_frame_count_mismatch:"
                f"{after_joints.shape[0]}!={object_frames}"
            )
            continue
        if after_joints.shape[1] <= max(DEFAULT_FINGERTIP_INDICES):
            errors.append(
                f"{side}_joints_missing_openpose21_fingertips:"
                f"{after_joints.shape[1]}"
            )
            continue

        valid_key = f"{side}_valid"
        before_valid_values = (
            np.asarray(before_hands[valid_key]).reshape(-1)
            if valid_key in before_hands
            else np.ones(object_frames, dtype=np.float64)
        )
        after_valid_values = (
            np.asarray(after_hands[valid_key]).reshape(-1)
            if valid_key in after_hands
            else np.ones(object_frames, dtype=np.float64)
        )
        for label, valid_values in (
            ("before", before_valid_values),
            ("after", after_valid_values),
        ):
            if valid_values.size != object_frames:
                errors.append(
                    f"{side}_{label}_valid_frame_count_mismatch:"
                    f"{valid_values.size}!={object_frames}"
                )
        if (
            before_valid_values.size != object_frames
            or after_valid_values.size != object_frames
        ):
            continue
        before_valid = np.isfinite(before_valid_values) & (before_valid_values > 0.5)
        after_valid = np.isfinite(after_valid_values) & (after_valid_values > 0.5)
        if not np.array_equal(before_valid, after_valid):
            errors.append(f"{side}_valid_mask_mismatch")
        valid = before_valid & after_valid

        selected_before = before_joints[:, (0,) + DEFAULT_FINGERTIP_INDICES, :]
        selected_after = after_joints[:, (0,) + DEFAULT_FINGERTIP_INDICES, :]
        finite = np.isfinite(selected_before).all(axis=(1, 2)) & np.isfinite(
            selected_after
        ).all(axis=(1, 2))
        nonfinite_valid = valid & ~finite
        if nonfinite_valid.any():
            errors.append(f"{side}_nonfinite_joints_in_valid_frames")
        valid &= finite
        if not valid.any():
            side_summaries[side] = {
                "status": "unavailable",
                "valid_frames": 0,
                "reason": "no_valid_hand_frames",
            }
            continue

        before_joints_valid = before_joints[valid]
        after_joints_valid = after_joints[valid]
        expected_after_joints = _transform_points(
            before_joints_valid,
            effective_transform,
            name=f"{side}_pre_canonical_joints",
        )
        selected_indices = (0,) + DEFAULT_FINGERTIP_INDICES
        canonical_error = np.linalg.norm(
            after_joints_valid[:, selected_indices]
            - expected_after_joints[:, selected_indices],
            axis=-1,
        )
        hand_canonical_errors.append(canonical_error.reshape(-1))
        wrist_before = np.linalg.norm(
            before_joints_valid[:, 0] - before_centers[valid], axis=-1
        )
        wrist_after = np.linalg.norm(
            after_joints_valid[:, 0] - after_centers[valid], axis=-1
        )
        tips_before = np.linalg.norm(
            before_joints_valid[:, DEFAULT_FINGERTIP_INDICES]
            - before_centers[valid, None, :],
            axis=-1,
        )
        tips_after = np.linalg.norm(
            after_joints_valid[:, DEFAULT_FINGERTIP_INDICES]
            - after_centers[valid, None, :],
            axis=-1,
        )
        wrist_error = np.abs(wrist_after - wrist_before)
        fingertip_error = np.abs(tips_after - tips_before)
        wrist_errors.append(wrist_error)
        fingertip_errors.append(fingertip_error.reshape(-1))
        compared_sides.append(side)
        valid_frame_union |= valid
        side_summaries[side] = {
            "status": "ok",
            "valid_frames": int(np.count_nonzero(valid)),
            "wrist_object": {
                "before_distance_m": _distance_stats(wrist_before),
                "after_distance_m": _distance_stats(wrist_after),
                "absolute_error_m": _distance_stats(wrist_error),
            },
            "fingertips_object": {
                "indices": list(DEFAULT_FINGERTIP_INDICES),
                "before_distance_m": _distance_stats(tips_before),
                "after_distance_m": _distance_stats(tips_after),
                "absolute_error_m": _distance_stats(fingertip_error),
            },
            "canonical_position_error_m": _distance_stats(canonical_error),
        }

        vertices_key = f"{side}_vertices"
        if vertices_key in before_hands or vertices_key in after_hands:
            if vertices_key not in before_hands or vertices_key not in after_hands:
                errors.append(f"{side}_hand_vertices_missing_on_one_boundary")
            else:
                before_vertices = np.asarray(
                    before_hands[vertices_key], dtype=np.float64
                )
                after_vertices = np.asarray(
                    after_hands[vertices_key], dtype=np.float64
                )
                if before_vertices.shape != after_vertices.shape:
                    errors.append(
                        f"{side}_hand_vertices_shape_mismatch:"
                        f"{before_vertices.shape}!={after_vertices.shape}"
                    )
                elif (
                    before_vertices.ndim != 3
                    or before_vertices.shape[0] != object_frames
                    or before_vertices.shape[-1] != 3
                ):
                    errors.append(
                        f"invalid_{vertices_key}_shape:{before_vertices.shape}"
                    )
                else:
                    finite_vertices = np.isfinite(before_vertices).all(axis=(1, 2)) & np.isfinite(
                        after_vertices
                    ).all(axis=(1, 2))
                    vertex_valid = valid & finite_vertices
                    if np.any(valid & ~finite_vertices):
                        errors.append(f"{side}_nonfinite_vertices_in_valid_frames")
                    if vertex_valid.any():
                        expected_after_vertices = _transform_points(
                            before_vertices[vertex_valid],
                            effective_transform,
                            name=f"{side}_pre_canonical_vertices",
                        )
                        vertex_error = np.linalg.norm(
                            after_vertices[vertex_valid] - expected_after_vertices,
                            axis=-1,
                        )
                        vertex_canonical_errors.append(vertex_error.reshape(-1))
                        side_summaries[side][
                            "vertices_canonical_position_error_m"
                        ] = _distance_stats(vertex_error)

    wrist_flat = (
        np.concatenate(wrist_errors) if wrist_errors else np.asarray([], dtype=np.float64)
    )
    fingertip_flat = (
        np.concatenate(fingertip_errors)
        if fingertip_errors
        else np.asarray([], dtype=np.float64)
    )
    all_errors = np.concatenate((wrist_flat, fingertip_flat))
    hand_canonical_flat = (
        np.concatenate(hand_canonical_errors)
        if hand_canonical_errors
        else np.asarray([], dtype=np.float64)
    )
    vertex_canonical_flat = (
        np.concatenate(vertex_canonical_errors)
        if vertex_canonical_errors
        else np.asarray([], dtype=np.float64)
    )
    aggregate = _distance_stats(all_errors)
    thresholds = _invariance_thresholds(tolerance_value)

    if all_errors.size == 0:
        errors.append("no_valid_hand_object_samples")
    else:
        for metric in ("median", "p95", "max"):
            value = aggregate[metric]
            if value is not None and value > thresholds[metric]:
                errors.append(f"relative_distance_{metric}_error_exceeds_tolerance")
    if (
        hand_canonical_flat.size == 0
        or float(np.max(hand_canonical_flat)) > tolerance_value
    ):
        errors.append("hand_canonical_position_error_exceeds_tolerance")
    if vertex_canonical_flat.size and float(np.max(vertex_canonical_flat)) > tolerance_value:
        errors.append("hand_vertices_canonical_position_error_exceeds_tolerance")

    return {
        "status": "invalid" if errors else "ok",
        "errors": errors,
        "median": aggregate["median"],
        "p95": aggregate["p95"],
        "max": aggregate["max"],
        "thresholds_m": thresholds,
        "tolerance_m": tolerance_value,
        "hand_coordinate_frame": "explicit_adapter_boundary_frames",
        "canonical_transform_applied": transform is not None,
        "canonical_transform": transform.tolist() if transform is not None else None,
        "true_pre_post_hand_comparison": True,
        "frames": int(np.count_nonzero(valid_frame_union)),
        "samples": int(all_errors.size),
        "compared_sides": compared_sides,
        "wrist_object": _distance_stats(wrist_flat),
        "fingertips_object": _distance_stats(fingertip_flat),
        "hand_canonical_position_error_m": _distance_stats(hand_canonical_flat),
        "hand_vertices_canonical_position_error_m": _distance_stats(
            vertex_canonical_flat
        ),
        "object_pose": {
            "expected_relation": "canonical_transform @ before_object_pose",
            "translation_error_m": _distance_stats(object_translation_error),
            "rotation_error_rad": _distance_stats(object_rotation_error),
        },
        "sides": side_summaries,
    }
