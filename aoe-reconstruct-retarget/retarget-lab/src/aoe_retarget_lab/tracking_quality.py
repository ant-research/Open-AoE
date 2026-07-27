from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np


_THRESHOLD_EPS = 1e-12
_GEOM_DISTANCE_ZERO_EPS_M = 1e-9
HARD_TRACKING_ADMISSION_POLICY = "post_retarget_hard_metrics_v2"


@dataclass(frozen=True)
class TrackingThresholds:
    pos_median_m: float = 0.05
    pos_p95_m: float = 0.10
    rot_median_rad: float = 0.75
    rot_p95_rad: float = 2.0
    lost_pos_m: float = 0.10
    lost_fraction: float = 0.10


@dataclass(frozen=True)
class InteractionThresholds:
    reference_contact_m: float = 0.05
    executed_contact_m: float = 0.08
    distance_error_median_m: float = 0.03
    distance_error_p95_m: float = 0.08
    relative_translation_error_median_m: float = 0.03
    relative_translation_error_p95_m: float = 0.08
    min_reference_contact_frames: int = 3
    min_contact_retention: float = 0.50


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _required_metric(
    payload: dict[str, Any],
    key: str,
    *,
    label: str,
    errors: list[str],
) -> float | None:
    value = _finite_float(payload.get(key))
    if value is None:
        errors.append(f"missing_or_nonfinite {label}.{key}")
    return value


def validate_hard_tracking_admission(report: dict[str, Any] | None) -> dict[str, Any]:
    """Re-evaluate a tracking report against immutable post-retarget gates.

    This deliberately does not trust a producer's top-level ``status``.  Fresh
    reports must carry the raw metrics and the thresholds used to compute them;
    status-only and pre-schema reports remain useful diagnostics but are not
    admissible as demo evidence.
    """

    hard_thresholds: dict[str, Any] = {
        "pos_median_m_max": 0.05,
        "pos_p95_m_max": 0.10,
        "rot_median_rad_max": 0.75,
        "rot_p95_rad_max": 2.0,
        "lost_pos_m_max": 0.10,
        "lost_frame_fraction_max": 0.10,
        "distance_error_median_m_max": 0.03,
        "distance_error_p95_m_max": 0.08,
        "wrist_object_relative_translation_error_median_m_max": 0.03,
        "wrist_object_relative_translation_error_p95_m_max": 0.08,
        "contact_retention_fraction_min": 0.50,
        "reference_contact_m": 0.05,
        "executed_contact_m_max": 0.08,
        "min_reference_contact_frames_min": 3,
    }
    errors: list[str] = []
    payload = report if isinstance(report, dict) else {}
    if not payload:
        errors.append("missing object_tracking_quality report")
    if payload.get("status") != "ok":
        errors.append(
            f"object_tracking_quality status={payload.get('status', 'missing')}, expected ok"
        )
    if payload.get("available") is False:
        errors.append("object_tracking_quality available=false")
    if payload.get("finite") is False:
        errors.append("object_tracking_quality finite=false")
    if payload.get("errors"):
        errors.append(f"object_tracking_quality reports errors={payload.get('errors')}")

    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("missing object_tracking_quality.metrics (status-only/legacy report)")
        metrics = {}
    metric_limits = (
        ("pos_median_m", hard_thresholds["pos_median_m_max"]),
        ("pos_p95_m", hard_thresholds["pos_p95_m_max"]),
        ("rot_median_rad", hard_thresholds["rot_median_rad_max"]),
        ("rot_p95_rad", hard_thresholds["rot_p95_rad_max"]),
        ("lost_frame_fraction", hard_thresholds["lost_frame_fraction_max"]),
    )
    checked_metrics: dict[str, float | None] = {}
    for key, limit in metric_limits:
        value = _required_metric(metrics, key, label="metrics", errors=errors)
        checked_metrics[key] = value
        if value is not None and value < -_THRESHOLD_EPS:
            errors.append(f"metrics.{key}={value:.6f} is negative")
        if value is not None and value > limit + _THRESHOLD_EPS:
            errors.append(f"metrics.{key}={value:.6f} exceeds hard limit {limit:.6f}")

    reported_thresholds = payload.get("thresholds")
    if not isinstance(reported_thresholds, dict):
        errors.append("missing object_tracking_quality.thresholds")
        reported_thresholds = {}
    max_thresholds = (
        ("pos_median_m", hard_thresholds["pos_median_m_max"]),
        ("pos_p95_m", hard_thresholds["pos_p95_m_max"]),
        ("rot_median_rad", hard_thresholds["rot_median_rad_max"]),
        ("rot_p95_rad", hard_thresholds["rot_p95_rad_max"]),
        ("lost_pos_m", hard_thresholds["lost_pos_m_max"]),
        ("lost_fraction", hard_thresholds["lost_frame_fraction_max"]),
    )
    for key, hard_limit in max_thresholds:
        value = _required_metric(
            reported_thresholds, key, label="thresholds", errors=errors
        )
        if value is not None and value > hard_limit + _THRESHOLD_EPS:
            errors.append(
                f"thresholds.{key}={value:.6f} is looser than hard limit {hard_limit:.6f}"
            )

    interaction = payload.get("interaction_quality")
    if not isinstance(interaction, dict):
        errors.append("missing object_tracking_quality.interaction_quality")
        interaction = {}
    if interaction.get("status") != "ok":
        errors.append(
            f"interaction_quality status={interaction.get('status', 'missing')}, expected ok"
        )
    if interaction.get("available") is False:
        errors.append("interaction_quality available=false")
    if interaction.get("errors"):
        errors.append(f"interaction_quality reports errors={interaction.get('errors')}")

    interaction_thresholds = interaction.get("thresholds")
    if not isinstance(interaction_thresholds, dict):
        errors.append("missing interaction_quality.thresholds")
        interaction_thresholds = {}
    interaction_max_thresholds = (
        ("distance_error_median_m", hard_thresholds["distance_error_median_m_max"]),
        ("distance_error_p95_m", hard_thresholds["distance_error_p95_m_max"]),
        ("executed_contact_m", hard_thresholds["executed_contact_m_max"]),
    )
    for key, hard_limit in interaction_max_thresholds:
        value = _required_metric(
            interaction_thresholds,
            key,
            label="interaction_quality.thresholds",
            errors=errors,
        )
        if value is not None and value > hard_limit + _THRESHOLD_EPS:
            errors.append(
                f"interaction_quality.thresholds.{key}={value:.6f} "
                f"is looser than hard limit {hard_limit:.6f}"
            )
    interaction_min_thresholds = (
        ("min_contact_retention", hard_thresholds["contact_retention_fraction_min"]),
        (
            "min_reference_contact_frames",
            float(hard_thresholds["min_reference_contact_frames_min"]),
        ),
    )
    for key, hard_limit in interaction_min_thresholds:
        value = _required_metric(
            interaction_thresholds,
            key,
            label="interaction_quality.thresholds",
            errors=errors,
        )
        if value is not None and value + _THRESHOLD_EPS < hard_limit:
            errors.append(
                f"interaction_quality.thresholds.{key}={value:.6f} "
                f"is looser than hard minimum {hard_limit:.6f}"
            )
    reference_contact_m = _required_metric(
        interaction_thresholds,
        "reference_contact_m",
        label="interaction_quality.thresholds",
        errors=errors,
    )
    if (
        reference_contact_m is not None
        and abs(reference_contact_m - hard_thresholds["reference_contact_m"])
        > _THRESHOLD_EPS
    ):
        errors.append(
            "interaction_quality.thresholds.reference_contact_m="
            f"{reference_contact_m:.6f} differs from fixed hard threshold "
            f"{hard_thresholds['reference_contact_m']:.6f}"
        )

    active_sides = interaction.get("active_sides")
    if not isinstance(active_sides, list) or not active_sides:
        errors.append("interaction_quality has no active_sides")
        active_sides = []
    normalized_active_sides = [str(side) for side in active_sides]
    if len(normalized_active_sides) != len(set(normalized_active_sides)):
        errors.append("interaction_quality.active_sides contains duplicates")
    unsupported_active_sides = sorted(
        side for side in set(normalized_active_sides) if side not in {"left", "right"}
    )
    if unsupported_active_sides:
        errors.append(
            "interaction_quality.active_sides contains unsupported sides="
            f"{unsupported_active_sides}"
        )
    sides = interaction.get("sides")
    if not isinstance(sides, dict):
        errors.append("missing interaction_quality.sides")
        sides = {}
    checked_sides: dict[str, Any] = {}
    for raw_side in active_sides:
        side = str(raw_side)
        row = sides.get(side)
        if not isinstance(row, dict):
            errors.append(f"missing interaction_quality.sides.{side}")
            continue
        if row.get("status") != "ok":
            errors.append(
                f"interaction_quality.sides.{side}.status="
                f"{row.get('status', 'missing')}, expected ok"
            )
        if row.get("errors"):
            errors.append(
                f"interaction_quality.sides.{side} reports errors={row.get('errors')}"
            )
        side_values: dict[str, float | None] = {}
        for key, limit in (
            ("distance_error_median_m", hard_thresholds["distance_error_median_m_max"]),
            ("distance_error_p95_m", hard_thresholds["distance_error_p95_m_max"]),
        ):
            value = _required_metric(
                row, key, label=f"interaction_quality.sides.{side}", errors=errors
            )
            side_values[key] = value
            if value is not None and value < -_THRESHOLD_EPS:
                errors.append(
                    f"interaction_quality.sides.{side}.{key}={value:.6f} is negative"
                )
            if value is not None and value > limit + _THRESHOLD_EPS:
                errors.append(
                    f"interaction_quality.sides.{side}.{key}={value:.6f} "
                    f"exceeds hard limit {limit:.6f}"
                )
        retention = _required_metric(
            row,
            "contact_retention_fraction",
            label=f"interaction_quality.sides.{side}",
            errors=errors,
        )
        side_values["contact_retention_fraction"] = retention
        if retention is not None and (
            retention < -_THRESHOLD_EPS or retention > 1.0 + _THRESHOLD_EPS
        ):
            errors.append(
                f"interaction_quality.sides.{side}.contact_retention_fraction="
                f"{retention:.6f} is outside [0, 1]"
            )
        if (
            retention is not None
            and retention + _THRESHOLD_EPS
            < hard_thresholds["contact_retention_fraction_min"]
        ):
            errors.append(
                f"interaction_quality.sides.{side}.contact_retention_fraction="
                f"{retention:.6f} below hard minimum "
                f"{hard_thresholds['contact_retention_fraction_min']:.6f}"
            )
        checked_sides[side] = side_values

    relative_translation = interaction.get("wrist_object_relative_translation")
    if not isinstance(relative_translation, dict):
        errors.append(
            "missing interaction_quality.wrist_object_relative_translation"
        )
        relative_translation = {}
    if relative_translation.get("status") != "ok":
        errors.append(
            "wrist_object_relative_translation status="
            f"{relative_translation.get('status', 'missing')}, expected ok"
        )
    if relative_translation.get("available") is not True:
        errors.append("wrist_object_relative_translation available is not true")
    if relative_translation.get("finite") is not True:
        errors.append("wrist_object_relative_translation finite is not true")
    if relative_translation.get("errors"):
        errors.append(
            "wrist_object_relative_translation reports errors="
            f"{relative_translation.get('errors')}"
        )

    relative_active_sides = relative_translation.get("active_sides")
    if not isinstance(relative_active_sides, list):
        errors.append("missing wrist_object_relative_translation.active_sides")
        relative_active_sides = []
    normalized_relative_sides = [str(side) for side in relative_active_sides]
    if normalized_relative_sides != normalized_active_sides:
        errors.append(
            "wrist_object_relative_translation.active_sides does not exactly match "
            f"interaction_quality.active_sides: {normalized_relative_sides} != "
            f"{normalized_active_sides}"
        )

    relative_thresholds = relative_translation.get("thresholds")
    if not isinstance(relative_thresholds, dict):
        errors.append("missing wrist_object_relative_translation.thresholds")
        relative_thresholds = {}
    for key, hard_limit in (
        (
            "error_median_m",
            hard_thresholds[
                "wrist_object_relative_translation_error_median_m_max"
            ],
        ),
        (
            "error_p95_m",
            hard_thresholds["wrist_object_relative_translation_error_p95_m_max"],
        ),
    ):
        value = _required_metric(
            relative_thresholds,
            key,
            label="wrist_object_relative_translation.thresholds",
            errors=errors,
        )
        if value is not None and value > hard_limit + _THRESHOLD_EPS:
            errors.append(
                f"wrist_object_relative_translation.thresholds.{key}="
                f"{value:.6f} is looser than hard limit {hard_limit:.6f}"
            )

    relative_sides = relative_translation.get("sides")
    if not isinstance(relative_sides, dict):
        errors.append("missing wrist_object_relative_translation.sides")
        relative_sides = {}
    for side in normalized_active_sides:
        row = relative_sides.get(side)
        label = f"wrist_object_relative_translation.sides.{side}"
        if not isinstance(row, dict):
            errors.append(f"missing {label}")
            continue
        if row.get("status") != "ok":
            errors.append(
                f"{label}.status={row.get('status', 'missing')}, expected ok"
            )
        if row.get("available") is not True:
            errors.append(f"{label}.available is not true")
        if row.get("finite") is not True:
            errors.append(f"{label}.finite is not true")
        if row.get("errors"):
            errors.append(f"{label} reports errors={row.get('errors')}")
        frames = _nonnegative_int(row.get("frames"))
        if frames is None or frames == 0:
            errors.append(f"missing_or_invalid {label}.frames")
        relative_values: dict[str, float | None] = {}
        for key, hard_limit in (
            (
                "error_median_m",
                hard_thresholds[
                    "wrist_object_relative_translation_error_median_m_max"
                ],
            ),
            (
                "error_p95_m",
                hard_thresholds[
                    "wrist_object_relative_translation_error_p95_m_max"
                ],
            ),
        ):
            value = _required_metric(row, key, label=label, errors=errors)
            relative_values[key] = value
            if value is not None and value < -_THRESHOLD_EPS:
                errors.append(f"{label}.{key}={value:.6f} is negative")
            if value is not None and value > hard_limit + _THRESHOLD_EPS:
                errors.append(
                    f"{label}.{key}={value:.6f} exceeds hard limit "
                    f"{hard_limit:.6f}"
                )
        binding_errors = _exact_wrist_object_binding_errors(
            side, row.get("binding")
        )
        errors.extend(f"{label}.binding:{error}" for error in binding_errors)
        checked_sides.setdefault(side, {})[
            "wrist_object_relative_translation"
        ] = relative_values

    return {
        "policy": HARD_TRACKING_ADMISSION_POLICY,
        "status": "invalid" if errors else "ok",
        "admissible": not errors,
        "errors": errors,
        "hard_thresholds": hard_thresholds,
        "checked_metrics": checked_metrics,
        "checked_active_sides": checked_sides,
        "source_report_status": payload.get("status", "missing"),
    }


def load_qpos(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "qpos" not in data:
            raise ValueError(f"trajectory has no qpos: {path}")
        qpos = np.asarray(data["qpos"], dtype=np.float64)
    if qpos.ndim < 2:
        raise ValueError(f"expected qpos with at least two dimensions: {path} shape={qpos.shape}")
    return qpos.reshape(-1, qpos.shape[-1])


def resample_rows(array: np.ndarray, count: int) -> np.ndarray:
    array = np.asarray(array)
    if len(array) == count:
        return array
    if len(array) == 0 or count <= 0:
        raise ValueError(f"cannot resample shape={array.shape} to count={count}")
    indices = np.rint(np.linspace(0, len(array) - 1, count)).astype(int)
    return array[indices]


def rotation_geodesic_rad(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64).reshape(-1, 3, 3)
    second = np.asarray(second, dtype=np.float64).reshape(-1, 3, 3)
    relative = np.einsum("nij,njk->nik", np.transpose(first, (0, 2, 1)), second)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def summarize_pose_errors(
    position_errors: np.ndarray,
    rotation_errors: np.ndarray,
    thresholds: TrackingThresholds,
) -> dict[str, Any]:
    position_errors = np.asarray(position_errors, dtype=np.float64).reshape(-1)
    rotation_errors = np.asarray(rotation_errors, dtype=np.float64).reshape(-1)
    if len(position_errors) == 0 or len(position_errors) != len(rotation_errors):
        return {"status": "invalid", "available": False, "reason": "empty_or_mismatched_errors"}
    finite = np.isfinite(position_errors) & np.isfinite(rotation_errors)
    if not np.all(finite):
        return {
            "status": "invalid",
            "available": True,
            "finite": False,
            "frames": int(len(position_errors)),
            "nonfinite_frames": int(np.count_nonzero(~finite)),
            "errors": ["nonfinite_object_pose_error"],
        }

    metrics = {
        "pos_median_m": float(np.median(position_errors)),
        "pos_p95_m": float(np.quantile(position_errors, 0.95)),
        "pos_max_m": float(np.max(position_errors)),
        "pos_final_m": float(position_errors[-1]),
        "rot_median_rad": float(np.median(rotation_errors)),
        "rot_p95_rad": float(np.quantile(rotation_errors, 0.95)),
        "rot_max_rad": float(np.max(rotation_errors)),
        "rot_final_rad": float(rotation_errors[-1]),
        "lost_frame_fraction": float(np.mean(position_errors > thresholds.lost_pos_m)),
    }
    errors: list[str] = []
    checks = (
        ("pos_median_m", thresholds.pos_median_m),
        ("pos_p95_m", thresholds.pos_p95_m),
        ("rot_median_rad", thresholds.rot_median_rad),
        ("rot_p95_rad", thresholds.rot_p95_rad),
        ("lost_frame_fraction", thresholds.lost_fraction),
    )
    for name, limit in checks:
        if metrics[name] > limit + _THRESHOLD_EPS:
            errors.append(f"{name}={metrics[name]:.6f} exceeds {limit:.6f}")
    return {
        "status": "ok" if not errors else "invalid",
        "available": True,
        "finite": True,
        "frames": int(len(position_errors)),
        "metrics": metrics,
        "thresholds": {
            "pos_median_m": thresholds.pos_median_m,
            "pos_p95_m": thresholds.pos_p95_m,
            "rot_median_rad": thresholds.rot_median_rad,
            "rot_p95_rad": thresholds.rot_p95_rad,
            "lost_pos_m": thresholds.lost_pos_m,
            "lost_fraction": thresholds.lost_fraction,
        },
        "errors": errors,
    }


def summarize_interaction_distances(
    executed_by_side: dict[str, np.ndarray],
    reference_by_side: dict[str, np.ndarray],
    thresholds: InteractionThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or InteractionThresholds()
    sides: dict[str, Any] = {}
    errors: list[str] = []
    active_sides: list[str] = []
    for side in sorted(set(executed_by_side) | set(reference_by_side)):
        executed = np.asarray(executed_by_side.get(side, []), dtype=np.float64).reshape(-1)
        reference = np.asarray(reference_by_side.get(side, []), dtype=np.float64).reshape(-1)
        if len(executed) == 0 or len(executed) != len(reference):
            sides[side] = {"available": False, "reason": "empty_or_mismatched_distance_sequence"}
            continue
        finite = np.isfinite(executed) & np.isfinite(reference)
        if not np.all(finite):
            sides[side] = {
                "available": True,
                "finite": False,
                "nonfinite_frames": int(np.count_nonzero(~finite)),
            }
            errors.append(f"{side}:nonfinite_hand_object_distance")
            continue
        distance_error = np.abs(executed - reference)
        reference_contact = reference <= thresholds.reference_contact_m
        reference_contact_frames = int(np.count_nonzero(reference_contact))
        retention = (
            float(np.mean(executed[reference_contact] <= thresholds.executed_contact_m))
            if reference_contact_frames
            else None
        )
        row = {
            "available": True,
            "finite": True,
            "frames": int(len(executed)),
            "reference_distance_median_m": float(np.median(reference)),
            "executed_distance_median_m": float(np.median(executed)),
            "distance_error_median_m": float(np.median(distance_error)),
            "distance_error_p95_m": float(np.quantile(distance_error, 0.95)),
            "reference_contact_frames": reference_contact_frames,
            "reference_contact_fraction": float(np.mean(reference_contact)),
            "executed_contact_fraction": float(np.mean(executed <= thresholds.executed_contact_m)),
            "contact_retention_fraction": retention,
        }
        side_errors: list[str] = []
        if reference_contact_frames >= thresholds.min_reference_contact_frames:
            active_sides.append(side)
            if row["distance_error_median_m"] > thresholds.distance_error_median_m + _THRESHOLD_EPS:
                side_errors.append("distance_error_median_m")
            if row["distance_error_p95_m"] > thresholds.distance_error_p95_m + _THRESHOLD_EPS:
                side_errors.append("distance_error_p95_m")
            if retention is None or retention < thresholds.min_contact_retention:
                side_errors.append("contact_retention_fraction")
        row["status"] = "invalid" if side_errors else (
            "ok" if reference_contact_frames >= thresholds.min_reference_contact_frames else "not_applicable"
        )
        row["errors"] = side_errors
        errors.extend(f"{side}:{name}" for name in side_errors)
        sides[side] = row
    if not active_sides:
        errors.append("no_reference_hand_object_contact")
    return {
        "status": "ok" if not errors else "invalid",
        "available": bool(sides),
        "active_sides": active_sides,
        "sides": sides,
        "thresholds": {
            "reference_contact_m": thresholds.reference_contact_m,
            "executed_contact_m": thresholds.executed_contact_m,
            "distance_error_median_m": thresholds.distance_error_median_m,
            "distance_error_p95_m": thresholds.distance_error_p95_m,
            "min_reference_contact_frames": thresholds.min_reference_contact_frames,
            "min_contact_retention": thresholds.min_contact_retention,
        },
        "errors": errors,
        "method": "mujoco_collision_geom_surface_distance_vs_kinematic_reference",
    }


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    if result < 0:
        return None
    try:
        if float(value) != float(result):
            return None
    except (TypeError, ValueError):
        return None
    return result


def _exact_wrist_object_binding_errors(side: str, binding: Any) -> list[str]:
    if not isinstance(binding, dict):
        return ["missing_exact_wrist_object_binding"]
    errors: list[str] = []
    expected_wrist_site = f"{side}_palm"
    expected_object_body = f"{side}_object"
    if binding.get("side") != side:
        errors.append("binding_side_mismatch")
    if binding.get("wrist_site_name") != expected_wrist_site:
        errors.append(
            f"wrist_site_name_not_exact:{binding.get('wrist_site_name')!r}!={expected_wrist_site!r}"
        )
    if binding.get("object_body_name") != expected_object_body:
        errors.append(
            "object_body_name_not_exact:"
            f"{binding.get('object_body_name')!r}!={expected_object_body!r}"
        )
    for key in ("wrist_site_id", "wrist_body_id", "object_body_id"):
        if _nonnegative_int(binding.get(key)) is None:
            errors.append(f"missing_or_invalid_{key}")
    wrist_body_name = binding.get("wrist_body_name")
    if not isinstance(wrist_body_name, str) or not wrist_body_name.startswith(f"{side}_"):
        errors.append(f"wrist_body_name_not_exact_side:{wrist_body_name!r}")
    resolution_errors = binding.get("resolution_errors")
    if resolution_errors:
        errors.append(f"geometry_resolution_errors:{resolution_errors}")
    return errors


def _xyz_sequence(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[1] != 3 or not len(array):
        return None
    return array


def _rotation_sequence(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim != 3 or array.shape[1:] != (3, 3) or not len(array):
        return None
    return array


def summarize_wrist_object_relative_translation(
    executed_wrist_by_side: dict[str, np.ndarray],
    reference_wrist_by_side: dict[str, np.ndarray],
    executed_wrist_rotation_by_side: dict[str, np.ndarray],
    reference_wrist_rotation_by_side: dict[str, np.ndarray],
    executed_object_by_side: dict[str, np.ndarray],
    reference_object_by_side: dict[str, np.ndarray],
    active_sides: list[str],
    bindings_by_side: dict[str, dict[str, Any]],
    thresholds: InteractionThresholds | None = None,
) -> dict[str, Any]:
    """Measure the true same-side wrist/object relative translation vector error.

    The metric compares ``R_wrist.T @ (p_object - p_wrist)`` between executed
    and reference trajectories.  It is invariant to a common SE(3), and is
    intentionally not a scalar surface-distance or contact proxy.  Every
    active side must bind the exact ``<side>_palm`` site and
    ``<side>_object`` body; missing or ambiguous geometry fails closed.
    """

    thresholds = thresholds or InteractionThresholds()
    normalized_sides = [str(side) for side in active_sides]
    errors: list[str] = []
    if not normalized_sides:
        errors.append("no_active_sides_for_wrist_object_relative_translation")
    if len(normalized_sides) != len(set(normalized_sides)):
        errors.append("duplicate_active_sides")

    sides: dict[str, Any] = {}
    for side in normalized_sides:
        side_errors: list[str] = []
        if side not in {"left", "right"}:
            side_errors.append(f"unsupported_active_side:{side}")
        binding = bindings_by_side.get(side)
        side_errors.extend(_exact_wrist_object_binding_errors(side, binding))

        sequences: dict[str, np.ndarray] = {}
        for label, source in (
            ("executed_wrist", executed_wrist_by_side),
            ("reference_wrist", reference_wrist_by_side),
            ("executed_object", executed_object_by_side),
            ("reference_object", reference_object_by_side),
        ):
            sequence = _xyz_sequence(source.get(side))
            if sequence is None:
                side_errors.append(f"missing_or_malformed_{label}_positions")
            else:
                sequences[label] = sequence

        for label, source in (
            ("executed_wrist_rotation", executed_wrist_rotation_by_side),
            ("reference_wrist_rotation", reference_wrist_rotation_by_side),
        ):
            sequence = _rotation_sequence(source.get(side))
            if sequence is None:
                side_errors.append(f"missing_or_malformed_{label}")
            else:
                sequences[label] = sequence

        lengths = {len(sequence) for sequence in sequences.values()}
        if len(sequences) == 6 and len(lengths) != 1:
            side_errors.append("mismatched_wrist_object_sequence_lengths")
        finite = len(sequences) == 6 and all(
            bool(np.isfinite(sequence).all()) for sequence in sequences.values()
        )
        if len(sequences) == 6 and not finite:
            side_errors.append("nonfinite_wrist_object_positions")

        row: dict[str, Any] = {
            "status": "invalid",
            "available": len(sequences) == 6 and not _exact_wrist_object_binding_errors(
                side, binding
            ),
            "finite": finite,
            "binding": binding,
            "errors": side_errors,
        }
        if len(sequences) == 6 and len(lengths) == 1 and finite:
            executed_delta = sequences["executed_object"] - sequences["executed_wrist"]
            reference_delta = sequences["reference_object"] - sequences["reference_wrist"]
            executed_relative = np.einsum(
                "nij,nj->ni",
                np.transpose(sequences["executed_wrist_rotation"], (0, 2, 1)),
                executed_delta,
            )
            reference_relative = np.einsum(
                "nij,nj->ni",
                np.transpose(sequences["reference_wrist_rotation"], (0, 2, 1)),
                reference_delta,
            )
            vector_error = np.linalg.norm(executed_relative - reference_relative, axis=1)
            row.update(
                {
                    "frames": int(len(vector_error)),
                    "error_median_m": float(np.median(vector_error)),
                    "error_p95_m": float(np.quantile(vector_error, 0.95)),
                    "error_max_m": float(np.max(vector_error)),
                    "error_final_m": float(vector_error[-1]),
                }
            )
            if (
                row["error_median_m"]
                > thresholds.relative_translation_error_median_m + _THRESHOLD_EPS
            ):
                side_errors.append("error_median_m")
            if (
                row["error_p95_m"]
                > thresholds.relative_translation_error_p95_m + _THRESHOLD_EPS
            ):
                side_errors.append("error_p95_m")
        row["status"] = "ok" if not side_errors else "invalid"
        errors.extend(f"{side}:{error}" for error in side_errors)
        sides[side] = row

    return {
        "status": "ok" if not errors else "invalid",
        "available": bool(normalized_sides)
        and all(bool(row.get("available")) for row in sides.values()),
        "finite": bool(normalized_sides)
        and all(bool(row.get("finite")) for row in sides.values()),
        "active_sides": normalized_sides,
        "sides": sides,
        "thresholds": {
            "error_median_m": thresholds.relative_translation_error_median_m,
            "error_p95_m": thresholds.relative_translation_error_p95_m,
        },
        "errors": errors,
        "method": (
            "same_side_exact_palm_frame_object_translation_vector_error:"
            "norm(Rw_exec.T@(object-wrist)_exec-"
            "Rw_ref.T@(object-wrist)_ref)"
        ),
    }


def _visual_object_bodies(model: Any, mujoco: Any) -> tuple[list[int], list[dict[str, Any]]]:
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    selected: list[int] = []
    evidence: list[dict[str, Any]] = []
    for body_id in range(1, int(model.nbody)):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if "object" not in body_name.casefold():
            continue
        geom_start = int(model.body_geomadr[body_id])
        geom_count = int(model.body_geomnum[body_id])
        mesh_geoms: list[str] = []
        all_geoms: list[str] = []
        for geom_id in range(geom_start, geom_start + geom_count):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            all_geoms.append(geom_name)
            if int(model.geom_type[geom_id]) == mesh_type:
                mesh_geoms.append(geom_name)
        row = {
            "body_id": body_id,
            "body_name": body_name,
            "mesh_geoms": mesh_geoms,
            "all_geoms": all_geoms,
            "selected": bool(mesh_geoms),
        }
        evidence.append(row)
        if mesh_geoms:
            selected.append(body_id)
    return selected, evidence


def _body_pose_sequence(model: Any, mujoco: Any, qpos: np.ndarray, body_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if qpos.shape[1] != int(model.nq):
        raise ValueError(f"qpos width={qpos.shape[1]} does not match model.nq={model.nq}")
    data = mujoco.MjData(model)
    positions = np.empty((len(qpos), len(body_ids), 3), dtype=np.float64)
    rotations = np.empty((len(qpos), len(body_ids), 3, 3), dtype=np.float64)
    for frame_id, row in enumerate(qpos):
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        for object_id, body_id in enumerate(body_ids):
            positions[frame_id, object_id] = data.xpos[body_id]
            rotations[frame_id, object_id] = data.xmat[body_id].reshape(3, 3)
    return positions, rotations


def _site_pose_sequence(
    model: Any,
    mujoco: Any,
    qpos: np.ndarray,
    site_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if qpos.shape[1] != int(model.nq):
        raise ValueError(f"qpos width={qpos.shape[1]} does not match model.nq={model.nq}")
    data = mujoco.MjData(model)
    positions = np.empty((len(qpos), len(site_ids), 3), dtype=np.float64)
    rotations = np.empty((len(qpos), len(site_ids), 3, 3), dtype=np.float64)
    for frame_id, row in enumerate(qpos):
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        for site_index, site_id in enumerate(site_ids):
            positions[frame_id, site_index] = data.site_xpos[site_id]
            rotations[frame_id, site_index] = data.site_xmat[site_id].reshape(3, 3)
    return positions, rotations


def _interaction_geom_ids(
    model: Any,
    mujoco: Any,
    object_body_ids: list[int],
) -> tuple[dict[str, list[int]], dict[str, list[int]], dict[str, Any]]:
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    object_geoms_by_side: dict[str, list[int]] = {"left": [], "right": []}
    unbound_object_geoms: list[int] = []
    for body_id in object_body_ids:
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        geom_start = int(model.body_geomadr[body_id])
        geom_count = int(model.body_geomnum[body_id])
        candidates = list(range(geom_start, geom_start + geom_count))
        collision = [
            geom_id
            for geom_id in candidates
            if int(model.geom_contype[geom_id]) != 0 or int(model.geom_conaffinity[geom_id]) != 0
        ]
        selected = collision or [
            geom_id
            for geom_id in candidates
            if int(model.geom_type[geom_id]) == mesh_type
        ]
        matching_sides = [
            side for side in object_geoms_by_side if body_name == f"{side}_object"
        ]
        if len(matching_sides) == 1:
            object_geoms_by_side[matching_sides[0]].extend(selected)
        else:
            unbound_object_geoms.extend(selected)
    hands: dict[str, list[int]] = {"left": [], "right": []}
    for geom_id in range(int(model.ngeom)):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        for side in hands:
            if name.startswith(f"collision_hand_{side}_") and not name.endswith("arm_cyl"):
                hands[side].append(geom_id)
    evidence = {
        "object_geom_ids_by_side": object_geoms_by_side,
        "object_geom_names_by_side": {
            side: [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                for geom_id in geom_ids
            ]
            for side, geom_ids in object_geoms_by_side.items()
        },
        "unbound_object_geom_ids": unbound_object_geoms,
        "exact_same_side_binding": True,
        "hand_geom_counts": {side: len(ids) for side, ids in hands.items()},
    }
    return object_geoms_by_side, hands, evidence


def _hand_object_distance_sequences(
    model: Any,
    mujoco: Any,
    qpos: np.ndarray,
    object_geoms: list[int] | dict[str, list[int]],
    hand_geoms: dict[str, list[int]],
    anomaly_evidence: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    if isinstance(object_geoms, dict):
        has_object_geoms = any(bool(ids) for ids in object_geoms.values())
    else:
        has_object_geoms = bool(object_geoms)
    if not has_object_geoms:
        raise ValueError("no collision-capable object geoms")
    data = mujoco.MjData(model)
    values = {
        side: np.empty(len(qpos), dtype=np.float64)
        for side, ids in hand_geoms.items()
        if ids and (not isinstance(object_geoms, dict) or object_geoms.get(side))
    }
    fromto = np.zeros(6, dtype=np.float64)
    fallback_count_by_side = {side: 0 for side in values}
    fallback_frames_by_side = {side: set() for side in values}
    for frame_id, row in enumerate(qpos):
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        contact_pairs = {
            tuple(sorted((int(data.contact[index].geom1), int(data.contact[index].geom2))))
            for index in range(int(data.ncon))
        }
        for side, geom_ids in hand_geoms.items():
            if side not in values:
                continue
            side_object_geoms = (
                object_geoms.get(side, [])
                if isinstance(object_geoms, dict)
                else object_geoms
            )
            best = float("inf")
            for hand_geom in geom_ids:
                for object_geom in side_object_geoms:
                    fromto.fill(np.nan)
                    distance = float(
                        mujoco.mj_geomDistance(model, data, hand_geom, object_geom, 10.0, fromto)
                    )
                    segment_norm = float(np.linalg.norm(fromto[3:] - fromto[:3]))
                    pair = tuple(sorted((int(hand_geom), int(object_geom))))
                    if (
                        abs(distance) <= _GEOM_DISTANCE_ZERO_EPS_M
                        and math.isfinite(segment_norm)
                        and segment_norm > _GEOM_DISTANCE_ZERO_EPS_M
                        and pair not in contact_pairs
                    ):
                        # MuJoCo can occasionally return an exact zero while
                        # still reporting two separated closest points.  With
                        # no corresponding collision contact, treating that as
                        # contact creates false retention.  The closest-point
                        # segment is the physically consistent fallback.
                        distance = segment_norm
                        fallback_count_by_side[side] += 1
                        fallback_frames_by_side[side].add(frame_id)
                    best = min(best, distance)
            values[side][frame_id] = best
    if anomaly_evidence is not None:
        anomaly_evidence.update(
            {
                "method": "mujoco_geom_distance_with_false_zero_segment_fallback",
                "false_zero_epsilon_m": _GEOM_DISTANCE_ZERO_EPS_M,
                "false_zero_fallback_count": int(sum(fallback_count_by_side.values())),
                "false_zero_fallback_count_by_side": fallback_count_by_side,
                "false_zero_fallback_frame_count_by_side": {
                    side: len(frames) for side, frames in fallback_frames_by_side.items()
                },
                "false_zero_fallback_frame_sample_by_side": {
                    side: sorted(frames)[:32] for side, frames in fallback_frames_by_side.items()
                },
            }
        )
    return values


def _exact_wrist_object_pose_sequences(
    model: Any,
    mujoco: Any,
    executed_qpos: np.ndarray,
    reference_qpos: np.ndarray,
    active_sides: list[str],
    eligible_object_body_ids: list[int],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, dict[str, Any]],
]:
    executed_wrist: dict[str, np.ndarray] = {}
    reference_wrist: dict[str, np.ndarray] = {}
    executed_wrist_rotation: dict[str, np.ndarray] = {}
    reference_wrist_rotation: dict[str, np.ndarray] = {}
    executed_object: dict[str, np.ndarray] = {}
    reference_object: dict[str, np.ndarray] = {}
    bindings: dict[str, dict[str, Any]] = {}
    resolved: list[tuple[str, int, int]] = []
    eligible_ids = set(int(body_id) for body_id in eligible_object_body_ids)

    for side in active_sides:
        wrist_site_name = f"{side}_palm"
        object_body_name = f"{side}_object"
        wrist_site_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, wrist_site_name)
        )
        object_body_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
        )
        resolution_errors: list[str] = []
        if wrist_site_id < 0:
            resolution_errors.append(f"missing_site:{wrist_site_name}")
        if object_body_id < 0:
            resolution_errors.append(f"missing_body:{object_body_name}")
        elif object_body_id not in eligible_ids:
            resolution_errors.append(
                f"object_body_not_selected_mesh_body:{object_body_name}"
            )

        wrist_body_id: int | None = None
        wrist_body_name: str | None = None
        if wrist_site_id >= 0:
            wrist_body_id = int(model.site_bodyid[wrist_site_id])
            wrist_body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, wrist_body_id)
                or ""
            )
        binding: dict[str, Any] = {
            "side": side,
            "wrist_site_name": wrist_site_name,
            "wrist_site_id": wrist_site_id if wrist_site_id >= 0 else None,
            "wrist_body_name": wrist_body_name,
            "wrist_body_id": wrist_body_id,
            "object_body_name": object_body_name,
            "object_body_id": object_body_id if object_body_id >= 0 else None,
            "resolution_errors": resolution_errors,
        }
        bindings[side] = binding
        if not resolution_errors:
            resolved.append((side, wrist_site_id, object_body_id))

    if resolved:
        wrist_site_ids = [wrist_site_id for _, wrist_site_id, _ in resolved]
        object_body_ids = [object_body_id for _, _, object_body_id in resolved]
        executed_wrist_array, executed_wrist_rotation_array = _site_pose_sequence(
            model, mujoco, executed_qpos, wrist_site_ids
        )
        reference_wrist_array, reference_wrist_rotation_array = _site_pose_sequence(
            model, mujoco, reference_qpos, wrist_site_ids
        )
        executed_object_array, _ = _body_pose_sequence(
            model, mujoco, executed_qpos, object_body_ids
        )
        reference_object_array, _ = _body_pose_sequence(
            model, mujoco, reference_qpos, object_body_ids
        )
        for index, (side, _, _) in enumerate(resolved):
            executed_wrist[side] = executed_wrist_array[:, index]
            reference_wrist[side] = reference_wrist_array[:, index]
            executed_wrist_rotation[side] = executed_wrist_rotation_array[:, index]
            reference_wrist_rotation[side] = reference_wrist_rotation_array[:, index]
            executed_object[side] = executed_object_array[:, index]
            reference_object[side] = reference_object_array[:, index]

    return (
        executed_wrist,
        reference_wrist,
        executed_wrist_rotation,
        reference_wrist_rotation,
        executed_object,
        reference_object,
        bindings,
    )


def evaluate_mjwp_object_tracking(
    scene: Path,
    mjwp_trajectory: Path,
    reference_trajectory: Path,
    thresholds: TrackingThresholds | None = None,
    interaction_thresholds: InteractionThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or TrackingThresholds()
    interaction_thresholds = interaction_thresholds or InteractionThresholds()
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(scene))
        body_ids, body_evidence = _visual_object_bodies(model, mujoco)
        if not body_ids:
            return {
                "status": "invalid",
                "available": False,
                "reason": "no_mesh_object_body",
                "body_candidates": body_evidence,
            }
        mjwp_qpos = load_qpos(mjwp_trajectory)
        reference_qpos = resample_rows(load_qpos(reference_trajectory), len(mjwp_qpos))
        if not np.isfinite(mjwp_qpos).all() or not np.isfinite(reference_qpos).all():
            return {
                "status": "invalid",
                "available": True,
                "finite": False,
                "reason": "nonfinite_qpos",
            }
        mjwp_position, mjwp_rotation = _body_pose_sequence(model, mujoco, mjwp_qpos, body_ids)
        ref_position, ref_rotation = _body_pose_sequence(model, mujoco, reference_qpos, body_ids)
        per_body_position = np.linalg.norm(mjwp_position - ref_position, axis=2)
        per_body_rotation = np.stack(
            [
                rotation_geodesic_rad(mjwp_rotation[:, body_id], ref_rotation[:, body_id])
                for body_id in range(len(body_ids))
            ],
            axis=1,
        )
        position_error = np.max(per_body_position, axis=1)
        rotation_error = np.max(per_body_rotation, axis=1)
        result = summarize_pose_errors(position_error, rotation_error, thresholds)
        object_geoms, hand_geoms, interaction_evidence = _interaction_geom_ids(
            model, mujoco, body_ids
        )
        executed_distance_evidence: dict[str, Any] = {}
        reference_distance_evidence: dict[str, Any] = {}
        interaction = summarize_interaction_distances(
            _hand_object_distance_sequences(
                model,
                mujoco,
                mjwp_qpos,
                object_geoms,
                hand_geoms,
                executed_distance_evidence,
            ),
            _hand_object_distance_sequences(
                model,
                mujoco,
                reference_qpos,
                object_geoms,
                hand_geoms,
                reference_distance_evidence,
            ),
            interaction_thresholds,
        )
        interaction_evidence["distance_queries"] = {
            "executed": executed_distance_evidence,
            "reference": reference_distance_evidence,
        }
        (
            executed_wrist_by_side,
            reference_wrist_by_side,
            executed_wrist_rotation_by_side,
            reference_wrist_rotation_by_side,
            executed_object_by_side,
            reference_object_by_side,
            bindings_by_side,
        ) = _exact_wrist_object_pose_sequences(
            model,
            mujoco,
            mjwp_qpos,
            reference_qpos,
            list(interaction.get("active_sides") or []),
            body_ids,
        )
        relative_translation = summarize_wrist_object_relative_translation(
            executed_wrist_by_side,
            reference_wrist_by_side,
            executed_wrist_rotation_by_side,
            reference_wrist_rotation_by_side,
            executed_object_by_side,
            reference_object_by_side,
            list(interaction.get("active_sides") or []),
            bindings_by_side,
            interaction_thresholds,
        )
        interaction["wrist_object_relative_translation"] = relative_translation
        if relative_translation.get("status") != "ok":
            interaction["status"] = "invalid"
            interaction.setdefault("errors", []).extend(
                f"wrist_object_relative_translation:{error}"
                for error in relative_translation.get("errors", [])
            )
        interaction["geom_evidence"] = interaction_evidence
        result["interaction_quality"] = interaction
        if interaction.get("status") != "ok":
            result["status"] = "invalid"
            result.setdefault("errors", []).extend(
                f"interaction:{error}" for error in interaction.get("errors", [])
            )
        selected_evidence = {row["body_id"]: row for row in body_evidence}
        result.update(
            {
                "scene": str(scene),
                "mjwp_trajectory": str(mjwp_trajectory),
                "reference_trajectory": str(reference_trajectory),
                "object_bodies": [selected_evidence[body_id] for body_id in body_ids],
                "method": "mujoco_mesh_object_body_world_pose",
            }
        )
        return result
    except Exception as exc:
        return {
            "status": "invalid",
            "available": False,
            "reason": f"evaluation_failed:{type(exc).__name__}:{exc}",
            "scene": str(scene),
            "mjwp_trajectory": str(mjwp_trajectory),
            "reference_trajectory": str(reference_trajectory),
        }
