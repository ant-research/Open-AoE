from __future__ import annotations

import math
from typing import Any


def assess_ego_retarget_input(
    *,
    scale_fit: dict[str, Any],
    visual_interaction: dict[str, Any],
    geometry_interaction: dict[str, Any],
    hand_projection: dict[str, Any],
    selected_hand: str | None,
    hoi_contact_alignment: dict[str, Any],
    min_visual_overlap_frames: float = 3.0,
    min_surface_contact_frames: int = 3,
    max_surface_contact_distance_m: float = 0.05,
    min_projection_inside_fraction: float = 0.25,
    min_projection_positive_depth_fraction: float = 0.95,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if not scale_fit.get("available"):
        if scale_fit.get("reason") == "object_scale_fit_disabled_preserve_source":
            warnings.append("object_scale_fit_disabled_preserve_source")
        else:
            errors.append(f"object_scale_qc_unavailable:{scale_fit.get('reason')}")
    elif scale_fit.get("reason") == "unstable_projected_scale_ratio":
        errors.append("object_scale_ratio_temporally_unstable")
    if scale_fit.get("applied") or scale_fit.get("mesh_modified"):
        errors.append("object_only_mesh_scale_refinement_applied")

    visual_scores = visual_interaction.get("scores") or {}
    overlap_by_hand = {
        side: float((visual_scores.get(side) or {}).get("overlap_frames", 0.0) or 0.0)
        for side in ("left", "right")
    }
    if selected_hand == "bimanual":
        overlap_frames = min(overlap_by_hand.values())
        weak_sides = [
            side for side, value in overlap_by_hand.items()
            if value < min_visual_overlap_frames
        ]
        if weak_sides:
            errors.append(
                "weak_visual_bimanual_overlap:"
                + ",".join(f"{side}:{overlap_by_hand[side]:.0f}" for side in weak_sides)
            )
    else:
        overlap_frames = overlap_by_hand.get(str(selected_hand), 0.0)
    if selected_hand not in {"left", "right", "bimanual"}:
        errors.append(f"interaction_hand_not_single:{selected_hand}")
    elif selected_hand in {"left", "right"} and overlap_frames < min_visual_overlap_frames:
        errors.append(
            f"weak_visual_hand_object_overlap:{selected_hand}:{overlap_frames:.0f}"
        )

    selected_sides = (
        ["left", "right"]
        if selected_hand == "bimanual"
        else [str(selected_hand)]
        if selected_hand in {"left", "right"}
        else []
    )
    geometry_available = bool(geometry_interaction.get("available"))
    geometry_threshold = geometry_interaction.get("distance_thresh")
    try:
        geometry_threshold_m = float(geometry_threshold)
    except (TypeError, ValueError):
        geometry_threshold_m = None
    geometry_scores = geometry_interaction.get("scores") or {}
    geometry_coverage = geometry_interaction.get("coverage_by_hand") or {}
    try:
        geometry_source_frame_count = int(geometry_interaction.get("source_frame_count"))
    except (TypeError, ValueError):
        geometry_source_frame_count = 0
    surface_contact_frames_by_hand: dict[str, int] = {}
    surface_evaluated_frames_by_hand: dict[str, int] = {}
    for side in ("left", "right"):
        try:
            contact_value = float(
                (geometry_scores.get(side) or {}).get("in_hand_frames", 0.0)
                or 0.0
            )
            evaluated_value = float(
                (geometry_scores.get(side) or {}).get("frames", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            contact_value = 0.0
            evaluated_value = 0.0
        surface_contact_frames_by_hand[side] = (
            int(round(contact_value))
            if math.isfinite(contact_value) and contact_value >= 0.0
            else 0
        )
        surface_evaluated_frames_by_hand[side] = (
            int(round(evaluated_value))
            if math.isfinite(evaluated_value) and evaluated_value >= 0.0
            else 0
        )
    geometry_selected_hand = geometry_interaction.get("selected_hand")
    if not geometry_available:
        errors.append(
            "surface_contact_qc_unavailable:"
            + str(geometry_interaction.get("error") or geometry_interaction.get("reason") or "unknown")
        )
    elif geometry_interaction.get("sampled_all_frames") is not True:
        errors.append("surface_contact_qc_not_full_frame_coverage")
    elif geometry_threshold_m is None or not math.isfinite(geometry_threshold_m):
        errors.append("surface_contact_qc_invalid_distance_threshold")
    elif geometry_threshold_m > max_surface_contact_distance_m + 1e-12:
        errors.append(
            "surface_contact_qc_threshold_too_loose:"
            f"{geometry_threshold_m:.6f}>{max_surface_contact_distance_m:.6f}"
        )
    else:
        for side in selected_sides:
            coverage = geometry_coverage.get(side)
            if not isinstance(coverage, dict):
                errors.append(f"surface_contact_qc_coverage_missing:{side}")
                continue
            try:
                planned_frames = int(coverage.get("planned_frame_count"))
                evaluated_frames = int(coverage.get("evaluated_frame_count"))
                failed_frames = int(coverage.get("failed_frame_count"))
                coverage_source_frames = int(coverage.get("source_frame_count"))
            except (TypeError, ValueError):
                errors.append(f"surface_contact_qc_coverage_invalid:{side}")
                continue
            failure_reason_counts = coverage.get("failure_reasons")
            try:
                recorded_failure_total = sum(
                    int(value)
                    for value in failure_reason_counts.values()
                )
            except (AttributeError, TypeError, ValueError):
                recorded_failure_total = -1
            complete_coverage = bool(
                coverage.get("complete_source_coverage") is True
                and geometry_source_frame_count > 0
                and coverage_source_frames == geometry_source_frame_count
                and planned_frames == geometry_source_frame_count
                and evaluated_frames == geometry_source_frame_count
                and failed_frames == 0
                and recorded_failure_total == failed_frames
                and evaluated_frames
                == surface_evaluated_frames_by_hand[side]
            )
            if not complete_coverage:
                errors.append(
                    "surface_contact_qc_incomplete_evaluation:"
                    f"{side}:evaluated={evaluated_frames}/"
                    f"{geometry_source_frame_count}:failed={failed_frames}:"
                    f"planned={planned_frames}"
                )
            contact_frames = surface_contact_frames_by_hand[side]
            if contact_frames < min_surface_contact_frames:
                errors.append(
                    "insufficient_surface_contact_frames:"
                    f"{side}:{contact_frames}<{min_surface_contact_frames}"
                    f"@{geometry_threshold_m:.3f}m"
                )

        if selected_hand in {"left", "right"} and geometry_selected_hand in {"left", "right"}:
            selected_side = str(selected_hand)
            other_side = "right" if selected_side == "left" else "left"
            selected_contacts = surface_contact_frames_by_hand[selected_side]
            other_contacts = surface_contact_frames_by_hand[other_side]
            obvious_opposite_dominance = (
                geometry_selected_hand == other_side
                and other_contacts >= min_surface_contact_frames
                and (
                    selected_contacts < min_surface_contact_frames
                    or other_contacts >= max(min_surface_contact_frames, 2 * max(selected_contacts, 1))
                )
            )
            if obvious_opposite_dominance:
                errors.append(
                    "selected_hand_visual_geometry_conflict:"
                    f"selected={selected_side},geometry={other_side},"
                    f"contacts={selected_contacts}:{other_contacts}"
                )

    projection_reports = hand_projection.get("sides") or {}
    projection_inside_fraction_median_by_hand: dict[str, float | None] = {}
    projection_inside_fraction_min_by_hand: dict[str, float | None] = {}
    projection_inside_fraction_max_by_hand: dict[str, float | None] = {}
    projection_positive_depth_fraction_min_by_hand: dict[str, float | None] = {}
    for side in selected_sides:
        report = projection_reports.get(side) or {}
        if not report.get("available"):
            projection_inside_fraction_median_by_hand[side] = None
            projection_inside_fraction_min_by_hand[side] = None
            projection_inside_fraction_max_by_hand[side] = None
            projection_positive_depth_fraction_min_by_hand[side] = None
            errors.append(
                "hand_projection_qc_unavailable:"
                f"{side}:{report.get('reason') or 'unknown'}"
            )
            continue
        frames = report.get("frames") or []
        expected_frame_count = report.get("expected_frame_count")
        complete_sample_coverage = bool(
            report.get("anchor_hand") == side
            and report.get("sampling_policy") == "start_middle_end"
            and expected_frame_count == 3
            and report.get("available_frame_count") == 3
            and report.get("complete_sample_coverage") is True
            and len(frames) == 3
            and all(frame.get("available") is True for frame in frames)
        )
        if not complete_sample_coverage:
            errors.append(
                "hand_projection_qc_incomplete_start_middle_end:"
                f"{side}:available={report.get('available_frame_count')}/"
                f"{expected_frame_count}"
            )
        inside_values = []
        positive_depth_values = []
        for frame in frames:
            if not frame.get("available") or frame.get("inside_fraction") is None:
                continue
            try:
                inside_fraction = float(frame.get("inside_fraction"))
                positive_depth_fraction = float(
                    frame.get("positive_depth_fraction")
                )
            except (TypeError, ValueError):
                continue
            if math.isfinite(inside_fraction) and math.isfinite(
                positive_depth_fraction
            ):
                inside_values.append(inside_fraction)
                positive_depth_values.append(positive_depth_fraction)
        if len(inside_values) != 3 or len(positive_depth_values) != 3:
            errors.append(
                "hand_projection_qc_incomplete_measurements:"
                f"{side}:{len(inside_values)}/3"
            )
        if not inside_values:
            projection_inside_fraction_median_by_hand[side] = None
            projection_inside_fraction_min_by_hand[side] = None
            projection_inside_fraction_max_by_hand[side] = None
            projection_positive_depth_fraction_min_by_hand[side] = None
            errors.append(f"hand_projection_qc_no_valid_frames:{side}")
            continue
        inside_values.sort()
        middle = len(inside_values) // 2
        if len(inside_values) % 2:
            inside_median = inside_values[middle]
        else:
            inside_median = (inside_values[middle - 1] + inside_values[middle]) * 0.5
        projection_inside_fraction_median_by_hand[side] = inside_median
        inside_min = inside_values[0]
        projection_inside_fraction_min_by_hand[side] = inside_min
        inside_max = inside_values[-1]
        projection_inside_fraction_max_by_hand[side] = inside_max
        positive_depth_min = min(positive_depth_values)
        projection_positive_depth_fraction_min_by_hand[side] = positive_depth_min
        if positive_depth_min < min_projection_positive_depth_fraction:
            errors.append(
                "hand_projection_insufficient_positive_depth:"
                f"{side}:{positive_depth_min:.6f}<"
                f"{min_projection_positive_depth_fraction:.6f}"
            )
        if inside_min < min_projection_inside_fraction:
            errors.append(
                "hand_projection_mostly_outside_image:"
                f"{side}:{inside_min:.6f}<{min_projection_inside_fraction:.6f}"
            )

    hoi_reason = str(hoi_contact_alignment.get("reason") or "")
    if hoi_reason == "offset_exceeds_max":
        errors.append("hoi_coordinate_offset_exceeds_max")
    elif hoi_contact_alignment.get("applied"):
        errors.append("object_only_hoi_translation_applied")
    elif hoi_contact_alignment.get("would_apply"):
        warnings.append("object_only_hoi_translation_rejected_diagnostic")

    return {
        "status": "invalid" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "selected_hand": selected_hand,
        "visual_overlap_frames": overlap_frames,
        "visual_overlap_by_hand": overlap_by_hand,
        "min_visual_overlap_frames": min_visual_overlap_frames,
        "surface_contact_distance_threshold_m": geometry_threshold_m,
        "max_surface_contact_distance_m": max_surface_contact_distance_m,
        "surface_contact_frames_by_hand": surface_contact_frames_by_hand,
        "surface_evaluated_frames_by_hand": surface_evaluated_frames_by_hand,
        "min_surface_contact_frames": min_surface_contact_frames,
        "surface_contact_geometry_selected_hand": geometry_selected_hand,
        "projection_inside_fraction_median_by_hand": projection_inside_fraction_median_by_hand,
        "projection_inside_fraction_min_by_hand": projection_inside_fraction_min_by_hand,
        "projection_inside_fraction_max_by_hand": projection_inside_fraction_max_by_hand,
        "projection_positive_depth_fraction_min_by_hand": projection_positive_depth_fraction_min_by_hand,
        "min_projection_inside_fraction": min_projection_inside_fraction,
        "min_projection_positive_depth_fraction": min_projection_positive_depth_fraction,
        "scale_fit_reason": scale_fit.get("reason"),
        "scale_multiplier": scale_fit.get("multiplier"),
        "scale_p90_p10_ratio": scale_fit.get("p90_p10_ratio"),
        "hoi_alignment_reason": hoi_reason,
        "hoi_offset_norm": hoi_contact_alignment.get("offset_norm"),
    }
