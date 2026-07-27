#!/usr/bin/env python3
"""Validate production SAM3 prompt provenance before expensive mesh reconstruction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_MODE = "semantic_text_with_bbox_instance_selection"


def parse_bbox(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def bbox_iou(first: list[float], second: list[float]) -> float:
    if len(first) != 4 or len(second) != 4:
        return 0.0
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def bbox_intersection_over_first(first: list[float], second: list[float]) -> float:
    """Return how much of ``first`` is contained by ``second``.

    AoE action annotations are interaction regions rather than tight object
    boxes.  Comparing mask area with the action-box area therefore rejects
    legitimate small objects (for example the red paint pot).  Containment
    still rejects a SAM3 candidate outside the annotated interaction region
    without assuming the annotation is a tight segmentation box.
    """
    if len(first) != 4 or len(second) != 4:
        return 0.0
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    return intersection / first_area if first_area > 0 else 0.0


def validate_prompt_qc(
    qc: dict[str, Any],
    expected_text: str,
    expected_bbox: list[float],
    min_mask_to_bbox_area_ratio: float = 0.25,
    min_selected_bbox_containment: float = 0.50,
    min_multi_candidate_bbox_iou: float = 0.01,
    max_temporal_centroid_jump_px: float = 100.0,
    max_consecutive_mask_area_ratio: float = 2.0,
    max_mask_to_bbox_area_ratio_for_auto_review: float = 2.0,
    min_bbox_iou_for_auto_review: float = 0.10,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    actual_text = qc.get("text")
    actual_bbox = qc.get("target_bbox") or []
    selected_id = qc.get("selected_sam3_obj_id")
    prompt_frame_idx = qc.get("prompt_frame_idx")
    selected_candidate = next(
        (
            row
            for row in (qc.get("prompt_candidates") or [])
            if row.get("sam3_obj_id") == selected_id
        ),
        None,
    )
    prompt_frame_stat = next(
        (
            row
            for row in (qc.get("frame_stats") or [])
            if row.get("frame_idx") == prompt_frame_idx
        ),
        None,
    )
    target_bbox_area = 0.0
    if len(expected_bbox) == 4:
        target_bbox_area = max(0.0, expected_bbox[2] - expected_bbox[0]) * max(
            0.0, expected_bbox[3] - expected_bbox[1]
        )
    selected_mask_pixels = int((selected_candidate or {}).get("mask_pixels") or 0)
    prompt_candidate_count = len(qc.get("prompt_candidates") or [])
    selected_bbox_iou = bbox_iou(
        list((selected_candidate or {}).get("bbox_xyxy") or []), expected_bbox
    )
    selected_bbox_containment = bbox_intersection_over_first(
        list((selected_candidate or {}).get("bbox_xyxy") or []), expected_bbox
    )
    selected_bbox = list((selected_candidate or {}).get("bbox_xyxy") or [])
    selected_centroid_inside_annotation = False
    if len(selected_bbox) == 4 and len(expected_bbox) == 4:
        selected_center_x = (float(selected_bbox[0]) + float(selected_bbox[2])) * 0.5
        selected_center_y = (float(selected_bbox[1]) + float(selected_bbox[3])) * 0.5
        selected_centroid_inside_annotation = (
            float(expected_bbox[0]) <= selected_center_x <= float(expected_bbox[2])
            and float(expected_bbox[1]) <= selected_center_y <= float(expected_bbox[3])
        )
    mask_to_bbox_area_ratio = (
        selected_mask_pixels / target_bbox_area if target_bbox_area > 0 else 0.0
    )
    frame_stats = sorted(
        (
            row
            for row in (qc.get("frame_stats") or [])
            if row.get("valid") and int(row.get("mask_pixels") or 0) > 0
        ),
        key=lambda row: int(row.get("frame_idx") or 0),
    )
    consecutive_area_ratios: list[dict[str, Any]] = []
    for previous, current in zip(frame_stats, frame_stats[1:]):
        previous_index = int(previous.get("frame_idx") or 0)
        current_index = int(current.get("frame_idx") or 0)
        if current_index != previous_index + 1:
            continue
        previous_area = float(previous.get("mask_pixels") or 0)
        current_area = float(current.get("mask_pixels") or 0)
        ratio = max(previous_area / current_area, current_area / previous_area)
        consecutive_area_ratios.append(
            {
                "from": previous_index,
                "to": current_index,
                "ratio": ratio,
            }
        )
    worst_area_change = max(
        consecutive_area_ratios,
        key=lambda row: float(row["ratio"]),
        default=None,
    )
    max_centroid_jump_px = float(qc.get("max_centroid_jump_px") or 0.0)

    if not qc.get("qa_pass"):
        errors.append(f"SAM3 temporal QA failed: {qc.get('fail_reasons')}")
    if (actual_text or "").strip().casefold() != expected_text.strip().casefold():
        errors.append(f"SAM3 text prompt={actual_text!r}, expected {expected_text!r}")
    if len(expected_bbox) != 4:
        errors.append(f"fresh run requires a transformed target bbox, got {expected_bbox}")
    elif len(actual_bbox) != 4 or any(
        not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-3)
        for actual, expected in zip(actual_bbox, expected_bbox)
    ):
        errors.append(f"SAM3 target_bbox={actual_bbox}, expected {expected_bbox}")
    if qc.get("target_point") is not None:
        errors.append(f"production SAM3 unexpectedly used target_point={qc.get('target_point')}")
    if selected_id is None:
        errors.append("SAM3 did not select a prompt-frame object")
    if not selected_candidate or int(selected_candidate.get("mask_pixels") or 0) <= 0:
        errors.append(f"selected SAM3 candidate is missing or empty: {selected_candidate}")
    elif (
        selected_bbox_containment < min_selected_bbox_containment
        and not selected_centroid_inside_annotation
    ):
        errors.append(
            "selected SAM3 mask lies outside the annotated interaction region: "
            f"bbox_containment={selected_bbox_containment:.3f} < "
            f"{min_selected_bbox_containment:.3f}"
        )
    elif mask_to_bbox_area_ratio < min_mask_to_bbox_area_ratio:
        warnings.append(
            "manual_visual_review: selected SAM3 mask is small relative to the "
            "annotation interaction region: "
            f"ratio={mask_to_bbox_area_ratio:.3f} < {min_mask_to_bbox_area_ratio:.3f}"
        )
    if prompt_candidate_count > 1 and selected_bbox_iou < min_multi_candidate_bbox_iou:
        errors.append(
            "multiple SAM3 prompt candidates exist but the selected mask does not overlap the "
            f"annotation bbox: iou={selected_bbox_iou:.3f} < {min_multi_candidate_bbox_iou:.3f}"
        )
    if mask_to_bbox_area_ratio > max_mask_to_bbox_area_ratio_for_auto_review:
        warnings.append(
            "manual_visual_review: selected SAM3 mask is much larger than the annotation bbox: "
            f"ratio={mask_to_bbox_area_ratio:.3f} > "
            f"{max_mask_to_bbox_area_ratio_for_auto_review:.3f}"
        )
    if selected_candidate and selected_bbox_iou < min_bbox_iou_for_auto_review:
        warnings.append(
            "manual_visual_review: selected SAM3 mask has weak annotation-bbox overlap: "
            f"iou={selected_bbox_iou:.3f} < {min_bbox_iou_for_auto_review:.3f}"
        )
    if not prompt_frame_stat or not prompt_frame_stat.get("valid"):
        errors.append(f"SAM3 prompt frame has no final written mask: {prompt_frame_stat}")
    elif int(prompt_frame_stat.get("mask_pixels") or 0) <= 0:
        errors.append(f"SAM3 prompt frame final mask is empty: {prompt_frame_stat}")
    if len(frame_stats) >= 8 and max_centroid_jump_px > max_temporal_centroid_jump_px:
        errors.append(
            "SAM3 mask has a large temporal centroid jump consistent with target switching: "
            f"{max_centroid_jump_px:.1f}px > {max_temporal_centroid_jump_px:.1f}px"
        )
    if (
        len(frame_stats) >= 8
        and worst_area_change is not None
        and float(worst_area_change["ratio"]) > max_consecutive_mask_area_ratio
    ):
        errors.append(
            "SAM3 mask area changes too abruptly between consecutive frames: "
            f"frames {worst_area_change['from']}->{worst_area_change['to']} "
            f"ratio={worst_area_change['ratio']:.3f} > {max_consecutive_mask_area_ratio:.3f}"
        )

    return {
        "status": "ok" if not errors else "failed",
        "required_mode": REQUIRED_MODE,
        "expected_text": expected_text,
        "actual_text": actual_text,
        "expected_bbox": expected_bbox,
        "actual_bbox": actual_bbox,
        "selected_sam3_obj_id": selected_id,
        "selected_candidate": selected_candidate,
        "prompt_frame_idx": prompt_frame_idx,
        "prompt_frame_stat": prompt_frame_stat,
        "target_bbox_area": target_bbox_area,
        "selected_mask_pixels": selected_mask_pixels,
        "mask_to_bbox_area_ratio": mask_to_bbox_area_ratio,
        "min_mask_to_bbox_area_ratio": min_mask_to_bbox_area_ratio,
        "max_mask_to_bbox_area_ratio_for_auto_review": max_mask_to_bbox_area_ratio_for_auto_review,
        "prompt_candidate_count": prompt_candidate_count,
        "selected_bbox_iou": selected_bbox_iou,
        "selected_bbox_containment": selected_bbox_containment,
        "selected_centroid_inside_annotation": selected_centroid_inside_annotation,
        "min_selected_bbox_containment": min_selected_bbox_containment,
        "min_multi_candidate_bbox_iou": min_multi_candidate_bbox_iou,
        "min_bbox_iou_for_auto_review": min_bbox_iou_for_auto_review,
        "temporal_valid_frame_count": len(frame_stats),
        "max_centroid_jump_px": max_centroid_jump_px,
        "max_temporal_centroid_jump_px": max_temporal_centroid_jump_px,
        "worst_consecutive_mask_area_change": worst_area_change,
        "max_consecutive_mask_area_ratio": max_consecutive_mask_area_ratio,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-text", required=True)
    parser.add_argument("--expected-bbox", required=True)
    parser.add_argument("--min-mask-to-bbox-area-ratio", type=float, default=0.25)
    parser.add_argument("--min-selected-bbox-containment", type=float, default=0.50)
    parser.add_argument("--min-multi-candidate-bbox-iou", type=float, default=0.01)
    parser.add_argument("--max-temporal-centroid-jump-px", type=float, default=100.0)
    parser.add_argument("--max-consecutive-mask-area-ratio", type=float, default=2.0)
    parser.add_argument("--max-mask-to-bbox-area-ratio-for-auto-review", type=float, default=2.0)
    parser.add_argument("--min-bbox-iou-for-auto-review", type=float, default=0.10)
    args = parser.parse_args()

    if not args.qc.exists():
        raise SystemExit(f"missing SAM3 QC summary: {args.qc}")
    qc = json.loads(args.qc.read_text(encoding="utf-8"))
    payload = validate_prompt_qc(
        qc,
        args.expected_text,
        parse_bbox(args.expected_bbox),
        min_mask_to_bbox_area_ratio=args.min_mask_to_bbox_area_ratio,
        min_selected_bbox_containment=args.min_selected_bbox_containment,
        min_multi_candidate_bbox_iou=args.min_multi_candidate_bbox_iou,
        max_temporal_centroid_jump_px=args.max_temporal_centroid_jump_px,
        max_consecutive_mask_area_ratio=args.max_consecutive_mask_area_ratio,
        max_mask_to_bbox_area_ratio_for_auto_review=(
            args.max_mask_to_bbox_area_ratio_for_auto_review
        ),
        min_bbox_iou_for_auto_review=args.min_bbox_iou_for_auto_review,
    )
    payload["mask_qc_summary"] = str(args.qc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if payload["errors"]:
        raise SystemExit("SAM3 production prompt gate failed: " + "; ".join(payload["errors"]))
    print(f"SAM3 production prompt gate passed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
