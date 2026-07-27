from __future__ import annotations

import math
from typing import Iterable


def scale_bbox_xyxy(
    box: tuple[float, float, float, float],
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("bbox source and target sizes must be positive")
    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    x1, y1, x2, y2 = box
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = max(0.0, min(1.0, fraction)) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def robust_projected_scale_fit(
    ratios: Iterable[float],
    *,
    min_frames: int = 5,
    min_multiplier: float = 0.20,
    max_multiplier: float = 5.0,
    max_p90_p10_ratio: float = 1.75,
    deadband_low: float = 0.85,
    deadband_high: float = 1.18,
) -> dict[str, object]:
    """Decide whether a single projected mesh-scale correction is trustworthy."""
    values = sorted(float(value) for value in ratios if math.isfinite(float(value)) and float(value) > 0.0)
    if len(values) < min_frames:
        return {
            "available": False,
            "applied": False,
            "reason": "insufficient_valid_frames",
            "valid_frames": len(values),
            "min_frames": min_frames,
        }

    median = _quantile(values, 0.5)
    p10 = _quantile(values, 0.1)
    p90 = _quantile(values, 0.9)
    spread = p90 / max(p10, 1e-9)
    summary: dict[str, object] = {
        "available": True,
        "valid_frames": len(values),
        "raw_multiplier_median": median,
        "raw_multiplier_p10": p10,
        "raw_multiplier_p90": p90,
        "p90_p10_ratio": spread,
        "max_p90_p10_ratio": max_p90_p10_ratio,
        "deadband": [deadband_low, deadband_high],
    }
    if spread > max_p90_p10_ratio:
        return {
            **summary,
            "applied": False,
            "reason": "unstable_projected_scale_ratio",
        }

    multiplier = min(max(median, min_multiplier), max_multiplier)
    summary["multiplier"] = multiplier
    summary["clipped"] = not math.isclose(multiplier, median, rel_tol=1e-12, abs_tol=1e-12)
    if deadband_low <= multiplier <= deadband_high:
        return {
            **summary,
            "applied": False,
            "reason": "within_scale_deadband",
        }
    return {
        **summary,
        "applied": True,
        "reason": "stable_projected_scale_mismatch",
    }
