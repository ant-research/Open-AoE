from __future__ import annotations

import math


def contact_fraction_hand(fractions: dict[str, float]) -> str | None:
    left = float(fractions.get("left", 0.0) or 0.0)
    right = float(fractions.get("right", 0.0) or 0.0)
    best = max(left, right)
    if best <= 0:
        return None
    if left >= 0.25 and right >= 0.25 and min(left, right) >= 0.60 * best:
        return "bimanual"
    return "left" if left >= right else "right"


def dominant_hand_from_geometry(
    fractions: dict[str, float],
    mean_min_distance: dict[str, float],
    stats: dict[str, dict[str, float]] | None = None,
) -> str | None:
    left_fraction = float(fractions.get("left", 0.0) or 0.0)
    right_fraction = float(fractions.get("right", 0.0) or 0.0)
    left_distance = float(mean_min_distance.get("left", math.inf) or math.inf)
    right_distance = float(mean_min_distance.get("right", math.inf) or math.inf)
    if right_fraction >= 0.35 and math.isfinite(right_distance) and right_distance < left_distance:
        if right_fraction >= left_fraction + 0.25 or right_distance <= left_distance * 0.35:
            if _is_complementary("right", fractions, stats) and not _is_decisively_closer("right", fractions, mean_min_distance):
                return None
            return "right"
    if left_fraction >= 0.35 and math.isfinite(left_distance) and left_distance < right_distance:
        if left_fraction >= right_fraction + 0.25 or left_distance <= right_distance * 0.35:
            if _is_complementary("left", fractions, stats) and not _is_decisively_closer("left", fractions, mean_min_distance):
                return None
            return "left"
    return None


def geometry_choice(
    geometry: dict,
    manifest_selected: str | None = None,
) -> dict:
    if not geometry.get("available"):
        return {
            "available": False,
            "reason": "missing_geometry_interaction",
            "manifest_selected": manifest_selected,
        }
    selected = geometry.get("selected_hand")
    fractions = geometry.get("contact_fraction") or {}
    distances = geometry.get("mean_min_distance") or {}
    scores = geometry.get("scores") or {}
    choice = {
        "available": True,
        "selected_hand": selected,
        "manifest_selected": manifest_selected,
        "contact_fraction": fractions,
        "mean_min_distance": distances,
        "reason": "geometry_not_strong",
    }
    if selected == "bimanual":
        dominant = dominant_hand_from_geometry(fractions, distances, scores)
        if dominant is not None:
            choice["selected_hand"] = dominant
            choice["reason"] = f"adapter-geometry-dominant-{dominant}"
            return choice
        left_fraction = float(fractions.get("left", 0.0) or 0.0)
        right_fraction = float(fractions.get("right", 0.0) or 0.0)
        if min(left_fraction, right_fraction) >= 0.35:
            if _complementary_either(fractions, scores):
                choice["reason"] = "adapter-geometry-bimanual-complementary-hands"
            else:
                choice["reason"] = "adapter-geometry-bimanual"
            return choice
        choice["selected_hand"] = None
        return choice
    if selected not in {"left", "right"}:
        choice["selected_hand"] = None
        return choice
    if single_hand_is_strong(selected, fractions, distances, scores):
        choice["reason"] = "adapter-geometry"
        return choice
    choice["selected_hand"] = None
    return choice


def single_hand_is_strong(
    selected: str,
    fractions: dict[str, float],
    distances: dict[str, float],
    scores: dict[str, dict[str, float]],
) -> bool:
    other = "right" if selected == "left" else "left"
    selected_fraction = float(fractions.get(selected, 0.0) or 0.0)
    other_fraction = float(fractions.get(other, 0.0) or 0.0)
    selected_distance = float(distances.get(selected, math.inf) or math.inf)
    other_distance = float(distances.get(other, math.inf) or math.inf)
    selected_score = float((scores.get(selected) or {}).get("score", 0.0) or 0.0)
    other_score = float((scores.get(other) or {}).get("score", 0.0) or 0.0)
    strong_contact = selected_fraction >= 0.35 and selected_fraction >= other_fraction + 0.20
    much_closer = math.isfinite(selected_distance) and selected_distance <= max(0.05, other_distance * 0.65)
    stronger_score = selected_score > 0 and selected_score >= other_score * 1.25
    return strong_contact or much_closer or stronger_score


def _is_complementary(
    dominant: str,
    fractions: dict[str, float],
    stats: dict[str, dict[str, float]] | None,
) -> bool:
    if not stats:
        return False
    other = "left" if dominant == "right" else "right"
    dominant_fraction = float(fractions.get(dominant, 0.0) or 0.0)
    other_fraction = float(fractions.get(other, 0.0) or 0.0)
    dominant_frames = float((stats.get(dominant) or {}).get("frames", 0.0) or 0.0)
    other_frames = float((stats.get(other) or {}).get("frames", 0.0) or 0.0)
    return other_fraction >= 0.70 and dominant_frames < other_frames * 0.75 and dominant_fraction >= 0.35


def _is_decisively_closer(
    dominant: str,
    fractions: dict[str, float],
    distances: dict[str, float],
) -> bool:
    other = "left" if dominant == "right" else "right"
    dominant_fraction = float(fractions.get(dominant, 0.0) or 0.0)
    dominant_distance = float(distances.get(dominant, math.inf) or math.inf)
    other_distance = float(distances.get(other, math.inf) or math.inf)
    return (
        dominant_fraction >= 0.75
        and math.isfinite(dominant_distance)
        and math.isfinite(other_distance)
        and dominant_distance <= other_distance * 0.20
    )


def _complementary_either(
    fractions: dict[str, float],
    stats: dict[str, dict[str, float]],
) -> bool:
    return _is_complementary("left", fractions, stats) or _is_complementary("right", fractions, stats)
