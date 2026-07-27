#!/usr/bin/env python3
"""Select a gravity frame that is valid for egocentric AoE clips."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

def gravity_vector(roll_deg: float, pitch_deg: float) -> list[float]:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    return [
        -math.sin(roll) * math.cos(pitch),
        -math.cos(roll) * math.cos(pitch),
        math.sin(pitch),
    ]


def angular_error_deg(vectors: list[list[float]], reference: list[float]) -> list[float]:
    reference_norm = math.sqrt(sum(value * value for value in reference))
    errors = []
    for vector in vectors:
        vector_norm = math.sqrt(sum(value * value for value in vector))
        dot = sum(a * b for a, b in zip(vector, reference)) / (vector_norm * reference_norm)
        errors.append(math.degrees(math.acos(max(-1.0, min(1.0, dot)))))
    return errors


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def frame_index(item: dict) -> int:
    stem = Path(str(item.get("path", ""))).stem
    try:
        return int(stem)
    except ValueError:
        return -1


def select_gravity(payload: dict, reference_frame: int, max_p95_deg: float) -> dict:
    samples = [
        item
        for item in payload.get("per_frame", [])
        if math.isfinite(float(item.get("roll_deg", math.nan)))
        and math.isfinite(float(item.get("pitch_deg", math.nan)))
    ]
    if not samples:
        raise ValueError("GeoCalib output has no finite per-frame gravity samples")

    aggregate = [float(value) for value in payload["vec3d"]]
    vectors = [
        gravity_vector(float(item["roll_deg"]), float(item["pitch_deg"]))
        for item in samples
    ]
    errors = angular_error_deg(vectors, aggregate)
    p95 = percentile(errors, 0.95)
    median = statistics.median(errors)
    dynamic = p95 > max_p95_deg

    selected = min(samples, key=lambda item: abs(frame_index(item) - reference_frame))
    selected_idx = frame_index(selected)
    original = {
        "vec3d": aggregate,
        "roll_deg": float(payload["roll_deg"]),
        "pitch_deg": float(payload["pitch_deg"]),
    }
    if dynamic:
        chosen = gravity_vector(float(selected["roll_deg"]), float(selected["pitch_deg"]))
        payload["vec3d"] = chosen
        payload["roll_deg"] = float(selected["roll_deg"])
        payload["pitch_deg"] = float(selected["pitch_deg"])

    confidences = [
        float(item["confidence"])
        for item in samples
        if math.isfinite(float(item.get("confidence", math.nan)))
    ]
    payload["gravity_quality"] = {
        "status": "dynamic_camera_reference_frame" if dynamic else "static_camera_aggregate",
        "selection_policy": (
            "nearest_object_reconstruction_reference_frame" if dynamic else "geocalib_aggregate"
        ),
        "reference_frame": int(reference_frame),
        "selected_frame": int(selected_idx) if dynamic else None,
        "sample_count": len(samples),
        "angular_error_median_deg": median,
        "angular_error_p95_deg": p95,
        "max_aggregate_p95_deg": float(max_p95_deg),
        "median_confidence": statistics.median(confidences) if confidences else None,
        "original_aggregate": original,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gravity-json", type=Path, required=True)
    parser.add_argument("--reference-frame", type=int, required=True)
    parser.add_argument("--max-aggregate-p95-deg", type=float, default=25.0)
    args = parser.parse_args()

    payload = json.loads(args.gravity_json.read_text(encoding="utf-8"))
    payload = select_gravity(payload, args.reference_frame, args.max_aggregate_p95_deg)
    args.gravity_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["gravity_quality"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
