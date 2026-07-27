#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_geometry_contact(
    manifest: dict,
    *,
    min_contact_fraction: float,
    max_mean_distance_m: float,
) -> dict:
    geometry = manifest.get("geometry_interaction_hand") or {}
    selected = str(manifest.get("selected_hand") or "")
    sides = [selected] if selected in {"left", "right"} else ["left", "right"]
    fractions = geometry.get("contact_fraction") or {}
    distances = geometry.get("mean_min_distance") or {}
    selected_fractions = [float(fractions.get(side, 0.0) or 0.0) for side in sides]
    selected_distances = [float(distances.get(side, float("inf")) or float("inf")) for side in sides]
    errors: list[str] = []
    if geometry.get("available") is not True:
        errors.append("geometry_interaction_hand_unavailable")
    if max(selected_fractions, default=0.0) < min_contact_fraction:
        errors.append("no_geometric_hand_object_contact")
    if min(selected_distances, default=float("inf")) > max_mean_distance_m:
        errors.append("hand_object_geometry_separated")
    return {
        "status": "invalid" if errors else "ok",
        "errors": errors,
        "selected_hand": selected,
        "evaluated_sides": sides,
        "contact_fraction": {side: fractions.get(side) for side in sides},
        "mean_min_distance_m": {side: distances.get(side) for side in sides},
        "thresholds": {
            "min_contact_fraction": min_contact_fraction,
            "max_mean_distance_m": max_mean_distance_m,
        },
        "reason": "cross_source_hand_object_geometry_gate",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reject cross-source retarget adapters whose hand and object geometry are separated."
    )
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--min-contact-fraction", type=float, default=0.01)
    parser.add_argument("--max-mean-distance-m", type=float, default=0.25)
    args = parser.parse_args()

    manifest = json.loads(args.adapter_manifest.read_text(encoding="utf-8"))
    qc = validate_geometry_contact(
        manifest,
        min_contact_fraction=args.min_contact_fraction,
        max_mean_distance_m=args.max_mean_distance_m,
    )
    manifest["cross_source_geometry_qc"] = qc
    args.adapter_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return 0 if qc["status"] == "ok" else 4


if __name__ == "__main__":
    raise SystemExit(main())
