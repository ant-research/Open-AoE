#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoe_retarget_lab.tracking_quality import (  # noqa: E402
    InteractionThresholds,
    TrackingThresholds,
    evaluate_mjwp_object_tracking,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MJWP object tracking using actual MuJoCo mesh body poses.")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--mjwp", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alignment-manifest", type=Path)
    parser.add_argument("--pos-median-threshold", type=float, default=0.05)
    parser.add_argument("--pos-p95-threshold", type=float, default=0.10)
    parser.add_argument("--rot-median-threshold", type=float, default=0.75)
    parser.add_argument("--rot-p95-threshold", type=float, default=2.0)
    parser.add_argument("--lost-pos-threshold", type=float, default=0.10)
    parser.add_argument("--lost-fraction-threshold", type=float, default=0.10)
    parser.add_argument("--hoi-reference-contact-threshold", type=float, default=0.05)
    parser.add_argument("--hoi-executed-contact-threshold", type=float, default=0.08)
    parser.add_argument("--hoi-distance-median-threshold", type=float, default=0.03)
    parser.add_argument("--hoi-distance-p95-threshold", type=float, default=0.08)
    parser.add_argument(
        "--hoi-relative-translation-median-threshold", type=float, default=0.03
    )
    parser.add_argument(
        "--hoi-relative-translation-p95-threshold", type=float, default=0.08
    )
    parser.add_argument("--hoi-min-reference-contact-frames", type=int, default=3)
    parser.add_argument("--hoi-min-contact-retention", type=float, default=0.50)
    args = parser.parse_args()

    result = evaluate_mjwp_object_tracking(
        args.scene.resolve(),
        args.mjwp.resolve(),
        args.reference.resolve(),
        TrackingThresholds(
            pos_median_m=args.pos_median_threshold,
            pos_p95_m=args.pos_p95_threshold,
            rot_median_rad=args.rot_median_threshold,
            rot_p95_rad=args.rot_p95_threshold,
            lost_pos_m=args.lost_pos_threshold,
            lost_fraction=args.lost_fraction_threshold,
        ),
        InteractionThresholds(
            reference_contact_m=args.hoi_reference_contact_threshold,
            executed_contact_m=args.hoi_executed_contact_threshold,
            distance_error_median_m=args.hoi_distance_median_threshold,
            distance_error_p95_m=args.hoi_distance_p95_threshold,
            relative_translation_error_median_m=(
                args.hoi_relative_translation_median_threshold
            ),
            relative_translation_error_p95_m=(
                args.hoi_relative_translation_p95_threshold
            ),
            min_reference_contact_frames=args.hoi_min_reference_contact_frames,
            min_contact_retention=args.hoi_min_contact_retention,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.alignment_manifest:
        manifest = {}
        if args.alignment_manifest.exists():
            manifest = json.loads(args.alignment_manifest.read_text(encoding="utf-8"))
        manifest["object_tracking_quality"] = result
        manifest["object_tracking_quality_file"] = str(args.output.resolve())
        args.alignment_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ok" else 4


if __name__ == "__main__":
    raise SystemExit(main())
