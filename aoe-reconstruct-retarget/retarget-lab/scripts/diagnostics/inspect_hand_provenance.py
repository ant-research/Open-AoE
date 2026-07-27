#!/usr/bin/env python3
"""Classify and validate the hand geometry consumed by a retarget cell."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def scalar_text(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data:
        return ""
    value = np.asarray(data[key])
    if value.size != 1:
        return ""
    return str(value.reshape(-1)[0])


def find_hand_npz(raw_dir: Path) -> Path:
    candidates = sorted(raw_dir.glob("*/all_hand_meshes.npz"))
    if not candidates:
        candidates = sorted(raw_dir.glob("all_hand_meshes.npz"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one all_hand_meshes.npz below {raw_dir}, found {len(candidates)}"
        )
    return candidates[0].resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(raw_dir: Path, requested_source: str) -> dict[str, object]:
    hand_npz = find_hand_npz(raw_dir)
    with np.load(hand_npz, allow_pickle=False) as data:
        source_hands_npz = scalar_text(data, "source_hands_npz")
        geometry_source = scalar_text(data, "hand_geometry_source")
        pose_space = scalar_text(data, "pose_space")
        camera_space_source = scalar_text(data, "camera_space_source")
        camera_space_mode = scalar_text(data, "camera_space_mode")
        frame_ids = np.asarray(data["source_frame_ids"]) if "source_frame_ids" in data else np.empty(0)

    geometry_lower = geometry_source.lower()
    source_lower = source_hands_npz.lower()
    if "egoinfinity" in geometry_lower:
        actual_source = "egoinfinity_estimated"
    elif "ego_hands_reconstruction" in source_lower and source_lower.endswith("hands.npz"):
        actual_source = "aoe_hawor"
    elif geometry_source or source_hands_npz:
        actual_source = "external_estimated"
    else:
        actual_source = "unknown"

    expected = {
        "aoe": {"aoe_hawor"},
        "estimated": {"egoinfinity_estimated", "external_estimated"},
    }[requested_source]
    errors = []
    if actual_source not in expected:
        errors.append(
            f"requested hand_source={requested_source}, but NPZ geometry classifies as {actual_source}"
        )

    return {
        "status": "ok" if not errors else "invalid",
        "requested_hand_source": requested_source,
        "actual_hand_source": actual_source,
        "hand_npz": str(hand_npz),
        "hand_npz_sha256": sha256(hand_npz),
        "source_hands_npz": source_hands_npz or None,
        "hand_geometry_source": geometry_source or None,
        "pose_space": pose_space or None,
        "camera_space_source": camera_space_source or None,
        "camera_space_mode": camera_space_mode or None,
        "source_frame_count": int(frame_ids.size),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--requested-source", required=True, choices=["aoe", "estimated"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        report = inspect(args.raw_dir.expanduser().resolve(), args.requested_source)
    except Exception as exc:
        report = {
            "status": "invalid",
            "requested_hand_source": args.requested_source,
            "actual_hand_source": "unknown",
            "errors": [str(exc)],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 4


if __name__ == "__main__":
    raise SystemExit(main())
