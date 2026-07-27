#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np


def load_result(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def save_result(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def mask_centroid(obj: dict) -> np.ndarray | None:
    packed = obj.get("mask_packed")
    shape = obj.get("mask_shape")
    if packed is None or shape is None:
        return None
    h, w = [int(x) for x in shape]
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))[: h * w]
    mask = bits.reshape(h, w).astype(bool)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return np.array([(xs.min() + xs.max()) * 0.5, (ys.min() + ys.max()) * 0.5], dtype=np.float32)


def prompt_for(data: dict, oid) -> str:
    mapping = data.get("sam3_prompt_mapping") or []
    try:
        idx = int(oid)
    except Exception:
        return ""
    if 0 <= idx < len(mapping) and isinstance(mapping[idx], dict):
        return str(mapping[idx].get("prompt", ""))
    return ""


def prompt_score_for(data: dict, oid) -> float:
    mapping = data.get("sam3_prompt_mapping") or []
    try:
        idx = int(oid)
    except Exception:
        return 0.0
    if 0 <= idx < len(mapping) and isinstance(mapping[idx], dict):
        try:
            return float(mapping[idx].get("score", 0.0) or 0.0)
        except Exception:
            return 0.0
    return 0.0


def centroid_track(data: dict, oid) -> list[np.ndarray | None]:
    out = []
    for frame in data.get("frame_data") or []:
        sd = frame.get("sam3_obj_data") or {}
        obj = sd.get(oid) or sd.get(str(oid))
        out.append(mask_centroid(obj) if isinstance(obj, dict) else None)
    return out


def stable_prefix(centroids: list[np.ndarray | None], max_jump_px: float) -> tuple[int, float, list[dict]]:
    last = None
    max_jump = 0.0
    jumps = []
    for i, c in enumerate(centroids):
        if c is None:
            continue
        if last is not None:
            jump = float(np.linalg.norm(c - last))
            max_jump = max(max_jump, jump)
            jumps.append({"frame_idx": i, "jump_px": jump})
            if max_jump_px > 0 and jump > max_jump_px:
                return i, max_jump, jumps
        last = c
    return len(centroids), max_jump, jumps


def target_point_score(centroids: list[np.ndarray | None], target_point: tuple[float, float] | None) -> float:
    if target_point is None:
        return 0.0
    target = np.array(target_point, dtype=np.float32)
    distances = [float(np.linalg.norm(c - target)) for c in centroids if c is not None]
    if distances:
        return -float(np.median(distances))
    return -1e6


def parse_point(value: str | None) -> tuple[float, float] | None:
    if not value:
        return None
    vals = [float(x) for x in value.split(",")]
    if len(vals) != 2:
        raise ValueError(f"expected x,y target point, got {value!r}")
    return vals[0], vals[1]


def choose_objects(
    data: dict,
    target_prompt: str | None,
    target_point,
    max_jump_px: float,
    selection_mode: str,
) -> tuple[set, list[dict]]:
    mesh_info = data.get("sam3_mesh_info") or {}
    rows = []
    for oid, info in mesh_info.items():
        prompt = prompt_for(data, oid)
        if target_prompt and prompt != target_prompt:
            continue
        centroids = centroid_track(data, oid)
        prefix_len, max_jump, jumps = stable_prefix(centroids, max_jump_px)
        valid = sum(c is not None for c in centroids)
        rows.append(
            {
                "oid": int(oid),
                "prompt": prompt,
                "prompt_score": float(prompt_score_for(data, oid)),
                "valid_masks": int(valid),
                "stable_prefix_len": int(prefix_len),
                "max_jump_px": float(max_jump),
                "target_point_score": float(target_point_score(centroids, target_point)),
                "n_points": int(info.get("n_points", 0) or 0),
                "jumps": jumps,
            }
        )
    if not rows:
        return set(), rows
    if target_point is None:
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                r["stable_prefix_len"],
                r["prompt_score"],
                r["valid_masks"],
                -r["max_jump_px"],
                r["n_points"],
            ),
            reverse=True,
        )
    else:
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                r["target_point_score"],
                r["stable_prefix_len"],
                r["prompt_score"],
                r["valid_masks"],
                -r["max_jump_px"],
                r["n_points"],
            ),
            reverse=True,
        )
    for rank, row in enumerate(rows_sorted):
        row["selection_rank"] = rank
    if target_prompt and selection_mode == "best":
        return {rows_sorted[0]["oid"]}, rows
    return {row["oid"] for row in rows if row["valid_masks"] > 0 and row["stable_prefix_len"] > 0}, rows


def normalize_oid_keys(mapping: dict) -> dict:
    out = {}
    for k, v in mapping.items():
        try:
            out[int(k)] = v
        except Exception:
            out[k] = v
    return out


def apply_filter(data: dict, keep_oids: set[int], max_jump_px: float, freeze_after_drift: bool) -> dict:
    data = dict(data)
    mesh_info = normalize_oid_keys(data.get("sam3_mesh_info") or {})
    pose_info = normalize_oid_keys(data.get("pose_track_info") or {})
    frame_data = data.get("frame_data") or []

    removed = sorted([int(oid) for oid in mesh_info.keys() if int(oid) not in keep_oids])
    data["sam3_mesh_info"] = {oid: info for oid, info in mesh_info.items() if int(oid) in keep_oids}
    data["pose_track_info"] = {oid: info for oid, info in pose_info.items() if int(oid) in keep_oids}

    for frame in frame_data:
        sd = frame.get("sam3_obj_data")
        if not isinstance(sd, dict):
            continue
        for oid in removed:
            sd.pop(oid, None)
            sd.pop(str(oid), None)

    repair = {}
    if freeze_after_drift:
        for oid in keep_oids:
            info = data["pose_track_info"].get(oid) or data["pose_track_info"].get(str(oid))
            if not isinstance(info, dict) or info.get("T_seq") is None:
                continue
            centroids = centroid_track(data, oid)
            prefix_len, max_jump, jumps = stable_prefix(centroids, max_jump_px)
            T_seq = np.asarray(info["T_seq"], dtype=np.float32).copy()
            if 0 < prefix_len < len(T_seq):
                T_seq[prefix_len:] = T_seq[prefix_len - 1]
                info["T_seq"] = T_seq
                info["tracking_status"] = "drift_repaired_freeze"
                info["drift_repair"] = {
                    "method": "freeze_after_centroid_jump",
                    "valid_until_frame": int(prefix_len - 1),
                    "max_centroid_jump_px": float(max_jump),
                    "jump_threshold_px": float(max_jump_px),
                }
            repair[int(oid)] = {
                "stable_prefix_len": int(prefix_len),
                "max_centroid_jump_px": float(max_jump),
                "jumps": jumps,
            }

    data["egoinfinity_object_filter"] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "keep_oids": sorted(int(x) for x in keep_oids),
        "removed_oids": removed,
        "max_jump_px": float(max_jump_px),
        "freeze_after_drift": bool(freeze_after_drift),
        "repair": repair,
    }
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter EgoInfinity duplicate/spurious objects and repair centroid-jump drift.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--target-prompt", default=None)
    parser.add_argument("--target-point", default=None)
    parser.add_argument(
        "--selection-mode",
        choices=["all", "best"],
        default="all",
        help="Default all: keep every object matching target-prompt. best preserves the legacy single-object behavior.",
    )
    parser.add_argument("--max-centroid-jump-px", type=float, default=320.0)
    parser.add_argument("--no-freeze-after-drift", action="store_true")
    parser.add_argument("--backup", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    out_path = args.output or args.input
    data = load_result(args.input)
    target_point = parse_point(args.target_point)
    keep, rows = choose_objects(
        data,
        args.target_prompt,
        target_point,
        args.max_centroid_jump_px,
        args.selection_mode,
    )
    if args.target_prompt and not keep:
        raise RuntimeError(f"No EgoInfinity object matched target prompt {args.target_prompt!r}")

    filtered = apply_filter(
        data,
        keep_oids=keep,
        max_jump_px=args.max_centroid_jump_px,
        freeze_after_drift=not args.no_freeze_after_drift,
    )
    if args.backup and out_path == args.input:
        backup = args.input.with_suffix(args.input.suffix + ".pre_object_filter")
        shutil.copy2(args.input, backup)
    save_result(out_path, filtered)

    report = {
        "input": str(args.input),
        "output": str(out_path),
        "target_prompt": args.target_prompt,
        "target_point": target_point,
        "selection_mode": args.selection_mode,
        "keep_oids": sorted(int(x) for x in keep),
        "candidates": rows,
        "filter": filtered.get("egoinfinity_object_filter"),
    }
    report_path = args.report or (out_path.parent / "object_filter_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
