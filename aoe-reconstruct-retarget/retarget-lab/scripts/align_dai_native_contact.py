#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


POSE_TO_CAMERA_ROW = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) >= 4:
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (3,) or len(array) < 4:
        raise ValueError(f"object mesh has too few vertices: {path}")
    return array


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def subsample(points: np.ndarray, limit: int = 768) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) <= limit:
        return points
    return points[np.linspace(0, len(points) - 1, limit, dtype=np.int64)]


def nearest_offset(hand: np.ndarray, obj: np.ndarray) -> tuple[np.ndarray, float]:
    best_distance = float("inf")
    best_offset = None
    for start in range(0, len(hand), 256):
        diff = hand[start : start + 256, None, :] - obj[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        flat = int(np.argmin(dist2))
        hand_idx, obj_idx = np.unravel_index(flat, dist2.shape)
        distance = float(np.sqrt(dist2[hand_idx, obj_idx]))
        if distance < best_distance:
            best_distance = distance
            best_offset = diff[hand_idx, obj_idx]
    if best_offset is None:
        raise ValueError("empty hand or object geometry")
    return best_offset, best_distance


def layout_entries(layout: dict) -> list[dict]:
    objects = layout.get("objects") or []
    if isinstance(objects, dict):
        objects = list(objects.values())
    entries = [row for row in objects if isinstance(row, dict)]
    entries.sort(key=lambda row: int(row.get("frame_idx", row.get("index", 0)) or 0))
    return entries


def transformed_object(mesh: np.ndarray, entry: dict, mesh_scale: float) -> np.ndarray:
    local = entry.get("local_to_scene") or {}
    rotation = quat_wxyz_to_matrix(
        np.asarray(local.get("quat_wxyz_camera_frame") or [1, 0, 0, 0], dtype=np.float64)
    )
    translation = np.asarray(local.get("translation_camera_frame"), dtype=np.float64).reshape(3)
    return (rotation @ (mesh * float(mesh_scale)).T).T + translation


def pose_translation_from_camera_frame(translation_camera: np.ndarray) -> list[float]:
    t_pose = np.asarray(translation_camera, dtype=np.float64).reshape(3) @ POSE_TO_CAMERA_ROW.T
    return [float(t_pose[1]), float(t_pose[2]), float(t_pose[0])]


def assess_object_pose_continuity(
    layout: dict,
    max_translation_step_m: float = 2.0,
    p95_translation_step_m: float = 0.20,
    max_rotation_step_deg: float = 150.0,
    p95_rotation_step_deg: float = 60.0,
    max_lost_frame_fraction: float = 0.05,
) -> dict:
    entries = layout_entries(layout)
    all_frame_ids: list[int] = []
    frame_ids: list[int] = []
    translations: list[np.ndarray] = []
    quaternions: list[np.ndarray] = []
    invalid_frames: list[int] = []
    for row in entries:
        frame_id = int(row.get("frame_idx", row.get("index", len(frame_ids))) or 0)
        all_frame_ids.append(frame_id)
        local = row.get("local_to_scene") or {}
        translation = local.get("translation_camera_frame") or local.get("translation")
        quaternion = local.get("quat_wxyz_camera_frame") or local.get("quat_wxyz")
        try:
            t = np.asarray(translation, dtype=np.float64).reshape(3)
            q = np.asarray(quaternion, dtype=np.float64).reshape(4)
        except (TypeError, ValueError):
            invalid_frames.append(frame_id)
            continue
        q_norm = float(np.linalg.norm(q))
        if not np.isfinite(t).all() or not np.isfinite(q).all() or q_norm <= 1e-12:
            invalid_frames.append(frame_id)
            continue
        frame_ids.append(frame_id)
        translations.append(t)
        quaternions.append(q / q_norm)

    errors: list[str] = []
    if len(frame_ids) < 2:
        errors.append("insufficient_finite_object_poses")
        return {
            "status": "invalid",
            "errors": errors,
            "frame_count": len(entries),
            "finite_pose_count": len(frame_ids),
            "invalid_frames": invalid_frames,
        }

    order = np.argsort(np.asarray(frame_ids))
    ordered_ids = np.asarray(frame_ids, dtype=np.int64)[order]
    t = np.stack(translations, axis=0)[order]
    q = np.stack(quaternions, axis=0)[order]
    expected_count = int(max(all_frame_ids) - min(all_frame_ids) + 1)
    lost_count = max(expected_count - len(set(frame_ids)), 0)
    lost_fraction = lost_count / max(expected_count, 1)

    translation_steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    quaternion_dots = np.abs(np.sum(q[:-1] * q[1:], axis=1))
    rotation_steps = np.degrees(2.0 * np.arccos(np.clip(quaternion_dots, -1.0, 1.0)))
    translation_max = float(np.max(translation_steps))
    translation_p95 = float(np.percentile(translation_steps, 95))
    rotation_max = float(np.max(rotation_steps))
    rotation_p95 = float(np.percentile(rotation_steps, 95))

    if lost_fraction > max_lost_frame_fraction:
        errors.append(f"object_pose_lost_frame_fraction:{lost_fraction:.6f}")
    if translation_max > max_translation_step_m:
        errors.append(f"object_pose_translation_step_max_m:{translation_max:.6f}")
    if translation_p95 > p95_translation_step_m:
        errors.append(f"object_pose_translation_step_p95_m:{translation_p95:.6f}")
    if rotation_max > max_rotation_step_deg:
        errors.append(f"object_pose_rotation_step_max_deg:{rotation_max:.6f}")
    if rotation_p95 > p95_rotation_step_deg:
        errors.append(f"object_pose_rotation_step_p95_deg:{rotation_p95:.6f}")

    max_translation_index = int(np.argmax(translation_steps))
    max_rotation_index = int(np.argmax(rotation_steps))
    return {
        "status": "invalid" if errors else "ok",
        "errors": errors,
        "frame_count": len(entries),
        "finite_pose_count": len(frame_ids),
        "frame_range": [int(min(all_frame_ids)), int(max(all_frame_ids))],
        "invalid_frames": invalid_frames,
        "lost_frame_fraction": float(lost_fraction),
        "translation_step_m": {
            "median": float(np.median(translation_steps)),
            "p95": translation_p95,
            "max": translation_max,
            "max_frame_pair": [
                int(ordered_ids[max_translation_index]),
                int(ordered_ids[max_translation_index + 1]),
            ],
        },
        "rotation_step_deg": {
            "median": float(np.median(rotation_steps)),
            "p95": rotation_p95,
            "max": rotation_max,
            "max_frame_pair": [
                int(ordered_ids[max_rotation_index]),
                int(ordered_ids[max_rotation_index + 1]),
            ],
        },
        "thresholds": {
            "max_translation_step_m": float(max_translation_step_m),
            "p95_translation_step_m": float(p95_translation_step_m),
            "max_rotation_step_deg": float(max_rotation_step_deg),
            "p95_rotation_step_deg": float(p95_rotation_step_deg),
            "max_lost_frame_fraction": float(max_lost_frame_fraction),
        },
    }


def assess_dai_native_retarget_input(
    contact_alignment: dict,
    overlap_frames: float,
    min_visual_overlap_frames: int = 3,
    object_pose_quality: dict | None = None,
) -> dict:
    errors: list[str] = []
    if float(overlap_frames) < min_visual_overlap_frames:
        errors.append(f"weak_visual_hand_object_overlap:{int(overlap_frames)}")
    if contact_alignment.get("reason") == "offset_exceeds_max":
        errors.append("hoi_coordinate_offset_exceeds_max")
    if object_pose_quality is not None and object_pose_quality.get("status") != "ok":
        errors.extend(
            f"source_{error}" for error in (object_pose_quality.get("errors") or ["object_pose_invalid"])
        )
    return {
        "status": "invalid" if errors else "ok",
        "errors": errors,
        "visual_overlap_frames": float(overlap_frames),
        "min_visual_overlap_frames": int(min_visual_overlap_frames),
        "hoi_contact_alignment_reason": contact_alignment.get("reason"),
        "required_offset_norm": contact_alignment.get("required_offset_norm"),
        "max_offset": contact_alignment.get("max_offset"),
        "object_pose_quality": object_pose_quality,
    }


def align_layout(
    layout: dict,
    hand_npz: Path,
    mesh_path: Path,
    selected_hand: str,
    overlap_frames: float,
    mode: str = "auto",
    min_bad_mean_distance: float = 0.10,
    target_surface_distance: float = 0.025,
    max_offset: float = 0.10,
) -> tuple[dict, dict]:
    """Diagnose a contact offset while preserving the production layout.

    Object-only contact translation is not a coordinate transform: it changes
    the source HOI.  All non-``none`` modes therefore compute only a
    hypothetical candidate and return the original layout unchanged.
    """
    if mode == "none":
        return layout, {
            "applied": False,
            "diagnostic_only": True,
            "layout_modified": False,
            "reason": "disabled",
            "mode": mode,
            "policy": "object_only_hoi_translation_forbidden",
        }
    if selected_hand not in {"left", "right"}:
        return layout, {"applied": False, "reason": "selected_hand_not_single", "selected_hand": selected_hand}
    if mode in {"auto", "diagnostic"} and overlap_frames < 3:
        return layout, {"applied": False, "reason": "weak_visual_overlap", "overlap_frames": overlap_frames}

    entries = layout_entries(layout)
    if not entries:
        return layout, {"applied": False, "reason": "empty_layout"}
    mesh_scale = float((layout.get("translation_scale_optimization") or {}).get("mesh_scale", 1.0) or 1.0)
    if not np.isfinite(mesh_scale) or mesh_scale <= 0:
        return layout, {"applied": False, "reason": "invalid_mesh_scale", "mesh_scale": mesh_scale}
    with np.load(hand_npz, allow_pickle=False) as hands:
        key = f"{selected_hand}_vertices"
        valid_key = f"{selected_hand}_valid"
        if key not in hands:
            return layout, {"applied": False, "reason": "missing_selected_hand_vertices"}
        hand_vertices = np.asarray(hands[key], dtype=np.float64)
        valid = np.asarray(hands[valid_key]) > 0 if valid_key in hands else np.ones(len(hand_vertices), dtype=bool)

    mesh = load_obj_vertices(mesh_path)
    count = min(len(entries), len(hand_vertices))
    sample_ids = np.arange(count)
    if len(sample_ids) > 40:
        sample_ids = np.linspace(0, count - 1, 40, dtype=int)
    offsets = []
    before_distances = []
    for idx in sample_ids:
        if not valid[int(idx)]:
            continue
        obj = subsample(transformed_object(mesh, entries[int(idx)], mesh_scale))
        hand = subsample(hand_vertices[int(idx)])
        offset, distance = nearest_offset(hand, obj)
        if np.isfinite(distance) and np.isfinite(offset).all():
            offsets.append(offset)
            before_distances.append(distance)
    if not offsets:
        return layout, {"applied": False, "reason": "no_valid_nearest_offsets"}

    before = np.asarray(before_distances, dtype=np.float64)
    before_stats = {"min": float(before.min()), "median": float(np.median(before)), "mean": float(before.mean())}
    if mode in {"auto", "diagnostic"} and before_stats["mean"] < min_bad_mean_distance:
        return layout, {
            "applied": False,
            "reason": "geometry_already_close",
            "before": before_stats,
            "threshold": float(min_bad_mean_distance),
        }

    raw_offset = np.median(np.stack(offsets), axis=0)
    raw_norm = float(np.linalg.norm(raw_offset))
    if not np.isfinite(raw_norm) or raw_norm <= 1e-8:
        return layout, {"applied": False, "reason": "degenerate_offset", "before": before_stats}
    required_offset_norm = max(0.0, raw_norm - float(target_surface_distance))
    if mode in {"auto", "diagnostic"} and required_offset_norm > max_offset:
        return layout, {
            "applied": False,
            "reason": "offset_exceeds_max",
            "selected_hand": selected_hand,
            "overlap_frames": float(overlap_frames),
            "raw_offset_norm": raw_norm,
            "required_offset_norm": required_offset_norm,
            "max_offset": float(max_offset),
            "before": before_stats,
        }
    applied_offset_norm = min(required_offset_norm, float(max_offset))
    clipped = required_offset_norm > max_offset
    offset = raw_offset * (applied_offset_norm / raw_norm)

    candidate = json.loads(json.dumps(layout))
    candidate_entries = layout_entries(candidate)
    for row in candidate_entries:
        local = row.get("local_to_scene") or {}
        translation = np.asarray(local.get("translation_camera_frame"), dtype=np.float64).reshape(3)
        aligned_translation = translation + offset
        local["translation_camera_frame"] = [float(x) for x in aligned_translation]
        if "translation" in local:
            local["translation"] = pose_translation_from_camera_frame(aligned_translation)
    after_distances = []
    for idx in sample_ids:
        if not valid[int(idx)]:
            continue
        obj = subsample(transformed_object(mesh, candidate_entries[int(idx)], mesh_scale))
        hand = subsample(hand_vertices[int(idx)])
        _, distance = nearest_offset(hand, obj)
        after_distances.append(distance)
    after = np.asarray(after_distances, dtype=np.float64)
    after_stats = {"min": float(after.min()), "median": float(np.median(after)), "mean": float(after.mean())}
    improvement = 1.0 - after_stats["mean"] / max(before_stats["mean"], 1e-9)
    if mode in {"auto", "diagnostic"} and improvement < 0.25:
        return layout, {
            "applied": False,
            "reason": "insufficient_contact_improvement",
            "before": before_stats,
            "candidate_after": after_stats,
            "improvement_fraction": float(improvement),
        }

    provenance = {
        "applied": False,
        "would_apply": True,
        "diagnostic_only": True,
        "layout_modified": False,
        "reason": "object_only_translation_forbidden",
        "policy": "preserve_source_hoi_relative_transform",
        "mode": mode,
        "method": "diagnostic_constant_median_nearest_surface_translation",
        "selected_hand": selected_hand,
        "overlap_frames": float(overlap_frames),
        "frames_used": len(offsets),
        "proposed_offset": [float(x) for x in offset],
        "proposed_offset_norm": float(np.linalg.norm(offset)),
        "offset": [float(x) for x in offset],
        "offset_norm": float(np.linalg.norm(offset)),
        "raw_offset_norm": raw_norm,
        "required_offset_norm": required_offset_norm,
        "max_offset": float(max_offset),
        "offset_clipped": clipped,
        "before": before_stats,
        "hypothetical_after": after_stats,
        "improvement_fraction": float(improvement),
        "object_track_source": "dai_native",
        "object_mesh_source": "dai_native",
        "retarget_object_source": "dai_native",
    }
    return layout, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose DAI-native hand-object contact without modifying the layout.")
    parser.add_argument("--layout", required=True, type=Path)
    parser.add_argument("--hand-npz", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--selected-hand", required=True, choices=["left", "right"])
    parser.add_argument("--overlap-frames", required=True, type=float)
    parser.add_argument("--mode", choices=["none", "diagnostic", "auto", "force"], default="diagnostic")
    parser.add_argument("--min-bad-mean-distance", type=float, default=0.10)
    parser.add_argument("--target-surface-distance", type=float, default=0.025)
    parser.add_argument("--max-offset", type=float, default=0.10)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    layout = json.loads(args.layout.read_text(encoding="utf-8"))
    aligned, report = align_layout(
        layout, args.hand_npz, args.mesh, args.selected_hand, args.overlap_frames,
        args.mode, args.min_bad_mean_distance, args.target_surface_distance, args.max_offset,
    )
    args.layout.write_text(json.dumps(aligned, indent=2), encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
