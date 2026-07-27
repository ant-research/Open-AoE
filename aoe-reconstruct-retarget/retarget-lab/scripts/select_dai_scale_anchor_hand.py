#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def project_vertices(vertices: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    z = vertices[:, 2]
    valid = np.isfinite(vertices).all(axis=1) & (z > 1e-6)
    uv = np.full((vertices.shape[0], 2), np.nan, dtype=np.float64)
    uv[:, 0] = intrinsics[0, 0] * vertices[:, 0] / np.maximum(z, 1e-6) + intrinsics[0, 2]
    uv[:, 1] = intrinsics[1, 1] * vertices[:, 1] / np.maximum(z, 1e-6) + intrinsics[1, 2]
    return uv, valid


def bbox_from_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)


def bbox_from_points(uv: np.ndarray, valid: np.ndarray, width: int, height: int) -> np.ndarray | None:
    keep = valid & np.isfinite(uv).all(axis=1)
    keep &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if not keep.any():
        return None
    points = uv[keep]
    return np.array(
        [points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()],
        dtype=np.float64,
    )


def bbox_overlap(a: np.ndarray, b: np.ndarray) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def bbox_area(box: np.ndarray) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float64)


def read_object_mask(mask_root: Path, object_id: str, frame_id: int) -> np.ndarray | None:
    frame_dir = mask_root / f"frame_{frame_id:06d}_masks"
    candidates = [
        frame_dir / f"{object_id}.png",
        frame_dir / object_id / f"{object_id}.png",
    ]
    for path in candidates:
        if path.exists():
            return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None


def interaction_scores(
    hand_npz: Path,
    frames_dir: Path,
    masks_dir: Path,
    object_id: str,
    max_frames: int = 32,
) -> dict[str, object]:
    with np.load(hand_npz, allow_pickle=False) as data:
        frame_count = max(
            [int(data[f"{side}_vertices"].shape[0]) for side in ("left", "right") if f"{side}_vertices" in data]
            or [0]
        )
        if frame_count <= 0:
            raise ValueError(f"no left/right hand vertices in {hand_npz}")
        sample_ids = np.arange(frame_count, dtype=int)
        if sample_ids.size > max_frames:
            sample_ids = np.unique(np.linspace(0, frame_count - 1, max_frames, dtype=int))
        scores: dict[str, dict[str, float]] = {
            "left": {"score": 0.0, "frames": 0.0, "overlap_frames": 0.0, "center_distance_sum": 0.0},
            "right": {"score": 0.0, "frames": 0.0, "overlap_frames": 0.0, "center_distance_sum": 0.0},
        }
        for frame_id in sample_ids.tolist():
            image = cv2.imread(str(frames_dir / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
            intrinsics_path = frames_dir / f"{frame_id:06d}_intrinsics.npy"
            if image is None or not intrinsics_path.exists():
                continue
            object_box = bbox_from_mask(read_object_mask(masks_dir, object_id, frame_id))
            if object_box is None:
                continue
            height, width = image.shape[:2]
            image_diag = max(float(np.hypot(width, height)), 1.0)
            object_area = max(bbox_area(object_box), 1.0)
            object_center = bbox_center(object_box)
            intrinsics = np.load(intrinsics_path)
            for side in ("left", "right"):
                vertices_key = f"{side}_vertices"
                valid_key = f"{side}_valid"
                if vertices_key not in data or frame_id >= data[vertices_key].shape[0]:
                    continue
                if valid_key in data and float(data[valid_key][frame_id]) <= 0:
                    continue
                uv, valid = project_vertices(data[vertices_key][frame_id], intrinsics)
                hand_box = bbox_from_points(uv, valid, width, height)
                if hand_box is None:
                    continue
                overlap = bbox_overlap(hand_box, object_box)
                distance = float(np.linalg.norm(bbox_center(hand_box) - object_center))
                proximity = max(0.0, 1.0 - distance / (0.35 * image_diag))
                scores[side]["score"] += 4.0 * overlap / object_area + proximity
                scores[side]["frames"] += 1.0
                scores[side]["center_distance_sum"] += distance
                if overlap > 0:
                    scores[side]["overlap_frames"] += 1.0

    normalized = {
        side: values["score"] / max(values["frames"], 1.0)
        for side, values in scores.items()
    }
    mean_center_distance = {
        side: values["center_distance_sum"] / max(values["frames"], 1.0)
        for side, values in scores.items()
    }
    return {
        "sampled_frames": sample_ids.tolist(),
        "scores": scores,
        "normalized": normalized,
        "mean_center_distance_px": mean_center_distance,
    }


def select_anchor(requested: str, evidence: dict[str, object]) -> tuple[str, dict[str, object]]:
    if requested in {"left", "right"}:
        return requested, {"reason": "explicit_single_hand", "ambiguous": False, "score_ratio": None}
    normalized = evidence["normalized"]
    left = float(normalized["left"])
    right = float(normalized["right"])
    if max(left, right) <= 0:
        raise ValueError("neither projected hand has measurable proximity to the object mask")
    selected = "left" if left >= right else "right"
    other = right if selected == "left" else left
    ratio = max(left, right) / max(other, 1e-9)
    return selected, {
        "reason": "visual_object_mask_proximity",
        "ambiguous": bool(ratio < 1.25),
        "score_ratio": float(ratio),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve DAI's single-hand scale anchor from projected hand/object-mask proximity."
    )
    parser.add_argument("--hand-npz", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--requested", choices=["auto", "left", "right", "bimanual"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=32)
    args = parser.parse_args()

    evidence = interaction_scores(
        args.hand_npz,
        args.frames_dir,
        args.masks_dir,
        args.object_id,
        max_frames=max(1, args.max_frames),
    )
    selected, decision = select_anchor(args.requested, evidence)
    payload = {
        "status": "ok",
        "requested_anchor_hand": args.requested,
        "selected_anchor_hand": selected,
        "method": "projected_hand_bbox_vs_object_mask_bbox",
        "decision": decision,
        "evidence": evidence,
        "inputs": {
            "hand_npz": str(args.hand_npz.resolve()),
            "frames_dir": str(args.frames_dir.resolve()),
            "masks_dir": str(args.masks_dir.resolve()),
            "object_id": args.object_id,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
