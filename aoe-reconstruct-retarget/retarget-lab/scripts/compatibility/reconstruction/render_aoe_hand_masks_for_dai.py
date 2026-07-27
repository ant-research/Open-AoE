#!/usr/bin/env python3
"""Render OpenAoE MANO hand meshes into do-as-i-do hand mask files.

This is an input adapter only: do-as-i-do's translation-scale optimizer expects
SAM-style masks named left_hand_0.png / right_hand_0.png next to each object
mask. OpenAoE gives us per-frame MANO meshes instead, so we project those meshes
with the saved camera intrinsics and rasterize their faces into binary masks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render AoE MANO hand meshes to do-as-i-do hand masks."
    )
    parser.add_argument("--clip-dir", required=True, type=Path)
    parser.add_argument("--hand-meshes", type=Path, default=None)
    parser.add_argument("--layout-json", type=Path, default=None)
    parser.add_argument("--frames", type=str, default=None,
                        help="Frame range like 35:60. Ignored if --layout-json is set.")
    parser.add_argument("--mask-root", type=Path, default=None)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--dilate", type=int, default=3,
                        help="Odd dilation kernel size; 0 disables dilation.")
    return parser.parse_args()


def frame_indices(args: argparse.Namespace) -> list[int]:
    if args.layout_json is not None:
        with args.layout_json.open() as f:
            data = json.load(f)
        frames = []
        for obj in data.get("objects", []):
            idx = obj.get("frame_idx", obj.get("frame_index"))
            if idx is not None:
                frames.append(int(idx))
        return sorted(set(frames))

    if args.frames is None:
        raise ValueError("Provide either --layout-json or --frames.")
    start_s, end_s = args.frames.split(":", 1)
    start, end = int(start_s), int(end_s)
    return list(range(start, end))


def load_intrinsics(frames_dir: Path, frame_idx: int) -> tuple[np.ndarray, tuple[int, int]]:
    intr_path = frames_dir / f"{frame_idx:06d}_intrinsics.npy"
    if not intr_path.exists():
        raise FileNotFoundError(intr_path)
    intr = np.load(intr_path)

    img_path = frames_dir / f"{frame_idx:06d}.png"
    if not img_path.exists():
        img_path = frames_dir / f"{frame_idx:06d}.jpg"
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]
    return intr, (h, w)


def parse_resolution(value: str | None) -> tuple[float, float] | None:
    if not value or "x" not in value:
        return None
    left, right = value.lower().split("x", 1)
    try:
        return float(left), float(right)
    except ValueError:
        return None


def hawor_hand_intrinsics(hand_data: np.lib.npyio.NpzFile, fallback: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    source_path = hand_data["source_hands_npz"] if "source_hands_npz" in hand_data else None
    if source_path is None:
        return fallback
    source = Path(str(source_path))
    if not source.exists():
        return fallback
    try:
        source_hands = np.load(source, allow_pickle=True)
    except Exception:
        return fallback
    if "focal" not in source_hands:
        return fallback
    focal = float(np.asarray(source_hands["focal"]).reshape(-1)[0])
    source_width = source_height = None
    info_path = source.parent.parent.parent / "ego_process" / "ego_undistorted_video" / "undistorted_video_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            resolution = parse_resolution((info.get("cameraParams") or {}).get("resolution"))
            if resolution is not None:
                source_width, source_height = resolution
        except Exception:
            pass
    h, w = image_shape
    if source_width is None or source_height is None:
        source_width, source_height = float(w), float(h)
    fx = focal * float(w) / max(float(source_width), 1e-6)
    fy = focal * float(h) / max(float(source_height), 1e-6)
    return np.asarray([[fx, 0.0, w * 0.5], [0.0, fy, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)


def rasterize_mesh_mask(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

    verts = np.asarray(vertices, dtype=np.float64)
    z = verts[:, 2]
    valid_z = z > 1e-5
    uv = np.empty((verts.shape[0], 2), dtype=np.float64)
    uv[:, 0] = fx * verts[:, 0] / np.maximum(z, 1e-5) + cx
    uv[:, 1] = fy * verts[:, 1] / np.maximum(z, 1e-5) + cy

    for tri in np.asarray(faces, dtype=np.int64):
        if not np.all(valid_z[tri]):
            continue
        pts = uv[tri]
        if np.all(pts[:, 0] < 0) or np.all(pts[:, 0] >= w):
            continue
        if np.all(pts[:, 1] < 0) or np.all(pts[:, 1] >= h):
            continue
        if not np.isfinite(pts).all():
            continue
        pts_i = np.round(pts).astype(np.int32)
        cv2.fillConvexPoly(mask, pts_i, 255, lineType=cv2.LINE_AA)
    return mask


def main() -> None:
    args = parse_args()
    clip_dir = args.clip_dir
    hand_meshes = args.hand_meshes or clip_dir / "raw" / "all_hand_meshes.npz"
    mask_root = args.mask_root or clip_dir / "video_segmentation" / "masks"
    frames_dir = args.frames_dir or clip_dir / "all_frames"

    hand_data = np.load(hand_meshes)
    frames = frame_indices(args)
    kernel = None
    if args.dilate and args.dilate > 0:
        k = args.dilate if args.dilate % 2 == 1 else args.dilate + 1
        kernel = np.ones((k, k), np.uint8)

    print(f"hand meshes: {hand_meshes}")
    print(f"frames: {frames[0]}..{frames[-1]} ({len(frames)})")

    for frame_idx in frames:
        intr, image_shape = load_intrinsics(frames_dir, frame_idx)
        intr = hawor_hand_intrinsics(hand_data, intr, image_shape)
        frame_mask_dir = mask_root / f"frame_{frame_idx:06d}_masks"
        frame_mask_dir.mkdir(parents=True, exist_ok=True)

        for hand in ("left", "right"):
            verts_all = hand_data[f"{hand}_vertices"]
            faces = hand_data[f"{hand}_faces"]
            valid_key = f"{hand}_valid"
            if valid_key in hand_data and not bool(hand_data[valid_key][min(frame_idx, len(hand_data[valid_key]) - 1)]):
                continue
            verts = verts_all[min(frame_idx, verts_all.shape[0] - 1)]
            mask = rasterize_mesh_mask(verts, faces, intr, image_shape)
            if kernel is not None and mask.any():
                mask = cv2.dilate(mask, kernel, iterations=1)
            out = frame_mask_dir / f"{hand}_hand_0.png"
            cv2.imwrite(str(out), mask)
        print(f"rendered hand masks for frame {frame_idx:06d}")


if __name__ == "__main__":
    main()
