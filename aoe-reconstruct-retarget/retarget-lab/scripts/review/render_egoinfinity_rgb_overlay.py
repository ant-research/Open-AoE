#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import io
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def decode_rgb(blob: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))


def load_ply_vertices(path: Path, max_points: int = 12000) -> np.ndarray:
    with path.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"invalid PLY header: {path}")
            header.append(line)
            if line.strip() == b"end_header":
                break
        offset = handle.tell()
    text = b"".join(header).decode("ascii", errors="replace")
    if "format binary_little_endian" not in text:
        raise ValueError(f"only binary_little_endian PLY is supported: {path}")
    vertex_count = int([line for line in text.splitlines() if line.startswith("element vertex")][0].split()[-1])
    properties = [line for line in text.splitlines() if line.startswith("property")]
    dtype = np.dtype([(f"p{i}", "<f4") for i in range(len(properties))])
    arr = np.fromfile(path, dtype=dtype, offset=offset, count=vertex_count)
    vertices = np.stack([arr["p0"], arr["p1"], arr["p2"]], axis=1).astype(np.float32)
    if max_points > 0 and len(vertices) > max_points:
        step = max(1, len(vertices) // max_points)
        vertices = vertices[::step][:max_points]
    return vertices


def project_camera_points(points: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-4)
    uv = np.full((len(points), 2), -100000, dtype=np.int32)
    if valid.any():
        pts = points[valid]
        uv_float = np.stack([cx + fx * pts[:, 0] / pts[:, 2], cy + fy * pts[:, 1] / pts[:, 2]], axis=1)
        uv[valid] = np.round(uv_float).astype(np.int32)
    return uv, valid


def align_uv_to_joints(
    vertices_cam: np.ndarray,
    joints_3d: np.ndarray,
    joints_2d: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    uv_vertices, valid_vertices = project_camera_points(vertices_cam, fx, fy, cx, cy)
    joints_3d = np.asarray(joints_3d, dtype=np.float32)
    joints_2d = np.asarray(joints_2d, dtype=np.float32)
    if joints_3d.shape[0] < 6 or joints_2d.shape[0] != joints_3d.shape[0]:
        return uv_vertices, valid_vertices
    uv_joints, valid_joints = project_camera_points(joints_3d, fx, fy, cx, cy)
    good = valid_joints & np.isfinite(joints_2d).all(axis=1)
    if int(good.sum()) < 4:
        return uv_vertices, valid_vertices
    src = uv_joints[good].astype(np.float32)
    dst = joints_2d[good].astype(np.float32)
    affine, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if affine is None:
        delta = np.median(dst - src, axis=0)
        uv_vertices = np.round(uv_vertices.astype(np.float32) + delta).astype(np.int32)
        return uv_vertices, valid_vertices
    uv_float = uv_vertices.astype(np.float32)
    uv_aligned = uv_float @ affine[:, :2].T + affine[:, 2]
    return np.round(uv_aligned).astype(np.int32), valid_vertices


def estimate_joint_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or len(src) == 0:
        return None
    if len(src) >= 4:
        affine, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if affine is not None:
            return affine.astype(np.float32)
    delta = np.median(dst - src, axis=0)
    return np.array([[1.0, 0.0, float(delta[0])], [0.0, 1.0, float(delta[1])]], dtype=np.float32)


def apply_affine_to_uv(uv: np.ndarray, affine: np.ndarray | None) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.int32)
    if affine is None or len(uv) == 0:
        return uv
    uv_float = uv.astype(np.float32)
    uv_aligned = uv_float @ affine[:, :2].T + affine[:, 2]
    return np.round(uv_aligned).astype(np.int32)


def estimate_frame_affine(
    joints_3d_per_hand: list[np.ndarray],
    joints_2d_per_hand: list[np.ndarray],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray | None:
    src_chunks = []
    dst_chunks = []
    for joints_3d, joints_2d in zip(joints_3d_per_hand, joints_2d_per_hand):
        joints_3d = np.asarray(joints_3d, dtype=np.float32)
        joints_2d = np.asarray(joints_2d, dtype=np.float32)
        if joints_3d.ndim != 2 or joints_2d.shape != joints_3d.shape:
            continue
        uv_joints, valid_joints = project_camera_points(joints_3d, fx, fy, cx, cy)
        good = valid_joints & np.isfinite(joints_2d).all(axis=1)
        if int(good.sum()) < 4:
            continue
        src_chunks.append(uv_joints[good].astype(np.float32))
        dst_chunks.append(joints_2d[good].astype(np.float32))
    if not src_chunks:
        return None
    src = np.concatenate(src_chunks, axis=0)
    dst = np.concatenate(dst_chunks, axis=0)
    return estimate_joint_affine(src, dst)


def draw_point_cloud(
    rgb: np.ndarray,
    points_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    point_radius: int,
    uv_override: np.ndarray | None = None,
    valid_override: np.ndarray | None = None,
) -> None:
    h, w = rgb.shape[:2]
    if uv_override is None or valid_override is None:
        uv, valid = project_camera_points(points_cam, fx, fy, cx, cy)
    else:
        uv = np.asarray(uv_override, dtype=np.int32)
        valid = np.asarray(valid_override, dtype=bool)
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    pts = uv[inside]
    if len(pts) == 0:
        return
    color = (30, 225, 85)
    if point_radius <= 1:
        rgb[pts[:, 1], pts[:, 0]] = np.array(color, dtype=np.uint8)
        return
    for x, y in pts:
        cv2.circle(rgb, (int(x), int(y)), point_radius, color, -1, cv2.LINE_AA)


def draw_triangle_mesh(
    rgb: np.ndarray,
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    fill_color: tuple[int, int, int],
    edge_color: tuple[int, int, int],
    alpha: float = 0.18,
    uv_override: np.ndarray | None = None,
    valid_override: np.ndarray | None = None,
) -> None:
    if vertices_cam is None or len(vertices_cam) == 0 or faces is None or len(faces) == 0:
        return
    h, w = rgb.shape[:2]
    vertices_cam = np.asarray(vertices_cam, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if uv_override is None or valid_override is None:
        uv, valid = project_camera_points(vertices_cam, fx, fy, cx, cy)
    else:
        uv = np.asarray(uv_override, dtype=np.int32)
        valid = np.asarray(valid_override, dtype=bool)
    if not valid.any():
        return
    overlay = rgb.copy()
    for face in faces:
        if np.any(face < 0) or np.any(face >= len(vertices_cam)) or not valid[face].all():
            continue
        tri = uv[face]
        if (tri[:, 0].max() < -w or tri[:, 0].min() > 2 * w or tri[:, 1].max() < -h or tri[:, 1].min() > 2 * h):
            continue
        cv2.fillConvexPoly(overlay, tri.astype(np.int32), fill_color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, rgb, 1.0 - alpha, 0, dst=rgb)

    drawn = set()
    for face in faces:
        if np.any(face < 0) or np.any(face >= len(vertices_cam)) or not valid[face].all():
            continue
        for a, b in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            key = (a, b) if a < b else (b, a)
            if key in drawn:
                continue
            drawn.add(key)
            pa = tuple(uv[a])
            pb = tuple(uv[b])
            if (-w < pa[0] < 2 * w and -h < pa[1] < 2 * h and -w < pb[0] < 2 * w and -h < pb[1] < 2 * h):
                cv2.line(rgb, pa, pb, edge_color, 1, cv2.LINE_AA)


def mask_centroid_from_obj_data(obj: dict) -> np.ndarray | None:
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


def stable_prefix_len(result: dict, obj_id, max_jump_px: float) -> tuple[int, float]:
    frames = result.get("frame_data") or []
    centroids = []
    for frame in frames:
        obj_data = frame.get("sam3_obj_data") or {}
        obj = obj_data.get(obj_id) or obj_data.get(str(obj_id))
        centroids.append(mask_centroid_from_obj_data(obj) if isinstance(obj, dict) else None)
    last = None
    max_jump = 0.0
    for i, centroid in enumerate(centroids):
        if centroid is None:
            continue
        if last is not None:
            jump = float(np.linalg.norm(centroid - last))
            max_jump = max(max_jump, jump)
            if max_jump_px > 0 and jump > max_jump_px:
                return i, max_jump
        last = centroid
    return len(frames), max_jump


def object_prompt(result: dict, obj_id) -> str:
    mapping = result.get("sam3_prompt_mapping") or []
    try:
        idx = int(obj_id)
    except Exception:
        return ""
    if 0 <= idx < len(mapping) and isinstance(mapping[idx], dict):
        return str(mapping[idx].get("prompt", ""))
    return ""


def object_prompt_score(result: dict, obj_id) -> float:
    mapping = result.get("sam3_prompt_mapping") or []
    try:
        idx = int(obj_id)
    except Exception:
        return 0.0
    if 0 <= idx < len(mapping) and isinstance(mapping[idx], dict):
        return float(mapping[idx].get("score", 0.0) or 0.0)
    return 0.0


def select_object_ids(
    result: dict,
    target_prompt: str | None,
    object_id: int | None,
    max_centroid_jump_px: float,
    skip_spurious: bool,
    select_best: bool,
) -> set:
    mesh_info = result.get("sam3_mesh_info", {}) or {}
    pose_info_all = result.get("pose_track_info", {}) or {}
    if object_id is not None:
        return {object_id}

    rows = []
    for obj_id, info in mesh_info.items():
        pose_info = pose_info_all.get(obj_id) or pose_info_all.get(str(obj_id))
        if not isinstance(info, dict) or not isinstance(pose_info, dict):
            continue
        if skip_spurious and pose_info.get("spurious_flag"):
            continue
        prompt = object_prompt(result, obj_id)
        if target_prompt and prompt != target_prompt:
            continue
        stable_len, max_jump = stable_prefix_len(result, obj_id, max_centroid_jump_px)
        rows.append(
            (
                stable_len,
                object_prompt_score(result, obj_id),
                -max_jump,
                float(info.get("n_points", 0) or 0),
                obj_id,
            )
        )
    if target_prompt and rows and select_best:
        rows.sort(reverse=True)
        return {rows[0][-1]}
    return {row[-1] for row in rows} if rows else set(mesh_info.keys())


def pose_sequence_for(result: dict, obj_id, pose_info: dict) -> np.ndarray | None:
    if pose_info.get("T_seq") is not None:
        return np.asarray(pose_info["T_seq"], dtype=np.float32)

    frames = result.get("frame_data") or []
    transforms = []
    last = None
    for frame in frames:
        sam3_obj_data = frame.get("sam3_obj_data") or {}
        obj = sam3_obj_data.get(obj_id) or sam3_obj_data.get(str(obj_id))
        if isinstance(obj, dict) and obj.get("pose_R") is not None and obj.get("pose_t") is not None:
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = np.asarray(obj["pose_R"], dtype=np.float32)
            transform[:3, 3] = np.asarray(obj["pose_t"], dtype=np.float32).reshape(3)
            last = transform
        elif last is not None:
            transform = last.copy()
        else:
            transform = None
        if transform is not None:
            transforms.append(transform)

    if not transforms:
        return None
    return np.stack(transforms, axis=0).astype(np.float32)


def build_object_items(
    result: dict,
    max_points: int,
    scale_multiplier: float,
    target_prompt: str | None = None,
    object_id: int | None = None,
    max_centroid_jump_px: float = 0.0,
    stop_on_drift: bool = True,
    skip_spurious: bool = True,
    select_best: bool = False,
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    items = []
    mesh_info = result.get("sam3_mesh_info", {}) or {}
    pose_info_all = result.get("pose_track_info", {}) or {}
    selected = select_object_ids(
        result,
        target_prompt=target_prompt,
        object_id=object_id,
        max_centroid_jump_px=max_centroid_jump_px,
        skip_spurious=skip_spurious,
        select_best=select_best,
    )
    for obj_id, info in sorted(mesh_info.items(), key=lambda item: str(item[0])):
        if obj_id not in selected and str(obj_id) not in {str(x) for x in selected}:
            continue
        pose_info = pose_info_all.get(obj_id) or pose_info_all.get(str(obj_id))
        if not isinstance(info, dict) or not isinstance(pose_info, dict):
            continue
        if skip_spurious and pose_info.get("spurious_flag"):
            continue
        ply_path = Path(info["ply_path"])
        if not ply_path.exists():
            continue
        scale = float(pose_info.get("scale_correction") or info.get("canonical_scale") or 1.0)
        vertices = load_ply_vertices(ply_path, max_points) * (scale * scale_multiplier)
        transforms = pose_sequence_for(result, obj_id, pose_info)
        if transforms is None or len(transforms) == 0:
            continue
        valid_until, _max_jump = stable_prefix_len(result, obj_id, max_centroid_jump_px)
        if not stop_on_drift:
            valid_until = len(transforms)
        items.append((vertices, transforms, int(obj_id), int(valid_until)))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Render EgoInfinity RGB overlay from reconstructed object mesh and hand meshes.")
    parser.add_argument("--pipeline-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--object-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--target-prompt", default=None, help="If set, render all objects matching this SAM3 prompt.")
    parser.add_argument("--object-id", type=int, default=None, help="If set, render only this SAM3 object id.")
    parser.add_argument(
        "--select-best-object",
        action="store_true",
        help="Legacy behavior: with --target-prompt, render only the most stable matching object.",
    )
    parser.add_argument("--max-centroid-jump-px", type=float, default=320.0)
    parser.add_argument("--no-stop-on-drift", action="store_true")
    parser.add_argument("--include-spurious", action="store_true")
    parser.add_argument("--background", choices=["rgb", "black", "white"], default="rgb")
    parser.add_argument(
        "--object-point-radius",
        type=int,
        default=1,
        help="Pixel radius for projected object point cloud. Keep 1 for point-only overlay without mask-like fill.",
    )
    parser.add_argument(
        "--no-align-hands-to-joints",
        dest="align_hands_to_joints",
        action="store_false",
        help="Disable 2D-joint affine correction for raw hand-projection diagnostics.",
    )
    parser.set_defaults(align_hands_to_joints=True)
    args = parser.parse_args()

    with gzip.open(args.pipeline_result, "rb") as handle:
        result = pickle.load(handle)
    frames = result["frame_data"]
    object_items = build_object_items(
        result,
        args.max_points,
        args.object_scale_multiplier,
        target_prompt=args.target_prompt,
        object_id=args.object_id,
        max_centroid_jump_px=args.max_centroid_jump_px,
        stop_on_drift=not args.no_stop_on_drift,
        skip_spurious=not args.include_spurious,
        select_best=args.select_best_object,
    )
    mano_faces = np.asarray(result.get("mano_faces"), dtype=np.int32)
    if not object_items:
        raise RuntimeError("no EgoInfinity reconstructed object mesh + pose sequence found")

    first = decode_rgb(frames[0]["img_rgb"])
    h, w = first.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {args.output}")

    fx = fy = float(result["dp_focal"])
    cx = float(result["cx"])
    cy = float(result["cy"])
    for i, frame in enumerate(frames):
        rgb = decode_rgb(frame["img_rgb"]).copy()
        if args.background == "black":
            rgb = np.zeros_like(rgb)
        elif args.background == "white":
            rgb = np.full_like(rgb, 255)
        joints_3d_per_hand = frame.get("joints_3d_pred", []) or []
        joints_2d_per_hand = frame.get("joints_2d_pred", []) or []
        frame_affine = None
        if args.align_hands_to_joints:
            frame_affine = estimate_frame_affine(joints_3d_per_hand, joints_2d_per_hand, fx, fy, cx, cy)
        for vertices, transforms, obj_id, valid_until in object_items:
            if i >= valid_until:
                continue
            transform = transforms[min(i, len(transforms) - 1)]
            points_cam = (transform[:3, :3] @ vertices.T).T + transform[:3, 3]
            uv_override = valid_override = None
            if frame_affine is not None:
                uv_raw, valid_override = project_camera_points(points_cam, fx, fy, cx, cy)
                uv_override = apply_affine_to_uv(uv_raw, frame_affine)
            draw_point_cloud(
                rgb,
                points_cam,
                fx,
                fy,
                cx,
                cy,
                args.object_point_radius,
                uv_override=uv_override,
                valid_override=valid_override,
            )

        for hand_idx, vertices in enumerate(frame.get("vertices_3d", []) or []):
            if vertices is None:
                continue
            is_right = bool((frame.get("hand_is_right", []) or [False])[hand_idx])
            fill = (245, 115, 45) if is_right else (50, 130, 255)
            edge = (255, 165, 70) if is_right else (90, 180, 255)
            uv_override = valid_override = None
            if frame_affine is not None:
                uv_raw, valid_override = project_camera_points(vertices, fx, fy, cx, cy)
                uv_override = apply_affine_to_uv(uv_raw, frame_affine)
            elif args.align_hands_to_joints and hand_idx < len(joints_3d_per_hand) and hand_idx < len(joints_2d_per_hand):
                uv_override, valid_override = align_uv_to_joints(
                    vertices,
                    joints_3d_per_hand[hand_idx],
                    joints_2d_per_hand[hand_idx],
                    fx,
                    fy,
                    cx,
                    cy,
                )
            draw_triangle_mesh(
                rgb,
                vertices,
                mano_faces,
                fx,
                fy,
                cx,
                cy,
                fill,
                edge,
                uv_override=uv_override,
                valid_override=valid_override,
            )

        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
