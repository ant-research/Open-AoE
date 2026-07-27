#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def object_id_from_task(task: str) -> str:
    for marker in ["_bimanual", "_right", "_left"]:
        if marker in task:
            return task.split(marker, 1)[0]
    return task


def object_id_from_config(raw_dir: Path, task: str) -> str:
    config_path = raw_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            object_names = config.get("object_names") or []
            if object_names:
                return str(object_names[0])
        except Exception:
            pass
    return object_id_from_task(task)


def load_obj(path: Path, max_vertices: int = 0) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for token in line.split()[1:]:
                    raw = token.split("/")[0]
                    if not raw:
                        continue
                    value = int(raw)
                    idx.append(value - 1 if value > 0 else len(vertices) + value)
                if len(idx) >= 3:
                    faces.append(idx[:3])
    verts = np.asarray(vertices, dtype=np.float32)
    fcs = np.asarray(faces, dtype=np.int32)
    if max_vertices > 0 and len(verts) > max_vertices:
        step = max(1, len(verts) // max_vertices)
        keep = np.zeros(len(verts), dtype=bool)
        keep[::step] = True
        remap = np.full(len(verts), -1, dtype=np.int32)
        remap[keep] = np.arange(int(keep.sum()), dtype=np.int32)
        valid_faces = keep[fcs].all(axis=1)
        fcs = remap[fcs[valid_faces]]
        verts = verts[keep]
    return verts, fcs


def quat_wxyz_to_matrix(quat: list[float] | np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def project_camera_points(points: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-4)
    uv = np.full((len(points), 2), -100000, dtype=np.int32)
    if valid.any():
        pts = points[valid]
        uv_float = np.stack(
            [K[0, 0] * pts[:, 0] / pts[:, 2] + K[0, 2], K[1, 1] * pts[:, 1] / pts[:, 2] + K[1, 2]],
            axis=1,
        )
        uv[valid] = np.round(uv_float).astype(np.int32)
    return uv, valid


def draw_hand_mesh(
    rgb: np.ndarray,
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    fill_color: tuple[int, int, int],
    edge_color: tuple[int, int, int],
) -> None:
    if vertices_cam is None or faces is None or len(vertices_cam) == 0 or len(faces) == 0:
        return
    h, w = rgb.shape[:2]
    uv, valid = project_camera_points(vertices_cam, K)
    if not valid.any():
        return
    overlay = rgb.copy()
    for face in faces:
        if np.any(face < 0) or np.any(face >= len(vertices_cam)) or not valid[face].all():
            continue
        tri = uv[face]
        if tri[:, 0].max() < -w or tri[:, 0].min() > 2 * w or tri[:, 1].max() < -h or tri[:, 1].min() > 2 * h:
            continue
        cv2.fillConvexPoly(overlay, tri.astype(np.int32), fill_color, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, rgb, 0.82, 0, dst=rgb)

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
            if -w < pa[0] < 2 * w and -h < pa[1] < 2 * h and -w < pb[0] < 2 * w and -h < pb[1] < 2 * h:
                cv2.line(rgb, pa, pb, edge_color, 1, cv2.LINE_AA)


def draw_object_mesh(
    rgb: np.ndarray,
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    max_faces: int,
) -> None:
    h, w = rgb.shape[:2]
    uv, valid = project_camera_points(vertices_cam, K)
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

    if faces is not None and len(faces) > 0:
        step = max(1, len(faces) // max_faces) if max_faces > 0 else 1
        drawn = set()
        for face in faces[::step]:
            if np.any(face < 0) or np.any(face >= len(vertices_cam)) or not valid[face].all():
                continue
            for a, b in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
                key = (a, b) if a < b else (b, a)
                if key in drawn:
                    continue
                drawn.add(key)
                pa = tuple(uv[a])
                pb = tuple(uv[b])
                if -w < pa[0] < 2 * w and -h < pa[1] < 2 * h and -w < pb[0] < 2 * w and -h < pb[1] < 2 * h:
                    cv2.line(rgb, pa, pb, (30, 255, 120), 1, cv2.LINE_AA)

    pts = uv[inside]
    if len(pts) == 0:
        return
    point_step = max(1, len(pts) // 250)
    for p in pts[::point_step]:
        cv2.circle(rgb, tuple(p), 1, (30, 230, 80), -1, cv2.LINE_AA)


def read_intrinsics(frame_dir: Path, frame_id: int, shape: tuple[int, int, int]) -> np.ndarray:
    npy = frame_dir / f"{frame_id:06d}_intrinsics.npy"
    if npy.exists():
        K = np.load(npy).astype(np.float32)
        if K.shape == (3, 3):
            return K
    h, w = shape[:2]
    focal = float(max(w, h) * 0.75)
    return np.array([[focal, 0.0, w * 0.5], [0.0, focal, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)


def parse_resolution(value: str | None) -> tuple[float, float] | None:
    if not value or "x" not in value:
        return None
    left, right = value.lower().split("x", 1)
    try:
        return float(left), float(right)
    except ValueError:
        return None


def scale_intrinsics_to_shape(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    source_width: float,
    source_height: float,
    shape: tuple[int, int, int],
) -> np.ndarray:
    h, w = shape[:2]
    sx = float(w) / max(float(source_width), 1e-6)
    sy = float(h) / max(float(source_height), 1e-6)
    return np.array(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def read_hand_intrinsics(hands: np.lib.npyio.NpzFile, shape: tuple[int, int, int]) -> np.ndarray | None:
    source_path = hands["source_hands_npz"] if "source_hands_npz" in hands else None
    if source_path is None:
        return None
    source = Path(str(source_path))
    if not source.exists():
        return None
    segment_dir = source.parent.parent.parent
    info_path = segment_dir / "ego_process" / "ego_undistorted_video" / "undistorted_video_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text())
            camera = info.get("cameraParams", {})
            resolution = parse_resolution(camera.get("resolution"))
            if resolution is not None:
                source_width, source_height = resolution
                fx = float(camera.get("fx_pixels"))
                fy = float(camera.get("fy_pixels", fx))
                cx = float(camera.get("cx_pixels", source_width * 0.5))
                cy = float(camera.get("cy_pixels", source_height * 0.5))
                return scale_intrinsics_to_shape(fx, fy, cx, cy, source_width, source_height, shape)
        except Exception:
            pass
    try:
        data = np.load(source, allow_pickle=True)
    except Exception:
        return None
    if "focal" not in data:
        return None
    focal = float(np.asarray(data["focal"]).reshape(-1)[0])
    h, w = shape[:2]
    return np.array([[focal, 0.0, w * 0.5], [0.0, focal, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)


def read_layout_intrinsics(pose_item: dict, shape: tuple[int, int, int]) -> np.ndarray | None:
    intr = pose_item.get("intrinsics_normalized")
    if not isinstance(intr, dict):
        return None
    h, w = shape[:2]
    fx = float(intr.get("fx_norm", 0.0)) * w
    fy = float(intr.get("fy_norm", 0.0)) * h
    if fx <= 0 or fy <= 0:
        return None
    cx = float(intr.get("cx_norm", 0.5)) * w
    cy = float(intr.get("cy_norm", 0.5)) * h
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def layout_scale(local_to_scene: dict, override: float) -> float:
    if override > 0:
        return float(override)
    scale = local_to_scene.get("scale", 1.0)
    if isinstance(scale, (list, tuple)) and scale:
        return float(scale[0])
    return float(scale)


def load_layout(raw_dir: Path, object_id: str) -> tuple[dict[int, dict], float]:
    layout_path = raw_dir / "obj_tracking_out" / object_id / "combined_visualization" / "layout_camera_frame_optimized.json"
    if not layout_path.exists():
        layout_path = raw_dir / "obj_tracking_out" / object_id / "combined_visualization" / "layout_camera_frame.json"
    if not layout_path.exists():
        layout_path = raw_dir / "obj_tracking_out" / object_id / "combined_visualization" / "layout.json"
    if not layout_path.exists():
        return {}, 0.0
    layout = json.loads(layout_path.read_text())
    optimized_scale = 0.0
    scale_meta = layout.get("translation_scale_optimization") or {}
    if layout.get("frame") == "camera_frame" and "mesh_scale" in scale_meta:
        optimized_scale = float(scale_meta["mesh_scale"])
    return {int(item["frame_idx"]): item for item in layout.get("objects", [])}, optimized_scale


def find_object_mesh(raw_dir: Path, object_id: str) -> Path | None:
    candidates = sorted((raw_dir / "video_segmentation" / "masks").glob(f"frame_*_masks/{object_id}/{object_id}.obj"))
    return candidates[0] if candidates else None


def resolve_object_source(clip_dir: Path, raw_dir: Path, object_id: str, override: Path | None) -> Path:
    if override is not None:
        return override
    clip_layout = clip_dir / "obj_tracking_out" / object_id / "combined_visualization"
    clip_masks = clip_dir / "video_segmentation" / "masks"
    if clip_layout.is_dir() and clip_masks.is_dir():
        return clip_dir
    return raw_dir


def find_hand_mesh_npz(raw_dir: Path, task: str) -> Path:
    candidates = [
        raw_dir / task / "all_hand_meshes.npz",
        raw_dir / "raw" / "all_hand_meshes.npz",
        raw_dir / "all_hand_meshes.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "missing hand mesh file; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Do-as-I-Do RGB overlay from reconstructed hand and object meshes.")
    parser.add_argument("--clip-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--task", default="foundation_jar_bimanual_leftscale")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--object-source-dir", type=Path, default=None, help="Optional reconstruction dir for object layout/mesh; default prefers clip-dir.")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-id-mode", choices=["auto", "source", "sequential"], default="auto")
    parser.add_argument("--hand-focal", type=float, default=0.0)
    parser.add_argument("--object-scale", type=float, default=0.0, help="Override object scale; 0 uses layout local_to_scene.scale.")
    parser.add_argument("--max-object-faces", type=int, default=1500)
    parser.add_argument("--background", choices=["rgb", "black", "white"], default="rgb")
    parser.add_argument(
        "--hand-intrinsics-source",
        choices=["frame", "source", "first-frame"],
        default="frame",
        help="Camera intrinsics for hand mesh projection. 'frame' uses all_frames/NNNNNN_intrinsics.npy.",
    )
    args = parser.parse_args()

    object_id = object_id_from_config(args.raw_dir, args.task)
    hand_npz_path = find_hand_mesh_npz(args.raw_dir, args.task)
    frame_dir = args.clip_dir / "all_frames"
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"missing RGB frame directory: {frame_dir}")

    hands = np.load(hand_npz_path, allow_pickle=True)
    source_frame_ids = hands["source_frame_ids"] if "source_frame_ids" in hands else np.arange(len(hands["left_vertices"]))
    if args.max_frames > 0:
        source_frame_ids = source_frame_ids[: args.max_frames]
    sequential_frame_ids = np.arange(len(source_frame_ids), dtype=np.int32)
    if args.frame_id_mode == "source":
        rgb_frame_ids = source_frame_ids
    elif args.frame_id_mode == "sequential":
        rgb_frame_ids = sequential_frame_ids
    else:
        first_source = frame_dir / f"{int(source_frame_ids[0]):06d}.png"
        rgb_frame_ids = source_frame_ids if first_source.exists() else sequential_frame_ids

    left_vertices = hands["left_vertices"]
    right_vertices = hands["right_vertices"]
    left_faces = hands["left_faces"]
    right_faces = hands["right_faces"]
    left_valid = hands["left_valid"] if "left_valid" in hands else np.ones(len(left_vertices), dtype=np.float32)
    right_valid = hands["right_valid"] if "right_valid" in hands else np.ones(len(right_vertices), dtype=np.float32)

    object_source_dir = resolve_object_source(args.clip_dir, args.raw_dir, object_id, args.object_source_dir)
    layout_by_frame, optimized_mesh_scale = load_layout(object_source_dir, object_id)
    object_mesh_path = find_object_mesh(object_source_dir, object_id)
    object_vertices = object_faces = None
    if object_mesh_path is not None:
        object_vertices, object_faces = load_obj(object_mesh_path)
    if args.object_scale <= 0 and optimized_mesh_scale > 0:
        print(f"Using optimized Do-as-I-Do mesh_scale={optimized_mesh_scale:.6f}")

    first_frame = cv2.imread(str(frame_dir / f"{int(rgb_frame_ids[0]):06d}.png"), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise FileNotFoundError(f"missing first RGB frame under {frame_dir}")
    h, w = first_frame.shape[:2]
    fixed_hand_K = None
    source_hand_K = read_hand_intrinsics(hands, first_frame.shape)
    if args.hand_focal > 0:
        fixed_hand_K = np.array([[args.hand_focal, 0.0, w * 0.5], [0.0, args.hand_focal, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    elif args.hand_intrinsics_source == "source":
        fixed_hand_K = source_hand_K
    elif args.hand_intrinsics_source == "first-frame":
        fixed_hand_K = read_intrinsics(frame_dir, int(rgb_frame_ids[0]), first_frame.shape)
    if fixed_hand_K is None and args.hand_intrinsics_source != "frame":
        fixed_hand_K = read_intrinsics(frame_dir, int(rgb_frame_ids[0]), first_frame.shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open {args.output}")

    for out_idx, frame_id_raw in enumerate(rgb_frame_ids):
        frame_id = int(frame_id_raw)
        bgr = cv2.imread(str(frame_dir / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if args.background == "black":
            rgb = np.zeros_like(rgb)
        elif args.background == "white":
            rgb = np.full_like(rgb, 255)
        object_K = read_intrinsics(frame_dir, frame_id, rgb.shape)
        hand_K = fixed_hand_K if fixed_hand_K is not None else read_intrinsics(frame_dir, frame_id, rgb.shape)
        if hand_K is None:
            hand_K = source_hand_K if source_hand_K is not None else object_K

        if out_idx < len(left_vertices) and float(left_valid[out_idx]) > 0:
            draw_hand_mesh(rgb, left_vertices[out_idx], left_faces, hand_K, (50, 130, 255), (95, 185, 255))
        if out_idx < len(right_vertices) and float(right_valid[out_idx]) > 0:
            draw_hand_mesh(rgb, right_vertices[out_idx], right_faces, hand_K, (245, 115, 45), (255, 165, 70))

        pose_item = layout_by_frame.get(frame_id)
        if pose_item is not None and object_vertices is not None:
            layout_K = read_layout_intrinsics(pose_item, rgb.shape)
            if layout_K is not None:
                object_K = layout_K
            local = pose_item["local_to_scene"]
            rotation = quat_wxyz_to_matrix(local["quat_wxyz_camera_frame"])
            translation = np.asarray(local["translation_camera_frame"], dtype=np.float32)
            object_scale = args.object_scale if args.object_scale > 0 else optimized_mesh_scale
            object_cam = (rotation @ (object_vertices * layout_scale(local, object_scale)).T).T + translation
            draw_object_mesh(rgb, object_cam, object_faces, object_K, args.max_object_faces)

        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    writer.release()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
