#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import pickle
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


COLORS = [
    (30, 225, 85),
    (255, 180, 40),
    (80, 170, 255),
    (255, 80, 140),
    (180, 120, 255),
    (40, 220, 220),
]


PLY_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def load_result(path: Path) -> dict:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_rgb(blob: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))


def decode_depth_png(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    arr = np.frombuffer(blob, dtype=np.uint8)
    depth_mm = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        return None
    return depth_mm.astype(np.float32) / 1000.0


def unpack_mask(obj: dict | None) -> np.ndarray | None:
    if not isinstance(obj, dict):
        return None
    packed = obj.get("mask_packed")
    shape = obj.get("mask_shape")
    if packed is None or shape is None:
        return None
    h, w = [int(x) for x in shape]
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))[: h * w]
    return bits.reshape(h, w).astype(bool)


def load_ply_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        header = []
        vertex_count = 0
        props = []
        fmt = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"invalid PLY header: {path}")
            header.append(line)
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("format "):
                fmt = text.split()[1]
            elif text.startswith("element vertex"):
                vertex_count = int(text.split()[-1])
            elif text.startswith("property ") and vertex_count >= 0:
                parts = text.split()
                if len(parts) >= 3 and parts[1] in PLY_DTYPES:
                    props.append((parts[-1], PLY_DTYPES[parts[1]]))
            elif text == "end_header":
                break
        offset = handle.tell()
    if fmt != "binary_little_endian":
        raise ValueError(f"only binary_little_endian PLY is supported: {path}")
    if not vertex_count or not props:
        raise ValueError(f"PLY missing vertices/properties: {path}")
    names = [name for name, _dtype in props]
    for need in ("x", "y", "z"):
        if need not in names:
            raise ValueError(f"PLY missing {need} property: {path}")
    dtype = np.dtype([(name, dtype) for name, dtype in props])
    arr = np.fromfile(path, dtype=dtype, offset=offset, count=vertex_count)
    return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)


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


def oid_get(mapping: dict, obj_id):
    return mapping.get(obj_id) or mapping.get(str(obj_id)) or mapping.get(int(obj_id))


def pose_sequence_for(result: dict, obj_id, pose_info: dict) -> np.ndarray | None:
    if isinstance(pose_info, dict) and pose_info.get("T_seq") is not None:
        return np.asarray(pose_info["T_seq"], dtype=np.float64)
    transforms = []
    last = None
    for frame in result.get("frame_data") or []:
        obj = oid_get(frame.get("sam3_obj_data") or {}, obj_id)
        if isinstance(obj, dict) and obj.get("pose_R") is not None and obj.get("pose_t") is not None:
            t = np.eye(4, dtype=np.float64)
            t[:3, :3] = np.asarray(obj["pose_R"], dtype=np.float64)
            t[:3, 3] = np.asarray(obj["pose_t"], dtype=np.float64).reshape(3)
            last = t
        elif last is not None:
            t = last.copy()
        else:
            t = None
        if t is not None:
            transforms.append(t)
    return np.stack(transforms, axis=0) if transforms else None


def mask_stats(mask: np.ndarray, depth: np.ndarray | None, focal: float) -> dict:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {"area_px": 0}
    bbox_px = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    bw = float(bbox_px[2] - bbox_px[0] + 1)
    bh = float(bbox_px[3] - bbox_px[1] + 1)
    out = {
        "area_px": int(len(xs)),
        "bbox_px": bbox_px,
        "bbox_size_px": [bw, bh],
        "centroid_px": [float(xs.mean()), float(ys.mean())],
    }
    if depth is not None:
        z = depth[ys, xs]
        z = z[np.isfinite(z) & (z > 0.1)]
        if len(z) >= 30:
            z_med = float(np.median(z))
            out["median_depth_m"] = z_med
            out["expected_size_from_mask_m"] = {
                "width": bw * z_med / focal,
                "height": bh * z_med / focal,
                "max": max(bw, bh) * z_med / focal,
            }
    return out


def expected_size_from_mask(mask: np.ndarray, depth: np.ndarray | None, fx: float, cx: float, cy: float) -> dict | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or depth is None:
        return None
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z > 0.1)
    if int(valid.sum()) < 30:
        return None
    xs_v = xs[valid]
    ys_v = ys[valid]
    z_v = z[valid]
    z_med = float(np.median(z_v))
    bw = float(xs.max() - xs.min() + 1)
    bh = float(ys.max() - ys.min() + 1)
    points = np.stack([(xs_v - cx) * z_v / fx, (ys_v - cy) * z_v / fx, z_v], axis=1)
    if len(points) > 5000:
        points = points[np.random.RandomState(23).choice(len(points), 5000, replace=False)]
    bbox3 = points.max(axis=0) - points.min(axis=0)
    return {
        "median_depth_m": z_med,
        "bbox_size_px": [bw, bh],
        "expected_size_from_2d_m": {
            "width": bw * z_med / fx,
            "height": bh * z_med / fx,
            "max": max(bw, bh) * z_med / fx,
        },
        "observed_pointcloud_bbox_m": bbox3.tolist(),
        "observed_pointcloud_max_m": float(bbox3.max()),
    }


def project_points(points_cam: np.ndarray, fx: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-4)
    uv = np.zeros((len(points_cam), 2), dtype=np.float64)
    uv[valid, 0] = cx + fx * points_cam[valid, 0] / z[valid]
    uv[valid, 1] = cy + fx * points_cam[valid, 1] / z[valid]
    return uv, valid


def projected_bbox(
    vertices: np.ndarray,
    scale: float,
    transform: np.ndarray,
    fx: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> dict | None:
    if len(vertices) > 20000:
        vertices = vertices[:: max(1, len(vertices) // 20000)]
    points_cam = (transform[:3, :3] @ (vertices * scale).T).T + transform[:3, 3]
    uv, valid = project_points(points_cam, fx, cx, cy)
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if not inside.any():
        return {"visible_points": 0, "valid_points": int(valid.sum())}
    uvi = uv[inside]
    bbox = [float(uvi[:, 0].min()), float(uvi[:, 1].min()), float(uvi[:, 0].max()), float(uvi[:, 1].max())]
    return {
        "visible_points": int(inside.sum()),
        "valid_points": int(valid.sum()),
        "inside_fraction": float(inside.sum() / max(int(valid.sum()), 1)),
        "bbox_px": bbox,
        "bbox_size_px": [bbox[2] - bbox[0] + 1.0, bbox[3] - bbox[1] + 1.0],
    }


def draw_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    overlay = rgb.copy()
    overlay[mask] = np.asarray(color, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.35, rgb, 0.65, 0, dst=rgb)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, color, 2, cv2.LINE_AA)
    ys, xs = np.where(mask)
    if len(xs):
        cv2.putText(rgb, label, (int(xs.min()), max(24, int(ys.min()) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def draw_mesh_projection(
    rgb: np.ndarray,
    vertices: np.ndarray,
    scale: float,
    transform: np.ndarray,
    fx: float,
    cx: float,
    cy: float,
    color: tuple[int, int, int],
    label: str,
) -> None:
    h, w = rgb.shape[:2]
    if len(vertices) > 15000:
        vertices = vertices[:: max(1, len(vertices) // 15000)]
    pts = (transform[:3, :3] @ (vertices * scale).T).T + transform[:3, 3]
    uv, valid = project_points(pts, fx, cx, cy)
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    if not inside.any():
        return
    uvi = np.round(uv[inside]).astype(np.int32)
    uvi[:, 0] = np.clip(uvi[:, 0], 0, w - 1)
    uvi[:, 1] = np.clip(uvi[:, 1], 0, h - 1)
    canvas = np.zeros((h, w), dtype=np.uint8)
    canvas[uvi[:, 1], uvi[:, 0]] = 255
    canvas = cv2.dilate(canvas, np.ones((3, 3), np.uint8), iterations=1)
    overlay = rgb.copy()
    overlay[canvas > 0] = np.asarray(color, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.50, rgb, 0.50, 0, dst=rgb)
    y, x = np.where(canvas > 0)
    cv2.putText(rgb, label, (int(x.min()), max(24, int(y.min()) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)


def default_sample_frames(n_frames: int, count: int) -> list[int]:
    if n_frames <= 0:
        return []
    idxs = np.linspace(0, n_frames - 1, min(count, n_frames)).round().astype(int).tolist()
    return sorted(set(int(x) for x in idxs))


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose EgoInfinity SAM3.1/SAM3D object scale and pose without changing the pkl.")
    parser.add_argument("--pipeline-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-prompt", default=None)
    parser.add_argument("--sample-count", type=int, default=7)
    parser.add_argument("--scale-threshold", type=float, default=1.8)
    parser.add_argument(
        "--trusted-pickle",
        action="store_true",
        required=True,
        help="Acknowledge that the pipeline result pickle is trusted local input.",
    )
    args = parser.parse_args()

    result = load_result(args.pipeline_result)
    frames = result.get("frame_data") or []
    if not frames:
        raise RuntimeError("pipeline result has no frame_data")
    fx = float(result["dp_focal"])
    cx = float(result["cx"])
    cy = float(result["cy"])
    first_rgb = decode_rgb(frames[0]["img_rgb"])
    height, width = first_rgb.shape[:2]
    mesh_info = result.get("sam3_mesh_info") or {}
    pose_info_all = result.get("pose_track_info") or {}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    key_dir = args.output_dir / "keyframes"
    key_dir.mkdir(parents=True, exist_ok=True)

    sample_frames = default_sample_frames(len(frames), args.sample_count)
    object_rows = []
    per_frame_masks: dict[int, dict[int, np.ndarray]] = {}
    selected_oids = []
    for obj_id, info in mesh_info.items():
        prompt = object_prompt(result, obj_id)
        if args.target_prompt and prompt != args.target_prompt:
            continue
        selected_oids.append(int(obj_id))
    selected_oids = sorted(selected_oids)
    if not selected_oids:
        raise RuntimeError("no matching EgoInfinity objects")

    for obj_id in selected_oids:
        info = oid_get(mesh_info, obj_id)
        pose_info = oid_get(pose_info_all, obj_id) or {}
        ply = Path(str(info.get("ply_path", "")))
        vertices = load_ply_vertices(ply)
        raw_bbox = vertices.max(axis=0) - vertices.min(axis=0)
        raw_centroid = vertices.mean(axis=0)
        canonical_scale = float(info.get("canonical_scale") or 1.0)
        pose_scale = float(pose_info.get("scale_correction") or canonical_scale)
        transforms = pose_sequence_for(result, obj_id, pose_info)

        frame_rows = []
        expected_max_values = []
        observed_max_values = []
        proj_max_values = []
        for frame_idx in sample_frames:
            frame = frames[frame_idx]
            obj = oid_get(frame.get("sam3_obj_data") or {}, obj_id)
            mask = unpack_mask(obj)
            if mask is None or not mask.any():
                frame_rows.append({"frame": frame_idx, "has_mask": False})
                continue
            per_frame_masks.setdefault(frame_idx, {})[obj_id] = mask
            depth = decode_depth_png(frame.get("depth_png"))
            exp = expected_size_from_mask(mask, depth, fx, cx, cy)
            row = {
                "frame": frame_idx,
                "has_mask": True,
                "mask": mask_stats(mask, depth, fx),
            }
            if exp is not None:
                row["physical_size_from_mask_depth"] = exp
                expected_max_values.append(exp["expected_size_from_2d_m"]["max"])
                observed_max_values.append(exp["observed_pointcloud_max_m"])
            if transforms is not None and frame_idx < len(transforms):
                proj = projected_bbox(vertices, pose_scale, transforms[frame_idx], fx, cx, cy, width, height)
                row["projected_mesh_pose_scale"] = proj
                if proj is not None and "bbox_size_px" in proj:
                    proj_max_values.append(max(proj["bbox_size_px"]))
            frame_rows.append(row)

        expected_75 = float(np.percentile(expected_max_values, 75)) if expected_max_values else None
        observed_75 = float(np.percentile(observed_max_values, 75)) if observed_max_values else None
        pose_real_bbox = raw_bbox * pose_scale
        canonical_real_bbox = raw_bbox * canonical_scale
        scale_check = {}
        if expected_75 is not None and raw_bbox.max() > 1e-9:
            current_max = float(pose_real_bbox.max())
            canonical_max = float(canonical_real_bbox.max())
            scale_check = {
                "expected_max_m_75p": expected_75,
                "observed_pointcloud_max_m_75p": observed_75,
                "pose_scale_real_max_m": current_max,
                "canonical_scale_real_max_m": canonical_max,
                "pose_scale_over_expected": current_max / max(expected_75, 1e-9),
                "canonical_scale_over_expected": canonical_max / max(expected_75, 1e-9),
                "nonhard_physical_scale_candidate": expected_75 / float(raw_bbox.max()),
                "would_trigger_scale_sanity_threshold": bool(current_max / max(expected_75, 1e-9) > args.scale_threshold),
            }
        object_rows.append(
            {
                "oid": obj_id,
                "prompt": object_prompt(result, obj_id),
                "prompt_score": object_prompt_score(result, obj_id),
                "ply_path": str(ply),
                "ply_sha256": sha256(ply),
                "n_vertices": int(len(vertices)),
                "mesh_raw_bbox": raw_bbox.tolist(),
                "mesh_raw_centroid": raw_centroid.tolist(),
                "canonical_scale": canonical_scale,
                "pose_scale_correction": pose_scale,
                "mesh_bbox_with_canonical_scale_m": canonical_real_bbox.tolist(),
                "mesh_bbox_with_pose_scale_m": pose_real_bbox.tolist(),
                "pose_track_keys": sorted(str(k) for k in pose_info.keys()),
                "pose_track_mode": pose_info.get("mode"),
                "pose_track_note": pose_info.get("note"),
                "T_seq_shape": list(np.asarray(pose_info.get("T_seq")).shape) if pose_info.get("T_seq") is not None else None,
                "sample_frames": frame_rows,
                "projected_mesh_max_px_samples": proj_max_values,
                "scale_check": scale_check,
            }
        )

    overlaps = []
    for frame_idx, masks in sorted(per_frame_masks.items()):
        for a, b in combinations(sorted(masks), 2):
            ma, mb = masks[a], masks[b]
            inter = int((ma & mb).sum())
            union = int((ma | mb).sum())
            overlaps.append({"frame": frame_idx, "oid_a": a, "oid_b": b, "iou": inter / max(union, 1), "intersection_px": inter})

    for frame_idx in sample_frames:
        rgb = decode_rgb(frames[frame_idx]["img_rgb"]).copy()
        mesh_rgb = rgb.copy()
        for pos, obj_id in enumerate(selected_oids):
            color = COLORS[pos % len(COLORS)]
            obj = oid_get(frames[frame_idx].get("sam3_obj_data") or {}, obj_id)
            mask = unpack_mask(obj)
            if mask is not None and mask.any():
                draw_mask(rgb, mask, color, f"oid {obj_id}")
            info = oid_get(mesh_info, obj_id)
            pose_info = oid_get(pose_info_all, obj_id) or {}
            transforms = pose_sequence_for(result, obj_id, pose_info)
            if transforms is not None and frame_idx < len(transforms):
                vertices = load_ply_vertices(Path(str(info.get("ply_path"))))
                scale = float(pose_info.get("scale_correction") or info.get("canonical_scale") or 1.0)
                draw_mesh_projection(mesh_rgb, vertices, scale, transforms[frame_idx], fx, cx, cy, color, f"oid {obj_id}")
        cv2.imwrite(str(key_dir / f"frame_{frame_idx:04d}_mask_overlay.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(key_dir / f"frame_{frame_idx:04d}_mesh_projection.png"), cv2.cvtColor(mesh_rgb, cv2.COLOR_RGB2BGR))

    report = {
        "scope": "diagnostic_only_object_scale_evidence",
        "automatic_scale_application_allowed": False,
        "pipeline_result": str(args.pipeline_result),
        "pipeline_result_sha256": sha256(args.pipeline_result),
        "target_prompt": args.target_prompt,
        "camera": {"fx": fx, "fy": fx, "cx": cx, "cy": cy, "width": width, "height": height},
        "sample_frames": sample_frames,
        "selected_oids": selected_oids,
        "objects": object_rows,
        "mask_pair_overlaps": overlaps,
    }
    report_path = args.output_dir / "scale_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
