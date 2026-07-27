#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from hand_selection_utils import geometry_choice


def object_id_from_config(raw_dir: Path, task: str) -> str:
    config_path = raw_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            object_names = config.get("object_names") or []
            if object_names:
                return str(object_names[0])
        except Exception:
            pass
    return task


def hand_type_from_task(task: str) -> str | None:
    for marker, value in (
        ("_bimanual", "bimanual"),
        ("_right", "right"),
        ("_left", "left"),
    ):
        if marker in task:
            return value
    return None


def anchor_hand_from_config(raw_dir: Path) -> str:
    config_path = raw_dir / "config.json"
    if not config_path.exists():
        return "bimanual"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        value = str(config.get("anchor_hand") or "bimanual").lower()
        if value in {"left", "right", "bimanual"}:
            return value
    except Exception:
        pass
    return "bimanual"


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


def build_hand_index_map(
    source_frame_ids: np.ndarray,
    rgb_frame_ids: np.ndarray,
    hand_frame_count: int,
) -> dict[int, int]:
    rgb = [int(x) for x in np.asarray(rgb_frame_ids).reshape(-1)]
    if rgb and min(rgb) >= 0 and max(rgb) < hand_frame_count:
        return {frame_id: frame_id for frame_id in rgb}
    mapping = {}
    for idx, frame_id in enumerate(np.asarray(source_frame_ids).reshape(-1)):
        if idx >= hand_frame_count:
            break
        mapping[int(frame_id)] = idx
    return mapping


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
    if "ego_hand_projection_K" in hands:
        try:
            K = np.asarray(hands["ego_hand_projection_K"], dtype=np.float32)
            size = np.asarray(hands["ego_hand_projection_size"], dtype=np.float32).reshape(-1)
            if K.shape == (3, 3) and size.size >= 2:
                return scale_intrinsics_to_shape(
                    float(K[0, 0]),
                    float(K[1, 1]),
                    float(K[0, 2]),
                    float(K[1, 2]),
                    float(size[0]),
                    float(size[1]),
                    shape,
                )
        except Exception:
            pass
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
            info = json.loads(info_path.read_text(encoding="utf-8"))
            camera = info.get("cameraParams", {})
            resolution = parse_resolution(camera.get("resolution"))
            if resolution is not None:
                source_width, source_height = resolution
                fx = float(camera.get("fx_pixels"))
                fy = float(camera.get("fy_pixels", fx))
                cx = float(camera.get("cx_pixels", source_width * 0.5))
                cy = float(camera.get("cy_pixels", source_height * 0.5))
                return scale_intrinsics_to_shape(
                    fx, fy, cx, cy, source_width, source_height, shape
                )
        except Exception:
            pass
    return None


def read_intrinsics(
    frame_dir: Path,
    frame_id: int,
    shape: tuple[int, int, int],
    hands: np.lib.npyio.NpzFile,
) -> np.ndarray:
    npy = frame_dir / f"{frame_id:06d}_intrinsics.npy"
    if npy.exists():
        K = np.load(npy).astype(np.float32)
        if K.shape == (3, 3):
            return K
    source_K = read_hand_intrinsics(hands, shape)
    if source_K is not None:
        return source_K
    h, w = shape[:2]
    focal = float(max(w, h) * 0.75)
    return np.array([[focal, 0.0, w * 0.5], [0.0, focal, h * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)


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


def bbox_from_points(
    uv: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if not inside.any():
        return None
    pts = uv[inside].astype(np.float32)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def bbox_overlap(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> float:
    if a is None or b is None:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_center(box: tuple[float, float, float, float]) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def object_mask_bbox(raw_dir: Path, object_id: str, frame_id: int) -> tuple[float, float, float, float] | None:
    mask_root = raw_dir / "video_segmentation" / "masks" / f"frame_{frame_id:06d}_masks"
    candidates = [
        mask_root / f"{object_id}.png",
        mask_root / object_id / f"{object_id}.png",
    ]
    mask = None
    for mask_path in candidates:
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            break
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.where(mask > 0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def hand_mask_bbox(raw_dir: Path, side: str, frame_id: int) -> tuple[float, float, float, float] | None:
    mask_root = raw_dir / "video_segmentation" / "masks" / f"frame_{frame_id:06d}_masks"
    candidates = [
        mask_root / f"{side}_hand_0.png",
        mask_root / f"{side}_hand.png",
    ]
    mask = None
    for mask_path in candidates:
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            break
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.where(mask > 0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def bbox_area(box: tuple[float, float, float, float] | None) -> float:
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_gap(a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None) -> float:
    if a is None or b is None:
        return float("inf")
    dx = max(float(b[0] - a[2]), float(a[0] - b[2]), 0.0)
    dy = max(float(b[1] - a[3]), float(a[1] - b[3]), 0.0)
    return float(np.hypot(dx, dy))


def resolve_rgb_frame_ids(frame_dir: Path, source_frame_ids: np.ndarray) -> np.ndarray:
    if len(source_frame_ids) <= 0:
        return source_frame_ids
    first_source = frame_dir / f"{int(source_frame_ids[0]):06d}.png"
    if first_source.exists():
        return source_frame_ids
    return np.arange(len(source_frame_ids), dtype=np.int32)


def score_hand_sides(raw_dir: Path, task: str, max_samples: int) -> dict:
    object_id = object_id_from_config(raw_dir, task)
    hands = np.load(find_hand_mesh_npz(raw_dir, task), allow_pickle=True)
    frame_dir = raw_dir / "all_frames"
    source_frame_ids = hands["source_frame_ids"] if "source_frame_ids" in hands else np.arange(len(hands["left_vertices"]))
    rgb_frame_ids = resolve_rgb_frame_ids(frame_dir, source_frame_ids)
    hand_index_by_frame = build_hand_index_map(source_frame_ids, rgb_frame_ids, len(hands["left_vertices"]))

    sample_ids = list(range(len(rgb_frame_ids)))
    if len(sample_ids) > max_samples:
        sample_ids = np.linspace(0, len(rgb_frame_ids) - 1, max_samples, dtype=int).tolist()
    scores: dict[str, dict[str, float]] = {
        "left": {"score": 0.0, "frames": 0.0, "overlap_frames": 0.0},
        "right": {"score": 0.0, "frames": 0.0, "overlap_frames": 0.0},
    }
    mask_scores: dict[str, dict[str, float]] = {
        "left": {"score": 0.0, "frames": 0.0, "overlap_frames": 0.0, "touch_frames": 0.0, "mean_gap": 0.0},
        "right": {"score": 0.0, "frames": 0.0, "overlap_frames": 0.0, "touch_frames": 0.0, "mean_gap": 0.0},
    }
    for out_idx in sample_ids:
        frame_id = int(rgb_frame_ids[out_idx])
        hand_idx = hand_index_by_frame.get(frame_id)
        if hand_idx is None:
            continue
        img = cv2.imread(str(frame_dir / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
        if img is None:
            continue
        height, width = img.shape[:2]
        obj_box = object_mask_bbox(raw_dir, object_id, frame_id)
        if obj_box is None:
            continue
        obj_area = max(bbox_area(obj_box), 1.0)
        image_diag = max(float(np.hypot(width, height)), 1.0)
        obj_center = bbox_center(obj_box)
        K = read_intrinsics(frame_dir, frame_id, img.shape, hands)
        for side in ("left", "right"):
            hand_mask_box = hand_mask_bbox(raw_dir, side, frame_id)
            if hand_mask_box is not None:
                hand_area = max(bbox_area(hand_mask_box), 1.0)
                overlap = bbox_overlap(hand_mask_box, obj_box)
                gap = bbox_gap(hand_mask_box, obj_box)
                hand_center = bbox_center(hand_mask_box)
                dist = float(np.linalg.norm(hand_center - obj_center))
                proximity = max(0.0, 1.0 - dist / (0.40 * image_diag))
                overlap_score = overlap / min(obj_area, hand_area)
                touch = 1.0 if gap <= 18.0 or overlap > 0.0 else 0.0
                gap_score = max(0.0, 1.0 - gap / (0.18 * image_diag))
                mask_scores[side]["score"] += overlap_score * 5.0 + touch * 2.0 + gap_score + proximity
                mask_scores[side]["frames"] += 1.0
                mask_scores[side]["mean_gap"] += 0.0 if not np.isfinite(gap) else gap
                if overlap > 0:
                    mask_scores[side]["overlap_frames"] += 1.0
                if touch > 0:
                    mask_scores[side]["touch_frames"] += 1.0

            vertices_key = f"{side}_vertices"
            valid_key = f"{side}_valid"
            if vertices_key not in hands or hand_idx >= len(hands[vertices_key]):
                continue
            valid_frames = hands[valid_key] if valid_key in hands else None
            if valid_frames is not None and float(valid_frames[hand_idx]) <= 0:
                continue
            uv, valid = project_camera_points(hands[vertices_key][hand_idx], K)
            hand_box = bbox_from_points(uv, valid, width, height)
            if hand_box is None:
                continue
            overlap = bbox_overlap(hand_box, obj_box)
            hand_center = bbox_center(hand_box)
            dist = float(np.linalg.norm(hand_center - obj_center))
            proximity = max(0.0, 1.0 - dist / (0.35 * image_diag))
            overlap_score = overlap / obj_area
            scores[side]["score"] += overlap_score * 4.0 + proximity
            scores[side]["frames"] += 1.0
            if overlap > 0:
                scores[side]["overlap_frames"] += 1.0
    for side, values in mask_scores.items():
        if values["frames"] > 0:
            values["mean_gap"] /= values["frames"]
    normalized = {
        side: (values["score"] / max(values["frames"], 1.0))
        for side, values in scores.items()
    }
    mask_normalized = {
        side: (values["score"] / max(values["frames"], 1.0))
        for side, values in mask_scores.items()
    }
    return {
        "object_id": object_id,
        "scores": scores,
        "normalized": normalized,
        "mask_scores": mask_scores,
        "mask_normalized": mask_normalized,
        "config_anchor_hand": anchor_hand_from_config(raw_dir),
    }


def adapter_geometry_hand_choice(raw_dir: Path) -> dict:
    manifest_path = raw_dir / "adapter_manifest.json"
    if not manifest_path.exists():
        return {"available": False, "reason": "missing_adapter_manifest"}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"available": False, "reason": f"invalid_adapter_manifest:{type(exc).__name__}"}
    geometry = manifest.get("geometry_interaction_hand") or {}
    choice = geometry_choice(geometry, manifest.get("selected_hand"))
    manifest_selected = manifest.get("selected_hand")
    visual_selected = (manifest.get("visual_interaction_hand") or {}).get("selected_hand")
    manifest_visual_agree = (
        manifest_selected in {"left", "right"}
        and visual_selected == manifest_selected
    )
    choice["manifest_visual_agree"] = manifest_visual_agree
    if choice.get("selected_hand") is None and manifest_visual_agree:
        choice.update(
            available=True,
            selected_hand=manifest_selected,
            reason="adapter-manifest-visual-agreement",
        )
    return choice


def strong_single_hand_mask_choice(result: dict) -> dict:
    """Return a dominant mask-side only when the evidence is unambiguous.

    Adapter geometry is measured in reconstructed 3D and can label both hands as
    interacting when an unrelated hand happens to be close to the object.  The
    RGB masks are the better slot identity signal in that case, but only a wide
    score/contact margin is strong enough to override a bimanual decision.
    """
    normalized = result.get("mask_normalized") or {}
    stats = result.get("mask_scores") or {}
    left_score = float(normalized.get("left", 0.0) or 0.0)
    right_score = float(normalized.get("right", 0.0) or 0.0)
    side = "left" if left_score >= right_score else "right"
    other = "right" if side == "left" else "left"
    best_score = max(left_score, right_score)
    other_score = min(left_score, right_score)
    best_stats = stats.get(side) or {}
    other_stats = stats.get(other) or {}
    frames = float(best_stats.get("frames", 0.0) or 0.0)
    touch = float(best_stats.get("touch_frames", 0.0) or 0.0)
    other_touch = float(other_stats.get("touch_frames", 0.0) or 0.0)
    overlap = float(best_stats.get("overlap_frames", 0.0) or 0.0)
    other_overlap = float(other_stats.get("overlap_frames", 0.0) or 0.0)
    ratio = best_score / max(other_score, 1e-6)
    min_touch = max(4.0, 0.35 * frames)
    min_margin = max(3.0, 0.20 * frames)
    strong = (
        frames >= 8.0
        and best_score >= 1.0
        and ratio >= 1.75
        and touch >= min_touch
        and (
            touch - other_touch >= min_margin
            or overlap - other_overlap >= min_margin
        )
    )
    return {
        "available": frames > 0.0,
        "strong": bool(strong),
        "selected_hand": side if strong else None,
        "score_ratio": ratio,
        "touch_margin": touch - other_touch,
        "overlap_margin": overlap - other_overlap,
        "sampled_frames": frames,
    }


def resolve_hand_type(raw_dir: Path, task: str, max_samples: int) -> dict:
    result = score_hand_sides(raw_dir, task, max_samples)
    adapter_choice = adapter_geometry_hand_choice(raw_dir)
    result["adapter_geometry_choice"] = adapter_choice
    mask_override = strong_single_hand_mask_choice(result)
    result["strong_single_hand_mask_choice"] = mask_override
    task_choice = hand_type_from_task(task)
    result["task_hand_type"] = task_choice
    mask_left = float(result["mask_normalized"]["left"])
    mask_right = float(result["mask_normalized"]["right"])
    left = float(result["normalized"]["left"])
    right = float(result["normalized"]["right"])
    mask_left_frames = float(result["mask_scores"]["left"]["frames"])
    mask_right_frames = float(result["mask_scores"]["right"]["frames"])
    left_overlap = float(result["mask_scores"]["left"]["overlap_frames"] or result["scores"]["left"]["overlap_frames"])
    right_overlap = float(result["mask_scores"]["right"]["overlap_frames"] or result["scores"]["right"]["overlap_frames"])
    left_touch = float(result["mask_scores"]["left"]["touch_frames"])
    right_touch = float(result["mask_scores"]["right"]["touch_frames"])
    config_choice = str(result["config_anchor_hand"])
    if max(mask_left_frames, mask_right_frames) > 0 and max(mask_left, mask_right) > 0.0:
        left_value, right_value = mask_left, mask_right
        source = "hand-object-mask"
    else:
        left_value, right_value = left, right
        source = "mesh-projection"
    if max(left_value, right_value) <= 0.0:
        selected = config_choice
        reason = "config-fallback"
    elif left_value >= right_value * 1.2:
        selected = "left"
        reason = source
    elif right_value >= left_value * 1.2:
        selected = "right"
        reason = source
    elif (left_touch > 0 and right_touch > 0) or (left_overlap > 0 and right_overlap > 0):
        selected = "bimanual"
        reason = f"{source}-bimanual"
    else:
        selected = "left" if left_value >= right_value else "right"
        reason = f"{source}-tie-break"
    adapter_selected = adapter_choice.get("selected_hand") if adapter_choice.get("available") else None
    adapter_manifest_visual_agree = bool(adapter_choice.get("manifest_visual_agree"))
    if adapter_selected in {"left", "right", "bimanual"}:
        mask_selected = mask_override.get("selected_hand") if mask_override.get("strong") else None
        if (
            mask_selected in {"left", "right"}
            and mask_selected != adapter_selected
            and not adapter_manifest_visual_agree
        ):
            selected = str(mask_selected)
            reason = f"mask-overrides-adapter-{adapter_selected}"
            source = "hand-object-mask"
        else:
            selected = adapter_selected
            reason = str(adapter_choice.get("reason") or "adapter-geometry")
            source = "adapter-geometry"
    elif task_choice in {"left", "right", "bimanual"} and max(left_value, right_value) <= 0.0:
        selected = task_choice
        reason = "task-suffix-fallback"
        source = "task"

    # A single-hand task suffix is an explicit interaction-side annotation.
    # Keep it when visual/geometry evidence is ambiguous; only a strong mask
    # signal from the opposite hand may override it. This prevents clips such
    # as phone_left from silently becoming bimanual merely because both hands
    # touch the object's 2D bounding box.
    if task_choice in {"left", "right"}:
        opposite = "right" if task_choice == "left" else "left"
        mask_selected = mask_override.get("selected_hand") if mask_override.get("strong") else None
        adapter_confirms_task = (
            adapter_manifest_visual_agree and adapter_selected == task_choice
        )
        if mask_selected == opposite and not adapter_confirms_task:
            selected = opposite
            reason = f"strong-mask-overrides-task-{task_choice}"
            source = "hand-object-mask"
        elif selected in {"bimanual", opposite}:
            selected = task_choice
            reason = "explicit-task-side-overrides-ambiguous-evidence"
            source = "task"
    result["selected_hand_type"] = selected
    result["selection_reason"] = reason
    result["selection_source"] = source if max(left_value, right_value) > 0.0 else "config"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve Do-as-I-Do retarget hand type from raw RGB/object interaction.")
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = resolve_hand_type(args.raw_dir.expanduser().resolve(), args.task, int(args.max_samples))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(str(result["selected_hand_type"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
