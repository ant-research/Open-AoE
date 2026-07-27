#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import json

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def object_id_from_task(task: str) -> str:
    for marker in ["_bimanual", "_right", "_left"]:
        if marker in task:
            return task.split(marker, 1)[0]
    return task


def anchor_hand_from_config(clip_dir: Path) -> str:
    config_path = clip_dir / "config.json"
    if not config_path.exists():
        return "both"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        value = str(config.get("anchor_hand") or "both").lower()
        if value == "bimanual":
            return "both"
        if value in {"left", "right", "both"}:
            return value
    except Exception:
        pass
    return "both"


def official_hands_arg(hand_side: str, clip_dir: Path) -> str:
    if hand_side in {"left", "right"}:
        return hand_side
    if hand_side in {"all", "bimanual"}:
        return "both"
    return anchor_hand_from_config(clip_dir)


def hand_focal_from_clip(clip_dir: Path) -> float | None:
    candidates: list[Path] = []
    raw_npz = clip_dir / "raw" / "all_hand_meshes.npz"
    if raw_npz.exists():
        try:
            with np.load(raw_npz, allow_pickle=False) as data:
                if "source_hands_npz" in data:
                    value = data["source_hands_npz"]
                    source = value.item() if getattr(value, "shape", ()) == () else value
                    candidates.append(Path(str(source)))
        except Exception as exc:
            print(f"[warn] could not read source_hands_npz from {raw_npz}: {exc}")
    metadata_path = clip_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("hands_npz"):
                candidates.append(Path(str(metadata["hands_npz"])))
        except Exception as exc:
            print(f"[warn] could not read hand focal metadata: {exc}")
    for path in candidates:
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                if "focal" in data:
                    focal = float(data["focal"])
                    if np.isfinite(focal) and focal > 0:
                        return focal
        except Exception as exc:
            print(f"[warn] could not read hand focal from {path}: {exc}")
    return None


def load_obj_vertices(path: Path) -> np.ndarray:
    verts = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"no vertices found in {path}")
    return np.asarray(verts, dtype=np.float32)


def quat_wxyz_to_matrix(quat: list[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.asarray(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def project_bbox(vertices_cam: np.ndarray, intrinsics: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float] | None:
    z = vertices_cam[:, 2]
    valid = np.isfinite(vertices_cam).all(axis=1) & (z > 1e-6)
    if valid.sum() < 10:
        return None
    pts = vertices_cam[valid]
    u = intrinsics[0, 0] * pts[:, 0] / pts[:, 2] + intrinsics[0, 2]
    v = intrinsics[1, 1] * pts[:, 1] / pts[:, 2] + intrinsics[1, 2]
    finite = np.isfinite(u) & np.isfinite(v)
    if finite.sum() < 10:
        return None
    u = u[finite]
    v = v[finite]
    inside = float(((u >= 0) & (u < width) & (v >= 0) & (v < height)).mean())
    return np.asarray([u.min(), v.min(), u.max(), v.max()], dtype=np.float32), inside


def bbox_gap(a: np.ndarray, b: np.ndarray) -> float:
    dx = max(float(b[0] - a[2]), float(a[0] - b[2]), 0.0)
    dy = max(float(b[1] - a[3]), float(a[1] - b[3]), 0.0)
    return float(np.hypot(dx, dy))


def choose_overlay_hand(clip_dir: Path, object_id: str, mesh: Path, layout: Path) -> str:
    hand_meshes = clip_dir / "raw" / "all_hand_meshes.npz"
    if not hand_meshes.exists():
        return anchor_hand_from_config(clip_dir)
    try:
        layout_data = json.loads(layout.read_text(encoding="utf-8"))
        objects = [obj for obj in layout_data["objects"] if "frame_idx" in obj or "frame_index" in obj]
        objects.sort(key=lambda obj: int(obj.get("frame_idx", obj.get("frame_index"))))
        verts = load_obj_vertices(mesh)
        hands = np.load(hand_meshes, allow_pickle=False)
    except Exception as exc:
        print(f"[warn] could not auto-select overlay hand: {exc}")
        return anchor_hand_from_config(clip_dir)

    if not objects:
        return anchor_hand_from_config(clip_dir)

    sample_indices = sorted(set(np.linspace(0, len(objects) - 1, num=min(12, len(objects)), dtype=int).tolist()))
    scores = {"left": [], "right": []}
    width, height = 1280, 720
    is_camera_frame_layout = layout_data.get("frame") == "camera_frame"
    scale_override = None
    try:
        scale_override = float((layout_data.get("translation_scale_optimization") or {}).get("mesh_scale"))
    except (TypeError, ValueError):
        scale_override = None

    for sample_idx in sample_indices:
        obj = objects[sample_idx]
        frame_idx = int(obj.get("frame_idx", obj.get("frame_index")))
        intr_path = clip_dir / "all_frames" / f"{frame_idx:06d}_intrinsics.npy"
        if not intr_path.exists():
            continue
        intr = np.load(intr_path)
        pose = obj["local_to_scene"]
        scale = scale_override if scale_override is not None else float(pose["scale"][0])
        if is_camera_frame_layout:
            quat = pose["quat_wxyz_camera_frame"]
            trans = np.asarray(pose["translation_camera_frame"], dtype=np.float32)
            obj_cam = (verts * scale) @ quat_wxyz_to_matrix(quat).T + trans
        else:
            quat = pose["new_quat"]
            tx, ty, tz = pose["translation"]
            pose_space = (verts * scale) @ quat_wxyz_to_matrix(quat).T + np.asarray([tz, tx, ty], dtype=np.float32)
            obj_cam = np.empty_like(pose_space)
            obj_cam[:, 0] = -pose_space[:, 1]
            obj_cam[:, 1] = -pose_space[:, 2]
            obj_cam[:, 2] = pose_space[:, 0]
        obj_proj = project_bbox(obj_cam, intr, width, height)
        if obj_proj is None:
            continue
        obj_bbox, obj_inside = obj_proj
        for hand in ["left", "right"]:
            key = f"{hand}_vertices"
            if key not in hands:
                continue
            hand_idx = min(frame_idx, hands[key].shape[0] - 1)
            hand_proj = project_bbox(hands[key][hand_idx], intr, width, height)
            if hand_proj is None:
                continue
            hand_bbox, hand_inside = hand_proj
            gap = bbox_gap(obj_bbox, hand_bbox)
            obj_center = np.asarray([(obj_bbox[0] + obj_bbox[2]) * 0.5, (obj_bbox[1] + obj_bbox[3]) * 0.5])
            hand_center = np.asarray([(hand_bbox[0] + hand_bbox[2]) * 0.5, (hand_bbox[1] + hand_bbox[3]) * 0.5])
            center_dist = float(np.linalg.norm(obj_center - hand_center))
            scores[hand].append(gap + 0.25 * center_dist + 500.0 * max(0.0, 0.65 - hand_inside) + 200.0 * max(0.0, 0.5 - obj_inside))

    means = {hand: (float(np.mean(vals)) if vals else float("inf")) for hand, vals in scores.items()}
    chosen = "left" if means["left"] <= means["right"] else "right"
    configured = anchor_hand_from_config(clip_dir)
    print(
        "[official-overlay] hand auto-select "
        f"object={object_id} configured={configured} chosen={chosen} "
        f"score_left={means['left']:.2f} score_right={means['right']:.2f}"
    )
    if not np.isfinite(means[chosen]):
        return configured
    return chosen


def official_hands_arg_for_object(hand_side: str, clip_dir: Path, object_id: str, mesh: Path, layout: Path) -> str:
    if hand_side in {"left", "right"}:
        return hand_side
    if hand_side in {"all", "bimanual"}:
        return "both"
    return choose_overlay_hand(clip_dir, object_id, mesh, layout)


def find_object_mesh(clip_dir: Path, object_id: str) -> Path | None:
    candidates = sorted((clip_dir / "video_segmentation" / "masks").glob(f"frame_*_masks/{object_id}/{object_id}.obj"))
    if candidates:
        return candidates[0]
    candidates = sorted((clip_dir / "video_segmentation" / "masks").glob(f"frame_*_masks/{object_id}.obj"))
    return candidates[0] if candidates else None


def metadata_from_clip(clip_dir: Path) -> dict:
    path = clip_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] could not read metadata from {path}: {exc}")
        return {}


def hawor_visualization_from_clip(clip_dir: Path) -> Path | None:
    metadata = metadata_from_clip(clip_dir)
    hands_npz = metadata.get("hands_npz")
    if hands_npz:
        candidate = Path(str(hands_npz)).parent / "visualization" / "hands_combined.mp4"
        if candidate.exists():
            return candidate
    return None


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return count


def read_video_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    return frame if ok else None


def object_mask_is_hand_like(clip_dir: Path) -> bool:
    metadata = metadata_from_clip(clip_dir)
    object_id = metadata.get("object_id")
    if object_id in {"blue_box", "detergent_bottle"}:
        print(
            "[official-overlay] object reconstruction is known to select hands for "
            f"object={object_id}. Skipping object mesh layer so the reconstruction "
            "overlay keeps the official hand projection clean."
        )
        return True
    config_path = clip_dir / "config.json"
    ref_frame = None
    if config_path.exists():
        try:
            ref_frame = int(json.loads(config_path.read_text(encoding="utf-8")).get("frame_number"))
        except Exception:
            ref_frame = None
    if object_id is None or ref_frame is None:
        return False
    mask_dir = clip_dir / "video_segmentation" / "masks" / f"frame_{ref_frame:06d}_masks"
    obj = cv2.imread(str(mask_dir / f"{object_id}.png"), cv2.IMREAD_GRAYSCALE)
    if obj is None:
        return False
    obj_mask = obj > 127
    if int(obj_mask.sum()) < 100:
        return False
    hand_union = np.zeros_like(obj_mask, dtype=bool)
    for hand in ["left_hand_0.png", "right_hand_0.png"]:
        hand_img = cv2.imread(str(mask_dir / hand), cv2.IMREAD_GRAYSCALE)
        if hand_img is None:
            continue
        if hand_img.shape != obj.shape:
            hand_img = cv2.resize(hand_img, (obj.shape[1], obj.shape[0]), interpolation=cv2.INTER_NEAREST)
        hand_union |= hand_img > 127
    overlap = float((obj_mask & hand_union).sum()) / float(obj_mask.sum())
    if overlap > 0.35:
        print(
            "[official-overlay] object mask looks hand-like; "
            f"object={object_id} ref_frame={ref_frame} hand_overlap={overlap:.3f}. "
            "Skipping object mesh layer to avoid rendering a hand as the object."
        )
        return True
    return False


def compose_hawor_hand_object_overlay(
    clip_dir: Path,
    object_overlay: Path,
    output: Path,
    fps: float,
) -> bool:
    hawor_video = hawor_visualization_from_clip(clip_dir)
    metadata = metadata_from_clip(clip_dir)
    start_frame = metadata.get("start_frame")
    raw_video = clip_dir / "raw.mp4"
    if hawor_video is None or start_frame is None or not object_overlay.exists() or not raw_video.exists():
        return False

    object_cap = cv2.VideoCapture(str(object_overlay))
    raw_cap = cv2.VideoCapture(str(raw_video))
    hawor_cap = cv2.VideoCapture(str(hawor_video))
    try:
        num_frames = int(object_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(object_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(object_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if num_frames <= 0 or width <= 0 or height <= 0:
            return False
        skip_object_layer = object_mask_is_hand_like(clip_dir)
        if output.exists() or output.is_symlink():
            output.unlink()
        writer = ffmpeg_writer(output, width, height, fps)
        assert writer.stdin is not None
        start_frame_f = float(start_frame)
        for frame_idx in range(num_frames):
            ok_obj, obj_bgr = object_cap.read()
            ok_raw, raw_bgr = raw_cap.read()
            if not ok_obj or not ok_raw:
                break
            hawor_idx = int(round((start_frame_f + frame_idx) * 0.5))
            hawor_bgr = read_video_frame_at(hawor_cap, hawor_idx)
            if hawor_bgr is None:
                break
            hh, hw = hawor_bgr.shape[:2]
            hand_bgr = hawor_bgr[hh // 2 :, :]
            hand_bgr = cv2.resize(hand_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
            if raw_bgr.shape[:2] != (height, width):
                raw_bgr = cv2.resize(raw_bgr, (width, height), interpolation=cv2.INTER_LINEAR)
            result = hand_bgr.copy()
            if not skip_object_layer:
                diff = cv2.absdiff(obj_bgr, raw_bgr)
                mask = diff.max(axis=2) > 18
                mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
                result[mask] = obj_bgr[mask]
            writer.stdin.write(np.ascontiguousarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).tobytes())
    finally:
        object_cap.release()
        raw_cap.release()
        hawor_cap.release()
        if "writer" in locals():
            writer.stdin.close()
            rc = writer.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {rc}")
    print(f"[official-overlay] composed HaWoR hand mesh with Do-as-I-Do object mesh: {output}")
    return output.exists()


def render_official_mesh_overlay(
    clip_dir: Path,
    object_id: str,
    output: Path,
    hand_side: str,
    python_bin: str,
) -> None:
    script = REPO_ROOT / "third_party" / "do-as-i-do" / "reconstruction" / "scripts" / "run_project_mesh_combined.py"
    mesh = find_object_mesh(clip_dir, object_id)
    layout = clip_dir / "obj_tracking_out" / object_id / "combined_visualization" / "layout_camera_frame_optimized.json"
    hand_meshes = clip_dir / "raw" / "all_hand_meshes.npz"
    frame_dir = clip_dir / "all_frames"
    missing = [str(path) for path in [script, mesh, layout, hand_meshes, frame_dir] if path is None or not Path(path).exists()]
    if missing:
        raise FileNotFoundError("missing official Do-as-I-Do projection inputs: " + ", ".join(missing))
    projection_dir = output.parent / "official_projected_mesh_overlay"
    object_projection_dir = output.parent / "official_projected_object_overlay"
    if projection_dir.exists():
        shutil.rmtree(projection_dir)
    if object_projection_dir.exists():
        shutil.rmtree(object_projection_dir)
    object_cmd = [
        python_bin,
        str(script),
        "--video",
        str(frame_dir),
        "--mesh",
        str(mesh),
        "--json",
        str(layout),
        "--output-base",
        str(object_projection_dir),
    ]
    try:
        layout_data = json.loads(layout.read_text(encoding="utf-8"))
        if "mesh_scale" in (layout_data.get("translation_scale_optimization") or {}):
            object_cmd.append("--use-optimized-mesh-scale")
    except Exception:
        pass
    subprocess.run(object_cmd, check=True)
    object_video = object_projection_dir / "video.mp4"
    if compose_hawor_hand_object_overlay(clip_dir, object_video, output, fps=30.0):
        return

    cmd = [
        python_bin,
        str(script),
        "--video",
        str(frame_dir),
        "--mesh",
        str(mesh),
        "--json",
        str(layout),
        "--output-base",
        str(projection_dir),
        "--hand-meshes",
        str(hand_meshes),
        "--hands",
        official_hands_arg_for_object(hand_side, clip_dir, object_id, mesh, layout),
        "--layer-order",
        "hand_front",
    ]
    hand_focal = hand_focal_from_clip(clip_dir)
    if hand_focal is not None:
        print(f"[official-overlay] using hand focal from AoE hands.npz: {hand_focal:.3f}")
        cmd += ["--hand-focal", f"{hand_focal:.8f}"]
    try:
        layout_data = json.loads(layout.read_text(encoding="utf-8"))
        if "mesh_scale" in (layout_data.get("translation_scale_optimization") or {}):
            cmd.append("--use-optimized-mesh-scale")
    except Exception:
        pass
    subprocess.run(cmd, check=True)
    official_video = projection_dir / "video.mp4"
    if not official_video.exists():
        raise FileNotFoundError(f"official projection did not create {official_video}")
    link_or_copy(official_video, output, copy=False)


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def ffmpeg_writer(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_depth(pointmaps: list[Path], output: Path, fps: float) -> None:
    if not pointmaps:
        raise FileNotFoundError("no *_pointmap.npy files found")
    sample = np.load(pointmaps[0])
    if sample.ndim != 3 or sample.shape[-1] < 3:
        raise ValueError(f"expected HxWx3 pointmap, got {sample.shape}")
    height, width = sample.shape[:2]
    proc = ffmpeg_writer(output, width, height, fps)
    assert proc.stdin is not None
    try:
        for path in pointmaps:
            arr = np.load(path)
            z = arr[..., 2].astype(np.float32)
            finite = np.isfinite(z)
            if finite.any():
                lo, hi = np.nanpercentile(z[finite], [2.0, 98.0])
                if hi <= lo:
                    hi = lo + 1e-6
                img = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
            else:
                img = np.zeros_like(z, dtype=np.float32)
            gray = (img * 255.0).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=-1)
            proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize Do-as-I-Do overlay/depth videos under experiments/.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--clip-dir", required=True, type=Path)
    parser.add_argument("--task", default="foundation_jar_bimanual_leftscale")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--hand-side", choices=["all", "auto", "anchor", "left", "right", "bimanual"], default="all")
    parser.add_argument("--projection-renderer", choices=["official", "legacy"], default="official")
    parser.add_argument(
        "--official-projection-python",
        default=None,
        help="Python with PyTorch3D for the official Do-as-I-Do projection script. Defaults to DAI_PROJECTION_PYTHON or a known local env.",
    )
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    clip_dir = args.clip_dir.expanduser().resolve()
    if not clip_dir.is_dir():
        raise FileNotFoundError(f"clip dir does not exist: {clip_dir}")

    object_id = object_id_from_task(args.task)
    exp = REPO_ROOT / "experiments" / args.run_name
    out_dir = exp / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep the historical debug_tapir_overlay.mp4 output path, but feed it the
    # best reconstruction/projection visualization available instead of
    # prioritizing TAPIR-only videos.
    overlay_candidates = [
        clip_dir / "obj_tracking_out" / object_id / "combined_visualization" / "projected_hand_object_optimized" / "video.mp4",
        clip_dir / "obj_tracking_out" / object_id / "combined_visualization" / "projected" / "video.mp4",
        clip_dir / f"output_tapir_{object_id}_overlay.mp4",
        clip_dir / f"output_tapir_{object_id}.mp4",
        clip_dir / "video_segmentation" / f"tracked_{object_id}_bootstrap.mp4",
        clip_dir / "raw.mp4",
    ]
    overlay = next((p for p in overlay_candidates if p.exists()), None)
    if overlay is not None:
        link_or_copy(overlay, out_dir / "debug_tapir_overlay.mp4", args.copy)

    mask_candidates = sorted((clip_dir / "video_segmentation").glob("tracked_*.mp4"))
    if mask_candidates:
        link_or_copy(mask_candidates[0], out_dir / "mask_overlay.mp4", args.copy)

    frame_dir = clip_dir / "all_frames"
    pointmaps = sorted(frame_dir.glob("*_pointmap.npy"))
    if args.max_frames > 0:
        pointmaps = pointmaps[: args.max_frames]
    if pointmaps:
        render_depth(pointmaps, out_dir / "depth.mp4", args.fps)

    indexed_raw_dir = out_dir / "raw_dir"
    raw_dir = indexed_raw_dir if indexed_raw_dir.exists() else clip_dir
    if raw_dir.exists():
        mesh_overlay_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "review" / "render_do_as_i_do_mesh_overlay.py"),
            "--clip-dir",
            str(clip_dir),
            "--raw-dir",
            str(raw_dir),
            "--task",
            args.task,
            "--output",
            str(out_dir / "mesh_overlay.mp4"),
            "--fps",
            str(args.fps),
            "--hand-intrinsics-source",
            "frame",
            "--max-object-faces",
            "1500",
        ]
        if args.max_frames > 0:
            mesh_overlay_cmd += ["--max-frames", str(args.max_frames)]
        if args.projection_renderer == "official":
            projection_python = args.official_projection_python or os.environ.get("DAI_PROJECTION_PYTHON") or os.environ.get("SAM3D_PYTHON") or sys.executable
            try:
                render_official_mesh_overlay(
                    clip_dir,
                    object_id,
                    out_dir / "mesh_overlay.mp4",
                    args.hand_side,
                    projection_python,
                )
            except subprocess.CalledProcessError as exc:
                print(
                    "[official-overlay] renderer rejected the reconstruction asset; "
                    f"using the read-only point-cloud review renderer instead (rc={exc.returncode})",
                    file=sys.stderr,
                )
                subprocess.run(mesh_overlay_cmd, check=True)
        else:
            subprocess.run(mesh_overlay_cmd, check=True)
        link_or_copy(out_dir / "mesh_overlay.mp4", out_dir / "overlay.mp4", args.copy)

        if args.projection_renderer == "legacy":
            mesh_pure_cmd = list(mesh_overlay_cmd)
            output_idx = mesh_pure_cmd.index("--output") + 1
            mesh_pure_cmd[output_idx] = str(out_dir / "mesh_pure_camera.mp4")
            mesh_pure_cmd += ["--background", "black"]
            subprocess.run(mesh_pure_cmd, check=True)

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
