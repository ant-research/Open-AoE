#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from align_dai_native_contact import (
    align_layout,
    assess_dai_native_retarget_input,
    assess_object_pose_continuity,
    layout_entries,
    quat_wxyz_to_matrix,
)
from aoe_retarget_lab.hoi_geometry import summarize_adapter_rigid_invariance
from select_do_as_i_do_hand_type import score_hand_sides


TIP_VERTICES = [745, 320, 443, 554, 671]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def safe_adapter_output_dir(source_dir: Path, requested: Path) -> Path:
    """Return a lexical output path that cannot alias or contain the source."""

    source_dir = source_dir.expanduser().resolve()
    output_dir = Path(os.path.abspath(str(requested.expanduser())))
    if output_dir.is_symlink():
        raise ValueError("adapter output-dir must not be a symlink")
    resolved_output = output_dir.resolve(strict=False)
    if (
        source_dir == resolved_output
        or _is_relative_to(resolved_output, source_dir)
        or _is_relative_to(source_dir, resolved_output)
    ):
        raise ValueError(
            "adapter output-dir must be disjoint from source-dir "
            "(neither may contain the other)"
        )
    return output_dir


def _file_binding(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    binding: dict[str, object] = {"path": str(resolved), "sha256": None}
    if not resolved.is_file():
        binding["error"] = "missing"
        return binding
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        binding["error"] = f"unreadable:{type(exc).__name__}"
        return binding
    binding["sha256"] = digest.hexdigest()
    return binding


def _exact_frame_grid(frames_dir: Path) -> list[Path]:
    frames = sorted(
        path
        for path in frames_dir.glob("*.png")
        if path.stem.isdigit() and len(path.stem) == 6
    )
    expected = [f"{index:06d}.png" for index in range(len(frames))]
    if not frames or [path.name for path in frames] != expected:
        raise ValueError(
            "raw video provenance requires a contiguous zero-based %06d.png frame grid"
        )
    return frames


def _aggregate_frame_sha256(frames: list[Path]) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.name.encode("utf-8"))
        digest.update(b"\0")
        frame_digest = hashlib.sha256()
        with frame.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                frame_digest.update(chunk)
        digest.update(frame_digest.digest())
    return digest.hexdigest()


def build_raw_video_provenance(raw_dir: Path) -> dict[str, object]:
    """Bind native raw.mp4 and validate optional rematerialization evidence."""

    raw_dir = raw_dir.expanduser().resolve()
    raw_video = raw_dir / "raw.mp4"
    raw_binding = _file_binding(raw_video)
    if not raw_binding.get("sha256"):
        raise ValueError(f"raw video binding is invalid: {raw_binding}")

    source_input_path = raw_dir / "fresh_input_manifest.json"
    source_input_binding = (
        _file_binding(source_input_path) if source_input_path.is_file() else None
    )
    materialization_path = raw_dir / "raw_video_materialization.json"
    if not materialization_path.is_file():
        return {
            "schema_version": 1,
            "status": "validated",
            "kind": "reconstruction_raw_video",
            "raw_video": raw_binding,
            "materialization_manifest": None,
            "source_input_manifest": source_input_binding,
            "frame_count": None,
            "fps": None,
        }

    materialization_binding = _file_binding(materialization_path)
    if not materialization_binding.get("sha256"):
        raise ValueError(
            f"raw video materialization manifest binding is invalid: {materialization_binding}"
        )
    materialization = load_json(materialization_path)
    output = materialization.get("output") or {}
    all_frames = materialization.get("all_frames") or {}
    errors: list[str] = []
    if materialization.get("schema_version") != 1:
        errors.append("raw_video_materialization_schema_invalid")
    if materialization.get("status") != "ok":
        errors.append("raw_video_materialization_status_invalid")
    if materialization.get("kind") != "materialized_exact_all_frames":
        errors.append("raw_video_materialization_kind_invalid")

    expected_path = output.get("path")
    if not expected_path:
        errors.append("raw_video_materialization_output_path_missing")
    elif Path(str(expected_path)).expanduser().resolve() != raw_video.resolve():
        errors.append("raw_video_materialization_output_path_mismatch")
    if output.get("sha256") != raw_binding.get("sha256"):
        errors.append("raw_video_materialization_output_sha256_mismatch")

    try:
        frame_count = int(output.get("frame_count"))
    except (TypeError, ValueError):
        frame_count = 0
    try:
        fps = float(output.get("fps"))
    except (TypeError, ValueError):
        fps = 0.0
    if frame_count <= 0:
        errors.append("raw_video_materialization_frame_count_invalid")
    if not np.isfinite(fps) or fps <= 0.0:
        errors.append("raw_video_materialization_fps_invalid")

    try:
        current_frames = _exact_frame_grid(raw_dir / "all_frames")
    except (OSError, ValueError):
        current_frames = []
        errors.append("raw_video_materialization_all_frames_grid_invalid")
    expected_frames_path = all_frames.get("path")
    if not expected_frames_path:
        errors.append("raw_video_materialization_all_frames_path_missing")
    elif (
        Path(str(expected_frames_path)).expanduser().resolve()
        != (raw_dir / "all_frames").resolve()
    ):
        errors.append("raw_video_materialization_all_frames_path_mismatch")
    if current_frames:
        try:
            recorded_frame_count = int(all_frames.get("frame_count"))
        except (TypeError, ValueError):
            recorded_frame_count = 0
        if (
            recorded_frame_count != len(current_frames)
            or frame_count != len(current_frames)
        ):
            errors.append("raw_video_materialization_all_frames_count_mismatch")
        if all_frames.get("aggregate_sha256") != _aggregate_frame_sha256(
            current_frames
        ):
            errors.append("raw_video_materialization_all_frames_sha256_mismatch")

    if errors:
        raise ValueError(";".join(errors))
    return {
        "schema_version": 1,
        "status": "validated",
        "kind": "materialized_exact_all_frames",
        "raw_video": raw_binding,
        "materialization_manifest": materialization_binding,
        "source_input_manifest": source_input_binding,
        "frame_count": frame_count,
        "fps": fps,
    }


def layout_object_pose_sequence(layout: dict) -> np.ndarray:
    """Parse the production camera-frame layout as a strict SE(3) sequence."""

    poses: list[np.ndarray] = []
    for index, entry in enumerate(layout_entries(layout)):
        local = entry.get("local_to_scene")
        if not isinstance(local, dict):
            raise ValueError(f"layout object {index} is missing local_to_scene")
        translation = local.get("translation_camera_frame")
        if translation is None:
            translation = local.get("translation")
        quaternion = local.get("quat_wxyz_camera_frame")
        if quaternion is None:
            quaternion = local.get("quat_wxyz")
        if quaternion is None:
            quaternion = local.get("new_quat")
        try:
            translation_array = np.asarray(translation, dtype=np.float64).reshape(3)
            quaternion_array = np.asarray(quaternion, dtype=np.float64).reshape(4)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"layout object {index} has an invalid pose") from exc
        quaternion_norm = float(np.linalg.norm(quaternion_array))
        if (
            not np.isfinite(translation_array).all()
            or not np.isfinite(quaternion_array).all()
            or quaternion_norm <= 1e-12
        ):
            raise ValueError(f"layout object {index} has a non-finite/degenerate pose")
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = quat_wxyz_to_matrix(quaternion_array / quaternion_norm)
        pose[:3, 3] = translation_array
        poses.append(pose)
    if not poses:
        raise ValueError("layout contains no object poses")
    return np.stack(poses, axis=0)


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")


def remove_existing(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()


def link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    remove_existing(dst)
    dst.symlink_to(src)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    remove_existing(dst)
    shutil.copy2(src, dst)


def stage_native_task_hand_mesh(output_dir: Path, task: str, hand_mesh: Path) -> Path:
    if not task or Path(task).name != task or task in {".", ".."} or "\\" in task:
        raise ValueError(f"unsafe native DAI task name: {task!r}")
    destination = output_dir / task / "all_hand_meshes.npz"
    link_file(hand_mesh.resolve(), destination)
    return destination


def copy_tree(src: Path, dst: Path) -> None:
    remove_existing(dst)
    shutil.copytree(src, dst, symlinks=True)


def infer_object_id(source_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    config = load_json(source_dir / "config.json")
    objects = config.get("object_names") or []
    if not objects:
        raise RuntimeError(f"cannot infer object id from {source_dir / 'config.json'}")
    return str(objects[0])


def choose_hand_npz(source_dir: Path, explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            source_dir / "raw" / "all_hand_meshes_46f.npz",
            source_dir / "raw" / "all_hand_meshes.npz",
            source_dir / "all_hand_meshes.npz",
        ]
    )
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("missing hand mesh npz; checked " + ", ".join(str(p) for p in candidates))


def expand_16_to_21(joints: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints)
    vertices = np.asarray(vertices)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"expected joints as (N,J,3), got {joints.shape}")
    if joints.shape[1] == 21:
        return joints.astype(np.float32, copy=False)
    if joints.shape[1] != 16:
        raise ValueError(f"cannot adapt {joints.shape[1]} joints to OpenPose-21")
    if vertices.shape[0] != joints.shape[0] or vertices.shape[1] <= max(TIP_VERTICES):
        raise ValueError(f"vertices {vertices.shape} incompatible with joints {joints.shape}")

    out = np.empty((joints.shape[0], 21, 3), dtype=np.float32)
    out[:, 0] = joints[:, 0]
    out[:, 1], out[:, 2], out[:, 3], out[:, 4] = (
        joints[:, 13],
        joints[:, 14],
        joints[:, 15],
        vertices[:, TIP_VERTICES[0]],
    )
    out[:, 5], out[:, 6], out[:, 7], out[:, 8] = (
        joints[:, 1],
        joints[:, 2],
        joints[:, 3],
        vertices[:, TIP_VERTICES[1]],
    )
    out[:, 9], out[:, 10], out[:, 11], out[:, 12] = (
        joints[:, 4],
        joints[:, 5],
        joints[:, 6],
        vertices[:, TIP_VERTICES[2]],
    )
    out[:, 13], out[:, 14], out[:, 15], out[:, 16] = (
        joints[:, 10],
        joints[:, 11],
        joints[:, 12],
        vertices[:, TIP_VERTICES[3]],
    )
    out[:, 17], out[:, 18], out[:, 19], out[:, 20] = (
        joints[:, 7],
        joints[:, 8],
        joints[:, 9],
        vertices[:, TIP_VERTICES[4]],
    )
    return out


def write_adapted_hand_npz(src: Path, dst: Path) -> dict:
    data = np.load(src, allow_pickle=True)
    out = {key: data[key] for key in data.files}
    stats: dict[str, object] = {"source": str(src), "sides": {}}
    if "source_frame_ids" in data:
        source_frame_ids = np.asarray(data["source_frame_ids"]).astype(int)
        stats["source_frame_ids"] = {
            "count": int(source_frame_ids.size),
            "first": int(source_frame_ids[0]) if source_frame_ids.size else None,
            "last": int(source_frame_ids[-1]) if source_frame_ids.size else None,
            "sample": source_frame_ids[: min(8, source_frame_ids.size)].tolist(),
        }
    for side in ("left", "right"):
        joints_key = f"{side}_joints"
        vertices_key = f"{side}_vertices"
        if joints_key not in out or vertices_key not in out:
            continue
        before = tuple(np.asarray(out[joints_key]).shape)
        out[joints_key] = expand_16_to_21(np.asarray(out[joints_key]), np.asarray(out[vertices_key]))
        after = tuple(np.asarray(out[joints_key]).shape)
        stats["sides"][side] = {"joints_before": before, "joints_after": after}

    out["mano16_to_openpose21_adapter"] = np.array(
        "MANO-16 to OpenPose-21 using MANO fingertip vertices [745,320,443,554,671]",
        dtype="<U96",
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    stats["output"] = str(dst)
    return stats


def hand_projection_qc(hand_npz: Path, frames_dir: Path, anchor_hand: str) -> dict:
    with np.load(hand_npz, allow_pickle=False) as data:
        vertices_key = f"{anchor_hand}_vertices"
        valid_key = f"{anchor_hand}_valid"
        if vertices_key not in data:
            return {
                "anchor_hand": anchor_hand,
                "available": False,
                "reason": "missing_hand_vertices",
            }
        vertices = np.asarray(data[vertices_key])
        valid_frames = (
            np.asarray(data[valid_key])
            if valid_key in data
            else np.ones(vertices.shape[0])
        )
    if vertices.ndim != 3 or vertices.shape[-1] != 3 or vertices.shape[0] <= 0:
        return {
            "anchor_hand": anchor_hand,
            "available": False,
            "reason": "invalid_hand_vertices_shape",
            "vertices_shape": list(vertices.shape),
        }
    frame_ids = sorted(set([0, vertices.shape[0] // 2, vertices.shape[0] - 1]))
    frames = []
    for frame_id in frame_ids:
        intrinsics_path = frames_dir / f"{frame_id:06d}_intrinsics.npy"
        shape = frame_shape(frames_dir, frame_id)
        if shape is None:
            frames.append(
                {
                    "frame": int(frame_id),
                    "available": False,
                    "reason": "missing_frame_shape",
                }
            )
            continue
        if not intrinsics_path.exists():
            frames.append(
                {
                    "frame": int(frame_id),
                    "available": False,
                    "reason": "missing_intrinsics",
                }
            )
            continue
        if frame_id >= valid_frames.reshape(-1).shape[0]:
            frames.append(
                {
                    "frame": int(frame_id),
                    "available": False,
                    "reason": "missing_hand_validity",
                }
            )
            continue
        validity = float(valid_frames.reshape(-1)[frame_id])
        if not np.isfinite(validity) or validity <= 0:
            frames.append(
                {
                    "frame": int(frame_id),
                    "available": False,
                    "reason": "invalid_hand_frame",
                }
            )
            continue
        h, w = shape
        try:
            intrinsics = np.asarray(np.load(intrinsics_path), dtype=np.float64)
        except (OSError, TypeError, ValueError):
            intrinsics = np.zeros((0, 0), dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            frames.append(
                {
                    "frame": int(frame_id),
                    "available": False,
                    "reason": "invalid_intrinsics",
                }
            )
            continue
        frame_vertices = np.asarray(vertices[frame_id], dtype=np.float64).reshape(-1, 3)
        vertex_count = int(frame_vertices.shape[0])
        finite_xyz = np.isfinite(frame_vertices).all(axis=1)
        positive_depth = finite_xyz & (frame_vertices[:, 2] > 1e-6)
        positive_depth_count = int(positive_depth.sum())
        positive_depth_fraction = (
            float(positive_depth_count / vertex_count) if vertex_count else 0.0
        )
        if positive_depth_count <= 0:
            frames.append(
                {
                    "frame": int(frame_id),
                    "available": False,
                    "reason": "no_positive_depth_vertices",
                    "vertex_count": vertex_count,
                    "finite_vertex_count": int(finite_xyz.sum()),
                    "positive_depth_vertex_count": positive_depth_count,
                    "positive_depth_fraction": positive_depth_fraction,
                }
            )
            continue
        uv, projected = project_vertices(frame_vertices, intrinsics)
        projected = projected & np.isfinite(uv).all(axis=1)
        if not projected.any():
            frames.append(
                {
                    "frame": int(frame_id),
                    "available": False,
                    "reason": "no_finite_projected_vertices",
                    "vertex_count": vertex_count,
                    "finite_vertex_count": int(finite_xyz.sum()),
                    "positive_depth_vertex_count": positive_depth_count,
                    "positive_depth_fraction": positive_depth_fraction,
                }
            )
            continue
        uv_valid = uv[projected]
        inside = (
            (uv_valid[:, 0] >= 0)
            & (uv_valid[:, 0] < w)
            & (uv_valid[:, 1] >= 0)
            & (uv_valid[:, 1] < h)
        )
        inside_count = int(inside.sum())
        frames.append(
            {
                "frame": int(frame_id),
                "available": True,
                "bbox_xyxy": [
                    float(uv_valid[:, 0].min()),
                    float(uv_valid[:, 1].min()),
                    float(uv_valid[:, 0].max()),
                    float(uv_valid[:, 1].max()),
                ],
                # Use every mesh vertex as the denominator.  Measuring only the
                # positive-depth subset lets one surviving vertex hide an
                # otherwise behind-camera hand.
                "inside_fraction": (
                    float(inside_count / vertex_count) if vertex_count else 0.0
                ),
                "inside_fraction_of_positive_depth": float(inside.mean()),
                "vertex_count": vertex_count,
                "finite_vertex_count": int(finite_xyz.sum()),
                "positive_depth_vertex_count": positive_depth_count,
                "positive_depth_fraction": positive_depth_fraction,
                "image_wh": [int(w), int(h)],
            }
        )
    available_frame_count = sum(bool(frame.get("available")) for frame in frames)
    expected_frame_count = len(frame_ids)
    return {
        "anchor_hand": anchor_hand,
        "available": True,
        "sampling_policy": "start_middle_end",
        "expected_frame_ids": [int(frame_id) for frame_id in frame_ids],
        "expected_frame_count": int(expected_frame_count),
        "available_frame_count": int(available_frame_count),
        "complete_sample_coverage": bool(
            expected_frame_count == 3 and available_frame_count == expected_frame_count
        ),
        "frames": frames,
    }


def frame_shape(frames_dir: Path, frame_id: int) -> tuple[int, int] | None:
    pointmap = frames_dir / f"{frame_id:06d}_pointmap.npy"
    if pointmap.exists():
        arr = np.load(pointmap, mmap_mode="r")
        return int(arr.shape[0]), int(arr.shape[1])
    rgb = cv2.imread(str(frames_dir / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
    if rgb is not None:
        h, w = rgb.shape[:2]
        return h, w
    return None


def project_vertices(vertices: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = vertices[:, 2]
    valid = z > 1e-6
    uv = np.zeros((vertices.shape[0], 2), dtype=np.float32)
    uv[:, 0] = intrinsics[0, 0] * vertices[:, 0] / np.maximum(z, 1e-6) + intrinsics[0, 2]
    uv[:, 1] = intrinsics[1, 1] * vertices[:, 1] / np.maximum(z, 1e-6) + intrinsics[1, 2]
    return uv, valid


def generate_hand_masks(hand_npz: Path, frames_dir: Path, masks_dir: Path) -> dict:
    data = np.load(hand_npz, allow_pickle=False)
    stats: dict[str, object] = {"output_dir": str(masks_dir), "sides": {}}
    for side in ("left", "right"):
        vertices_key = f"{side}_vertices"
        faces_key = f"{side}_faces"
        valid_key = f"{side}_valid"
        if vertices_key not in data or faces_key not in data:
            continue
        vertices = np.asarray(data[vertices_key])
        faces = np.asarray(data[faces_key], dtype=np.int32)
        valid_frames = np.asarray(data[valid_key]) if valid_key in data else np.ones(vertices.shape[0])
        written = 0
        for frame_id in range(vertices.shape[0]):
            if float(valid_frames[frame_id]) <= 0:
                continue
            intrinsics_path = frames_dir / f"{frame_id:06d}_intrinsics.npy"
            if not intrinsics_path.exists():
                continue
            shape = frame_shape(frames_dir, frame_id)
            if shape is None:
                continue
            h, w = shape
            intrinsics = np.load(intrinsics_path)
            uv, valid = project_vertices(vertices[frame_id], intrinsics)
            mask = np.zeros((h, w), dtype=np.uint8)
            for tri in faces:
                if np.any(tri < 0) or np.any(tri >= len(vertices[frame_id])) or not bool(valid[tri].all()):
                    continue
                pts = np.rint(uv[tri]).astype(np.int32)
                if pts[:, 0].max() < 0 or pts[:, 0].min() >= w or pts[:, 1].max() < 0 or pts[:, 1].min() >= h:
                    continue
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                cv2.fillConvexPoly(mask, pts, 255)
            mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            out_dir = masks_dir / f"frame_{frame_id:06d}_masks"
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{side}_hand_0.png"), mask)
            written += 1
        stats["sides"][side] = {"frames_written": written, "frames_total": int(vertices.shape[0])}
    return stats


def link_tracking_assets(source_dir: Path, output_dir: Path, object_id: str, optimize_scale: bool) -> None:
    src_obj = source_dir / "obj_tracking_out" / object_id
    dst_obj = output_dir / "obj_tracking_out" / object_id
    dst_layout = dst_obj / "combined_visualization"
    dst_layout.mkdir(parents=True, exist_ok=True)
    require(src_obj / "combined_visualization" / "layout_camera_frame.json", "Do-as-I-Do layout_camera_frame.json")

    for path in sorted(src_obj.glob("frame_*_samples.pt")) + sorted(src_obj.glob("guided_poses.pt")):
        link_file(path.resolve(), dst_obj / path.name)

    for path in sorted((src_obj / "combined_visualization").iterdir()):
        if optimize_scale and path.name == "layout_camera_frame_optimized.json":
            continue
        link_file(path.resolve(), dst_layout / path.name)

    optimized = dst_layout / "layout_camera_frame_optimized.json"
    if not optimize_scale:
        # The production identity adapter must never inherit a stale optimized
        # layout (including legacy object-only contact translations).
        remove_existing(optimized)
        copy_file(src_obj / "combined_visualization" / "layout_camera_frame.json", optimized)


def choose_ref_frame(config: dict, explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    value = config.get("frame_number", 0)
    return int(value)


def choose_object_mesh(prepared_dir: Path, object_id: str, ref_frame: int) -> Path:
    preferred = prepared_dir / "video_segmentation" / "masks" / f"frame_{ref_frame:06d}_masks" / object_id / f"{object_id}.obj"
    if preferred.exists():
        return preferred
    candidates = sorted((prepared_dir / "video_segmentation" / "masks").glob(f"frame_*_masks/{object_id}/{object_id}.obj"))
    if not candidates:
        raise FileNotFoundError(f"missing object mesh under {prepared_dir / 'video_segmentation/masks'}")
    return candidates[0]


def run_scale_optimization(
    python_bin: str,
    prepared_dir: Path,
    object_id: str,
    anchor_hand: str,
    ref_frame: int,
    viz_dir: Path | None,
) -> dict:
    script = REPO_ROOT / "third_party" / "do-as-i-do" / "reconstruction" / "scripts" / "optimize_translation_scale.py"
    require(script, "Do-as-I-Do optimize_translation_scale.py")
    layout_dir = prepared_dir / "obj_tracking_out" / object_id / "combined_visualization"
    cmd = [
        python_bin,
        str(script),
        "--layout-json",
        str(layout_dir / "layout_camera_frame.json"),
        "--mesh",
        str(choose_object_mesh(prepared_dir, object_id, ref_frame)),
        "--pointmap-dir",
        str(prepared_dir / "all_frames"),
        "--mask-dir",
        str(prepared_dir / "video_segmentation" / "masks"),
        "--mask-name",
        object_id,
        "--hand-meshes",
        str(prepared_dir / "raw" / "all_hand_meshes.npz"),
        "--anchor-hand",
        anchor_hand,
        "--ref-frame",
        str(ref_frame),
        "--output",
        str(layout_dir / "layout_camera_frame_optimized.json"),
        "--frames-dir",
        str(prepared_dir / "all_frames"),
    ]
    if viz_dir is not None:
        cmd += ["--viz-dir", str(viz_dir)]
    optimized = layout_dir / "layout_camera_frame_optimized.json"
    clean_layout = layout_dir / "layout_camera_frame.json"

    def restore_clean_production_layout() -> None:
        remove_existing(optimized)
        shutil.copy2(clean_layout, optimized)

    def write_fallback(reason: str, returncode: int | None) -> dict:
        restore_clean_production_layout()
        print(
            f"[warn] optimize_translation_scale.py did not apply ({reason}); "
            f"using unoptimized camera-frame layout at {optimized}",
            file=sys.stderr,
        )
        return {
            "requested": True,
            "status": "fallback_unoptimized",
            "succeeded": False,
            "candidate_generated": False,
            "applied": False,
            "method": "none",
            "fallback": True,
            "reason": reason,
            "returncode": returncode,
            "output": str(optimized),
            "mesh_scale": 1.0,
            "candidate_output": None,
            "candidate_mesh_scale": None,
        }

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        return write_fallback("optimize_translation_scale_failed", exc.returncode)

    if not optimized.is_file():
        return write_fallback("optimizer_output_missing", 0)
    try:
        optimized_layout = load_json(optimized)
        metadata = optimized_layout.get("translation_scale_optimization")
        if not isinstance(metadata, dict):
            raise ValueError("missing translation_scale_optimization metadata")
        method = str(metadata.get("method") or "").strip()
        mesh_scale = float(metadata.get("mesh_scale"))
        if (
            not method
            or method == "none"
            or metadata.get("status") == "fallback_unoptimized"
            or not np.isfinite(mesh_scale)
            or mesh_scale <= 0.0
        ):
            raise ValueError("invalid optimizer success metadata")
    except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return write_fallback(
            f"optimizer_output_invalid:{type(exc).__name__}",
            0,
        )
    candidate_output = prepared_dir / "adapter_qc" / "scale_optimization_candidate_layout.json"
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(optimized, candidate_output)
    # Object-only translation/scale optimization is diagnostic.  The exact
    # source layout is always restored before production retargeting consumes
    # ``layout_camera_frame_optimized.json``.
    restore_clean_production_layout()
    return {
        "requested": True,
        "status": "diagnostic_candidate_generated",
        "succeeded": True,
        "candidate_generated": True,
        "applied": False,
        "method": method,
        "fallback": False,
        "reason": None,
        "returncode": 0,
        "output": str(optimized),
        "mesh_scale": 1.0,
        "candidate_output": str(candidate_output),
        "candidate_mesh_scale": mesh_scale,
    }


def scale_refinement_provenance(
    *,
    requested: bool,
    optimization: dict,
    contact_diagnostic: dict,
) -> dict:
    """Record scale refinement from the optimizer outcome, not CLI intent."""

    applied = optimization.get("applied")
    fallback = optimization.get("fallback")
    if not isinstance(applied, bool) or not isinstance(fallback, bool):
        raise ValueError("scale optimization outcome must contain boolean applied/fallback")
    if applied and fallback:
        raise ValueError("scale optimization cannot be both applied and fallback")
    if applied and not requested:
        raise ValueError("unrequested scale optimization cannot be recorded as applied")
    if applied:
        raise ValueError("object-only scale optimization cannot modify production HOI")
    method = str(optimization.get("method") or "none")
    return {
        "applied": False,
        "method": "none",
        "object_scale_optimization_requested": bool(requested),
        "object_scale_optimization_applied": False,
        "fallback": fallback,
        "scale_optimization": optimization,
        "diagnostic_candidate_method": (
            method if optimization.get("candidate_generated") is True else "none"
        ),
        "hand_contact_translation_applied": False,
        "diagnostic_only": True,
        "contact_diagnostic": contact_diagnostic,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare an experiment-local Do-as-I-Do raw_dir from a reconstruction scene."
    )
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--object-id")
    parser.add_argument("--task", default="")
    parser.add_argument("--hand-npz", type=Path)
    parser.add_argument("--fallback-gravity-json", type=Path)
    parser.add_argument("--anchor-hand", default="")
    parser.add_argument(
        "--hand-source",
        choices=["aoe", "estimated", "unknown"],
        default="unknown",
        help="Exact matrix route hand source recorded in adapter provenance.",
    )
    parser.add_argument("--ref-frame", type=int)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--scale-viz-dir", type=Path)
    parser.add_argument("--no-optimize-scale", action="store_true")
    parser.add_argument(
        "--hoi-contact-alignment",
        choices=["none", "diagnostic", "auto", "force"],
        default="none",
        help="Measure contact offset without modifying the production object layout; auto/force are diagnostic aliases.",
    )
    parser.add_argument("--hoi-align-min-bad-mean-distance", type=float, default=0.10)
    parser.add_argument("--hoi-align-target-surface-distance", type=float, default=0.025)
    parser.add_argument("--hoi-align-max-offset", type=float, default=0.10)
    parser.add_argument("--min-visual-overlap-frames", type=int, default=3)
    parser.add_argument("--max-object-translation-step-m", type=float, default=2.0)
    parser.add_argument("--p95-object-translation-step-m", type=float, default=0.20)
    parser.add_argument("--max-object-rotation-step-deg", type=float, default=150.0)
    parser.add_argument("--p95-object-rotation-step-deg", type=float, default=60.0)
    parser.add_argument("--max-object-lost-frame-fraction", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    output_dir = safe_adapter_output_dir(source_dir, args.output_dir)
    require(source_dir / "config.json", "Do-as-I-Do config.json")
    require(source_dir / "raw.mp4", "Do-as-I-Do raw.mp4")
    require(source_dir / "all_frames", "Do-as-I-Do all_frames")
    require(source_dir / "video_segmentation", "Do-as-I-Do video_segmentation")

    if output_dir.exists() or output_dir.is_symlink():
        if not args.force:
            raise FileExistsError(f"output dir already exists: {output_dir}; pass --force to replace it")
        remove_existing(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(source_dir / "config.json")
    object_id = infer_object_id(source_dir, args.object_id)
    source_layout_path = (
        source_dir
        / "obj_tracking_out"
        / object_id
        / "combined_visualization"
        / "layout_camera_frame.json"
    )
    source_object_poses = layout_object_pose_sequence(load_json(source_layout_path))
    hand_npz = choose_hand_npz(source_dir, args.hand_npz)
    anchor_hand = args.anchor_hand or str(config.get("anchor_hand") or "left")
    ref_frame = choose_ref_frame(config, args.ref_frame)
    optimize_scale = not args.no_optimize_scale

    link_file((source_dir / "all_frames").resolve(), output_dir / "all_frames")
    link_file((source_dir / "raw.mp4").resolve(), output_dir / "raw.mp4")
    copy_tree(source_dir / "video_segmentation", output_dir / "video_segmentation")
    copy_file(source_dir / "config.json", output_dir / "config.json")
    for provenance_name in (
        "fresh_input_manifest.json",
        "raw_video_materialization.json",
    ):
        provenance_source = source_dir / provenance_name
        if provenance_source.is_file():
            copy_file(provenance_source, output_dir / provenance_name)
    raw_video_provenance = build_raw_video_provenance(output_dir)

    gravity_src = source_dir / "gravity.json"
    if gravity_src.exists():
        copy_file(gravity_src, output_dir / "gravity.json")
    elif args.fallback_gravity_json is not None and args.fallback_gravity_json.exists():
        copy_file(args.fallback_gravity_json.resolve(), output_dir / "gravity.json")
    else:
        raise FileNotFoundError("missing gravity.json; pass --fallback-gravity-json")

    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    for path in sorted((source_dir / "raw").glob("*")):
        if path.name == "all_hand_meshes.npz":
            continue
        link_file(path.resolve(), output_dir / "raw" / path.name)

    adapted_hand_path = output_dir / "raw" / "all_hand_meshes.npz"
    hand_stats = write_adapted_hand_npz(hand_npz, adapted_hand_path)
    native_task = args.task or f"{object_id}_{anchor_hand}"
    stage_native_task_hand_mesh(output_dir, native_task, adapted_hand_path)
    before_hand_path = output_dir / "adapter_qc" / "before_hand_geometry.npz"
    before_hand_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(adapted_hand_path, before_hand_path)
    projection_qc = hand_projection_qc(output_dir / "raw" / "all_hand_meshes.npz", output_dir / "all_frames", anchor_hand)
    mask_stats = generate_hand_masks(
        output_dir / "raw" / "all_hand_meshes.npz",
        output_dir / "all_frames",
        output_dir / "video_segmentation" / "masks",
    )
    link_tracking_assets(source_dir, output_dir, object_id, optimize_scale)

    scale_optimization = {
        "requested": False,
        "status": "not_requested",
        "succeeded": False,
        "candidate_generated": False,
        "applied": False,
        "method": "none",
        "fallback": False,
        "reason": None,
        "returncode": None,
        "output": None,
        "mesh_scale": 1.0,
        "candidate_output": None,
        "candidate_mesh_scale": None,
    }
    if optimize_scale:
        scale_optimization = run_scale_optimization(
            args.python_bin,
            output_dir,
            object_id,
            anchor_hand,
            ref_frame,
            args.scale_viz_dir,
        )

    layout_path = output_dir / "obj_tracking_out" / object_id / "combined_visualization" / "layout_camera_frame_optimized.json"
    layout = load_json(layout_path)
    visual_interaction = score_hand_sides(
        output_dir,
        args.task or f"{object_id}_{anchor_hand}",
        32,
    )
    overlap_frames = float(
        (visual_interaction.get("mask_scores") or {}).get(anchor_hand, {}).get("overlap_frames", 0.0) or 0.0
    )
    mesh_scale = float((layout.get("translation_scale_optimization") or {}).get("mesh_scale", 1.0) or 1.0)
    aligned_layout, contact_alignment = align_layout(
        layout,
        output_dir / "raw" / "all_hand_meshes.npz",
        choose_object_mesh(output_dir, object_id, ref_frame),
        anchor_hand,
        overlap_frames,
        args.hoi_contact_alignment,
        args.hoi_align_min_bad_mean_distance,
        args.hoi_align_target_surface_distance,
        args.hoi_align_max_offset,
    )
    contact_alignment = {
        **contact_alignment,
        "applied": False,
        "diagnostic_only": True,
        "layout_modified": False,
        "policy": "preserve_source_hoi_relative_transform",
    }
    aligned_layout.pop("hoi_contact_alignment", None)
    after_object_poses = layout_object_pose_sequence(aligned_layout)
    adapter_rigid_invariance = summarize_adapter_rigid_invariance(
        before_hand_path,
        adapted_hand_path,
        before_object_poses=source_object_poses,
        after_object_poses=after_object_poses,
        canonical_transform=None,
    )
    adapter_rigid_invariance.update(
        {
            "measurement_scope": "source_clean_layout_to_production_layout",
            "comparison_semantics": "true_pre_adapter_vs_post_adapter_hand_and_object",
            "hand_coordinate_frame": "explicit_adapter_boundary_frames",
            "before_object_coordinate_frame": "do_as_i_do_source_camera",
            "after_object_coordinate_frame": "do_as_i_do_source_camera",
            "before_layout": str(source_layout_path),
            "after_layout": str(layout_path),
            "object_pose_frames_before": int(source_object_poses.shape[0]),
            "object_pose_frames_after": int(after_object_poses.shape[0]),
            "before_hand": str(before_hand_path),
            "after_hand": str(adapted_hand_path),
        }
    )
    layout_path.write_text(json.dumps(aligned_layout, indent=2), encoding="utf-8")

    object_pose_quality = assess_object_pose_continuity(
        aligned_layout,
        max_translation_step_m=args.max_object_translation_step_m,
        p95_translation_step_m=args.p95_object_translation_step_m,
        max_rotation_step_deg=args.max_object_rotation_step_deg,
        p95_rotation_step_deg=args.p95_object_rotation_step_deg,
        max_lost_frame_fraction=args.max_object_lost_frame_fraction,
    )
    retarget_input_qc = assess_dai_native_retarget_input(
        contact_alignment,
        overlap_frames,
        args.min_visual_overlap_frames,
        object_pose_quality,
    )
    if retarget_input_qc.get("max_offset") is None:
        retarget_input_qc["max_offset"] = args.hoi_align_max_offset
    if adapter_rigid_invariance.get("status") != "ok":
        retarget_input_qc.setdefault("errors", []).append(
            "adapter_rigid_invariance_invalid"
        )
        retarget_input_qc["status"] = "invalid"

    manifest = {
        "adapter": "prepare_do_as_i_do_scene_adapter.py",
        "adapter_schema_version": 3,
        "production_hoi_policy": "preserve_source_relative_transform",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "raw_video_provenance": raw_video_provenance,
        "object_id": object_id,
        "anchor_hand": anchor_hand,
        "ref_frame": ref_frame,
        "optimize_scale": optimize_scale,
        "object_track_source": "dai_native",
        "object_mesh_source": "dai_native",
        "retarget_object_source": "dai_native",
        "hand_source": args.hand_source,
        "canonical_transform": {
            "applied": False,
            "kind": "identity",
            "matrix": np.eye(4, dtype=np.float64).tolist(),
            "from_frame": "do_as_i_do_source_camera",
            "to_frame": "do_as_i_do_source_camera",
            "applies_to": [
                "hand_joints",
                "hand_vertices",
                "object_pose",
                "contact_points",
                "object_mesh_world_via_object_pose",
            ],
        },
        "hoi_refinement": scale_refinement_provenance(
            requested=optimize_scale,
            optimization=scale_optimization,
            contact_diagnostic=contact_alignment,
        ),
        "adapter_rigid_invariance": adapter_rigid_invariance,
        "adapter_boundary_inputs": {
            "before_hand": str(before_hand_path),
            "after_hand": str(adapted_hand_path),
            "before_object_layout": str(source_layout_path),
            "after_object_layout": str(layout_path),
        },
        "hand_npz": hand_stats,
        "source_frame_maps": [str(path) for path in sorted(source_dir.glob("frame_map*.json"))],
        "hand_projection_qc": projection_qc,
        "hand_masks": mask_stats,
        "visual_interaction": visual_interaction,
        "hoi_contact_alignment": contact_alignment,
        "object_pose_quality": object_pose_quality,
        "retarget_input_qc": retarget_input_qc,
        "gravity": load_json(output_dir / "gravity.json"),
        "layout": str(layout_path),
    }
    (output_dir / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if retarget_input_qc["status"] == "ok" else 4


if __name__ == "__main__":
    raise SystemExit(main())
