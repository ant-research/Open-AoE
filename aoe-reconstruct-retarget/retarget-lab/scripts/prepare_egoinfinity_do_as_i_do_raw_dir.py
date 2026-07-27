#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Keep the same pose-frame <-> camera-frame convention as official Do-as-I-Do
# conversion helpers so exported Ego object layouts match DAI's renderer and
# retargeting pipeline expectations.
POSE_TO_CAMERA_ROW = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

from prepare_do_as_i_do_scene_adapter import (  # noqa: E402
    build_raw_video_provenance as build_common_raw_video_provenance,
    choose_hand_npz,
    choose_ref_frame,
    copy_file,
    copy_tree,
    generate_hand_masks,
    hand_projection_qc,
    infer_object_id,
    link_file,
    load_json,
    remove_existing,
    require,
    run_scale_optimization,
    safe_adapter_output_dir,
    scale_refinement_provenance,
    stage_native_task_hand_mesh,
    write_adapted_hand_npz,
)
from hand_selection_utils import (  # noqa: E402
    contact_fraction_hand,
    dominant_hand_from_geometry,
    single_hand_is_strong,
)
from aoe_retarget_lab.hoi_geometry import (  # noqa: E402
    summarize_adapter_rigid_invariance,
    validate_rigid_transform,
)
from aoe_retarget_lab.object_scale import robust_projected_scale_fit, scale_bbox_xyxy  # noqa: E402
from aoe_retarget_lab.retarget_input_qc import assess_ego_retarget_input  # noqa: E402
from aoe_retarget_lab.route_input_qc import (  # noqa: E402
    CAMERA_INTRINSICS_BINDING_POLICY,
    CAMERA_INTRINSICS_BINDING_SCHEMA_VERSION,
    CAMERA_INTRINSICS_NORMALIZATION_POLICY,
    build_camera_intrinsics_binding,
    camera_intrinsics_fingerprint,
    camera_ray_depth_transform,
)


def oid_get(mapping: dict, obj_id):
    return mapping.get(obj_id) or mapping.get(str(obj_id)) or mapping.get(int(obj_id))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mask_centroid_from_obj_data(obj: dict | None) -> np.ndarray | None:
    if not isinstance(obj, dict):
        return None
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
    centroids = []
    for frame in result.get("frame_data") or []:
        obj = oid_get(frame.get("sam3_obj_data") or {}, obj_id)
        centroids.append(mask_centroid_from_obj_data(obj))
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
    return len(result.get("frame_data") or []), max_jump


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


def _camera_source_file_binding(path: Path) -> dict[str, object]:
    """Bind one camera-evidence file by resolved path and current SHA256."""

    resolved = path.expanduser().resolve()
    binding: dict[str, object] = {
        "path": str(resolved),
        "sha256": None,
    }
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


def build_raw_video_provenance(raw_dir: Path) -> dict[str, object]:
    """Use the exact same raw-video schema and gates as the native adapter."""

    return build_common_raw_video_provenance(raw_dir)


def _result_image_size(result: dict) -> tuple[int, int] | None:
    """Resolve the image size attached to EgoInfinity's camera model."""

    background = result.get("bg_rgb")
    if background is not None:
        array = np.asarray(background)
        if array.ndim >= 2 and array.shape[0] > 0 and array.shape[1] > 0:
            return int(array.shape[1]), int(array.shape[0])
    for frame in result.get("frame_data") or []:
        if not isinstance(frame, dict):
            continue
        for key in ("img_rgb", "image_rgb", "rgb", "image", "frame_rgb"):
            value = frame.get(key)
            if value is None:
                continue
            array = np.asarray(value)
            if array.ndim >= 2 and array.shape[0] > 0 and array.shape[1] > 0:
                return int(array.shape[1]), int(array.shape[0])
        for obj in (frame.get("sam3_obj_data") or {}).values():
            if not isinstance(obj, dict):
                continue
            shape = obj.get("mask_shape")
            if isinstance(shape, (list, tuple, np.ndarray)) and len(shape) >= 2:
                height, width = int(shape[0]), int(shape[1])
                if width > 0 and height > 0:
                    return width, height
    return None


def egoinfinity_object_camera_intrinsics(
    result: dict,
    pipeline_result: Path,
) -> dict:
    """Extract the exact pinhole model used to back-project Ego object poses."""

    pipeline_binding = _camera_source_file_binding(pipeline_result)
    size = _result_image_size(result)
    intrinsics: dict[str, object] = {"model": "pinhole"}
    extraction_errors: list[str] = []
    try:
        focal = float(result.get("dp_focal"))
    except (TypeError, ValueError):
        focal = float("nan")
    if not np.isfinite(focal) or focal <= 0.0:
        extraction_errors.append("missing_or_invalid_dp_focal")
    else:
        intrinsics.update({"fx": focal, "fy": focal})
    for key in ("cx", "cy"):
        try:
            value = float(result.get(key))
        except (TypeError, ValueError):
            value = float("nan")
        if not np.isfinite(value):
            extraction_errors.append(f"missing_or_invalid_{key}")
        else:
            intrinsics[key] = value
    if size is None:
        extraction_errors.append("missing_image_size")
    else:
        intrinsics.update({"width": int(size[0]), "height": int(size[1])})
    if not pipeline_binding.get("sha256"):
        extraction_errors.append("pipeline_result_file_binding_invalid")
    return {
        "source": {
            "kind": "egoinfinity_pipeline_result",
            "path": pipeline_binding["path"],
            "sha256": pipeline_binding.get("sha256"),
            "fields": ["dp_focal", "cx", "cy"],
            "file_bindings": {"pipeline_result": pipeline_binding},
            "extraction_errors": extraction_errors,
        },
        "intrinsics": intrinsics,
    }


def _npz_scalar_text(archive: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in archive:
        return None
    value = np.asarray(archive[key])
    if value.size != 1:
        return None
    text = str(value.reshape(()).item()).strip()
    return text or None


def _npz_scalar_float(archive: np.lib.npyio.NpzFile, key: str) -> float | None:
    if key not in archive:
        return None
    try:
        value = float(np.asarray(archive[key]).reshape(()))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _hawor_calibration_path(source_hands_npz: Path) -> Path | None:
    candidates: list[Path] = []
    if source_hands_npz.parent.name == "ego_hands_reconstruction":
        candidates.append(
            source_hands_npz.parent.parent
            / "ego_undistorted_video"
            / "undistorted_video_info.json"
        )
    for parent in source_hands_npz.parents:
        if parent.name == "ego_process":
            candidates.append(
                parent
                / "ego_undistorted_video"
                / "undistorted_video_info.json"
            )
            break
    return next((path.resolve() for path in candidates if path.is_file()), None)


def aoe_hand_camera_intrinsics(hand_npz: Path) -> dict:
    """Trace an adapted AoE hand artifact back to its calibrated source K."""

    hand_npz = hand_npz.expanduser().resolve()
    extraction_errors: list[str] = []
    adapted_binding = _camera_source_file_binding(hand_npz)
    if not adapted_binding.get("sha256"):
        extraction_errors.append("adapted_hand_npz_file_binding_invalid")
    source_hands_npz = hand_npz
    track_focal: float | None = None
    try:
        with np.load(hand_npz, allow_pickle=True) as archive:
            source_text = _npz_scalar_text(archive, "source_hands_npz")
            track_focal = _npz_scalar_float(archive, "focal")
    except (OSError, ValueError, TypeError) as exc:
        source_text = None
        extraction_errors.append(f"hand_npz_unreadable:{type(exc).__name__}")
    if source_text:
        candidate = Path(source_text).expanduser()
        if not candidate.is_absolute():
            candidate = hand_npz.parent / candidate
        source_hands_npz = candidate.resolve()
    if source_hands_npz.is_file() and source_hands_npz != hand_npz:
        try:
            with np.load(source_hands_npz, allow_pickle=True) as source_archive:
                source_focal = _npz_scalar_float(source_archive, "focal")
                if source_focal is not None:
                    track_focal = source_focal
        except (OSError, ValueError, TypeError) as exc:
            extraction_errors.append(
                f"source_hands_npz_unreadable:{type(exc).__name__}"
            )
    elif not source_hands_npz.is_file():
        extraction_errors.append("source_hands_npz_missing")

    source_hands_binding = _camera_source_file_binding(source_hands_npz)
    if not source_hands_binding.get("sha256"):
        extraction_errors.append("source_hands_npz_file_binding_invalid")

    calibration_path = _hawor_calibration_path(source_hands_npz)
    calibration_binding = (
        _camera_source_file_binding(calibration_path)
        if calibration_path is not None
        else {"path": None, "sha256": None, "error": "missing"}
    )
    intrinsics: dict[str, object] = {"model": "pinhole"}
    if calibration_path is None:
        extraction_errors.append("undistorted_video_calibration_missing")
    else:
        try:
            calibration = load_json(calibration_path)
            camera = calibration.get("cameraParams") or {}
            resolution = str(camera.get("resolution") or "")
            width_text, separator, height_text = resolution.lower().partition("x")
            if not separator:
                raise ValueError("cameraParams.resolution is not WIDTHxHEIGHT")
            intrinsics.update(
                {
                    "width": int(width_text),
                    "height": int(height_text),
                    "fx": float(camera["fx_pixels"]),
                    "fy": float(camera.get("fy_pixels", camera["fx_pixels"])),
                    "cx": float(camera["cx_pixels"]),
                    "cy": float(camera["cy_pixels"]),
                }
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            extraction_errors.append(
                f"undistorted_video_calibration_invalid:{type(exc).__name__}"
            )
    if not calibration_binding.get("sha256"):
        extraction_errors.append("calibration_file_binding_invalid")

    track_focal_relative_error = None
    if track_focal is None:
        extraction_errors.append("hawor_track_focal_missing")
    elif "fx" in intrinsics:
        track_focal_relative_error = abs(track_focal - float(intrinsics["fx"])) / max(
            abs(float(intrinsics["fx"])), 1.0e-12
        )
        if track_focal_relative_error > 0.05:
            extraction_errors.append("hawor_track_focal_calibration_mismatch")
    return {
        "source": {
            "kind": "aoe_hawor_calibrated_camera",
            "adapted_hand_npz": adapted_binding["path"],
            "adapted_hand_npz_sha256": adapted_binding.get("sha256"),
            "source_hands_npz": source_hands_binding["path"],
            "source_hands_npz_sha256": source_hands_binding.get("sha256"),
            "calibration_path": calibration_binding.get("path"),
            "calibration_sha256": calibration_binding.get("sha256"),
            "file_bindings": {
                "adapted_hand_npz": adapted_binding,
                "source_hands_npz": source_hands_binding,
                "calibration": calibration_binding,
            },
            "track_focal": track_focal,
            "track_focal_relative_error": track_focal_relative_error,
            "extraction_errors": extraction_errors,
        },
        "intrinsics": intrinsics,
    }


def build_ego_aoe_camera_intrinsics_binding(
    result: dict,
    pipeline_result: Path,
    hand_npz: Path,
) -> dict:
    """Bind Ego object and AoE hand camera models without changing geometry."""

    object_camera = egoinfinity_object_camera_intrinsics(result, pipeline_result)
    hand_camera = aoe_hand_camera_intrinsics(hand_npz)
    binding = build_camera_intrinsics_binding(object_camera, hand_camera)
    extraction_errors: list[str] = []
    for label, camera in (
        ("object_camera", object_camera),
        ("hand_camera", hand_camera),
    ):
        source = camera.get("source") if isinstance(camera, dict) else None
        if not isinstance(source, dict):
            extraction_errors.append(f"{label}_source_provenance_missing")
            continue
        extraction_errors.extend(
            f"{label}_{item}" for item in source.get("extraction_errors") or []
        )
    if extraction_errors:
        binding["errors"] = list(dict.fromkeys([*(binding.get("errors") or []), *extraction_errors]))
        binding["status"] = "invalid"
    return binding


def normalize_aoe_hand_camera_to_ego(
    hand_npz: Path,
    binding: dict,
    frames_dir: Path,
) -> dict:
    """Convert AoE hand XYZ and frame K to the Ego object camera model.

    The conversion preserves every hand vertex/joint pixel and its metric Z.
    It is applied before the adapter's before/after invariance boundary, so
    downstream invariance still detects any later object-only manipulation.
    """

    object_entry = binding.get("object_camera") or {}
    hand_entry = binding.get("hand_camera") or {}
    object_camera = object_entry.get("intrinsics") or {}
    hand_camera = hand_entry.get("intrinsics") or {}
    matrix = camera_ray_depth_transform(hand_camera, object_camera)
    with np.load(hand_npz, allow_pickle=True) as archive:
        output = {key: archive[key] for key in archive.files}
    transformed_fields = []
    source_points = []
    transformed_points = []
    for side in ("left", "right"):
        for suffix in ("vertices", "joints", "trans"):
            key = f"{side}_{suffix}"
            if key not in output:
                continue
            values = np.asarray(output[key])
            if values.shape[-1:] != (3,):
                continue
            converted = np.einsum(
                "ij,...j->...i", matrix, values.astype(np.float64)
            )
            output[key] = converted.astype(values.dtype, copy=False)
            transformed_fields.append(key)
            if suffix in {"vertices", "joints"}:
                source_points.append(values.reshape(-1, 3))
                transformed_points.append(converted.reshape(-1, 3))

    output["camera_intrinsics_normalization_kind"] = np.array(
        "ray_depth_source_to_egoinfinity", dtype="<U48"
    )
    output["camera_intrinsics_normalization_matrix"] = matrix
    output["camera_intrinsics_source_fingerprint"] = np.array(
        camera_intrinsics_fingerprint(hand_camera), dtype="<U64"
    )
    output["camera_intrinsics_target_fingerprint"] = np.array(
        camera_intrinsics_fingerprint(object_camera), dtype="<U64"
    )
    temp = hand_npz.with_name(hand_npz.name + ".camera-normalized.npz")
    np.savez(temp, **output)
    temp.replace(hand_npz)

    source_xyz = np.concatenate(source_points, axis=0)
    target_xyz = np.concatenate(transformed_points, axis=0)
    finite = (
        np.isfinite(source_xyz).all(axis=1)
        & np.isfinite(target_xyz).all(axis=1)
        & (source_xyz[:, 2] > 1.0e-8)
    )
    source_xyz = source_xyz[finite]
    target_xyz = target_xyz[finite]
    source_uv = np.column_stack(
        [
            float(hand_camera["fx"]) * source_xyz[:, 0] / source_xyz[:, 2]
            + float(hand_camera["cx"]),
            float(hand_camera["fy"]) * source_xyz[:, 1] / source_xyz[:, 2]
            + float(hand_camera["cy"]),
        ]
    )
    target_uv = np.column_stack(
        [
            float(object_camera["fx"]) * target_xyz[:, 0] / target_xyz[:, 2]
            + float(object_camera["cx"]),
            float(object_camera["fy"]) * target_xyz[:, 1] / target_xyz[:, 2]
            + float(object_camera["cy"]),
        ]
    )
    reprojection_error = np.linalg.norm(source_uv - target_uv, axis=1)

    source_frames_dir = frames_dir.resolve()
    if frames_dir.is_symlink():
        frames_dir.unlink()
        frames_dir.mkdir(parents=True)
        image_size = None
        for candidate in sorted(source_frames_dir.glob("*.png")):
            image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
            if image is not None and image.shape[0] > 0 and image.shape[1] > 0:
                image_size = (int(image.shape[1]), int(image.shape[0]))
                break
        if image_size is None:
            raise ValueError("camera normalization could not determine all_frames image size")
        scale_x = image_size[0] / float(object_camera["width"])
        scale_y = image_size[1] / float(object_camera["height"])
        target_k = np.array(
            [
                [float(object_camera["fx"]) * scale_x, 0.0, float(object_camera["cx"]) * scale_x],
                [0.0, float(object_camera["fy"]) * scale_y, float(object_camera["cy"]) * scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        intrinsics_count = 0
        for source in sorted(source_frames_dir.iterdir()):
            destination = frames_dir / source.name
            if source.name.endswith("_intrinsics.npy"):
                np.save(destination, target_k)
                intrinsics_count += 1
            else:
                link_file(source.resolve(), destination)
    else:
        raise ValueError("camera normalization requires an unmodified all_frames symlink")

    original_comparison = dict(binding.get("comparison") or {})
    binding.update(
        {
            "policy": CAMERA_INTRINSICS_NORMALIZATION_POLICY,
            "status": "ok",
            "geometry_modified": True,
            "errors": [],
            "original_comparison": original_comparison,
            "comparison": {
                "comparable": True,
                "matches": True,
                "effective_hand_camera": "object_camera",
            },
            "normalization": {
                "kind": "ray_depth_source_to_target_camera",
                "source_camera_fingerprint": camera_intrinsics_fingerprint(hand_camera),
                "target_camera_fingerprint": camera_intrinsics_fingerprint(object_camera),
                "matrix": matrix.tolist(),
                "transformed_fields": transformed_fields,
                "output_hand_npz": str(hand_npz),
                "output_hand_npz_sha256": sha256_file(hand_npz),
                "rewritten_intrinsics_count": intrinsics_count,
                "output_image_wh": list(image_size),
                "output_intrinsics": target_k.tolist(),
                "reprojection_sample_count": int(reprojection_error.size),
                "reprojection_error_px_max": float(reprojection_error.max(initial=0.0)),
                "depth_max_abs_error_m": float(
                    np.max(np.abs(source_xyz[:, 2] - target_xyz[:, 2]), initial=0.0)
                ),
                "object_geometry_modified": False,
                "object_pose_modified": False,
                "hoi_translation_applied": False,
                "scale_fit_applied": False,
            },
        }
    )
    return binding


def write_gravity_metadata(path: Path, source_path: Path, max_tilt_deg: float | None = None) -> dict:
    payload = load_json(source_path)
    payload["source"] = "copied_from_do_as_i_do_reconstruction"
    payload["source_gravity_json"] = str(source_path)
    tilt = max(
        abs(float(payload.get("roll_deg", 0.0) or 0.0)),
        abs(float(payload.get("pitch_deg", 0.0) or 0.0)),
    )
    if max_tilt_deg is not None and max_tilt_deg > 0 and tilt > max_tilt_deg:
        original = json.loads(json.dumps(payload))
        payload.update(
            {
                "vec3d": [0.0, -1.0, 0.0],
                "roll_deg": 0.0,
                "pitch_deg": 0.0,
                "gravity_clamped": True,
                "gravity_clamp_reason": f"tilt {tilt:.2f} deg exceeds max {float(max_tilt_deg):.2f} deg",
                "original_gravity": original,
            }
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def ego_mesh_path(pipeline_result: Path, obj_id: int) -> Path:
    return pipeline_result.expanduser().resolve().parent / "sam3_meshes" / f"obj_{int(obj_id)}.ply"


def write_convex_hull_obj_from_ego_mesh(src: Path, dst: Path) -> dict:
    loaded = trimesh.load(src, process=False)
    vertices = None
    faces = None
    if hasattr(loaded, "vertices") and len(loaded.vertices):
        vertices = np.asarray(loaded.vertices, dtype=np.float64)
        faces = np.asarray(getattr(loaded, "faces", []), dtype=np.int64)
    elif hasattr(loaded, "geometry"):
        chunks = []
        face_chunks = []
        offset = 0
        for geom in loaded.geometry.values():
            if not hasattr(geom, "vertices") or not len(geom.vertices):
                continue
            verts = np.asarray(geom.vertices, dtype=np.float64)
            chunks.append(verts)
            geom_faces = np.asarray(getattr(geom, "faces", []), dtype=np.int64)
            if geom_faces.size:
                face_chunks.append(geom_faces + offset)
            offset += len(verts)
        if chunks:
            vertices = np.concatenate(chunks, axis=0)
            faces = np.concatenate(face_chunks, axis=0) if face_chunks else np.zeros((0, 3), dtype=np.int64)
    if vertices is None or len(vertices) < 4:
        raise RuntimeError(f"EgoInfinity mesh has too few vertices: {src}")

    if faces is not None and len(faces) > 0:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    else:
        mesh = trimesh.PointCloud(vertices).convex_hull
    if len(mesh.faces) == 0:
        mesh = trimesh.PointCloud(vertices).convex_hull
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    return {
        "source": str(src),
        "output": str(dst),
        "method": "convex_hull",
        "vertices_in": int(len(vertices)),
        "vertices_out": int(len(mesh.vertices)),
        "faces_out": int(len(mesh.faces)),
        "extents": extents.tolist(),
        "diag": float(np.linalg.norm(extents)),
    }


def is_box_prompt(prompt: str | None) -> bool:
    return "box" in str(prompt or "").lower()


def write_boxlike_obj_from_ego_mesh(src: Path, dst: Path) -> dict:
    loaded = trimesh.load(src, process=False)
    if hasattr(loaded, "vertices"):
        vertices = np.asarray(getattr(loaded, "vertices", []), dtype=np.float64)
    elif hasattr(loaded, "geometry"):
        chunks = [
            np.asarray(getattr(geom, "vertices", []), dtype=np.float64)
            for geom in loaded.geometry.values()
            if hasattr(geom, "vertices") and len(geom.vertices)
        ]
        vertices = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=np.float64)
    else:
        vertices = np.zeros((0, 3), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise RuntimeError(f"EgoInfinity mesh has too few vertices for box fallback: {src}")

    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center = (lo + hi) * 0.5
    extents = np.maximum(hi - lo, 1e-6)
    largest = float(extents.max())
    if largest <= 1e-8:
        raise RuntimeError(f"EgoInfinity box mesh has degenerate extents: {src}")

    min_extent = max(0.45 * largest, 1e-6)
    mid_extent = max(0.65 * largest, min_extent)
    order = np.argsort(extents)
    adjusted = extents.copy()
    adjusted[order[0]] = max(adjusted[order[0]], min_extent)
    adjusted[order[1]] = max(adjusted[order[1]], mid_extent)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = center
    mesh = trimesh.creation.box(extents=adjusted, transform=transform)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)
    return {
        "source": str(src),
        "output": str(dst),
        "method": "boxlike_cuboid_fallback",
        "vertices_in": int(len(vertices)),
        "vertices_out": int(len(mesh.vertices)),
        "faces_out": int(len(mesh.faces)),
        "extents": adjusted.tolist(),
        "original_extents": extents.tolist(),
        "diag": float(np.linalg.norm(adjusted)),
        "original_diag": float(np.linalg.norm(extents)),
        "min_extent_ratio": float(min_extent / largest),
        "mid_extent_ratio": float(mid_extent / largest),
    }


def should_use_boxlike_mesh(ego_mesh: Path, prompt: str | None) -> bool:
    if not is_box_prompt(prompt):
        return False
    try:
        mesh = trimesh.load(ego_mesh, process=False)
        vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
    except Exception:
        return False
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        return False
    extents = np.sort(vertices.max(axis=0) - vertices.min(axis=0))
    largest = float(extents[-1])
    if largest <= 1e-8:
        return False
    middle_ratio = float(extents[-2] / largest)
    smallest_ratio = float(extents[0] / largest)
    return middle_ratio < 0.65 or smallest_ratio < 0.45


def replace_object_mesh_with_ego_mesh(output_dir: Path, object_id: str, ego_mesh: Path, prompt: str | None = None) -> dict:
    mask_root = output_dir / "video_segmentation" / "masks"
    targets = sorted(mask_root.glob(f"frame_*_masks/{object_id}/{object_id}.obj"))
    if not targets:
        raise FileNotFoundError(f"cannot find Do-as-I-Do object OBJ under {mask_root} for {object_id}")
    use_boxlike = should_use_boxlike_mesh(ego_mesh, prompt)
    stats = []
    for target in targets:
        if use_boxlike:
            stats.append(write_boxlike_obj_from_ego_mesh(ego_mesh, target))
        else:
            stats.append(write_convex_hull_obj_from_ego_mesh(ego_mesh, target))
    return {
        "ego_mesh": str(ego_mesh),
        "object_id": object_id,
        "prompt": str(prompt or ""),
        "boxlike_fallback": bool(use_boxlike),
        "targets": [str(p) for p in targets],
        "stats": stats,
    }


def infer_interaction_hands(pose_info: dict) -> dict:
    values = []
    first_single = None
    for item in pose_info.get("grasp_hand_per_frame") or []:
        if item is None:
            continue
        text = str(item).strip().lower()
        if text in {"l", "left"}:
            values.append("left")
            first_single = first_single or "left"
        elif text in {"r", "right"}:
            values.append("right")
            first_single = first_single or "right"
        elif text == "both":
            values.append("both")
    counts = Counter(values)
    state_counts = pose_info.get("state_counts") or {}
    left_score = int(counts["left"] or state_counts.get("grasped_l", 0) or 0)
    right_score = int(counts["right"] or state_counts.get("grasped_r", 0) or 0)
    both_score = int(counts["both"] or state_counts.get("grasped_both", 0) or 0)

    # The official retargeter supports bimanual. When EgoInfinity says the
    # object is mostly grasped by both hands, forcing one side can select a
    # non-interacting hand or make the object follow a hand-like track.
    if both_score > max(left_score, right_score):
        selected = "bimanual"
    elif left_score > right_score:
        selected = "left"
    elif right_score > left_score:
        selected = "right"
    elif first_single is not None:
        selected = first_single
    elif both_score > 0:
        selected = "bimanual"
    else:
        selected = "left"
    if left_score > right_score:
        scale_anchor = "left"
    elif right_score > left_score:
        scale_anchor = "right"
    elif first_single is not None:
        scale_anchor = first_single
    else:
        scale_anchor = "left"
    return {
        "selected_hand": selected,
        "scale_anchor_hand": scale_anchor,
        "counts": dict(counts),
        "state_counts": dict(state_counts),
        "left_score": int(left_score),
        "right_score": int(right_score),
        "both_score": int(both_score),
    }


def ego_mesh_scale(pose_info: dict, mesh_info: dict | None) -> tuple[float, str]:
    # Prefer the scale-sanity corrected value when present. The original SAM3D
    # canonical scale can overestimate household bottles by several times,
    # which breaks both official projections and retarget contact geometry.
    for key in ("scale_correction",):
        value = pose_info.get(key) if isinstance(pose_info, dict) else None
        try:
            scale = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(scale) and scale > 0:
            return scale, f"pose_track_info.{key}"
    for key in ("scale_correction_orig_sam3d",):
        value = pose_info.get(key) if isinstance(pose_info, dict) else None
        try:
            scale = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(scale) and scale > 0:
            return scale, f"pose_track_info.{key}"
    if isinstance(mesh_info, dict):
        try:
            scale = float(mesh_info.get("canonical_scale"))
        except (TypeError, ValueError):
            scale = 0.0
        if np.isfinite(scale) and scale > 0:
            return scale, "sam3_mesh_info.canonical_scale"
    return 1.0, "fallback"


def run_ego_scale_optimization_diagnostic(
    *,
    requested: bool,
    python_bin: str,
    prepared_dir: Path,
    object_id: str,
    anchor_hand: str,
    ref_frame: int,
    viz_dir: Path | None,
    production_mesh_scale: float,
    production_mesh_scale_source: str,
) -> dict:
    """Run optional scale analysis without changing the production mesh scale.

    ``run_scale_optimization`` restores the clean production layout and returns
    only a diagnostic candidate.  This wrapper additionally binds that outcome
    to the exact production scale/source so provenance cannot imply that the
    candidate scale was consumed by the Ego route.
    """

    production_scale = float(production_mesh_scale)
    production_source = str(production_mesh_scale_source).strip()
    if not np.isfinite(production_scale) or production_scale <= 0.0:
        raise ValueError("production mesh scale must be positive and finite")
    if not production_source:
        raise ValueError("production mesh scale source must be non-empty")

    optimization = {
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
    if requested:
        optimization = run_scale_optimization(
            python_bin,
            prepared_dir,
            object_id,
            anchor_hand,
            ref_frame,
            viz_dir,
        )
    if not isinstance(optimization, dict):
        raise ValueError("scale optimization diagnostic must return a mapping")
    if optimization.get("requested") is not bool(requested):
        raise ValueError("scale optimization requested flag does not match CLI intent")
    if optimization.get("applied") is not False:
        raise ValueError("scale optimization diagnostic modified production HOI")

    return {
        "requested": bool(requested),
        "applied": False,
        "diagnostic_only": True,
        "production_mesh_scale": production_scale,
        "production_mesh_scale_source": production_source,
        "optimizer_outcome": optimization,
        "candidate_generated": optimization.get("candidate_generated") is True,
        "candidate_output": optimization.get("candidate_output"),
        "candidate_mesh_scale": optimization.get("candidate_mesh_scale"),
    }


def initial_ego_mask_scale_fit_diagnostic(
    *,
    object_geometry_source: str,
    optimize_scale: bool,
    no_fit_object_scale_to_ego_mask: bool,
) -> dict:
    """Describe why the independent Ego-mask scale fit has not run yet."""

    if object_geometry_source != "ego":
        reason = "non_ego_object_geometry"
    elif optimize_scale:
        reason = "skipped_for_scale_optimization_diagnostic"
    elif no_fit_object_scale_to_ego_mask:
        reason = "object_scale_fit_disabled_preserve_source"
    else:
        reason = "pending_ego_mask_scale_fit"
    return {
        "available": False,
        "applied": False,
        "diagnostic_only": True,
        "mesh_modified": False,
        "reason": reason,
    }


def source_layout_path(source_dir: Path, object_id: str) -> Path:
    layout_dir = source_dir / "obj_tracking_out" / object_id / "combined_visualization"
    clean = layout_dir / "layout_camera_frame.json"
    optimized = layout_dir / "layout_camera_frame_optimized.json"
    # Never inherit a legacy object-only contact correction.  The clean layout
    # is the authoritative source trajectory; optimized is only a compatibility
    # fallback when no clean reconstruction layout exists.
    if clean.exists():
        return clean
    return optimized


def adapter_object_provenance(source_dir: Path, object_geometry_source: str) -> dict:
    if object_geometry_source == "ego":
        return {
            "object_track_source": "egoinfinity",
            "object_mesh_source": "egoinfinity",
            "retarget_object_source": "egoinfinity",
        }
    source_manifest = {}
    manifest_path = source_dir / "adapter_manifest.json"
    if manifest_path.exists():
        try:
            source_manifest = load_json(manifest_path)
        except Exception:
            source_manifest = {}
    retarget_object_source = str(source_manifest.get("retarget_object_source") or "dai_native")
    if retarget_object_source == "dai_native_contact_aligned":
        retarget_object_source = "dai_native"
    return {
        "object_track_source": str(source_manifest.get("object_track_source") or "dai_native"),
        "object_mesh_source": str(source_manifest.get("object_mesh_source") or "dai_native"),
        "retarget_object_source": retarget_object_source,
    }


def copy_tracking_assets_for_adapter(source_dir: Path, output_dir: Path, object_id: str, optimize_scale: bool) -> dict:
    src_obj = source_dir / "obj_tracking_out" / object_id
    dst_obj = output_dir / "obj_tracking_out" / object_id
    src_layout = src_obj / "combined_visualization" / "layout_camera_frame.json"
    require(src_layout, "Do-as-I-Do layout_camera_frame.json")

    remove_existing(dst_obj)
    shutil.copytree(src_obj, dst_obj, symlinks=False)

    dst_layout = dst_obj / "combined_visualization"
    optimized = dst_layout / "layout_camera_frame_optimized.json"
    if optimize_scale:
        remove_existing(optimized)
    else:
        # Identity/source-object adapters must start from the clean source
        # layout, never a stale optimized/object-only-refined trajectory.
        remove_existing(optimized)
        copy_file(src_layout, optimized)
    return {
        "mode": "copy",
        "source": str(src_obj),
        "destination": str(dst_obj),
        "dereference_symlinks": True,
        "optimize_scale": bool(optimize_scale),
        "optimized_layout_present": optimized.exists(),
    }


def transform_from_layout_entry(entry: dict) -> np.ndarray:
    local = entry.get("local_to_scene") or {}
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_wxyz_to_matrix(
        np.asarray(local.get("quat_wxyz_camera_frame") or local.get("new_quat") or [1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    transform[:3, 3] = np.asarray(
        local.get("translation_camera_frame") or local.get("translation") or [0.0, 0.0, 0.0],
        dtype=np.float64,
    ).reshape(3)
    return transform


def layout_object_pose_sequence(layout: dict) -> np.ndarray:
    """Parse a DAI camera-frame layout into a strict ordered SE(3) sequence."""

    poses: list[np.ndarray] = []
    for index, entry in enumerate(layout_object_entries(layout)):
        local = entry.get("local_to_scene")
        if not isinstance(local, dict):
            raise ValueError(f"layout object {index} is missing local_to_scene")
        translation = local.get("translation_camera_frame")
        if translation is None:
            translation = local.get("translation")
        quaternion = local.get("quat_wxyz_camera_frame")
        if quaternion is None:
            quaternion = local.get("new_quat")
        if quaternion is None:
            quaternion = local.get("quat_wxyz")
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


def summarize_source_object_layout_invariance(
    before_hand_npz: Path,
    after_hand_npz: Path,
    before_layout_path: Path,
    after_layout_path: Path,
) -> dict:
    """Measure source-object identity entirely in the DAI source frame.

    The two explicit hand files are the actual adapter boundaries.  Production
    source-object routes use identity canonicalization, so any hand-only
    mutation (including semantic slot swaps) is rejected along with any
    object-only mutation.
    """

    before_poses = layout_object_pose_sequence(load_json(before_layout_path))
    after_poses = layout_object_pose_sequence(load_json(after_layout_path))
    summary = summarize_adapter_rigid_invariance(
        before_hand_npz,
        after_hand_npz,
        before_object_poses=before_poses,
        after_object_poses=after_poses,
        canonical_transform=None,
    )
    summary.update(
        {
            "measurement_scope": "source_object_layout_before_after_adapter",
            "comparison_semantics": "true_pre_adapter_vs_post_adapter_hand_and_object",
            "hand_coordinate_frame": "explicit_adapter_boundary_frames",
            "before_object_coordinate_frame": "do_as_i_do_source_camera",
            "after_object_coordinate_frame": "do_as_i_do_source_camera",
            "before_layout": str(before_layout_path),
            "after_layout": str(after_layout_path),
            "object_pose_frames_before": int(before_poses.shape[0]),
            "object_pose_frames_after": int(after_poses.shape[0]),
            "before_hand": str(before_hand_npz),
            "after_hand": str(after_hand_npz),
        }
    )
    return summary


def pose_translation_from_camera_frame(translation_camera: np.ndarray) -> list[float]:
    t_cam = np.asarray(translation_camera, dtype=np.float64).reshape(3)
    t_pose = t_cam @ POSE_TO_CAMERA_ROW.T
    return [float(t_pose[1]), float(t_pose[2]), float(t_pose[0])]


def pose_quat_from_camera_rotation(rotation_camera: np.ndarray) -> list[float]:
    rot_pose = POSE_TO_CAMERA_ROW @ np.asarray(rotation_camera, dtype=np.float64).reshape(3, 3)
    return rotmat_to_quat_wxyz(rot_pose)


def align_transforms_to_source_layout(
    transforms: np.ndarray,
    source_dir: Path,
    object_id: str,
    ref_frame: int,
) -> tuple[np.ndarray, dict]:
    layout_path = source_layout_path(source_dir, object_id)
    if not layout_path.exists():
        return transforms, {"available": False, "reason": "missing_source_layout", "path": str(layout_path)}
    entries = layout_object_entries(load_json(layout_path))
    if not entries:
        return transforms, {"available": False, "reason": "empty_source_layout", "path": str(layout_path)}
    source_ref_idx = int(np.clip(ref_frame, 0, len(entries) - 1))
    ego_ref_idx = int(np.clip(ref_frame, 0, len(transforms) - 1))
    source_ref = transform_from_layout_entry(entries[source_ref_idx])
    ego_ref = np.asarray(transforms[ego_ref_idx], dtype=np.float64)
    alignment = source_ref @ np.linalg.inv(ego_ref)
    aligned = np.stack([alignment @ np.asarray(item, dtype=np.float64) for item in transforms], axis=0)
    return aligned, {
        "available": True,
        "path": str(layout_path),
        "source_ref_idx": source_ref_idx,
        "ego_ref_idx": ego_ref_idx,
        "source_ref_frame_idx": int(entries[source_ref_idx].get("frame_idx", entries[source_ref_idx].get("index", source_ref_idx)) or source_ref_idx),
        "alignment_transform": alignment.tolist(),
    }


def project_vertices(vertices: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = vertices[:, 2]
    valid = np.isfinite(vertices).all(axis=1) & (z > 1e-6)
    uv = np.zeros((vertices.shape[0], 2), dtype=np.float32)
    uv[valid, 0] = intrinsics[0, 0] * vertices[valid, 0] / z[valid] + intrinsics[0, 2]
    uv[valid, 1] = intrinsics[1, 1] * vertices[valid, 1] / z[valid] + intrinsics[1, 2]
    return uv, valid


def source_object_selection_score(
    result: dict,
    pipeline_result: Path,
    source_dir: Path,
    source_object_id: str,
    ref_frame: int,
    ego_obj_id: int,
    pose_info: dict,
    mesh_info: dict | None,
) -> dict:
    mask = read_object_mask(source_dir / "video_segmentation" / "masks", source_object_id, ref_frame)
    intrinsics_path = source_dir / "all_frames" / f"{int(ref_frame):06d}_intrinsics.npy"
    if mask is None:
        return {"available": False, "reason": "missing_source_object_mask", "ref_frame": int(ref_frame)}
    if not intrinsics_path.exists():
        return {"available": False, "reason": "missing_source_intrinsics", "ref_frame": int(ref_frame)}
    mesh_path = ego_mesh_path(pipeline_result, int(ego_obj_id))
    if not mesh_path.exists():
        return {"available": False, "reason": "missing_ego_mesh", "mesh_path": str(mesh_path)}
    try:
        vertices = load_mesh_vertices(mesh_path)
    except Exception as exc:
        return {"available": False, "reason": str(exc), "mesh_path": str(mesh_path)}
    transforms = pose_sequence_for(result, ego_obj_id, pose_info)
    if len(transforms) <= 0:
        return {"available": False, "reason": "missing_pose_sequence"}
    transforms, alignment_meta = align_transforms_to_source_layout(transforms, source_dir, source_object_id, ref_frame)

    frame_idx = int(np.clip(ref_frame, 0, len(transforms) - 1))
    mesh_scale, mesh_scale_source = ego_mesh_scale(pose_info, mesh_info)
    transform = np.asarray(transforms[frame_idx], dtype=np.float64)
    points = (transform[:3, :3] @ (vertices * float(mesh_scale)).T).T + transform[:3, 3]
    intrinsics = np.load(intrinsics_path).astype(np.float64)
    height, width = mask.shape[:2]
    uv, valid = project_vertices(points, intrinsics)
    projected_box = bbox_from_points(uv, valid, width, height)
    source_box = bbox_from_mask(mask)
    overlap = bbox_overlap(projected_box, source_box)
    source_area = max(bbox_area(source_box), 1.0)
    projected_area = max(bbox_area(projected_box), 1.0)
    diag = max(float(np.hypot(width, height)), 1.0)
    if projected_box is None or source_box is None:
        proximity = 0.0
    else:
        proximity = max(0.0, 1.0 - float(np.linalg.norm(bbox_center(projected_box) - bbox_center(source_box))) / (0.35 * diag))
    return {
        "available": True,
        "ref_frame": int(ref_frame),
        "ego_frame": int(frame_idx),
        "image_area": float(height * width),
        "mesh_scale": float(mesh_scale),
        "mesh_scale_source": mesh_scale_source,
        "source_box": [float(x) for x in source_box] if source_box is not None else None,
        "projected_box": [float(x) for x in projected_box] if projected_box is not None else None,
        "overlap": float(overlap),
        "source_area": float(source_area),
        "projected_area": float(projected_area),
        "overlap_ratio": float(overlap / source_area),
        "iou_like": float(overlap / max(source_area + projected_area - overlap, 1.0)),
        "center_proximity": float(proximity),
        "source_layout_alignment": alignment_meta,
    }


def bbox_from_ego_mask(
    result: dict,
    obj_id: int,
    frame_idx: int,
    target_size: tuple[int, int] | None = None,
) -> tuple[float, float, float, float] | None:
    frames = result.get("frame_data") or []
    if frame_idx < 0 or frame_idx >= len(frames):
        return None
    obj = oid_get(frames[frame_idx].get("sam3_obj_data") or {}, obj_id)
    if not isinstance(obj, dict):
        return None
    packed = obj.get("mask_packed")
    shape = obj.get("mask_shape")
    if packed is None or shape is None:
        return None
    h, w = [int(x) for x in shape]
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))[: h * w]
    mask = bits.reshape(h, w)
    box = bbox_from_mask(mask)
    if box is not None and target_size is not None and target_size != (w, h):
        box = scale_bbox_xyxy(box, source_size=(w, h), target_size=target_size)
    return box


def bbox_from_mask(mask: np.ndarray | None) -> tuple[float, float, float, float] | None:
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.where(mask > 0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def bbox_from_2d_points(points: np.ndarray | None) -> tuple[float, float, float, float] | None:
    if points is None:
        return None
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    valid = np.isfinite(pts).all(axis=1)
    if not valid.any():
        return None
    pts = pts[valid]
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def candidate_visual_interaction_score(result: dict, obj_id: int) -> dict:
    frames = result.get("frame_data") or []
    if not frames:
        return {"available": False, "reason": "missing_frame_data"}
    sample_ids = list(range(len(frames)))
    if len(sample_ids) > 32:
        sample_ids = np.linspace(0, len(frames) - 1, 32, dtype=int).tolist()
    scores: dict[str, dict[str, float]] = {
        "left": {"score": 0.0, "frames": 0.0, "overlap_frames": 0.0},
        "right": {"score": 0.0, "frames": 0.0, "overlap_frames": 0.0},
    }
    for frame_idx in sample_ids:
        frame = frames[int(frame_idx)]
        obj = oid_get(frame.get("sam3_obj_data") or {}, obj_id)
        obj_box = bbox_from_ego_mask(result, obj_id, int(frame_idx))
        if not isinstance(obj, dict) or obj_box is None:
            continue
        mask_shape = obj.get("mask_shape")
        if mask_shape is None:
            continue
        mask_shape = np.asarray(mask_shape).reshape(-1)
        if mask_shape.size < 2:
            continue
        height, width = int(mask_shape[0]), int(mask_shape[1])
        object_area = max(bbox_area(obj_box), 1.0)
        image_diag = max(float(np.hypot(width, height)), 1.0)
        object_center = bbox_center(obj_box)
        hands_right = frame.get("hand_is_right") or []
        joints_2d_all = frame.get("joints_2d_pred") or []
        for hand_idx, joints_2d in enumerate(joints_2d_all):
            hand_box = bbox_from_2d_points(joints_2d)
            if hand_box is None:
                continue
            side = "right" if (hand_idx < len(hands_right) and bool(hands_right[hand_idx])) else "left"
            overlap = bbox_overlap(hand_box, obj_box)
            distance = float(np.linalg.norm(bbox_center(hand_box) - object_center))
            proximity = max(0.0, 1.0 - distance / (0.35 * image_diag))
            scores[side]["score"] += 4.0 * overlap / object_area + proximity
            scores[side]["frames"] += 1.0
            if overlap > 0:
                scores[side]["overlap_frames"] += 1.0
    normalized = {
        side: values["score"] / max(values["frames"], 1.0)
        for side, values in scores.items()
    }
    return {
        "available": True,
        "scores": scores,
        "normalized": normalized,
        "best_score": max(float(normalized["left"]), float(normalized["right"])),
    }


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise RuntimeError(f"OBJ has no vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def load_mesh_vertices(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".obj":
        return load_obj_vertices(path)
    loaded = trimesh.load(path, process=False)
    if hasattr(loaded, "vertices") and len(getattr(loaded, "vertices", [])):
        vertices = np.asarray(loaded.vertices, dtype=np.float64)
        if vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) > 0:
            return vertices
    if hasattr(loaded, "geometry"):
        chunks = []
        for geom in loaded.geometry.values():
            vertices = np.asarray(getattr(geom, "vertices", []), dtype=np.float64)
            if vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) > 0:
                chunks.append(vertices)
        if chunks:
            return np.concatenate(chunks, axis=0)
    raise RuntimeError(f"mesh has no vertices: {path}")


def fit_mesh_scale_to_ego_mask(
    result: dict,
    ego_obj_id: int,
    output_dir: Path,
    object_id: str,
    transforms: np.ndarray,
    base_scale: float,
) -> dict:
    mesh_paths = sorted((output_dir / "video_segmentation" / "masks").glob(f"frame_*_masks/{object_id}/{object_id}.obj"))
    if not mesh_paths:
        return {"available": False, "reason": "missing_object_obj"}
    try:
        vertices = load_obj_vertices(mesh_paths[0])
    except Exception as exc:
        return {"available": False, "reason": str(exc)}

    frame_count = int(len(transforms))
    ego_frame_count = len(result.get("frame_data") or [])
    if frame_count <= 0 or ego_frame_count <= 0:
        return {"available": False, "reason": "missing_frames"}
    sample_ids = list(range(frame_count))
    if len(sample_ids) > 24:
        sample_ids = np.linspace(0, frame_count - 1, 24, dtype=int).tolist()
    ego_indices = np.rint(np.linspace(0, ego_frame_count - 1, frame_count)).astype(int)
    ratios = []
    rows = []
    for frame_idx in sample_ids:
        image = cv2.imread(str(output_dir / "all_frames" / f"{frame_idx:06d}.png"), cv2.IMREAD_COLOR)
        intrinsics_path = output_dir / "all_frames" / f"{frame_idx:06d}_intrinsics.npy"
        if image is None or not intrinsics_path.exists():
            continue
        height, width = image.shape[:2]
        ego_box = bbox_from_ego_mask(
            result,
            ego_obj_id,
            int(ego_indices[frame_idx]),
            target_size=(width, height),
        )
        if ego_box is None:
            continue
        transform = np.asarray(transforms[frame_idx], dtype=np.float64)
        points = (transform[:3, :3] @ (vertices * float(base_scale)).T).T + transform[:3, 3]
        uv, valid = project_vertices(points, np.load(intrinsics_path))
        proj_box = bbox_from_points(uv, valid, width, height)
        ego_area = bbox_area(ego_box)
        proj_area = bbox_area(proj_box)
        if ego_area <= 1.0 or proj_area <= 1.0:
            continue
        ratio = float(np.sqrt(ego_area / proj_area))
        if not np.isfinite(ratio) or ratio <= 0:
            continue
        ratios.append(ratio)
        rows.append(
            {
                "frame": int(frame_idx),
                "ego_frame": int(ego_indices[frame_idx]),
                "ego_bbox": [float(x) for x in ego_box],
                "projected_bbox": [float(x) for x in proj_box],
                "area_ratio_sqrt": ratio,
            }
        )
    decision = robust_projected_scale_fit(ratios)
    if not decision.get("available"):
        return {**decision, "rows": rows}
    multiplier = float(decision.get("multiplier", 1.0))
    return {
        **decision,
        "base_scale": float(base_scale),
        "fitted_scale": float(base_scale) * multiplier,
        "frames": rows,
    }


def refine_object_translations_to_ego_mask(
    result: dict,
    ego_obj_id: int,
    output_dir: Path,
    object_id: str,
    transforms: np.ndarray,
    mesh_scale: float,
) -> tuple[np.ndarray, dict]:
    mesh_paths = sorted((output_dir / "video_segmentation" / "masks").glob(f"frame_*_masks/{object_id}/{object_id}.obj"))
    if not mesh_paths:
        return transforms, {"available": False, "reason": "missing_object_obj"}
    try:
        vertices = load_obj_vertices(mesh_paths[0]) * float(mesh_scale)
    except Exception as exc:
        return transforms, {"available": False, "reason": str(exc)}

    refined = np.asarray(transforms, dtype=np.float64).copy()
    ego_frame_count = len(result.get("frame_data") or [])
    frame_count = int(len(refined))
    if frame_count <= 0 or ego_frame_count <= 0:
        return refined, {"available": False, "reason": "missing_frames"}
    ego_indices = np.rint(np.linspace(0, ego_frame_count - 1, frame_count)).astype(int)
    rows = []
    used = 0
    for frame_idx in range(frame_count):
        image = cv2.imread(str(output_dir / "all_frames" / f"{frame_idx:06d}.png"), cv2.IMREAD_COLOR)
        intrinsics_path = output_dir / "all_frames" / f"{frame_idx:06d}_intrinsics.npy"
        if image is None or not intrinsics_path.exists():
            continue
        height, width = image.shape[:2]
        ego_box = bbox_from_ego_mask(
            result,
            ego_obj_id,
            int(ego_indices[frame_idx]),
            target_size=(width, height),
        )
        if ego_box is None or bbox_area(ego_box) <= 1.0:
            continue
        intrinsics = np.load(intrinsics_path)
        transform = refined[frame_idx]
        rotation = transform[:3, :3]
        translation = transform[:3, 3].copy()
        points = (rotation @ vertices.T).T + translation
        uv, valid = project_vertices(points, intrinsics)
        proj_box = bbox_from_points(uv, valid, width, height)
        if proj_box is None or bbox_area(proj_box) <= 1.0:
            continue

        ego_w = max(ego_box[2] - ego_box[0], 1.0)
        ego_h = max(ego_box[3] - ego_box[1], 1.0)
        proj_w = max(proj_box[2] - proj_box[0], 1.0)
        proj_h = max(proj_box[3] - proj_box[1], 1.0)
        z_factor = float(np.median([proj_w / ego_w, proj_h / ego_h]))
        z_factor = float(np.clip(z_factor, 0.7, 1.45))
        new_translation = translation.copy()
        new_translation[2] = max(1e-4, translation[2] * z_factor)

        points_z = (rotation @ vertices.T).T + new_translation
        uv_z, valid_z = project_vertices(points_z, intrinsics)
        proj_box_z = bbox_from_points(uv_z, valid_z, width, height)
        if proj_box_z is None:
            continue
        ego_cx = (ego_box[0] + ego_box[2]) * 0.5
        ego_cy = (ego_box[1] + ego_box[3]) * 0.5
        proj_cx = (proj_box_z[0] + proj_box_z[2]) * 0.5
        proj_cy = (proj_box_z[1] + proj_box_z[3]) * 0.5
        depth = max(float(np.median(points_z[valid_z, 2])) if valid_z.any() else new_translation[2], 1e-4)
        dx = float(np.clip((ego_cx - proj_cx) * depth / intrinsics[0, 0], -0.18, 0.18))
        dy = float(np.clip((ego_cy - proj_cy) * depth / intrinsics[1, 1], -0.18, 0.18))
        new_translation[0] += dx
        new_translation[1] += dy
        refined[frame_idx, :3, 3] = new_translation
        used += 1
        if len(rows) < 32:
            rows.append(
                {
                    "frame": int(frame_idx),
                    "ego_frame": int(ego_indices[frame_idx]),
                    "ego_bbox": [float(x) for x in ego_box],
                    "projected_bbox_before": [float(x) for x in proj_box],
                    "projected_bbox_after_z": [float(x) for x in proj_box_z],
                    "z_factor": z_factor,
                    "delta_translation": [
                        float(new_translation[0] - translation[0]),
                        float(new_translation[1] - translation[1]),
                        float(new_translation[2] - translation[2]),
                    ],
                }
            )
    return refined, {
        "available": used > 0,
        "method": "ego_mask_bbox_center_depth_refine",
        "frames_used": int(used),
        "sample_rows": rows,
    }


def apply_rigid_transform(points: np.ndarray, transform: np.ndarray | None) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if transform is None:
        return pts
    rotation = np.asarray(transform[:3, :3], dtype=np.float32)
    translation = np.asarray(transform[:3, 3], dtype=np.float32).reshape(1, 3)
    return (rotation @ pts.T).T + translation


MANO_GLOBAL_TRANSLATION_SUFFIXES = (
    "trans",
    "translation",
    "transl",
    "global_translation",
)


def _strict_mano_xyz_field(
    value: np.ndarray,
    *,
    field: str,
    frame_count: int,
) -> tuple[np.ndarray, np.dtype]:
    raw = np.asarray(value)
    if raw.shape != (frame_count, 3):
        raise ValueError(
            f"{field} must have shape ({frame_count}, 3), got {raw.shape}"
        )
    if not np.issubdtype(raw.dtype, np.floating):
        raise ValueError(f"{field} must use a floating dtype, got {raw.dtype}")
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains non-finite values")
    return array, raw.dtype


def _mano_rotvecs_to_matrices(rotvecs: np.ndarray, *, field: str) -> np.ndarray:
    vectors = np.asarray(rotvecs, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or not np.isfinite(vectors).all():
        raise ValueError(f"{field} must contain finite axis-angle vectors")
    matrices = np.empty((vectors.shape[0], 3, 3), dtype=np.float64)
    for frame, vector in enumerate(vectors):
        matrix, _ = cv2.Rodrigues(vector.reshape(3, 1))
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"{field} produced an invalid rotation at frame {frame}")
        matrices[frame] = matrix
    return matrices


def _mano_matrices_to_rotvecs(matrices: np.ndarray, *, field: str) -> np.ndarray:
    rotations = np.asarray(matrices, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(f"{field} rotation matrices have invalid shape {rotations.shape}")
    vectors = np.empty((rotations.shape[0], 3), dtype=np.float64)
    for frame, matrix in enumerate(rotations):
        vector, _ = cv2.Rodrigues(matrix)
        vector = np.asarray(vector, dtype=np.float64).reshape(3)
        if not np.isfinite(vector).all():
            raise ValueError(f"{field} produced a non-finite rotvec at frame {frame}")
        vectors[frame] = vector
    return vectors


def transform_mano_global_pose_fields(
    data: dict,
    canonical_transform: np.ndarray | None,
    *,
    frame_count: int,
) -> dict:
    """Apply one world-frame SE(3) to MANO global orient/translation fields.

    MANO articulation (``*_hand_pose``) and shape (``*_betas``) are local and
    intentionally unchanged.  ``*_rot`` is axis-angle global orientation, so
    the common rotation is left-multiplied before converting back to rotvec.
    """

    applied = canonical_transform is not None
    transform = (
        validate_rigid_transform(canonical_transform)
        if applied
        else np.eye(4, dtype=np.float64)
    )
    common_rotation = transform[:3, :3]
    common_translation = transform[:3, 3]
    report = {
        "status": "ok",
        "applied": bool(applied),
        "frame_count": int(frame_count),
        "rotation_expected_relation": "R_after = R_common @ R_before",
        "translation_expected_relation": "t_after = R_common @ t_before + t_common",
        "common_rotation": common_rotation.tolist(),
        "common_translation": common_translation.tolist(),
        "transformed_fields": [],
        "sides": {},
    }
    for side in ("left", "right"):
        geometry_present = any(
            f"{side}_{suffix}" in data for suffix in ("joints", "vertices")
        )
        rotation_key = f"{side}_rot"
        translation_keys = [
            f"{side}_{suffix}"
            for suffix in MANO_GLOBAL_TRANSLATION_SUFFIXES
            if f"{side}_{suffix}" in data
        ]
        if applied and geometry_present and rotation_key not in data:
            raise ValueError(
                f"shared canonical SE(3) requires MANO global orientation {rotation_key}"
            )
        if applied and geometry_present and not translation_keys:
            raise ValueError(
                f"shared canonical SE(3) requires a global translation field for {side}"
            )
        side_report = {
            "geometry_present": bool(geometry_present),
            "rotation_field": rotation_key if rotation_key in data else None,
            "translation_fields": translation_keys,
            "rotation_frame_count": 0,
            "rotation_matrix_max_abs_error": None,
            "translation_max_abs_error_m": None,
        }
        if rotation_key in data:
            before_rotvecs, rotation_dtype = _strict_mano_xyz_field(
                data[rotation_key], field=rotation_key, frame_count=frame_count
            )
            before_rotations = _mano_rotvecs_to_matrices(
                before_rotvecs, field=rotation_key
            )
            expected_rotations = np.einsum(
                "ij,fjk->fik", common_rotation, before_rotations
            )
            after_rotvecs = (
                _mano_matrices_to_rotvecs(expected_rotations, field=rotation_key)
                if applied
                else before_rotvecs.copy()
            )
            stored_rotvecs = after_rotvecs.astype(rotation_dtype, copy=False)
            after_rotations = _mano_rotvecs_to_matrices(
                stored_rotvecs, field=rotation_key
            )
            rotation_error = np.abs(after_rotations - expected_rotations)
            max_rotation_error = float(np.max(rotation_error, initial=0.0))
            if max_rotation_error > 2e-6:
                raise ValueError(
                    f"{rotation_key} common-rotation verification failed: {max_rotation_error:.3e}"
                )
            if applied:
                data[rotation_key] = stored_rotvecs
                report["transformed_fields"].append(rotation_key)
            side_report["rotation_frame_count"] = int(before_rotvecs.shape[0])
            side_report["rotation_matrix_max_abs_error"] = max_rotation_error

        translation_errors = []
        for translation_key in translation_keys:
            before_translation, translation_dtype = _strict_mano_xyz_field(
                data[translation_key],
                field=translation_key,
                frame_count=frame_count,
            )
            expected_translation = np.einsum(
                "ij,fj->fi", common_rotation, before_translation
            ) + common_translation
            after_translation = expected_translation if applied else before_translation.copy()
            stored_translation = after_translation.astype(translation_dtype, copy=False)
            max_translation_error = float(
                np.max(np.abs(stored_translation - expected_translation), initial=0.0)
            )
            if max_translation_error > 1e-6:
                raise ValueError(
                    f"{translation_key} common-translation verification failed: "
                    f"{max_translation_error:.3e}"
                )
            translation_errors.append(max_translation_error)
            if applied:
                data[translation_key] = stored_translation
                report["transformed_fields"].append(translation_key)
        if translation_errors:
            side_report["translation_max_abs_error_m"] = float(
                max(translation_errors)
            )
        report["sides"][side] = side_report
    return report


def replace_hand_geometry_with_ego(result: dict, hand_npz: Path, alignment_transform: np.ndarray | None = None) -> dict:
    frames = result.get("frame_data") or []
    if not frames:
        return {"available": False, "reason": "missing_frame_data"}
    data = dict(np.load(hand_npz, allow_pickle=True))
    if "left_vertices" not in data or "right_vertices" not in data:
        return {"available": False, "reason": "missing_base_hand_vertices"}
    target_len = int(data["left_vertices"].shape[0])
    if target_len <= 0:
        return {"available": False, "reason": "empty_base_hand_vertices"}

    left_vertices = np.zeros_like(data["left_vertices"], dtype=np.float32)
    right_vertices = np.zeros_like(data["right_vertices"], dtype=np.float32)
    left_joints = np.zeros_like(data["left_joints"], dtype=np.float32)
    right_joints = np.zeros_like(data["right_joints"], dtype=np.float32)
    left_joints_2d = np.zeros((target_len, data["left_joints"].shape[1], 2), dtype=np.float32)
    right_joints_2d = np.zeros((target_len, data["right_joints"].shape[1], 2), dtype=np.float32)
    left_valid = np.zeros(target_len, dtype=np.float32)
    right_valid = np.zeros(target_len, dtype=np.float32)
    ego_indices = np.rint(np.linspace(0, len(frames) - 1, target_len)).astype(int)
    counts = {"left": 0, "right": 0}
    fit_x = []
    fit_y = []
    fit_u = []
    fit_v = []
    cx = result.get("cx")
    cy = result.get("cy")
    for out_idx, ego_idx in enumerate(ego_indices):
        frame = frames[int(ego_idx)]
        vertices_all = frame.get("vertices_3d") or []
        joints_all = frame.get("joints_3d_pred") or []
        joints_2d_all = frame.get("joints_2d_pred") or []
        hands_right = frame.get("hand_is_right") or []
        for hand_idx, vertices in enumerate(vertices_all):
            if hand_idx >= len(joints_all):
                continue
            vertices_arr = np.asarray(vertices, dtype=np.float32)
            joints_arr = np.asarray(joints_all[hand_idx], dtype=np.float32)
            if vertices_arr.shape != left_vertices[out_idx].shape or joints_arr.shape != left_joints[out_idx].shape:
                continue
            vertices_arr = apply_rigid_transform(vertices_arr, alignment_transform)
            joints_arr = apply_rigid_transform(joints_arr, alignment_transform)
            is_right = bool(hands_right[hand_idx]) if hand_idx < len(hands_right) else False
            if is_right:
                right_vertices[out_idx] = vertices_arr
                right_joints[out_idx] = joints_arr
                right_valid[out_idx] = 1.0
                if hand_idx < len(joints_2d_all):
                    right_joints_2d[out_idx] = np.asarray(joints_2d_all[hand_idx], dtype=np.float32)
                counts["right"] += 1
            else:
                left_vertices[out_idx] = vertices_arr
                left_joints[out_idx] = joints_arr
                left_valid[out_idx] = 1.0
                if hand_idx < len(joints_2d_all):
                    left_joints_2d[out_idx] = np.asarray(joints_2d_all[hand_idx], dtype=np.float32)
                counts["left"] += 1
            if alignment_transform is None and cx is not None and cy is not None and hand_idx < len(joints_2d_all):
                uv_arr = np.asarray(joints_2d_all[hand_idx], dtype=np.float32)
                z = joints_arr[:, 2]
                valid = (
                    np.isfinite(joints_arr).all(axis=1)
                    & np.isfinite(uv_arr).all(axis=1)
                    & (np.abs(joints_arr[:, 0]) > 1e-5)
                    & (np.abs(joints_arr[:, 1]) > 1e-5)
                    & (z > 1e-4)
                )
                if valid.any():
                    fit_x.extend((joints_arr[valid, 0] / z[valid]).tolist())
                    fit_y.extend((joints_arr[valid, 1] / z[valid]).tolist())
                    fit_u.extend((uv_arr[valid, 0] - float(cx)).tolist())
                    fit_v.extend((uv_arr[valid, 1] - float(cy)).tolist())
    if counts["left"] == 0 and counts["right"] == 0:
        return {"available": False, "reason": "no_ego_hand_geometry"}
    data["left_vertices"] = left_vertices
    data["right_vertices"] = right_vertices
    data["left_joints"] = left_joints
    data["right_joints"] = right_joints
    data["left_joints_2d"] = left_joints_2d
    data["right_joints_2d"] = right_joints_2d
    data["left_valid"] = left_valid
    data["right_valid"] = right_valid
    data["hand_geometry_source"] = "EgoInfinity frame_data vertices_3d/joints_3d_pred"
    mano_global_pose_transform = transform_mano_global_pose_fields(
        data,
        alignment_transform,
        frame_count=target_len,
    )
    projection_qc = {"available": False}
    if fit_x and cx is not None and cy is not None and alignment_transform is None:
        fit_x_arr = np.asarray(fit_x, dtype=np.float64)
        fit_y_arr = np.asarray(fit_y, dtype=np.float64)
        fit_u_arr = np.asarray(fit_u, dtype=np.float64)
        fit_v_arr = np.asarray(fit_v, dtype=np.float64)
        fx = float(np.sum(fit_x_arr * fit_u_arr) / max(np.sum(fit_x_arr * fit_x_arr), 1e-9))
        fy = float(np.sum(fit_y_arr * fit_v_arr) / max(np.sum(fit_y_arr * fit_y_arr), 1e-9))
        source_width = max(1.0, float(cx) * 2.0)
        source_height = max(1.0, float(cy) * 2.0)
        data["ego_hand_projection_K"] = np.asarray(
            [[fx, 0.0, float(cx)], [0.0, fy, float(cy)], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        data["ego_hand_projection_size"] = np.asarray([source_width, source_height], dtype=np.float32)
        data["ego_hand_projection_source"] = "least_squares_fit_joints_2d_to_joints_3d"
        projection_qc = {
            "available": True,
            "source": "least_squares_fit_joints_2d_to_joints_3d",
            "fx": fx,
            "fy": fy,
            "cx": float(cx),
            "cy": float(cy),
            "source_width": source_width,
            "source_height": source_height,
            "points": int(len(fit_x)),
        }
    np.savez(hand_npz, **data)
    return {
        "available": True,
        "source": "EgoInfinity frame_data",
        "alignment_transform_applied": None if alignment_transform is None else np.asarray(alignment_transform, dtype=np.float64).tolist(),
        "ego_frames": int(len(frames)),
        "target_frames": target_len,
        "valid_counts": counts,
        "mano_global_pose_transform": mano_global_pose_transform,
        "projection": projection_qc,
        "output": str(hand_npz),
    }


def bbox_from_points(uv: np.ndarray, valid: np.ndarray, width: int, height: int) -> tuple[float, float, float, float] | None:
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if not inside.any():
        return None
    pts = uv[inside]
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def bbox_area(box: tuple[float, float, float, float] | None) -> float:
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox_overlap(a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None) -> float:
    if a is None or b is None:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_center(box: tuple[float, float, float, float]) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float32)


def read_object_mask(masks_dir: Path, object_id: str, frame_id: int) -> np.ndarray | None:
    frame_dir = masks_dir / f"frame_{frame_id:06d}_masks"
    for path in [frame_dir / f"{object_id}.png", frame_dir / object_id / f"{object_id}.png"]:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return mask
    return None


def unpack_ego_mask(result: dict, obj_id: int, ego_frame_idx: int) -> np.ndarray | None:
    frames = result.get("frame_data") or []
    if ego_frame_idx < 0 or ego_frame_idx >= len(frames):
        return None
    obj = oid_get(frames[ego_frame_idx].get("sam3_obj_data") or {}, obj_id)
    if not isinstance(obj, dict):
        return None
    packed = obj.get("mask_packed")
    shape = obj.get("mask_shape")
    if packed is None or shape is None:
        return None
    h, w = [int(x) for x in np.asarray(shape).reshape(-1)[:2]]
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))[: h * w]
    if bits.size != h * w:
        return None
    mask = bits.reshape(h, w).astype(np.uint8) * 255
    return mask


def hand_mask_union(mask_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    union = np.zeros((h, w), dtype=bool)
    for name in ("left_hand_0.png", "left_hand.png", "right_hand_0.png", "right_hand.png"):
        mask_path = mask_dir / name
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        union |= mask > 0
    return union


def write_ego_object_masks(
    result: dict,
    ego_obj_id: int,
    output_dir: Path,
    object_id: str,
    frame_count: int,
) -> dict:
    frames = result.get("frame_data") or []
    if not frames or frame_count <= 0:
        return {"available": False, "reason": "missing_ego_frame_data"}
    available_ego_indices = [
        idx for idx in range(len(frames))
        if unpack_ego_mask(result, int(ego_obj_id), idx) is not None
    ]
    if not available_ego_indices:
        return {"available": False, "reason": "missing_ego_object_masks"}
    ego_indices = np.rint(np.linspace(0, len(frames) - 1, frame_count)).astype(int)
    mask_root = output_dir / "video_segmentation" / "masks"
    frame_dir = output_dir / "all_frames"
    written = 0
    missing = 0
    subtracted_pixels = 0
    object_pixels_before = 0
    object_pixels_after = 0
    for out_idx, ego_idx in enumerate(ego_indices):
        rgb = cv2.imread(str(frame_dir / f"{out_idx:06d}.png"), cv2.IMREAD_COLOR)
        if rgb is None:
            missing += 1
            continue
        h, w = rgb.shape[:2]
        mask = unpack_ego_mask(result, int(ego_obj_id), int(ego_idx))
        if mask is None:
            nearest = min(available_ego_indices, key=lambda idx: abs(idx - int(ego_idx)))
            mask = unpack_ego_mask(result, int(ego_obj_id), int(nearest))
        if mask is None:
            missing += 1
            continue
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask > 0
        before = int(mask_bool.sum())
        out_mask_dir = mask_root / f"frame_{out_idx:06d}_masks"
        out_mask_dir.mkdir(parents=True, exist_ok=True)
        hands = hand_mask_union(out_mask_dir, (h, w))
        if hands.any():
            overlap = mask_bool & hands
            subtracted_pixels += int(overlap.sum())
            mask_bool &= ~hands
        after = int(mask_bool.sum())
        object_pixels_before += before
        object_pixels_after += after
        out_mask = (mask_bool.astype(np.uint8) * 255)
        cv2.imwrite(str(out_mask_dir / f"{object_id}.png"), out_mask)
        object_subdir = out_mask_dir / object_id
        if object_subdir.is_dir():
            cv2.imwrite(str(object_subdir / f"{object_id}.png"), out_mask)
        written += 1
    return {
        "available": written > 0,
        "source": "egoinfinity_selected_object_mask_minus_hand_masks",
        "written": written,
        "missing": missing,
        "subtracted_pixels": subtracted_pixels,
        "object_pixels_before": object_pixels_before,
        "object_pixels_after": object_pixels_after,
    }


def infer_visual_interaction_hand(hand_npz: Path, frames_dir: Path, masks_dir: Path, object_id: str) -> dict:
    try:
        data = np.load(hand_npz, allow_pickle=False)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    frame_count = 0
    for side in ("left", "right"):
        key = f"{side}_vertices"
        if key in data:
            frame_count = max(frame_count, int(data[key].shape[0]))
    if frame_count <= 0:
        return {"available": False, "error": "no hand vertices"}
    sample_ids = list(range(frame_count))
    if len(sample_ids) > 32:
        sample_ids = np.linspace(0, frame_count - 1, 32, dtype=int).tolist()
    scores: dict[str, dict[str, float]] = {
        "left": {"score": 0.0, "frames": 0.0, "overlap_frames": 0.0},
        "right": {"score": 0.0, "frames": 0.0, "overlap_frames": 0.0},
    }
    for frame_id in sample_ids:
        image = cv2.imread(str(frames_dir / f"{frame_id:06d}.png"), cv2.IMREAD_COLOR)
        intrinsics_path = frames_dir / f"{frame_id:06d}_intrinsics.npy"
        if image is None or not intrinsics_path.exists():
            continue
        mask = read_object_mask(masks_dir, object_id, frame_id)
        obj_box = bbox_from_mask(mask)
        if obj_box is None:
            continue
        height, width = image.shape[:2]
        image_diag = max(float(np.hypot(width, height)), 1.0)
        object_area = max(bbox_area(obj_box), 1.0)
        object_center = bbox_center(obj_box)
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
            overlap = bbox_overlap(hand_box, obj_box)
            distance = float(np.linalg.norm(bbox_center(hand_box) - object_center))
            proximity = max(0.0, 1.0 - distance / (0.35 * image_diag))
            scores[side]["score"] += 4.0 * overlap / object_area + proximity
            scores[side]["frames"] += 1.0
            if overlap > 0:
                scores[side]["overlap_frames"] += 1.0
    normalized = {
        side: values["score"] / max(values["frames"], 1.0)
        for side, values in scores.items()
    }
    left = normalized["left"]
    right = normalized["right"]
    if max(left, right) <= 0:
        selected = None
    elif left >= right * 1.25:
        selected = "left"
    elif right >= left * 1.25:
        selected = "right"
    else:
        selected = "bimanual"
    return {
        "available": True,
        "scores": scores,
        "normalized": normalized,
        "selected_hand": selected,
    }


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def layout_object_entries(layout: dict) -> list[dict]:
    objects = layout.get("objects") or []
    if isinstance(objects, dict):
        objects = list(objects.values())
    entries = [entry for entry in objects if isinstance(entry, dict)]
    entries.sort(key=lambda item: int(item.get("index", item.get("frame_idx", 0)) or 0))
    return entries


def mesh_path_for_layout_entry(raw_dir: Path, object_id: str, entry: dict) -> Path | None:
    frame_id = int(entry.get("frame_idx", entry.get("index", 0)) or 0)
    candidates = [
        raw_dir / "video_segmentation" / "masks" / f"frame_{frame_id:06d}_masks" / object_id / f"{object_id}.obj",
        raw_dir / "video_segmentation" / "masks" / f"frame_{frame_id:06d}_masks" / f"{object_id}.obj",
    ]
    mesh_name = entry.get("mesh_obj")
    if mesh_name:
        candidates.append(raw_dir / "obj_tracking_out" / object_id / "combined_visualization" / str(mesh_name))
    for path in candidates:
        if path.exists():
            return path
    fallback_root = raw_dir / "video_segmentation" / "masks"
    fallback = sorted(fallback_root.glob(f"frame_*_masks/{object_id}/{object_id}.obj"))
    if fallback:
        return fallback[0]
    fallback = sorted(fallback_root.glob(f"frame_*_masks/{object_id}.obj"))
    if fallback:
        return fallback[0]
    return None


def transformed_object_vertices(mesh_path: Path, entry: dict) -> np.ndarray | None:
    try:
        mesh = trimesh.load(mesh_path, process=False)
    except Exception:
        return None
    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        return None
    local = entry.get("local_to_scene") or {}
    scale = np.asarray(local.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(-1)
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    scale = scale[:3]
    translation = np.asarray(
        local.get("translation_camera_frame") or local.get("translation") or [0.0, 0.0, 0.0],
        dtype=np.float64,
    ).reshape(3)
    quat = np.asarray(
        local.get("quat_wxyz_camera_frame") or local.get("new_quat") or local.get("quat_wxyz") or [1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    ).reshape(4)
    rot = quat_wxyz_to_matrix(quat)
    return (vertices * scale.reshape(1, 3)) @ rot.T + translation.reshape(1, 3)


def hand_index_by_frame_id(data: np.lib.npyio.NpzFile, frame_count: int) -> dict[int, int]:
    source_frame_ids = data["source_frame_ids"] if "source_frame_ids" in data else np.arange(frame_count)
    mapping = {}
    for idx, frame_id in enumerate(np.asarray(source_frame_ids).reshape(-1)):
        if idx >= frame_count:
            break
        mapping[int(frame_id)] = idx
    for idx in range(frame_count):
        mapping.setdefault(idx, idx)
    return mapping


def subsample_vertices(vertices: np.ndarray, max_points: int) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if len(vertices) <= max_points:
        return vertices
    idx = np.linspace(0, len(vertices) - 1, max_points, dtype=np.int64)
    return vertices[idx]


def min_distance_between(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    best = float("inf")
    chunk = 256
    for start in range(0, len(a), chunk):
        diff = a[start : start + chunk, None, :] - b[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        best = min(best, float(np.sqrt(np.min(dist2))))
    return best


def nearest_offset_between(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray | None, float]:
    if len(a) == 0 or len(b) == 0:
        return None, float("inf")
    best = float("inf")
    best_offset = None
    chunk = 256
    for start in range(0, len(a), chunk):
        a_chunk = a[start : start + chunk]
        diff = a_chunk[:, None, :] - b[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        flat = int(np.argmin(dist2))
        local_i, j = np.unravel_index(flat, dist2.shape)
        dist = float(np.sqrt(dist2[local_i, j]))
        if dist < best:
            best = dist
            best_offset = a_chunk[local_i] - b[j]
    return best_offset, best


def align_object_to_interacting_hand(
    transforms: np.ndarray,
    hand_npz: Path,
    mesh_path: Path,
    mesh_scale: float,
    selected_hand: str,
    visual_interaction: dict,
    mode: str,
    min_bad_mean_distance: float,
    target_surface_distance: float,
    max_offset: float,
) -> tuple[np.ndarray, dict]:
    """Measure a possible contact offset without changing production poses.

    Historical versions applied the measured translation only to the object.
    That destroys the source hand-object relative transform and contaminated
    both DAI and SPIDER inputs.  Non-``none`` modes are now diagnostic aliases:
    they report the hypothetical translation but always return the original
    transforms byte-for-byte.
    """
    if mode == "none":
        return transforms, {
            "applied": False,
            "diagnostic_only": True,
            "transforms_modified": False,
            "reason": "disabled",
            "mode": mode,
            "policy": "object_only_hoi_translation_forbidden",
        }
    if selected_hand not in {"left", "right", "bimanual"}:
        return transforms, {"applied": False, "reason": "selected_hand_invalid", "selected_hand": selected_hand}
    if not visual_interaction.get("available"):
        return transforms, {"applied": False, "reason": "missing_visual_interaction", "selected_hand": selected_hand}
    scores = visual_interaction.get("scores") or {}
    selected_sides = ["left", "right"] if selected_hand == "bimanual" else [selected_hand]
    overlap_by_hand = {
        side: float((scores.get(side) or {}).get("overlap_frames", 0.0) or 0.0)
        for side in selected_sides
    }
    overlap_frames = min(overlap_by_hand.values()) if overlap_by_hand else 0.0
    if mode in {"auto", "diagnostic"} and overlap_frames < 3:
        return transforms, {
            "applied": False,
            "reason": "weak_visual_overlap",
            "selected_hand": selected_hand,
            "overlap_frames": overlap_frames,
            "overlap_by_hand": overlap_by_hand,
        }
    try:
        data = np.load(hand_npz, allow_pickle=False)
        mesh_vertices = load_mesh_vertices(mesh_path) * float(mesh_scale)
    except Exception as exc:
        return transforms, {"applied": False, "reason": f"load_failed: {exc}", "selected_hand": selected_hand}
    hand_vertices: dict[str, np.ndarray] = {}
    hand_valid: dict[str, np.ndarray] = {}
    for side in selected_sides:
        vertices_key = f"{side}_vertices"
        valid_key = f"{side}_valid"
        if vertices_key not in data:
            return transforms, {
                "applied": False,
                "reason": "missing_selected_hand_vertices",
                "selected_hand": selected_hand,
                "missing_side": side,
            }
        vertices = np.asarray(data[vertices_key], dtype=np.float64)
        hand_vertices[side] = vertices
        hand_valid[side] = (
            np.asarray(data[valid_key], dtype=np.float64) > 0
            if valid_key in data
            else np.ones((vertices.shape[0],), dtype=bool)
        )
    frame_count = min([len(transforms), *(vertices.shape[0] for vertices in hand_vertices.values())])
    if frame_count <= 0:
        return transforms, {"applied": False, "reason": "empty_inputs", "selected_hand": selected_hand}
    sample_ids = np.arange(frame_count)
    if len(sample_ids) > 40:
        sample_ids = np.linspace(0, frame_count - 1, 40, dtype=int)
    offsets = []
    distances = []
    for frame_idx in sample_ids:
        frame_hand_points = [
            hand_vertices[side][int(frame_idx)]
            for side in selected_sides
            if bool(hand_valid[side][int(frame_idx)])
        ]
        if len(frame_hand_points) != len(selected_sides):
            continue
        transform = np.asarray(transforms[int(frame_idx)], dtype=np.float64)
        obj_points = (transform[:3, :3] @ mesh_vertices.T).T + transform[:3, 3]
        obj_points = subsample_vertices(obj_points, 768)
        hand_points = subsample_vertices(np.concatenate(frame_hand_points, axis=0), 1024)
        offset, dist = nearest_offset_between(hand_points, obj_points)
        if offset is None or not np.isfinite(dist):
            continue
        offsets.append(offset)
        distances.append(dist)
    if not offsets:
        return transforms, {"applied": False, "reason": "no_valid_nearest_offsets", "selected_hand": selected_hand}
    distances_arr = np.asarray(distances, dtype=np.float64)
    before = {
        "min": float(np.min(distances_arr)),
        "median": float(np.median(distances_arr)),
        "mean": float(np.mean(distances_arr)),
    }
    if mode in {"auto", "diagnostic"} and before["mean"] < float(min_bad_mean_distance):
        return transforms, {
            "applied": False,
            "reason": "geometry_already_close",
            "selected_hand": selected_hand,
            "before": before,
            "threshold": float(min_bad_mean_distance),
        }
    offset_vec = np.median(np.stack(offsets, axis=0), axis=0)
    norm = float(np.linalg.norm(offset_vec))
    if not np.isfinite(norm) or norm <= 1e-8:
        return transforms, {"applied": False, "reason": "degenerate_offset", "selected_hand": selected_hand, "before": before}
    if norm > float(max_offset):
        return transforms, {
            "applied": False,
            "reason": "offset_exceeds_max",
            "selected_hand": selected_hand,
            "overlap_frames": overlap_frames,
            "frames_used": int(len(offsets)),
            "offset": [float(x) for x in offset_vec.reshape(3)],
            "offset_norm": norm,
            "max_offset": float(max_offset),
            "before": before,
        }
    clipped = False
    if target_surface_distance > 0.0 and norm > target_surface_distance:
        offset_vec = offset_vec * max(0.0, (norm - float(target_surface_distance)) / norm)
    hypothetical = np.asarray(transforms, dtype=np.float64).copy()
    hypothetical[:, :3, 3] += offset_vec.reshape(3)
    after_distances = []
    for frame_idx in sample_ids:
        frame_hand_points = [
            hand_vertices[side][int(frame_idx)]
            for side in selected_sides
            if bool(hand_valid[side][int(frame_idx)])
        ]
        if len(frame_hand_points) != len(selected_sides):
            continue
        transform = np.asarray(hypothetical[int(frame_idx)], dtype=np.float64)
        obj_points = (transform[:3, :3] @ mesh_vertices.T).T + transform[:3, 3]
        obj_points = subsample_vertices(obj_points, 768)
        hand_points = subsample_vertices(np.concatenate(frame_hand_points, axis=0), 1024)
        after_distances.append(min_distance_between(hand_points, obj_points))
    after_arr = np.asarray(after_distances, dtype=np.float64)
    return transforms, {
        "applied": False,
        "would_apply": True,
        "diagnostic_only": True,
        "transforms_modified": False,
        "reason": "object_only_translation_forbidden",
        "policy": "preserve_source_hoi_relative_transform",
        "mode": mode,
        "method": "diagnostic_median_nearest_surface_translation",
        "selected_hand": selected_hand,
        "overlap_frames": overlap_frames,
        "overlap_by_hand": overlap_by_hand,
        "frames_used": int(len(offsets)),
        "proposed_offset": [float(x) for x in offset_vec.reshape(3)],
        "proposed_offset_norm": float(np.linalg.norm(offset_vec)),
        # Compatibility aliases. They describe a proposal, never an applied
        # production transform; applied/transforms_modified remain false.
        "offset": [float(x) for x in offset_vec.reshape(3)],
        "offset_norm": float(np.linalg.norm(offset_vec)),
        "offset_clipped": clipped,
        "before": before,
        "hypothetical_after": {
            "min": float(np.min(after_arr)) if after_arr.size else None,
            "median": float(np.median(after_arr)) if after_arr.size else None,
            "mean": float(np.mean(after_arr)) if after_arr.size else None,
        },
        "target_surface_distance": float(target_surface_distance),
    }


def infer_geometry_interaction_hand(
    hand_npz: Path,
    layout_path: Path,
    raw_dir: Path,
    object_id: str,
    distance_thresh: float = 0.10,
    max_sample_frames: int | None = 32,
) -> dict:
    try:
        data = np.load(hand_npz, allow_pickle=False)
        layout = load_json(layout_path)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    entries = layout_object_entries(layout)
    if not entries:
        return {"available": False, "error": "layout has no object entries"}
    frame_count = 0
    for side in ("left", "right"):
        key = f"{side}_vertices"
        if key in data:
            frame_count = max(frame_count, int(data[key].shape[0]))
    if frame_count <= 0:
        return {"available": False, "error": "no hand vertices"}
    sample_ids = list(range(len(entries)))
    if max_sample_frames is not None and len(sample_ids) > max_sample_frames:
        sample_ids = np.linspace(
            0,
            len(entries) - 1,
            max_sample_frames,
            dtype=int,
        ).tolist()
    row_by_frame = hand_index_by_frame_id(data, frame_count)
    stats: dict[str, dict[str, float]] = {
        "left": {"frames": 0.0, "in_hand_frames": 0.0, "score": 0.0, "min_distance_sum": 0.0},
        "right": {"frames": 0.0, "in_hand_frames": 0.0, "score": 0.0, "min_distance_sum": 0.0},
    }
    failure_reasons: dict[str, Counter] = {
        "left": Counter(),
        "right": Counter(),
    }

    def record_failure(reason: str, sides: tuple[str, ...] = ("left", "right")) -> None:
        for side in sides:
            failure_reasons[side][reason] += 1

    for entry_idx in sample_ids:
        entry = entries[entry_idx]
        frame_id = int(entry.get("frame_idx", entry.get("index", entry_idx)) or entry_idx)
        hand_idx = row_by_frame.get(frame_id)
        if hand_idx is None:
            record_failure("missing_hand_frame")
            continue
        mesh_path = mesh_path_for_layout_entry(raw_dir, object_id, entry)
        if mesh_path is None:
            record_failure("missing_object_mesh")
            continue
        object_vertices = transformed_object_vertices(mesh_path, entry)
        if object_vertices is None or not np.isfinite(object_vertices).all():
            record_failure("invalid_object_mesh_or_pose")
            continue
        object_points = subsample_vertices(object_vertices, 768)
        for side in ("left", "right"):
            vertices_key = f"{side}_vertices"
            valid_key = f"{side}_valid"
            if vertices_key not in data or hand_idx >= data[vertices_key].shape[0]:
                record_failure("missing_hand_vertices", (side,))
                continue
            if valid_key in data:
                if hand_idx >= np.asarray(data[valid_key]).reshape(-1).shape[0]:
                    record_failure("missing_hand_validity", (side,))
                    continue
                validity = float(np.asarray(data[valid_key]).reshape(-1)[hand_idx])
                if not np.isfinite(validity) or validity <= 0:
                    record_failure("invalid_hand_frame", (side,))
                    continue
            try:
                hand_points = subsample_vertices(data[vertices_key][hand_idx], 768)
            except (TypeError, ValueError):
                record_failure("invalid_hand_vertices", (side,))
                continue
            if not np.isfinite(hand_points).all():
                record_failure("invalid_hand_vertices", (side,))
                continue
            dist = min_distance_between(hand_points, object_points)
            if not np.isfinite(dist):
                record_failure("nonfinite_vertex_distance", (side,))
                continue
            stats[side]["frames"] += 1.0
            stats[side]["min_distance_sum"] += dist
            stats[side]["score"] += max(0.0, 1.0 - dist / max(distance_thresh, 1e-6))
            if dist <= distance_thresh:
                stats[side]["in_hand_frames"] += 1.0
    fractions = {
        side: values["in_hand_frames"] / max(values["frames"], 1.0)
        for side, values in stats.items()
    }
    mean_min_distance = {
        side: values["min_distance_sum"] / max(values["frames"], 1.0)
        for side, values in stats.items()
    }
    selected = contact_fraction_hand(fractions)
    if selected == "bimanual":
        dominant = dominant_hand_from_geometry(fractions, mean_min_distance, stats)
        if dominant is not None:
            selected = dominant
    if selected is None:
        normalized = {
            side: values["score"] / max(values["frames"], 1.0)
            for side, values in stats.items()
        }
        selected = "left" if normalized["left"] >= normalized["right"] else "right"
        if max(normalized.values()) <= 0:
            selected = None
    planned_frame_count = int(len(sample_ids))
    source_frame_count = int(len(entries))
    coverage_by_hand = {}
    for side in ("left", "right"):
        evaluated_frame_count = int(round(stats[side]["frames"]))
        failed_frame_count = int(sum(failure_reasons[side].values()))
        coverage_by_hand[side] = {
            "source_frame_count": source_frame_count,
            "planned_frame_count": planned_frame_count,
            "evaluated_frame_count": evaluated_frame_count,
            "failed_frame_count": failed_frame_count,
            "complete_source_coverage": bool(
                planned_frame_count == source_frame_count
                and evaluated_frame_count == source_frame_count
                and failed_frame_count == 0
            ),
            "failure_reasons": {
                key: int(value) for key, value in sorted(failure_reasons[side].items())
            },
        }
    return {
        "available": True,
        "distance_thresh": distance_thresh,
        "source_frame_count": source_frame_count,
        "sampled_frame_count": planned_frame_count,
        "sampled_all_frames": planned_frame_count == source_frame_count,
        "distance_method": "subsampled_mesh_vertex_to_vertex_minimum_conservative",
        "coverage_by_hand": coverage_by_hand,
        "scores": stats,
        "contact_fraction": fractions,
        "mean_min_distance": mean_min_distance,
        "selected_hand": selected,
    }


def merge_selected_hand(interaction: dict, visual: dict) -> str:
    ego_selected = interaction.get("selected_hand") or "left"
    visual_selected = visual.get("selected_hand") if visual.get("available") else None
    if visual_selected in {"left", "right"}:
        normalized = visual.get("normalized") or {}
        best = float(normalized.get(visual_selected, 0.0) or 0.0)
        other = "right" if visual_selected == "left" else "left"
        other_score = float(normalized.get(other, 0.0) or 0.0)
        if best > 0 and (best >= other_score * 1.25 or ego_selected != "bimanual"):
            return visual_selected
    if visual_selected == "bimanual" and ego_selected == "bimanual":
        return "bimanual"
    return ego_selected


def resolve_visual_bimanual_tie(
    selected_hand: str | None,
    visual: dict,
    preferred_hand: str | None,
    min_overlap_frames: float = 3.0,
) -> str | None:
    if selected_hand != "bimanual" or preferred_hand not in {"left", "right"}:
        return selected_hand
    scores = visual.get("scores") or {}
    normalized = visual.get("normalized") or {}
    preferred_overlap = float(
        (scores.get(preferred_hand) or {}).get("overlap_frames", 0.0) or 0.0
    )
    preferred_score = float(normalized.get(preferred_hand, 0.0) or 0.0)
    other = "right" if preferred_hand == "left" else "left"
    other_score = float(normalized.get(other, 0.0) or 0.0)
    if (
        preferred_overlap >= min_overlap_frames
        and preferred_score > 0.0
        and preferred_score >= other_score * 0.8
    ):
        return preferred_hand
    return selected_hand


def resolve_final_selected_hand(
    initial_selected: str,
    interaction: dict,
    visual: dict,
    geometry: dict,
) -> str:
    geometry_selected = geometry.get("selected_hand") if geometry.get("available") else None
    visual_selected = visual.get("selected_hand") if visual.get("available") else None
    geometry_is_strong = False
    if geometry_selected in {"left", "right"}:
        fractions = geometry.get("contact_fraction") or {}
        distances = geometry.get("mean_min_distance") or {}
        scores = geometry.get("scores") or {}
        geometry_is_strong = single_hand_is_strong(geometry_selected, fractions, distances, scores)
        if geometry_is_strong:
            return geometry_selected
    if interaction.get("selected_hand") == "bimanual" and visual_selected in {"left", "right"}:
        normalized = visual.get("normalized") or {}
        best = float(normalized.get(visual_selected, 0.0) or 0.0)
        other = "right" if visual_selected == "left" else "left"
        other_score = float(normalized.get(other, 0.0) or 0.0)
        # In bimanual clips the 3D geometry score can be biased by a noisy
        # object scale/pose.  If the 2D mask evidence overwhelmingly points to
        # one hand, trust that interaction side instead of letting geometry
        # flip to the wrong hand and feed SPIDER/DAI a separated HOI reference.
        if (
            not geometry_is_strong
            and
            best >= max(1.0, other_score * 3.0)
            and (geometry_selected not in {"left", "right"} or geometry_selected != visual_selected)
        ):
            return visual_selected
    if geometry_selected in {"left", "right"}:
        fractions = geometry.get("contact_fraction") or {}
        distances = geometry.get("mean_min_distance") or {}
        scores = geometry.get("scores") or {}
        if single_hand_is_strong(geometry_selected, fractions, distances, scores):
            return geometry_selected
    if geometry_selected == "bimanual":
        fractions = geometry.get("contact_fraction") or {}
        distances = geometry.get("mean_min_distance") or {}
        scores = geometry.get("scores") or {}
        dominant = dominant_hand_from_geometry(fractions, distances, scores)
        if dominant is not None:
            return dominant
        left_fraction = float(fractions.get("left", 0.0) or 0.0)
        right_fraction = float(fractions.get("right", 0.0) or 0.0)
        if min(left_fraction, right_fraction) >= 0.35:
            return "bimanual"

    if visual_selected in {"left", "right"}:
        normalized = visual.get("normalized") or {}
        best = float(normalized.get(visual_selected, 0.0) or 0.0)
        other = "right" if visual_selected == "left" else "left"
        other_score = float(normalized.get(other, 0.0) or 0.0)
        if best > 0 and best >= other_score * 1.20:
            return visual_selected
    if geometry_selected in {"left", "right"}:
        return geometry_selected
    if geometry_selected == "bimanual":
        return "bimanual"
    return initial_selected


def strong_visual_single_hand(visual: dict) -> str | None:
    selected = visual.get("selected_hand") if visual.get("available") else None
    if selected not in {"left", "right"}:
        return None
    normalized = visual.get("normalized") or {}
    best = float(normalized.get(selected, 0.0) or 0.0)
    other = "right" if selected == "left" else "left"
    other_score = float(normalized.get(other, 0.0) or 0.0)
    if best > 0.25 and best >= max(0.25, other_score * 1.5):
        return selected
    return None


def strong_geometry_single_hand(geometry: dict) -> str | None:
    selected = geometry.get("selected_hand") if geometry.get("available") else None
    if selected not in {"left", "right"}:
        return None
    fractions = geometry.get("contact_fraction") or {}
    distances = geometry.get("mean_min_distance") or {}
    scores = geometry.get("scores") or {}
    if single_hand_is_strong(selected, fractions, distances, scores):
        return selected
    return None


def swap_hand_slots_npz(hand_npz: Path) -> dict:
    with np.load(hand_npz, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    swapped_keys = []
    for key in list(arrays.keys()):
        if not key.startswith("left_"):
            continue
        suffix = key[len("left_") :]
        other = f"right_{suffix}"
        if other not in arrays:
            continue
        arrays[key], arrays[other] = arrays[other], arrays[key]
        swapped_keys.extend([key, other])
    arrays["hand_slot_swap_applied"] = np.asarray(True)
    arrays["hand_slot_swap_reason"] = np.asarray(
        "2D visual hand-object evidence disagreed with 3D geometry hand-object evidence",
        dtype="<U96",
    )
    np.savez(hand_npz, **arrays)
    return {
        "applied": True,
        "hand_npz": str(hand_npz),
        "swapped_keys": sorted(swapped_keys),
    }


def maybe_swap_hand_slots_for_visual_geometry(
    hand_npz: Path,
    visual: dict,
    geometry: dict,
) -> dict:
    visual_hand = strong_visual_single_hand(visual)
    geometry_hand = strong_geometry_single_hand(geometry)
    if visual_hand is None or geometry_hand is None:
        return {
            "applied": False,
            "reason": "weak_or_non_single_hand_evidence",
            "visual_hand": visual_hand,
            "geometry_hand": geometry_hand,
        }
    if visual_hand == geometry_hand:
        return {
            "applied": False,
            "reason": "visual_and_geometry_agree",
            "visual_hand": visual_hand,
            "geometry_hand": geometry_hand,
        }
    return {
        "applied": False,
        "would_apply": True,
        "diagnostic_only": True,
        "hand_npz_modified": False,
        "reason": "visual_geometry_hand_slot_disagreement",
        "policy": "production_hand_slot_refinement_forbidden",
        "visual_hand": visual_hand,
        "geometry_hand": geometry_hand,
    }


def pose_sequence_for(result: dict, obj_id, pose_info: dict) -> np.ndarray:
    if isinstance(pose_info, dict) and pose_info.get("T_seq") is not None:
        return np.asarray(pose_info["T_seq"], dtype=np.float64)
    transforms = []
    last = None
    for frame in result.get("frame_data") or []:
        obj = oid_get(frame.get("sam3_obj_data") or {}, obj_id)
        if isinstance(obj, dict) and obj.get("pose_R") is not None and obj.get("pose_t") is not None:
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = np.asarray(obj["pose_R"], dtype=np.float64)
            transform[:3, 3] = np.asarray(obj["pose_t"], dtype=np.float64).reshape(3)
            last = transform
        elif last is not None:
            transform = last.copy()
        else:
            transform = None
        if transform is not None:
            transforms.append(transform)
    if not transforms:
        raise RuntimeError(f"EgoInfinity object {obj_id} has no pose sequence")
    return np.stack(transforms, axis=0)


def preferred_hand_rank(pose: dict, preferred_hand: str | None) -> tuple[int, int, int]:
    if preferred_hand not in {"left", "right"}:
        return (0, 0, 0)
    state_counts = pose.get("state_counts") or {}
    preferred_only = int(state_counts.get(f"grasped_{preferred_hand[0]}", 0) or 0)
    other_hand = "right" if preferred_hand == "left" else "left"
    other_only = int(state_counts.get(f"grasped_{other_hand[0]}", 0) or 0)
    both = int(state_counts.get("grasped_both", 0) or 0)
    preferred_total = preferred_only + both
    return (preferred_total, both, -other_only)


def box_mesh_shape_score(result: dict, pipeline_result: Path, obj_id, prompt: str) -> float:
    if "box" not in str(prompt).lower():
        return 0.0
    path = ego_mesh_path(pipeline_result, int(obj_id))
    try:
        mesh = trimesh.load(path, process=False)
        vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float64)
    except Exception:
        return 0.0
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 16:
        return 0.0
    extents = np.sort(vertices.max(axis=0) - vertices.min(axis=0))
    largest = float(extents[-1])
    middle = float(extents[-2])
    smallest = float(extents[0])
    if largest <= 1e-8:
        return 0.0
    middle_ratio = middle / largest
    smallest_ratio = smallest / largest
    # Blue-box candidates can include a single visible wall of the box. Those
    # thin sheets look good in 2D but give SPIDER a physically tiny object, so
    # prefer volumetric cuboids and demote near-planar meshes.
    volumetric = min(1.0, middle_ratio / 0.65) * min(1.0, smallest_ratio / 0.45)
    sheet_penalty = 0.5 if smallest_ratio < 0.25 else 0.0
    return float(np.clip(volumetric - sheet_penalty, 0.0, 1.0))


def select_object(
    result: dict,
    pipeline_result: Path,
    source_dir: Path,
    source_object_id: str,
    ref_frame: int,
    target_prompt: str | None,
    object_id: int | None,
    max_jump_px: float,
    preferred_hand: str | None = None,
) -> tuple[int, dict, dict]:
    mesh_info = result.get("sam3_mesh_info") or {}
    pose_info_all = result.get("pose_track_info") or {}
    if object_id is not None:
        info = oid_get(mesh_info, object_id)
        pose = oid_get(pose_info_all, object_id)
        if not isinstance(info, dict) or not isinstance(pose, dict):
            raise RuntimeError(f"requested EgoInfinity object id is unavailable: {object_id}")
        selection_qc = source_object_selection_score(
            result,
            pipeline_result,
            source_dir,
            source_object_id,
            ref_frame,
            int(object_id),
            pose,
            info,
        )
        return int(object_id), pose, selection_qc

    rows = []
    for obj_id, info in mesh_info.items():
        pose = oid_get(pose_info_all, obj_id)
        if not isinstance(info, dict) or not isinstance(pose, dict):
            continue
        if pose.get("spurious_flag"):
            continue
        prompt = object_prompt(result, obj_id)
        if target_prompt and prompt != target_prompt:
            continue
        shape_score = box_mesh_shape_score(result, pipeline_result, obj_id, prompt)
        stable_len, max_jump = stable_prefix_len(result, obj_id, max_jump_px)
        hand_rank = preferred_hand_rank(pose, preferred_hand)
        visual_interaction = candidate_visual_interaction_score(result, int(obj_id))
        visual_scores = visual_interaction.get("normalized") or {}
        preferred_visual = (
            float(visual_scores.get(preferred_hand, 0.0) or 0.0)
            if preferred_hand in {"left", "right"}
            else float(max(visual_scores.get("left", 0.0) or 0.0, visual_scores.get("right", 0.0) or 0.0))
        )
        best_visual = float(max(visual_scores.get("left", 0.0) or 0.0, visual_scores.get("right", 0.0) or 0.0))
        selection_qc = source_object_selection_score(
            result,
            pipeline_result,
            source_dir,
            source_object_id,
            ref_frame,
            int(obj_id),
            pose,
            info,
        )
        qc_available = 0
        overlap_ratio = 0.0
        center_proximity = 0.0
        if selection_qc.get("available"):
            source_area = float(selection_qc.get("source_area", 0.0) or 0.0)
            image_area = float(selection_qc.get("image_area", 0.0) or 0.0)
            source_area_frac = source_area / max(image_area, 1.0)
            # DAI object masks can cover most of the frame or a whole hand.
            # Treat those as weak evidence so Ego visual interaction can select
            # the actual object instead of a broad source-mask centroid match.
            if 0.001 <= source_area_frac <= 0.35:
                qc_available = 1
                overlap_ratio = float(selection_qc.get("overlap_ratio", 0.0) or 0.0)
                center_proximity = float(selection_qc.get("center_proximity", 0.0) or 0.0)
        rows.append(
            (
                qc_available,
                overlap_ratio,
                center_proximity,
                preferred_visual,
                best_visual,
                hand_rank[0],
                hand_rank[1],
                hand_rank[2],
                stable_len,
                object_prompt_score(result, obj_id),
                shape_score,
                -max_jump,
                float(info.get("n_points", 0) or 0),
                int(obj_id),
                pose,
                selection_qc,
            )
        )
    if not rows:
        raise RuntimeError(f"no matching EgoInfinity object for prompt={target_prompt!r}")
    rows.sort(reverse=True, key=lambda row: row[:14])
    return rows[0][13], rows[0][14], rows[0][15]


def rotmat_to_quat_wxyz(rot: np.ndarray) -> list[float]:
    m = np.asarray(rot, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1e-12)
    return quat.tolist()


def resample_transforms_nearest(transforms: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        raise ValueError(f"invalid target_len={target_len}")
    if len(transforms) == target_len:
        return transforms
    if len(transforms) == 1:
        return np.repeat(transforms, target_len, axis=0)
    src_idx = np.rint(np.linspace(0, len(transforms) - 1, target_len)).astype(int)
    return transforms[src_idx]


def rewrite_layout(layout_path: Path, result: dict, ego_obj_id: int, transforms: np.ndarray, mesh_scale: float) -> dict:
    layout = load_json(layout_path)
    # Never inherit a DAI-native or legacy object-only contact correction into
    # an Ego pose layout. Current diagnostic metadata lives in the adapter
    # manifest and does not mutate this production layout.
    layout.pop("hoi_contact_alignment", None)
    source_objects = layout_object_entries(layout)
    if source_objects:
        template_ids = np.rint(np.linspace(0, len(source_objects) - 1, len(transforms))).astype(int)
    else:
        template_ids = np.zeros(len(transforms), dtype=int)
    objects = []
    for frame_idx, transform in enumerate(transforms):
        if source_objects:
            template = source_objects[int(template_ids[frame_idx])]
        else:
            template = {"index": frame_idx, "frame_idx": frame_idx, "local_to_scene": {}}
        row = json.loads(json.dumps(template))
        row["index"] = int(frame_idx)
        row["frame_idx"] = int(frame_idx)
        row["camera_frame"] = f"cam{frame_idx}"
        local = row.setdefault("local_to_scene", {})
        trans = np.asarray(transform[:3, 3], dtype=np.float64).tolist()
        quat = rotmat_to_quat_wxyz(transform[:3, :3])
        pose_translation = pose_translation_from_camera_frame(transform[:3, 3])
        pose_quat = pose_quat_from_camera_rotation(transform[:3, :3])
        local["translation_camera_frame"] = trans
        local["quat_wxyz_camera_frame"] = quat
        local["translation"] = pose_translation
        local["quat_wxyz"] = pose_quat
        local["quat_xyzw"] = [pose_quat[1], pose_quat[2], pose_quat[3], pose_quat[0]]
        local["new_quat"] = pose_quat
        local["scale"] = [float(mesh_scale), float(mesh_scale), float(mesh_scale)]
        row["ego_source"] = {
            "object_id": int(ego_obj_id),
            "template_frame_idx": int(template.get("frame_idx", template.get("index", frame_idx)) or frame_idx),
        }
        objects.append(row)
    layout["objects"] = objects
    layout["frame"] = "camera_frame"
    layout["ego_retarget_adapter"] = {
        "source": "EgoInfinity pose_track_info.T_seq",
        "object_id": int(ego_obj_id),
        "frames": int(len(objects)),
        "hoi_object_only_translation_applied": False,
    }
    layout["translation_scale_optimization"] = {
        "method": "egoinfinity_mesh_scale",
        "mesh_scale": float(mesh_scale),
        "source": "EgoInfinity mesh and pose are used directly; DAI masks are not used for scale by default.",
    }
    layout_path.write_text(json.dumps(layout, indent=2), encoding="utf-8")
    return layout


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a Do-as-I-Do raw-dir whose object pose layout comes from EgoInfinity.")
    parser.add_argument("--source-dir", required=True, type=Path, help="Existing Do-as-I-Do reconstruction raw-dir.")
    parser.add_argument("--pipeline-result", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--task",
        help="Exact pristine DAI task directory that must receive all_hand_meshes.npz.",
    )
    parser.add_argument("--target-prompt")
    parser.add_argument("--object-id", type=int)
    parser.add_argument("--max-centroid-jump-px", type=float, default=320.0)
    parser.add_argument("--hand-npz", type=Path)
    parser.add_argument("--fallback-gravity-json", type=Path)
    parser.add_argument(
        "--max-gravity-tilt-deg",
        type=float,
        default=25.0,
        help="Clamp suspicious gravity estimates above this roll/pitch magnitude to the upright camera-frame fallback; <=0 disables.",
    )
    parser.add_argument("--anchor-hand", default="")
    parser.add_argument(
        "--retarget-hand",
        choices=["auto", "left", "right", "bimanual"],
        default="auto",
        help="Hand side to use for Do-as-I-Do launch; auto uses EgoInfinity grasp_hand_per_frame.",
    )
    parser.add_argument(
        "--hand-selection-source",
        choices=["ego", "hybrid", "visual", "geometry"],
        default="hybrid",
        help="Prefer a robust hybrid of EgoInfinity labels and visual cues by default; geometry can still refine the final hand after the Ego layout is written.",
    )
    parser.add_argument(
        "--hand-geometry-source",
        choices=["ego", "source"],
        default="ego",
        help="For Ego-to-DAI raw dirs, replace source hand mesh geometry with EgoInfinity frame_data by default.",
    )
    parser.add_argument(
        "--hand-source",
        choices=["aoe", "estimated", "unknown"],
        default="unknown",
        help="Exact route hand source recorded in adapter/DAI/SPIDER provenance.",
    )
    parser.add_argument(
        "--object-geometry-source",
        choices=["ego", "source"],
        default="ego",
        help="Use EgoInfinity object mesh/6DoF by default. Use source to keep the native DAI object track while adapting only hand geometry.",
    )
    parser.add_argument(
        "--no-auto-align-hand-slots",
        action="store_true",
        help="Disable the diagnostic left/right hand-slot disagreement check.",
    )
    parser.add_argument(
        "--source-layout-alignment",
        choices=["none", "source"],
        default="source",
        help="Rigid-align Ego object and hand geometry to the Do-as-I-Do source layout by default "
        "so downstream DAI and SPIDER retargeting receive hand/object trajectories in one frame. "
        "Use none only for raw coordinate diagnostics.",
    )
    parser.add_argument(
        "--hoi-contact-alignment",
        choices=["none", "diagnostic", "auto", "force"],
        default="none",
        help="Optionally diagnose a constant hand-object offset. The adapter never applies the proposed "
        "object-only translation to production DAI/SPIDER input. auto/force are retained as diagnostic-only "
        "compatibility aliases.",
    )
    parser.add_argument(
        "--hoi-align-min-bad-mean-distance",
        type=float,
        default=0.075,
        help="Report a proposed contact offset when mean hand-object surface distance is at least 7.5 cm.",
    )
    parser.add_argument("--hoi-align-target-surface-distance", type=float, default=0.025)
    parser.add_argument("--hoi-align-max-offset", type=float, default=0.30)
    parser.add_argument("--ref-frame", type=int)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--scale-viz-dir", type=Path)
    parser.add_argument(
        "--optimize-scale",
        action="store_true",
        help="Use Do-as-I-Do masks to optimize Ego mesh scale. Disabled by default because bad DAI masks can track hands.",
    )
    parser.add_argument(
        "--refine-object-to-dai-mask",
        action="store_true",
        help="Diagnostic-only compatibility option: measure a hypothetical object translation refinement; "
        "the production object track is never modified.",
    )
    parser.add_argument(
        "--no-fit-object-scale-to-ego-mask",
        action="store_true",
        help="Disable the default robust constant mesh-scale fit against EgoInfinity's selected object mask.",
    )
    parser.add_argument(
        "--require-retarget-input-qc",
        action="store_true",
        help="Return nonzero after writing the adapter manifest when Ego scale/contact input QC is invalid.",
    )
    parser.add_argument("--no-optimize-scale", action="store_true", help="Compatibility no-op; scale optimization is off by default.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (
        args.object_geometry_source == "source"
        and args.source_layout_alignment != "none"
    ):
        parser.error(
            "--object-geometry-source source requires --source-layout-alignment none; "
            "otherwise the canonical transform would be applied to the hand but not "
            "the retained source object"
        )

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
    if args.retarget_hand != "auto":
        preferred_hand = args.retarget_hand
    elif args.anchor_hand:
        preferred_hand = args.anchor_hand
    else:
        preferred_hand = str(config.get("anchor_hand") or "")
    if preferred_hand not in {"left", "right"}:
        preferred_hand = None

    object_id = infer_object_id(source_dir, None)
    ref_frame = choose_ref_frame(config, args.ref_frame)

    with gzip.open(args.pipeline_result.expanduser().resolve(), "rb") as handle:
        result = pickle.load(handle)
    ego_obj_id, pose_info, object_selection_qc = select_object(
        result,
        args.pipeline_result.expanduser().resolve(),
        source_dir,
        object_id,
        ref_frame,
        args.target_prompt,
        args.object_id,
        args.max_centroid_jump_px,
        preferred_hand=preferred_hand,
    )
    mesh_info = oid_get(result.get("sam3_mesh_info") or {}, ego_obj_id)
    interaction = infer_interaction_hands(pose_info)
    hand_npz = choose_hand_npz(source_dir, args.hand_npz)
    camera_binding_required = bool(
        args.object_geometry_source == "ego" and args.hand_source == "aoe"
    )
    if camera_binding_required:
        camera_intrinsics_binding = build_ego_aoe_camera_intrinsics_binding(
            result,
            args.pipeline_result,
            hand_npz,
        )
    else:
        camera_intrinsics_binding = {
            "schema_version": CAMERA_INTRINSICS_BINDING_SCHEMA_VERSION,
            "policy": CAMERA_INTRINSICS_BINDING_POLICY,
            "status": "not_applicable",
            "geometry_modified": False,
            "reason": "binding_required_only_for_egoinfinity_object_plus_aoe_hand",
            "object_camera": None,
            "hand_camera": None,
            "comparison": {"comparable": False, "matches": False},
            "errors": [],
        }
    visual_interaction = infer_visual_interaction_hand(
        hand_npz,
        source_dir / "all_frames",
        source_dir / "video_segmentation" / "masks",
        object_id,
    )
    if args.retarget_hand != "auto":
        selected_hand = args.retarget_hand
    elif args.hand_selection_source == "visual":
        selected_hand = visual_interaction.get("selected_hand") or interaction["selected_hand"]
    elif args.hand_selection_source == "hybrid":
        selected_hand = merge_selected_hand(interaction, visual_interaction)
    else:
        selected_hand = interaction["selected_hand"]
    selected_hand = resolve_visual_bimanual_tie(
        selected_hand,
        visual_interaction,
        preferred_hand,
    )
    initial_selected_hand = selected_hand
    if selected_hand in {"left", "right"}:
        default_anchor = selected_hand
    else:
        default_anchor = interaction.get("scale_anchor_hand") or str(config.get("anchor_hand") or "left")
    anchor_hand = args.anchor_hand or default_anchor
    optimize_scale = bool(args.optimize_scale and not args.no_optimize_scale)

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
    adapted_config = load_json(output_dir / "config.json")
    adapted_config["anchor_hand"] = selected_hand
    (output_dir / "config.json").write_text(json.dumps(adapted_config, indent=2), encoding="utf-8")
    gravity_src = source_dir / "gravity.json"
    if not gravity_src.exists() and (args.fallback_gravity_json is None or not args.fallback_gravity_json.exists()):
        raise FileNotFoundError("missing gravity.json; pass --fallback-gravity-json")
    gravity_manifest = write_gravity_metadata(
        output_dir / "gravity.json",
        gravity_src if gravity_src.exists() else args.fallback_gravity_json.resolve(),
        args.max_gravity_tilt_deg,
    )

    (output_dir / "raw").mkdir(parents=True, exist_ok=True)
    for path in sorted((source_dir / "raw").glob("*")):
        if path.name == "all_hand_meshes.npz":
            continue
        link_file(path.resolve(), output_dir / "raw" / path.name)

    ego_object_prompt = object_prompt(result, ego_obj_id) or args.target_prompt
    ego_mesh = ego_mesh_path(args.pipeline_result, ego_obj_id)
    if args.object_geometry_source == "ego":
        mesh_replacement = replace_object_mesh_with_ego_mesh(output_dir, object_id, ego_mesh, ego_object_prompt)
    else:
        mesh_replacement = {
            "applied": False,
            "reason": "object_geometry_source=source",
            "object_id": object_id,
            "targets": [],
        }

    adapted_hand_path = output_dir / "raw" / "all_hand_meshes.npz"
    hand_stats = write_adapted_hand_npz(hand_npz, adapted_hand_path)
    if (
        camera_binding_required
        and camera_intrinsics_binding.get("status") == "invalid"
        and camera_intrinsics_binding.get("errors") == ["camera_intrinsics_mismatch"]
    ):
        camera_intrinsics_binding = normalize_aoe_hand_camera_to_ego(
            adapted_hand_path,
            camera_intrinsics_binding,
            output_dir / "all_frames",
        )
        hand_stats["camera_intrinsics_normalization"] = dict(
            camera_intrinsics_binding.get("normalization") or {}
        )
    before_hand_path = output_dir / "adapter_qc" / "before_hand_geometry.npz"
    before_hand_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(adapted_hand_path, before_hand_path)
    before_ego_hand_geometry = {
        "available": False,
        "reason": "hand_geometry_source=source",
    }
    if args.hand_geometry_source == "ego":
        before_ego_hand_geometry = replace_hand_geometry_with_ego(
            result,
            before_hand_path,
            alignment_transform=None,
        )
    with np.load(adapted_hand_path, allow_pickle=False) as hand_data:
        hand_frame_count = int(
            np.asarray(
                hand_data["right_vertices"] if "right_vertices" in hand_data else hand_data["left_vertices"]
            ).shape[0]
        )
    transforms = pose_sequence_for(result, ego_obj_id, pose_info)
    raw_ego_pose_frames = int(transforms.shape[0])
    transforms = resample_transforms_nearest(transforms, hand_frame_count)
    transforms_before_canonical = np.asarray(transforms, dtype=np.float64).copy()
    if (
        args.object_geometry_source == "ego"
        and args.source_layout_alignment == "source"
        and args.hand_geometry_source != "ego"
    ):
        raise ValueError(
            "source-layout canonicalization requires Ego hand geometry so the same SE(3) is applied "
            "to the complete HOI; use --source-layout-alignment none for source hand geometry"
        )
    if args.source_layout_alignment == "source":
        transforms, source_layout_alignment = align_transforms_to_source_layout(
            transforms,
            source_dir,
            object_id,
            ref_frame,
        )
    else:
        source_layout_alignment = {
            "available": False,
            "reason": "disabled",
            "mode": args.source_layout_alignment,
        }
    alignment_transform = None
    if source_layout_alignment.get("available") and source_layout_alignment.get("alignment_transform") is not None:
        alignment_transform = np.asarray(source_layout_alignment["alignment_transform"], dtype=np.float64)
    canonical_transform = {
        "applied": alignment_transform is not None,
        "kind": "shared_se3" if alignment_transform is not None else "identity",
        "matrix": (
            alignment_transform.tolist()
            if alignment_transform is not None
            else np.eye(4, dtype=np.float64).tolist()
        ),
        "from_frame": "egoinfinity_camera",
        "to_frame": "do_as_i_do_source_camera" if alignment_transform is not None else "egoinfinity_camera",
        "applies_to": [
            "hand_joints",
            "hand_vertices",
            "mano_global_orient",
            "mano_global_translation",
            "object_pose",
            "contact_points_generated_post_transform",
            "object_mesh_world_via_object_pose",
        ],
        "input_camera_intrinsics_binding_status": camera_intrinsics_binding.get(
            "status"
        ),
    }
    ego_hand_geometry = {"available": False, "reason": "hand_geometry_source=source"}
    if args.hand_geometry_source == "ego":
        ego_hand_geometry = replace_hand_geometry_with_ego(
            result,
            adapted_hand_path,
            alignment_transform=alignment_transform,
        )
    native_task = args.task or f"{object_id}_{anchor_hand}"
    stage_native_task_hand_mesh(output_dir, native_task, adapted_hand_path)
    projection_qc = hand_projection_qc(output_dir / "raw" / "all_hand_meshes.npz", output_dir / "all_frames", anchor_hand)
    mask_stats = generate_hand_masks(
        output_dir / "raw" / "all_hand_meshes.npz",
        output_dir / "all_frames",
        output_dir / "video_segmentation" / "masks",
    )
    if args.object_geometry_source == "ego":
        ego_object_masks = write_ego_object_masks(
            result,
            int(ego_obj_id),
            output_dir,
            object_id,
            hand_frame_count,
        )
    else:
        ego_object_masks = {"available": False, "reason": "object_geometry_source=source"}
    tracking_asset_manifest = copy_tracking_assets_for_adapter(source_dir, output_dir, object_id, optimize_scale)
    if args.object_geometry_source == "ego":
        mesh_scale, mesh_scale_source = ego_mesh_scale(pose_info, mesh_info)
    else:
        mesh_scale, mesh_scale_source = 1.0, "dai_native_source_layout"
        if optimize_scale or args.refine_object_to_dai_mask:
            raise ValueError("source object geometry cannot use Ego object scale or pose refinement")
    scale_optimization_diagnostic = run_ego_scale_optimization_diagnostic(
        requested=optimize_scale,
        python_bin=args.python_bin,
        prepared_dir=output_dir,
        object_id=object_id,
        anchor_hand=anchor_hand,
        ref_frame=ref_frame,
        viz_dir=args.scale_viz_dir,
        production_mesh_scale=mesh_scale,
        production_mesh_scale_source=mesh_scale_source,
    )
    scale_optimization = scale_optimization_diagnostic["optimizer_outcome"]
    # Keep the reconstruction-derived production scale and its source intact.
    # Any optimizer candidate is bound above as diagnostic-only provenance.
    mesh_scale = float(scale_optimization_diagnostic["production_mesh_scale"])
    mesh_scale_source = str(
        scale_optimization_diagnostic["production_mesh_scale_source"]
    )
    ego_mask_scale_fit = initial_ego_mask_scale_fit_diagnostic(
        object_geometry_source=args.object_geometry_source,
        optimize_scale=optimize_scale,
        no_fit_object_scale_to_ego_mask=args.no_fit_object_scale_to_ego_mask,
    )
    fit_object_scale = bool(
        args.object_geometry_source == "ego"
        and not args.no_fit_object_scale_to_ego_mask
        and not optimize_scale
    )
    if fit_object_scale:
        ego_mask_scale_fit = fit_mesh_scale_to_ego_mask(
            result,
            int(ego_obj_id),
            output_dir,
            object_id,
            transforms,
            mesh_scale,
        )
        would_apply_scale = bool(ego_mask_scale_fit.get("applied"))
        ego_mask_scale_fit.update(
            {
                "applied": False,
                "would_apply": would_apply_scale,
                "diagnostic_only": True,
                "mesh_modified": False,
                "policy": "object_only_mesh_scale_refinement_forbidden",
                "production_scale": float(mesh_scale),
            }
        )
    if args.object_geometry_source == "ego" and args.refine_object_to_dai_mask:
        _hypothetical_refined_transforms, ego_mask_pose_refine = refine_object_translations_to_ego_mask(
            result,
            int(ego_obj_id),
            output_dir,
            object_id,
            transforms,
            mesh_scale,
        )
        ego_mask_pose_refine.update(
            {
                "applied": False,
                "would_apply": bool(ego_mask_pose_refine.get("available")),
                "diagnostic_only": True,
                "transforms_modified": False,
                "policy": "object_only_mask_pose_refinement_forbidden",
            }
        )
    else:
        ego_mask_pose_refine = {
            "available": False,
            "applied": False,
            "diagnostic_only": True,
            "transforms_modified": False,
            "reason": "dai_mask_refine_disabled",
        }
    layout_path = output_dir / "obj_tracking_out" / object_id / "combined_visualization" / "layout_camera_frame_optimized.json"
    if args.object_geometry_source == "ego":
        mesh_targets = mesh_replacement.get("targets") or []
        object_mesh_for_alignment = Path(mesh_targets[0]) if mesh_targets else ego_mesh
        transforms, hoi_contact_alignment = align_object_to_interacting_hand(
            transforms,
            output_dir / "raw" / "all_hand_meshes.npz",
            object_mesh_for_alignment,
            mesh_scale,
            selected_hand,
            visual_interaction,
            args.hoi_contact_alignment,
            args.hoi_align_min_bad_mean_distance,
            args.hoi_align_target_surface_distance,
            args.hoi_align_max_offset,
        )
        rewrite_layout(layout_path, result, ego_obj_id, transforms, mesh_scale)
    else:
        hoi_contact_alignment = {
            "applied": False,
            "reason": "preserve_dai_native_object_track",
        }
    geometry_interaction = infer_geometry_interaction_hand(
        output_dir / "raw" / "all_hand_meshes.npz",
        layout_path,
        output_dir,
        object_id,
    )
    hand_slot_alignment = {"applied": False, "reason": "disabled"}
    if not args.no_auto_align_hand_slots:
        hand_slot_alignment = maybe_swap_hand_slots_for_visual_geometry(
            output_dir / "raw" / "all_hand_meshes.npz",
            visual_interaction,
            geometry_interaction,
        )
        if hand_slot_alignment.get("applied"):
            mask_stats = generate_hand_masks(
                output_dir / "raw" / "all_hand_meshes.npz",
                output_dir / "all_frames",
                output_dir / "video_segmentation" / "masks",
            )
            projection_qc = hand_projection_qc(
                output_dir / "raw" / "all_hand_meshes.npz",
                output_dir / "all_frames",
                anchor_hand,
            )
            geometry_interaction = infer_geometry_interaction_hand(
                output_dir / "raw" / "all_hand_meshes.npz",
                layout_path,
                output_dir,
                object_id,
            )
    if args.retarget_hand == "auto" and args.hand_selection_source in {"geometry", "hybrid", "visual"}:
        refined_selected = resolve_final_selected_hand(
            selected_hand,
            interaction,
            visual_interaction,
            geometry_interaction,
        )
        if refined_selected in {"left", "right", "bimanual"}:
            selected_hand = refined_selected
            if selected_hand in {"left", "right"}:
                anchor_hand = selected_hand
            elif not args.anchor_hand:
                anchor_hand = interaction.get("scale_anchor_hand") or str(config.get("anchor_hand") or "left")
            projection_qc = hand_projection_qc(output_dir / "raw" / "all_hand_meshes.npz", output_dir / "all_frames", anchor_hand)
            adapted_config = load_json(output_dir / "config.json")
            adapted_config["anchor_hand"] = selected_hand
            (output_dir / "config.json").write_text(json.dumps(adapted_config, indent=2), encoding="utf-8")

    if args.object_geometry_source == "ego":
        adapter_rigid_invariance = summarize_adapter_rigid_invariance(
            before_hand_path,
            adapted_hand_path,
            before_object_poses=transforms_before_canonical,
            after_object_poses=transforms,
            canonical_transform=alignment_transform,
        )
        adapter_rigid_invariance.update(
            {
                "measurement_scope": "ego_object_pose_before_after_adapter",
                "comparison_semantics": "true_pre_adapter_vs_post_adapter_hand_and_object",
                "hand_coordinate_frame": "explicit_adapter_boundary_frames",
                "object_pose_frames_before": int(transforms_before_canonical.shape[0]),
                "object_pose_frames_after": int(transforms.shape[0]),
                "before_hand": str(before_hand_path),
                "after_hand": str(adapted_hand_path),
                "mano_global_pose_transform": ego_hand_geometry.get(
                    "mano_global_pose_transform"
                ),
            }
        )
    else:
        before_layout_path = source_layout_path(source_dir, object_id)
        adapter_rigid_invariance = summarize_source_object_layout_invariance(
            before_hand_path,
            adapted_hand_path,
            before_layout_path,
            layout_path,
        )

    if args.object_geometry_source == "ego":
        surface_contact_geometry = infer_geometry_interaction_hand(
            output_dir / "raw" / "all_hand_meshes.npz",
            layout_path,
            output_dir,
            object_id,
            distance_thresh=0.05,
            max_sample_frames=None,
        )
        if selected_hand == "bimanual":
            production_projection_sides = ["left", "right"]
        elif selected_hand in {"left", "right"}:
            production_projection_sides = [selected_hand]
        else:
            production_projection_sides = []
    else:
        surface_contact_geometry = {
            "available": False,
            "reason": "object_geometry_source_is_not_ego",
        }
        production_projection_sides = []
    production_hand_projection_qc = {
        "method": "project_hand_mesh_at_start_middle_end",
        "sides": {
            side: (
                projection_qc
                if projection_qc.get("anchor_hand") == side
                else hand_projection_qc(
                    output_dir / "raw" / "all_hand_meshes.npz",
                    output_dir / "all_frames",
                    side,
                )
            )
            for side in production_projection_sides
        },
    }

    retarget_input_qc = assess_ego_retarget_input(
        scale_fit=ego_mask_scale_fit,
        visual_interaction=visual_interaction,
        geometry_interaction=surface_contact_geometry,
        hand_projection=production_hand_projection_qc,
        selected_hand=selected_hand,
        hoi_contact_alignment=hoi_contact_alignment,
    ) if args.object_geometry_source == "ego" else {
        "status": "not_applicable",
        "reason": "object_geometry_source_is_not_ego",
    }
    if adapter_rigid_invariance.get("status") != "ok":
        retarget_input_qc.setdefault("errors", []).append("adapter_rigid_invariance_invalid")
        retarget_input_qc["status"] = "invalid"
    if ego_mask_pose_refine.get("applied"):
        retarget_input_qc.setdefault("errors", []).append("object_only_mask_pose_refinement_applied")
        retarget_input_qc["status"] = "invalid"
    if camera_binding_required and camera_intrinsics_binding.get("status") != "ok":
        retarget_input_qc.setdefault("errors", []).append(
            "camera_intrinsics_binding_invalid"
        )
        retarget_input_qc["camera_intrinsics_binding_errors"] = list(
            camera_intrinsics_binding.get("errors") or []
        )
        retarget_input_qc["status"] = "invalid"

    hoi_refinement = scale_refinement_provenance(
        requested=optimize_scale,
        optimization=scale_optimization,
        contact_diagnostic=hoi_contact_alignment,
    )
    hoi_refinement.update(
        {
            "object_mask_pose_refinement_applied": False,
            "object_mesh_scale_refinement_applied": False,
            "hand_slot_swap_applied": False,
            "hand_contact_translation_applied": False,
            "scale_optimization_diagnostic": scale_optimization_diagnostic,
            "object_scale_diagnostic": ego_mask_scale_fit,
            "object_pose_diagnostic": ego_mask_pose_refine,
            "hand_slot_diagnostic": hand_slot_alignment,
        }
    )

    manifest = {
        "adapter": "prepare_egoinfinity_do_as_i_do_raw_dir.py",
        "adapter_schema_version": 3,
        "production_hoi_policy": "preserve_source_relative_transform",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "pipeline_result": str(args.pipeline_result),
        "raw_video_provenance": raw_video_provenance,
        "object_geometry_source": args.object_geometry_source,
        "hand_geometry_source": args.hand_geometry_source,
        "hand_source": args.hand_source,
        **adapter_object_provenance(source_dir, args.object_geometry_source),
        "dai_object_id": object_id,
        "ego_object_id": int(ego_obj_id),
        "ego_prompt": object_prompt(result, ego_obj_id),
        "ego_prompt_score": object_prompt_score(result, ego_obj_id),
        "ego_pose_frames_raw": raw_ego_pose_frames,
        "ego_pose_frames_resampled": int(transforms.shape[0]),
        "hand_frame_count": hand_frame_count,
        "ego_pose_track_keys": sorted(str(k) for k in pose_info.keys()),
        "object_selection_qc": object_selection_qc,
        "interaction_hand": interaction,
        "visual_interaction_hand": visual_interaction,
        "geometry_interaction_hand": geometry_interaction,
        "surface_contact_geometry": surface_contact_geometry,
        "hand_slot_alignment": hand_slot_alignment,
        "hand_selection_source": args.hand_selection_source,
        "initial_selected_hand": initial_selected_hand,
        "selected_hand": selected_hand,
        "anchor_hand": anchor_hand,
        "ref_frame": ref_frame,
        "optimize_scale_before_ego_pose_override": optimize_scale,
        "optimize_scale_diagnostic_requested": optimize_scale,
        "mesh_scale": mesh_scale,
        "mesh_scale_source": mesh_scale_source,
        "scale_optimization_diagnostic": scale_optimization_diagnostic,
        "source_layout_alignment": source_layout_alignment,
        "canonical_transform": canonical_transform,
        "camera_intrinsics_binding": camera_intrinsics_binding,
        "ego_mask_scale_fit": ego_mask_scale_fit,
        "ego_mask_pose_refine": ego_mask_pose_refine,
        "hoi_contact_alignment": hoi_contact_alignment,
        "hoi_refinement": hoi_refinement,
        "adapter_rigid_invariance": adapter_rigid_invariance,
        "adapter_boundary_inputs": {
            "before_hand": str(before_hand_path),
            "after_hand": str(adapted_hand_path),
            "before_object_pose_source": (
                "egoinfinity_pose_sequence"
                if args.object_geometry_source == "ego"
                else str(source_layout_path(source_dir, object_id))
            ),
            "after_object_layout": str(layout_path),
        },
        "retarget_input_qc": retarget_input_qc,
        "tracking_assets": tracking_asset_manifest,
        "gravity": gravity_manifest,
        "object_mesh_replacement": mesh_replacement,
        "hand_npz": hand_stats,
        "ego_hand_geometry": ego_hand_geometry,
        "before_ego_hand_geometry": before_ego_hand_geometry,
        "hand_projection_qc": projection_qc,
        "production_hand_projection_qc": production_hand_projection_qc,
        "hand_masks": mask_stats,
        "ego_object_masks": ego_object_masks,
        "layout": str(layout_path),
    }
    (output_dir / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if adapter_rigid_invariance.get("status") != "ok":
        print(
            "Adapter rigid-invariance QC rejected production input: "
            + ", ".join(str(item) for item in adapter_rigid_invariance.get("errors") or []),
            file=sys.stderr,
        )
        return 4
    if args.require_retarget_input_qc and retarget_input_qc.get("status") != "ok":
        print(
            "Ego retarget input QC rejected adapter: "
            + ", ".join(str(item) for item in retarget_input_qc.get("errors") or []),
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
