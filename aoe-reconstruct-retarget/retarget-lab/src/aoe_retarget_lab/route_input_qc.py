"""Hard QC for route-specific DAI keypoints consumed by retarget backends.

The matrix-level ``assets/trajectory_6dof`` keypoint link is intentionally not
an admissible source here.  That link is shared by cells and has historically
pointed at a different hand source.  QC must start from the exact DAI cell and,
when a backend copies that file into a processed dataset, compare the copy with
the exact source byte-for-byte.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

import numpy as np

from aoe_retarget_lab.hoi_geometry import validate_rigid_transform


PathLike = Union[str, Path]


SPIDER_CORE_HOI_ARRAYS = (
    "qpos_wrist_left",
    "qpos_wrist_right",
    "qpos_finger_left",
    "qpos_finger_right",
    "qpos_obj_left",
    "qpos_obj_right",
)

_SPIDER_ALLOWED_CONTACT_PREFIXES = ("contact_",)
_SPIDER_ALLOWED_DERIVED_PREFIXES = ("mano_",)
_SPIDER_ALLOWED_DERIVED_FIELDS = {
    "qpos_pip_left",
    "qpos_pip_right",
    "qpos_dip_left",
    "qpos_dip_right",
    "centering_offset",
    "world_offset",
}


CAMERA_INTRINSICS_BINDING_SCHEMA_VERSION = 1
CAMERA_INTRINSICS_BINDING_POLICY = "normalized_pinhole_intrinsics_match"
CAMERA_INTRINSICS_NORMALIZATION_POLICY = "explicit_ray_depth_intrinsics_normalization"
CAMERA_FOCAL_RELATIVE_TOLERANCE = 0.05
CAMERA_PRINCIPAL_POINT_NORMALIZED_TOLERANCE = 0.01
CAMERA_ASPECT_RELATIVE_TOLERANCE = 0.01


def _canonical_camera_intrinsics(value: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize one camera model for route binding and fingerprinting."""

    errors: list[str] = []
    if not isinstance(value, Mapping):
        return None, ["camera_entry_missing_or_invalid"]
    raw = value.get("intrinsics") if isinstance(value.get("intrinsics"), Mapping) else value
    model = str(raw.get("model") or "pinhole")
    if model != "pinhole":
        errors.append(f"camera_model_unsupported:{model}")

    numeric: dict[str, float] = {}
    for key in ("fx", "fy", "cx", "cy"):
        try:
            number = float(raw.get(key))
        except (TypeError, ValueError):
            errors.append(f"camera_{key}_missing_or_invalid")
            continue
        if not np.isfinite(number):
            errors.append(f"camera_{key}_nonfinite")
        elif key in {"fx", "fy"} and number <= 0.0:
            errors.append(f"camera_{key}_nonpositive")
        else:
            numeric[key] = number

    dimensions: dict[str, int] = {}
    for key in ("width", "height"):
        try:
            number = int(raw.get(key))
        except (TypeError, ValueError):
            errors.append(f"camera_{key}_missing_or_invalid")
            continue
        if number <= 0:
            errors.append(f"camera_{key}_nonpositive")
        else:
            dimensions[key] = number

    if errors:
        return None, errors
    return {
        "model": "pinhole",
        "width": dimensions["width"],
        "height": dimensions["height"],
        "fx": numeric["fx"],
        "fy": numeric["fy"],
        "cx": numeric["cx"],
        "cy": numeric["cy"],
    }, []


def camera_intrinsics_fingerprint(value: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint of one pinhole camera model."""

    canonical, errors = _canonical_camera_intrinsics(value)
    if canonical is None:
        raise ValueError("invalid camera intrinsics: " + ",".join(errors))
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _camera_binding_entry(value: Any, *, label: str) -> tuple[dict[str, Any], list[str]]:
    canonical, errors = _canonical_camera_intrinsics(value)
    source = dict(value.get("source")) if isinstance(value, Mapping) and isinstance(value.get("source"), Mapping) else {}
    if canonical is None:
        return {
            "status": "invalid",
            "source": source,
            "intrinsics": None,
            "fingerprint": None,
            "errors": [f"{label}_{item}" for item in errors],
        }, [f"{label}_{item}" for item in errors]
    entry = {
        "status": "ok",
        "source": source,
        "intrinsics": canonical,
        "fingerprint": camera_intrinsics_fingerprint(canonical),
        "errors": [],
    }
    return entry, []


def build_camera_intrinsics_binding(
    object_camera: Any,
    hand_camera: Any,
) -> dict[str, Any]:
    """Build a fail-closed binding for independently reconstructed HOI inputs.

    Intrinsics are compared after normalization by image width/height, so an
    exact resize is admissible while a different projection model is not.  The
    function only records and compares camera evidence; it never transforms
    hand or object geometry.
    """

    object_entry, object_errors = _camera_binding_entry(
        object_camera, label="object_camera"
    )
    hand_entry, hand_errors = _camera_binding_entry(hand_camera, label="hand_camera")
    errors = [*object_errors, *hand_errors]
    comparison: dict[str, Any] = {
        "comparable": False,
        "matches": False,
        "thresholds": {
            "max_relative_focal_error": CAMERA_FOCAL_RELATIVE_TOLERANCE,
            "max_normalized_principal_point_error": CAMERA_PRINCIPAL_POINT_NORMALIZED_TOLERANCE,
            "max_relative_aspect_error": CAMERA_ASPECT_RELATIVE_TOLERANCE,
        },
    }
    object_intrinsics = object_entry.get("intrinsics")
    hand_intrinsics = hand_entry.get("intrinsics")
    if isinstance(object_intrinsics, Mapping) and isinstance(hand_intrinsics, Mapping):
        object_fx = float(object_intrinsics["fx"]) / float(object_intrinsics["width"])
        object_fy = float(object_intrinsics["fy"]) / float(object_intrinsics["height"])
        hand_fx = float(hand_intrinsics["fx"]) / float(hand_intrinsics["width"])
        hand_fy = float(hand_intrinsics["fy"]) / float(hand_intrinsics["height"])
        focal_errors = {
            "x": abs(object_fx - hand_fx) / max(abs(hand_fx), 1.0e-12),
            "y": abs(object_fy - hand_fy) / max(abs(hand_fy), 1.0e-12),
        }
        principal_errors = {
            "x": abs(
                float(object_intrinsics["cx"]) / float(object_intrinsics["width"])
                - float(hand_intrinsics["cx"]) / float(hand_intrinsics["width"])
            ),
            "y": abs(
                float(object_intrinsics["cy"]) / float(object_intrinsics["height"])
                - float(hand_intrinsics["cy"]) / float(hand_intrinsics["height"])
            ),
        }
        object_aspect = float(object_intrinsics["width"]) / float(object_intrinsics["height"])
        hand_aspect = float(hand_intrinsics["width"]) / float(hand_intrinsics["height"])
        aspect_error = abs(object_aspect - hand_aspect) / max(abs(hand_aspect), 1.0e-12)
        matches = bool(
            max(focal_errors.values()) <= CAMERA_FOCAL_RELATIVE_TOLERANCE
            and max(principal_errors.values())
            <= CAMERA_PRINCIPAL_POINT_NORMALIZED_TOLERANCE
            and aspect_error <= CAMERA_ASPECT_RELATIVE_TOLERANCE
        )
        comparison.update(
            {
                "comparable": True,
                "matches": matches,
                "relative_focal_error": focal_errors,
                "normalized_principal_point_error": principal_errors,
                "relative_aspect_error": aspect_error,
            }
        )
        if not matches:
            errors.append("camera_intrinsics_mismatch")

    return {
        "schema_version": CAMERA_INTRINSICS_BINDING_SCHEMA_VERSION,
        "policy": CAMERA_INTRINSICS_BINDING_POLICY,
        "status": "invalid" if errors else "ok",
        "geometry_modified": False,
        "object_camera": object_entry,
        "hand_camera": hand_entry,
        "comparison": comparison,
        "errors": errors,
    }


def camera_ray_depth_transform(
    source_camera: Mapping[str, Any],
    target_camera: Mapping[str, Any],
) -> np.ndarray:
    """Map camera-space XYZ while preserving source pixels and metric depth.

    For a point reconstructed with ``source_camera``, this returns ``A`` such
    that ``K_target @ (A @ xyz)`` has the same inhomogeneous pixel coordinate
    as ``K_source @ xyz`` and the same Z value.  This is an explicit
    calibration conversion, not an arbitrary HOI translation or scale fit.
    """

    source, source_errors = _canonical_camera_intrinsics(source_camera)
    target, target_errors = _canonical_camera_intrinsics(target_camera)
    if source is None or target is None:
        raise ValueError(
            "invalid camera intrinsics: "
            + ",".join([*source_errors, *target_errors])
        )
    return np.array(
        [
            [
                float(source["fx"]) / float(target["fx"]),
                0.0,
                (float(source["cx"]) - float(target["cx"]))
                / float(target["fx"]),
            ],
            [
                0.0,
                float(source["fy"]) / float(target["fy"]),
                (float(source["cy"]) - float(target["cy"]))
                / float(target["fy"]),
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _validate_camera_source_file(
    source: Mapping[str, Any],
    *,
    binding_key: str,
    path_field: str,
    label: str,
    errors: list[str],
) -> Path | None:
    """Resolve and re-hash one camera source recorded by the adapter."""

    bindings = source.get("file_bindings")
    record = bindings.get(binding_key) if isinstance(bindings, Mapping) else None
    if not isinstance(record, Mapping):
        errors.append(f"adapter_camera_{label}_file_binding_missing")
        return None
    path_value = record.get("path")
    sha256 = record.get("sha256")
    if not isinstance(path_value, str) or not path_value.strip():
        errors.append(f"adapter_camera_{label}_path_missing")
        return None
    path = Path(path_value).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    recorded_source_path = source.get(path_field)
    if not isinstance(recorded_source_path, str) or not recorded_source_path.strip():
        errors.append(f"adapter_camera_{label}_source_path_missing")
    else:
        try:
            source_resolved = Path(recorded_source_path).expanduser().resolve()
        except OSError:
            source_resolved = Path(recorded_source_path).expanduser().absolute()
        if source_resolved != resolved:
            errors.append(f"adapter_camera_{label}_source_path_binding_mismatch")
    if not resolved.is_file():
        errors.append(f"adapter_camera_{label}_source_file_missing:{resolved}")
        return None
    if not isinstance(sha256, str) or len(sha256) != 64:
        errors.append(f"adapter_camera_{label}_sha256_missing_or_invalid")
        return resolved
    try:
        current_sha256 = file_sha256(resolved)
    except OSError as exc:
        errors.append(
            f"adapter_camera_{label}_source_file_unreadable:{type(exc).__name__}"
        )
        return None
    if current_sha256 != sha256:
        errors.append(f"adapter_camera_{label}_sha256_mismatch")
    flat_sha_field = "sha256" if path_field == "path" else f"{path_field}_sha256"
    flat_sha = source.get(flat_sha_field)
    if flat_sha is not None and flat_sha != sha256:
        errors.append(f"adapter_camera_{label}_flat_sha256_binding_mismatch")
    return resolved


def _pipeline_result_image_size(result: Mapping[str, Any]) -> tuple[int, int] | None:
    background = result.get("bg_rgb")
    shape = getattr(background, "shape", None)
    if shape is not None and len(shape) >= 2 and int(shape[0]) > 0 and int(shape[1]) > 0:
        return int(shape[1]), int(shape[0])
    for frame in result.get("frame_data") or []:
        if not isinstance(frame, Mapping):
            continue
        for key in ("img_rgb", "image_rgb", "rgb", "image", "frame_rgb"):
            value = frame.get(key)
            value_shape = getattr(value, "shape", None)
            if (
                value_shape is not None
                and len(value_shape) >= 2
                and int(value_shape[0]) > 0
                and int(value_shape[1]) > 0
            ):
                return int(value_shape[1]), int(value_shape[0])
        objects = frame.get("sam3_obj_data") or {}
        if not isinstance(objects, Mapping):
            continue
        for obj in objects.values():
            if not isinstance(obj, Mapping):
                continue
            mask_shape = obj.get("mask_shape")
            if isinstance(mask_shape, (list, tuple, np.ndarray)) and len(mask_shape) >= 2:
                height, width = int(mask_shape[0]), int(mask_shape[1])
                if width > 0 and height > 0:
                    return width, height
    return None


def _read_pipeline_result_camera(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        result = pickle.load(handle)
    if not isinstance(result, Mapping):
        raise ValueError("pipeline result is not a mapping")
    size = _pipeline_result_image_size(result)
    if size is None:
        raise ValueError("pipeline result has no image size")
    return {
        "model": "pinhole",
        "width": size[0],
        "height": size[1],
        "fx": float(result["dp_focal"]),
        "fy": float(result["dp_focal"]),
        "cx": float(result["cx"]),
        "cy": float(result["cy"]),
    }


def _read_hawor_calibration_camera(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera = payload.get("cameraParams") if isinstance(payload, Mapping) else None
    if not isinstance(camera, Mapping):
        raise ValueError("calibration has no cameraParams")
    width_text, separator, height_text = str(camera.get("resolution") or "").lower().partition("x")
    if not separator:
        raise ValueError("calibration resolution is not WIDTHxHEIGHT")
    return {
        "model": "pinhole",
        "width": int(width_text),
        "height": int(height_text),
        "fx": float(camera["fx_pixels"]),
        "fy": float(camera.get("fy_pixels", camera["fx_pixels"])),
        "cx": float(camera["cx_pixels"]),
        "cy": float(camera["cy_pixels"]),
    }


def _npz_scalar_text(path: Path, key: str) -> str | None:
    with np.load(path, allow_pickle=True) as archive:
        if key not in archive:
            return None
        value = np.asarray(archive[key])
        if value.size != 1:
            return None
        text = str(value.reshape(()).item()).strip()
        return text or None


def _npz_scalar_float(path: Path, key: str) -> float | None:
    with np.load(path, allow_pickle=True) as archive:
        if key not in archive:
            return None
        value = np.asarray(archive[key])
        if value.size != 1:
            return None
        number = float(value.reshape(()).item())
        return number if np.isfinite(number) else None


def _validate_camera_source_bindings(value: Mapping[str, Any], errors: list[str]) -> None:
    """Re-read the bound files so self-consistent embedded K values cannot pass."""

    object_entry = value.get("object_camera")
    hand_entry = value.get("hand_camera")
    if not isinstance(object_entry, Mapping) or not isinstance(hand_entry, Mapping):
        return
    object_source = object_entry.get("source")
    hand_source = hand_entry.get("source")
    if not isinstance(object_source, Mapping) or not isinstance(hand_source, Mapping):
        errors.append("adapter_camera_source_provenance_missing_or_invalid")
        return
    if object_source.get("kind") != "egoinfinity_pipeline_result":
        errors.append("adapter_camera_object_source_kind_invalid")
    if hand_source.get("kind") != "aoe_hawor_calibrated_camera":
        errors.append("adapter_camera_hand_source_kind_invalid")

    pipeline_path = _validate_camera_source_file(
        object_source,
        binding_key="pipeline_result",
        path_field="path",
        label="pipeline_result",
        errors=errors,
    )
    adapted_hand_path = _validate_camera_source_file(
        hand_source,
        binding_key="adapted_hand_npz",
        path_field="adapted_hand_npz",
        label="adapted_hand_npz",
        errors=errors,
    )
    source_hands_path = _validate_camera_source_file(
        hand_source,
        binding_key="source_hands_npz",
        path_field="source_hands_npz",
        label="source_hands_npz",
        errors=errors,
    )
    calibration_path = _validate_camera_source_file(
        hand_source,
        binding_key="calibration",
        path_field="calibration_path",
        label="calibration",
        errors=errors,
    )

    if pipeline_path is not None:
        try:
            actual_object_camera = _read_pipeline_result_camera(pipeline_path)
            if camera_intrinsics_fingerprint(actual_object_camera) != object_entry.get(
                "fingerprint"
            ):
                errors.append("adapter_camera_pipeline_result_intrinsics_mismatch")
        except (
            EOFError,
            OSError,
            KeyError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ) as exc:
            errors.append(
                "adapter_camera_pipeline_result_invalid:" + type(exc).__name__
            )

    if calibration_path is not None:
        try:
            actual_hand_camera = _read_hawor_calibration_camera(calibration_path)
            if camera_intrinsics_fingerprint(actual_hand_camera) != hand_entry.get(
                "fingerprint"
            ):
                errors.append("adapter_camera_calibration_intrinsics_mismatch")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append("adapter_camera_calibration_invalid:" + type(exc).__name__)

    if adapted_hand_path is not None and source_hands_path is not None:
        try:
            recorded_source = _npz_scalar_text(adapted_hand_path, "source_hands_npz")
            if adapted_hand_path != source_hands_path:
                if not recorded_source:
                    errors.append("adapter_camera_adapted_hand_source_metadata_missing")
                else:
                    recorded_path = Path(recorded_source).expanduser()
                    if not recorded_path.is_absolute():
                        recorded_path = adapted_hand_path.parent / recorded_path
                    if recorded_path.resolve() != source_hands_path.resolve():
                        errors.append("adapter_camera_adapted_hand_source_path_mismatch")
        except (OSError, TypeError, ValueError) as exc:
            errors.append("adapter_camera_adapted_hand_npz_invalid:" + type(exc).__name__)

    if source_hands_path is not None and calibration_path is not None:
        try:
            source_focal = _npz_scalar_float(source_hands_path, "focal")
            calibration = _read_hawor_calibration_camera(calibration_path)
            if source_focal is None:
                errors.append("adapter_camera_source_hands_focal_missing")
            elif abs(source_focal - float(calibration["fx"])) / max(
                abs(float(calibration["fx"])), 1.0e-12
            ) > CAMERA_FOCAL_RELATIVE_TOLERANCE:
                errors.append("adapter_camera_source_hands_calibration_mismatch")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append("adapter_camera_source_hands_invalid:" + type(exc).__name__)


def expected_cell_key(trajectory: str, hand_source: str) -> str:
    """Return the exact Do-as-I-Do cell key for a route and hand source."""

    trajectory = str(trajectory).strip()
    hand_source = str(hand_source).strip()
    if not trajectory:
        raise ValueError("trajectory must be non-empty")
    if not hand_source:
        raise ValueError("hand_source must be non-empty")
    if "/" in trajectory or "/" in hand_source:
        raise ValueError("trajectory and hand_source must be path components")
    return f"traj_{trajectory}__hand_{hand_source}__retarget_do_as_i_do"


def file_sha256(path: PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash *path* without loading a potentially large keypoint file at once."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _path_text(path: Path) -> str:
    return str(path.expanduser())


def _is_unified_source_keypoints(path: Path) -> bool:
    """Recognize the route-ambiguous matrix convenience asset lexically.

    This deliberately checks the caller-provided path, not ``resolve()``.  A
    symlink at the unified location is still an ambiguous QC input even when
    its current target happens to be the desired cell.
    """

    parts = path.expanduser().parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ("assets", "trajectory_6dof"):
            return path.name == "source_trajectory_keypoints.npz"
    return False


def _cell_components(path: Path) -> list[str]:
    return [part for part in path.expanduser().parts if part.startswith("traj_") and "__hand_" in part]


def _resolved_path(path: Path) -> Path:
    """Resolve symlinks so indexed aliases cannot hide a cross-cell target."""

    try:
        return path.expanduser().resolve()
    except OSError:
        return path.expanduser().absolute()


def _cell_root(path: Path, cell: str) -> Path | None:
    """Return the resolved root ending in *cell*, if it occurs exactly once."""

    resolved = _resolved_path(path)
    matches = [parent for parent in (resolved, *resolved.parents) if parent.name == cell]
    return matches[0] if len(matches) == 1 else None


def _lexical_cell_root(path: Path, cell: str) -> Path | None:
    """Return the non-resolved cell root used by the caller-facing path."""

    absolute = path.expanduser().absolute()
    matches = [parent for parent in (absolute, *absolute.parents) if parent.name == cell]
    return matches[0] if len(matches) == 1 else None


def _validate_exact_cell_adapter_manifest(
    path: Path,
    *,
    exact_cell: str,
    errors: list[str],
) -> Path | None:
    """Validate the exact cell's ``raw_dir`` manifest symlink.

    The official DAI wrapper intentionally makes ``<cell>/raw_dir`` a symlink
    to the route-specific adapter directory.  Requiring the resolved target to
    retain the cell component would reject that legitimate layout.  Instead we
    anchor the caller-facing path to exactly one cell and ``raw_dir``, reject a
    resolved target that names any *other* cell, and rely on the manifest's
    hand/object provenance plus the true pre/post invariant for target
    identity.
    """

    lexical_cells = _cell_components(path)
    if exact_cell not in path.expanduser().absolute().parts:
        found = ",".join(lexical_cells) if lexical_cells else "none"
        errors.append(
            f"adapter_manifest_not_exact_dai_cell:expected={exact_cell}:found={found}"
        )
    unexpected_lexical = sorted(set(lexical_cells) - {exact_cell})
    if unexpected_lexical:
        errors.append(
            "adapter_manifest_cross_cell_path:" + ",".join(unexpected_lexical)
        )
    root = _lexical_cell_root(path, exact_cell)
    if root is None:
        errors.append(f"adapter_manifest_exact_dai_cell_root_unresolved:{exact_cell}")
        return None
    try:
        relative = path.expanduser().absolute().relative_to(root)
    except ValueError:
        errors.append("adapter_manifest_outside_exact_dai_cell_root")
        return root
    if relative.parts[:1] != ("raw_dir",) or relative.name != "adapter_manifest.json":
        errors.append(
            "adapter_manifest_location_invalid:expected=raw_dir/adapter_manifest.json:actual="
            + str(relative)
        )

    resolved_cells = _cell_components(_resolved_path(path))
    unexpected_resolved = sorted(set(resolved_cells) - {exact_cell})
    if unexpected_resolved:
        errors.append(
            "adapter_manifest_resolved_cross_cell:" + ",".join(unexpected_resolved)
        )
    return root.resolve()


def _validate_exact_cell_source(
    path: Path,
    *,
    exact_cell: str,
    artifact: str,
    errors: list[str],
) -> Path | None:
    """Require both the selected path and its symlink target to name one cell."""

    lexical_cells = _cell_components(path)
    resolved = _resolved_path(path)
    resolved_cells = _cell_components(resolved)
    if exact_cell not in path.parts:
        found = ",".join(lexical_cells) if lexical_cells else "none"
        errors.append(f"{artifact}_not_exact_dai_cell:expected={exact_cell}:found={found}")
    unexpected_lexical = sorted(set(lexical_cells) - {exact_cell})
    if unexpected_lexical:
        errors.append(f"{artifact}_cross_cell_path:" + ",".join(unexpected_lexical))
    if exact_cell not in resolved.parts:
        found = ",".join(resolved_cells) if resolved_cells else "none"
        errors.append(
            f"{artifact}_resolved_not_exact_dai_cell:expected={exact_cell}:found={found}"
        )
    unexpected_resolved = sorted(set(resolved_cells) - {exact_cell})
    if unexpected_resolved:
        errors.append(f"{artifact}_resolved_cross_cell:" + ",".join(unexpected_resolved))
    root = _cell_root(path, exact_cell)
    if root is None:
        errors.append(f"{artifact}_exact_dai_cell_root_unresolved:{exact_cell}")
    return root


def _validate_relative_artifact_path(
    path: Path,
    *,
    cell_root: Path | None,
    artifact: str,
    required_prefix: tuple[str, ...],
    required_name: str,
    errors: list[str],
) -> None:
    """Validate the resolved location inside an exact DAI cell."""

    if cell_root is None:
        return
    resolved = _resolved_path(path)
    try:
        relative = resolved.relative_to(cell_root)
    except ValueError:
        errors.append(f"{artifact}_outside_exact_dai_cell_root")
        return
    if relative.name != required_name:
        errors.append(f"{artifact}_filename_invalid:{relative.name}")
    prefix = relative.parts[: len(required_prefix)]
    if prefix != required_prefix:
        errors.append(
            f"{artifact}_location_invalid:expected_prefix="
            + "/".join(required_prefix)
            + ":actual="
            + str(relative)
        )


def _root_before_suffix(path: Path, suffix: tuple[str, ...]) -> Path | None:
    absolute = path.expanduser().absolute()
    if len(absolute.parts) < len(suffix) or tuple(absolute.parts[-len(suffix) :]) != suffix:
        return None
    root = absolute
    for _ in suffix:
        root = root.parent
    return root


def _validate_dai_processed_artifact(
    path: Path,
    *,
    exact_cell: str,
    source_cell_root: Path | None,
    artifact: str,
    required_prefix: tuple[str, ...],
    required_name: str,
    errors: list[str],
) -> Path | None:
    """Validate a canonical indexed DAI artifact in copy or symlink mode."""

    absolute = path.expanduser().absolute()
    lexical_cells = _cell_components(absolute)
    if exact_cell not in absolute.parts:
        found = ",".join(lexical_cells) if lexical_cells else "none"
        errors.append(f"{artifact}_not_exact_dai_processed_cell:expected={exact_cell}:found={found}")
    unexpected = sorted(set(lexical_cells) - {exact_cell})
    if unexpected:
        errors.append(f"{artifact}_cross_cell_path:" + ",".join(unexpected))

    lexical_root = _lexical_cell_root(absolute, exact_cell)
    if lexical_root is None:
        errors.append(f"{artifact}_exact_dai_processed_cell_root_unresolved:{exact_cell}")
        return None
    try:
        relative = absolute.relative_to(lexical_root)
    except ValueError:
        errors.append(f"{artifact}_outside_exact_dai_processed_cell_root")
        return lexical_root
    if relative.name != required_name:
        errors.append(f"{artifact}_filename_invalid:{relative.name}")
    if relative.parts[: len(required_prefix)] != required_prefix:
        errors.append(
            f"{artifact}_location_invalid:expected_prefix="
            + "/".join(required_prefix)
            + ":actual="
            + str(relative)
        )

    resolved = _resolved_path(path)
    resolved_cells = _cell_components(resolved)
    if exact_cell not in resolved.parts:
        found = ",".join(resolved_cells) if resolved_cells else "none"
        errors.append(
            f"{artifact}_resolved_not_exact_dai_cell:expected={exact_cell}:found={found}"
        )
    unexpected_resolved = sorted(set(resolved_cells) - {exact_cell})
    if unexpected_resolved:
        errors.append(f"{artifact}_resolved_cross_cell:" + ",".join(unexpected_resolved))

    resolved_cell_root = _cell_root(path, exact_cell)
    allowed_resolved_roots = {_resolved_path(lexical_root)}
    if source_cell_root is not None:
        allowed_resolved_roots.add(_resolved_path(source_cell_root))
    if resolved_cell_root is None or _resolved_path(resolved_cell_root) not in allowed_resolved_roots:
        errors.append(f"{artifact}_resolved_outside_source_or_processed_cell")

    processed_experiment = _root_before_suffix(
        lexical_root, ("assets", "cells", exact_cell)
    )
    source_experiment = (
        _root_before_suffix(
            source_cell_root,
            ("intermediates", "retargeting", "do_as_i_do", exact_cell),
        )
        if source_cell_root is not None
        else None
    )
    if processed_experiment is None:
        errors.append(f"{artifact}_processed_experiment_root_unresolved")
    if source_experiment is None:
        errors.append(f"{artifact}_source_experiment_root_unresolved")
    if (
        processed_experiment is not None
        and source_experiment is not None
        and _resolved_path(processed_experiment) != _resolved_path(source_experiment)
    ):
        errors.append(f"{artifact}_cross_experiment_path")
    return lexical_root


def _hash_file(path: Path, *, label: str, errors: list[str]) -> str | None:
    if not path.is_file():
        errors.append(f"{label}_missing:{_path_text(path)}")
        return None
    try:
        return file_sha256(path)
    except OSError as exc:
        errors.append(f"{label}_unreadable:{type(exc).__name__}")
        return None


def _array_sha256(value: np.ndarray) -> str:
    """Hash an array's C-order payload independently of NPZ container bytes."""

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_npz_arrays(
    path: Path,
    *,
    label: str,
    errors: list[str],
) -> dict[str, np.ndarray]:
    if not path.is_file():
        errors.append(f"{label}_missing:{_path_text(path)}")
        return {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, TypeError) as exc:
        errors.append(f"{label}_invalid_npz:{type(exc).__name__}:{exc}")
        return {}


def _spider_mutable_field_category(name: str) -> str | None:
    if name.startswith(_SPIDER_ALLOWED_CONTACT_PREFIXES):
        return "contact"
    if name.startswith(_SPIDER_ALLOWED_DERIVED_PREFIXES) or name in _SPIDER_ALLOWED_DERIVED_FIELDS:
        return "derived_or_auxiliary"
    return None


def _arrays_payload_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(left, right)
    )


def _contains_path_sequence(path: Path, sequence: tuple[str, ...]) -> bool:
    parts = _resolved_path(path).parts
    width = len(sequence)
    return any(tuple(parts[index : index + width]) == sequence for index in range(len(parts) - width + 1))


def _load_manifest(
    adapter_manifest: Mapping[str, Any] | PathLike,
    errors: list[str],
) -> tuple[dict[str, Any], str | None]:
    if isinstance(adapter_manifest, Mapping):
        return dict(adapter_manifest), None

    path = Path(adapter_manifest).expanduser()
    path_text = _path_text(path)
    if not path.is_file():
        errors.append(f"adapter_manifest_missing:{path_text}")
        return {}, path_text
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"adapter_manifest_invalid_json:{path_text}:{type(exc).__name__}")
        return {}, path_text
    if not isinstance(payload, dict):
        errors.append(f"adapter_manifest_not_object:{path_text}")
        return {}, path_text
    return payload, path_text


def _validate_canonical_transform(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append("adapter_canonical_transform_missing_or_invalid")
        return

    applied = value.get("applied")
    kind = value.get("kind")
    matrix = value.get("matrix")
    if not isinstance(applied, bool):
        errors.append("adapter_canonical_transform_applied_not_boolean")
    if kind not in {"identity", "shared_se3"}:
        errors.append(f"adapter_canonical_transform_kind_invalid:{kind}")
    matrix_array: np.ndarray | None = None
    try:
        matrix_array = validate_rigid_transform(matrix)
    except (TypeError, ValueError) as exc:
        errors.append(f"adapter_canonical_transform_not_se3:{exc}")
    if (
        matrix_array is not None
        and kind == "identity"
        and not np.allclose(matrix_array, np.eye(4), atol=1e-9, rtol=0.0)
    ):
        errors.append("adapter_canonical_transform_identity_matrix_not_identity")

    if applied is True and kind != "shared_se3":
        errors.append("adapter_canonical_transform_applied_without_shared_se3")
    if applied is False and kind != "identity":
        errors.append("adapter_canonical_transform_unapplied_not_identity")

    if applied is True:
        applies_to = value.get("applies_to")
        required = {
            "hand_joints",
            "hand_vertices",
            "mano_global_orient",
            "mano_global_translation",
            "object_pose",
            "contact_points_generated_post_transform",
            "object_mesh_world_via_object_pose",
        }
        actual = set(applies_to) if isinstance(applies_to, (list, tuple, set)) else set()
        missing = sorted(required - actual)
        if missing:
            errors.append("adapter_canonical_transform_incomplete_targets:" + ",".join(missing))


def _validate_shared_se3_mano_global_pose(
    canonical_transform: Any,
    ego_hand_geometry: Any,
    errors: list[str],
) -> None:
    """Verify recorded MANO global pose evidence for an applied shared SE(3)."""

    if not isinstance(canonical_transform, Mapping):
        return
    if canonical_transform.get("applied") is not True:
        return
    try:
        matrix = validate_rigid_transform(canonical_transform.get("matrix"))
    except (TypeError, ValueError):
        # The canonical-transform validator already records the primary error.
        return

    if not isinstance(ego_hand_geometry, Mapping):
        errors.append("adapter_ego_hand_geometry_missing_for_shared_se3")
        return
    report = ego_hand_geometry.get("mano_global_pose_transform")
    if not isinstance(report, Mapping):
        errors.append("adapter_mano_global_pose_transform_missing_or_invalid")
        return
    if report.get("status") != "ok":
        errors.append(
            "adapter_mano_global_pose_transform_not_ok:" + str(report.get("status"))
        )
    if report.get("applied") is not True:
        errors.append("adapter_mano_global_pose_transform_not_applied")

    frame_count = report.get("frame_count")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        errors.append("adapter_mano_global_pose_transform_frame_count_invalid")

    try:
        common_rotation = np.asarray(report.get("common_rotation"), dtype=np.float64)
        common_translation = np.asarray(
            report.get("common_translation"), dtype=np.float64
        )
    except (TypeError, ValueError):
        common_rotation = np.zeros((0, 0), dtype=np.float64)
        common_translation = np.zeros((0,), dtype=np.float64)
    if (
        common_rotation.shape != (3, 3)
        or not np.isfinite(common_rotation).all()
        or not np.allclose(common_rotation, matrix[:3, :3], atol=2.0e-6, rtol=0.0)
    ):
        errors.append("adapter_mano_global_pose_common_rotation_mismatch")
    if (
        common_translation.shape != (3,)
        or not np.isfinite(common_translation).all()
        or not np.allclose(
            common_translation, matrix[:3, 3], atol=1.0e-6, rtol=0.0
        )
    ):
        errors.append("adapter_mano_global_pose_common_translation_mismatch")

    transformed_fields_value = report.get("transformed_fields")
    transformed_fields = (
        {str(field) for field in transformed_fields_value}
        if isinstance(transformed_fields_value, (list, tuple, set))
        else set()
    )
    sides = report.get("sides")
    if not isinstance(sides, Mapping):
        errors.append("adapter_mano_global_pose_sides_missing_or_invalid")
        return

    geometry_side_count = 0
    for side in ("left", "right"):
        side_report = sides.get(side)
        if not isinstance(side_report, Mapping):
            errors.append(f"adapter_mano_global_pose_{side}_side_missing_or_invalid")
            continue
        geometry_present = side_report.get("geometry_present")
        if not isinstance(geometry_present, bool):
            errors.append(f"adapter_mano_global_pose_{side}_geometry_present_invalid")
            continue
        if not geometry_present:
            continue
        geometry_side_count += 1

        rotation_field = side_report.get("rotation_field")
        if not isinstance(rotation_field, str) or not rotation_field:
            errors.append(f"adapter_mano_global_pose_{side}_rotation_field_missing")
        elif rotation_field not in transformed_fields:
            errors.append(f"adapter_mano_global_pose_{side}_rotation_not_transformed")

        translation_fields_value = side_report.get("translation_fields")
        translation_fields = (
            [str(field) for field in translation_fields_value]
            if isinstance(translation_fields_value, (list, tuple))
            else []
        )
        if not translation_fields:
            errors.append(f"adapter_mano_global_pose_{side}_translation_fields_missing")
        elif any(field not in transformed_fields for field in translation_fields):
            errors.append(f"adapter_mano_global_pose_{side}_translation_not_transformed")

        rotation_frame_count = side_report.get("rotation_frame_count")
        if isinstance(frame_count, int) and rotation_frame_count != frame_count:
            errors.append(f"adapter_mano_global_pose_{side}_frame_count_mismatch")

        for metric, threshold in (
            ("rotation_matrix_max_abs_error", 2.0e-6),
            ("translation_max_abs_error_m", 1.0e-6),
        ):
            try:
                value = float(side_report.get(metric))
            except (TypeError, ValueError):
                value = float("inf")
            if not np.isfinite(value) or value > threshold:
                errors.append(
                    f"adapter_mano_global_pose_{side}_{metric}_invalid"
                )
    if geometry_side_count <= 0:
        errors.append("adapter_mano_global_pose_no_geometry_sides")


def _validate_camera_intrinsics_binding(value: Any, errors: list[str]) -> None:
    """Validate recorded camera evidence without trusting its claimed status."""

    if not isinstance(value, Mapping):
        errors.append("adapter_camera_intrinsics_binding_missing_or_invalid")
        return
    if value.get("schema_version") != CAMERA_INTRINSICS_BINDING_SCHEMA_VERSION:
        errors.append(
            "adapter_camera_intrinsics_binding_schema_invalid:"
            + str(value.get("schema_version"))
        )
    policy = value.get("policy")
    if policy not in {
        CAMERA_INTRINSICS_BINDING_POLICY,
        CAMERA_INTRINSICS_NORMALIZATION_POLICY,
    }:
        errors.append("adapter_camera_intrinsics_binding_policy_invalid")
    if policy == CAMERA_INTRINSICS_BINDING_POLICY and value.get("geometry_modified") is not False:
        errors.append("adapter_camera_intrinsics_binding_geometry_modified_or_unrecorded")
    if policy == CAMERA_INTRINSICS_NORMALIZATION_POLICY and value.get("geometry_modified") is not True:
        errors.append("adapter_camera_intrinsics_normalization_not_recorded")

    _validate_camera_source_bindings(value, errors)

    rebuilt = build_camera_intrinsics_binding(
        value.get("object_camera"), value.get("hand_camera")
    )
    for label in ("object_camera", "hand_camera"):
        recorded = value.get(label)
        rebuilt_entry = rebuilt.get(label)
        recorded_fingerprint = (
            recorded.get("fingerprint") if isinstance(recorded, Mapping) else None
        )
        expected_fingerprint = (
            rebuilt_entry.get("fingerprint")
            if isinstance(rebuilt_entry, Mapping)
            else None
        )
        if not isinstance(recorded_fingerprint, str) or not recorded_fingerprint:
            errors.append(f"adapter_{label}_fingerprint_missing")
        elif recorded_fingerprint != expected_fingerprint:
            errors.append(f"adapter_{label}_fingerprint_mismatch")

    if policy == CAMERA_INTRINSICS_NORMALIZATION_POLICY:
        normalization = value.get("normalization")
        if not isinstance(normalization, Mapping):
            errors.append("adapter_camera_intrinsics_normalization_missing")
            return
        object_entry = value.get("object_camera")
        hand_entry = value.get("hand_camera")
        object_camera = object_entry.get("intrinsics") if isinstance(object_entry, Mapping) else None
        hand_camera = hand_entry.get("intrinsics") if isinstance(hand_entry, Mapping) else None
        try:
            expected_matrix = camera_ray_depth_transform(hand_camera, object_camera)
            recorded_matrix = np.asarray(normalization.get("matrix"), dtype=np.float64)
        except (TypeError, ValueError):
            expected_matrix = np.zeros((3, 3), dtype=np.float64)
            recorded_matrix = np.zeros((0, 0), dtype=np.float64)
        if recorded_matrix.shape != (3, 3) or not np.allclose(
            recorded_matrix, expected_matrix, rtol=0.0, atol=1.0e-12
        ):
            errors.append("adapter_camera_intrinsics_normalization_matrix_invalid")
        if normalization.get("source_camera_fingerprint") != (
            hand_entry.get("fingerprint") if isinstance(hand_entry, Mapping) else None
        ):
            errors.append("adapter_camera_intrinsics_normalization_source_mismatch")
        if normalization.get("target_camera_fingerprint") != (
            object_entry.get("fingerprint") if isinstance(object_entry, Mapping) else None
        ):
            errors.append("adapter_camera_intrinsics_normalization_target_mismatch")
        output_path_value = normalization.get("output_hand_npz")
        output_hash = normalization.get("output_hand_npz_sha256")
        output_path = Path(output_path_value).expanduser() if isinstance(output_path_value, str) else None
        if output_path is None or not output_path.is_file():
            errors.append("adapter_camera_intrinsics_normalized_hand_missing")
        else:
            if not isinstance(output_hash, str) or file_sha256(output_path) != output_hash:
                errors.append("adapter_camera_intrinsics_normalized_hand_hash_mismatch")
            try:
                if _npz_scalar_text(output_path, "camera_intrinsics_normalization_kind") != "ray_depth_source_to_egoinfinity":
                    errors.append("adapter_camera_intrinsics_normalized_hand_metadata_missing")
                with np.load(output_path, allow_pickle=False) as archive:
                    embedded_matrix = np.asarray(
                        archive["camera_intrinsics_normalization_matrix"],
                        dtype=np.float64,
                    )
                if embedded_matrix.shape != (3, 3) or not np.allclose(
                    embedded_matrix, expected_matrix, rtol=0.0, atol=1.0e-12
                ):
                    errors.append("adapter_camera_intrinsics_normalized_hand_matrix_mismatch")
            except (KeyError, OSError, TypeError, ValueError):
                errors.append("adapter_camera_intrinsics_normalized_hand_invalid")
        required_fields = {
            "left_vertices", "left_joints", "left_trans",
            "right_vertices", "right_joints", "right_trans",
        }
        if not required_fields.issubset(set(normalization.get("transformed_fields") or [])):
            errors.append("adapter_camera_intrinsics_normalized_fields_incomplete")
        try:
            reprojection_error = float(normalization.get("reprojection_error_px_max"))
            depth_error = float(normalization.get("depth_max_abs_error_m"))
        except (TypeError, ValueError):
            reprojection_error = depth_error = float("inf")
        # Float32 hand archives introduce sub-millipixel roundoff after the
        # analytically exact ray/depth transform.
        if not np.isfinite(reprojection_error) or reprojection_error > 1.0e-3:
            errors.append("adapter_camera_intrinsics_reprojection_error_invalid")
        if not np.isfinite(depth_error) or depth_error > 1.0e-10:
            errors.append("adapter_camera_intrinsics_depth_error_invalid")
        if normalization.get("object_geometry_modified") is not False:
            errors.append("adapter_camera_intrinsics_object_geometry_modified")
        if normalization.get("object_pose_modified") is not False:
            errors.append("adapter_camera_intrinsics_object_pose_modified")
        if normalization.get("hoi_translation_applied") is not False:
            errors.append("adapter_camera_intrinsics_hoi_translation_applied")
        if normalization.get("scale_fit_applied") is not False:
            errors.append("adapter_camera_intrinsics_scale_fit_applied")
        comparison = value.get("comparison")
        if not isinstance(comparison, Mapping) or comparison.get("matches") is not True:
            errors.append("adapter_camera_intrinsics_binding_comparison_not_matched")
        if value.get("status") != "ok":
            errors.append(
                "adapter_camera_intrinsics_binding_not_ok:" + str(value.get("status"))
            )
        return

    if value.get("status") != "ok":
        errors.append(
            "adapter_camera_intrinsics_binding_not_ok:" + str(value.get("status"))
        )
    if rebuilt.get("status") != "ok":
        errors.append(
            "adapter_camera_intrinsics_binding_recomputed_invalid:"
            + ",".join(str(item) for item in rebuilt.get("errors") or [])
        )
    comparison = value.get("comparison")
    if not isinstance(comparison, Mapping) or comparison.get("matches") is not True:
        errors.append("adapter_camera_intrinsics_binding_comparison_not_matched")


def _validate_manifest_provenance(
    manifest: Mapping[str, Any],
    *,
    trajectory: str,
    hand_source: str,
    errors: list[str],
) -> dict[str, Any]:
    provenance_keys = (
        "hand_source",
        "object_track_source",
        "object_mesh_source",
        "retarget_object_source",
        "canonical_transform",
        "hoi_refinement",
        "hoi_contact_alignment",
        "adapter_rigid_invariance",
        "camera_intrinsics_binding",
    )
    provenance = {key: manifest.get(key) for key in provenance_keys}

    schema_version = manifest.get("adapter_schema_version")
    if not isinstance(schema_version, int) or schema_version < 3:
        errors.append(f"adapter_schema_version_too_old:{schema_version}")
    if manifest.get("production_hoi_policy") != "preserve_source_relative_transform":
        errors.append("adapter_production_hoi_policy_invalid")
    retarget_input_qc = manifest.get("retarget_input_qc")
    if not isinstance(retarget_input_qc, Mapping):
        errors.append("adapter_retarget_input_qc_missing_or_invalid")
    elif retarget_input_qc.get("status") != "ok":
        errors.append(
            "adapter_retarget_input_qc_not_ok:" + str(retarget_input_qc.get("status"))
        )

    actual_hand_source = manifest.get("hand_source")
    if not isinstance(actual_hand_source, str) or not actual_hand_source.strip():
        errors.append("adapter_hand_source_missing")
    elif actual_hand_source != hand_source:
        errors.append(
            f"adapter_hand_source_mismatch:expected={hand_source}:actual={actual_hand_source}"
        )

    object_source_keys = (
        "object_track_source",
        "object_mesh_source",
        "retarget_object_source",
    )
    for key in object_source_keys:
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"adapter_{key}_missing")

    expected_object_source = {
        "egoinfinity": "egoinfinity",
        "do_as_i_do": "dai_native",
    }.get(trajectory)
    if expected_object_source is None:
        errors.append(f"object_provenance_policy_unsupported_trajectory:{trajectory}")
    else:
        for key in object_source_keys:
            value = manifest.get(key)
            if isinstance(value, str) and value.strip() and value != expected_object_source:
                errors.append(
                    f"adapter_{key}_mismatch:expected={expected_object_source}:actual={value}"
                )

    _validate_canonical_transform(manifest.get("canonical_transform"), errors)
    _validate_shared_se3_mano_global_pose(
        manifest.get("canonical_transform"),
        manifest.get("ego_hand_geometry"),
        errors,
    )

    # Ego object poses and AoE/HaWoR hands come from independent reconstruction
    # stacks.  A shared string label such as "camera" is insufficient: the
    # calibrated pinhole models must be bound and numerically compatible before
    # their 3D values can be treated as one HOI input.
    if trajectory == "egoinfinity" and hand_source == "aoe":
        _validate_camera_intrinsics_binding(
            manifest.get("camera_intrinsics_binding"), errors
        )

    hoi_refinement = manifest.get("hoi_refinement")
    if not isinstance(hoi_refinement, Mapping):
        errors.append("adapter_hoi_refinement_missing_or_invalid")
    else:
        if hoi_refinement.get("applied") is not False:
            errors.append("adapter_hoi_refinement_applied_or_unrecorded")
        applied_fields = sorted(
            str(key)
            for key, value in hoi_refinement.items()
            if str(key).endswith("_applied") and value is True
        )
        if applied_fields:
            errors.append(
                "adapter_hoi_refinement_component_applied:" + ",".join(applied_fields)
            )

    contact_alignment = manifest.get("hoi_contact_alignment")
    if not isinstance(contact_alignment, Mapping):
        errors.append("adapter_hoi_contact_alignment_missing_or_invalid")
    else:
        if contact_alignment.get("applied") is not False:
            errors.append("adapter_hoi_contact_alignment_applied_or_unrecorded")
        modified_fields = sorted(
            key
            for key in ("transforms_modified", "layout_modified")
            if contact_alignment.get(key) is True
        )
        if modified_fields:
            errors.append(
                "adapter_hoi_contact_alignment_modified_production_input:"
                + ",".join(modified_fields)
            )

    rigid_invariance = manifest.get("adapter_rigid_invariance")
    if not isinstance(rigid_invariance, Mapping):
        errors.append("adapter_rigid_invariance_missing_or_invalid")
    else:
        if rigid_invariance.get("status") != "ok":
            errors.append(
                "adapter_rigid_invariance_not_ok:"
                + str(rigid_invariance.get("status"))
            )
        if rigid_invariance.get("true_pre_post_hand_comparison") is not True:
            errors.append("adapter_rigid_invariance_not_true_pre_post_hand_comparison")

    return provenance


def validate_route_input(
    *,
    adapter_manifest: Mapping[str, Any] | PathLike,
    source_keypoints: PathLike,
    processed_keypoints: PathLike | None,
    processed_backend: str,
    source_object_mesh: PathLike | None = None,
    processed_object_mesh: PathLike | None = None,
    trajectory: str,
    hand_source: str,
    task: str | None = None,
    hand_type: str | None = None,
    data_id: int | str | None = None,
) -> dict[str, Any]:
    """Validate exact-cell DAI inputs and their backend copies.

    The function never raises for missing or malformed run artifacts; those are
    reported as an ``invalid`` result.  Programmer errors in the route labels
    (empty labels or labels containing path separators) still raise
    ``ValueError`` through :func:`expected_cell_key`.  Both production backends
    require exact source and processed visual meshes; ``processed_backend`` is
    explicit so mesh presence can never silently select the SPIDER path policy.
    """

    exact_cell = expected_cell_key(trajectory, hand_source)
    spider_cell = f"traj_{trajectory}__hand_{hand_source}__retarget_spider"
    source_path = Path(source_keypoints).expanduser()
    processed_path = Path(processed_keypoints).expanduser() if processed_keypoints is not None else None
    source_mesh_path = (
        Path(source_object_mesh).expanduser() if source_object_mesh is not None else None
    )
    processed_mesh_path = (
        Path(processed_object_mesh).expanduser() if processed_object_mesh is not None else None
    )
    errors: list[str] = []
    processed_backend = str(processed_backend).strip()
    if processed_backend not in {"do_as_i_do", "spider"}:
        errors.append(f"processed_backend_invalid:{processed_backend or 'empty'}")
    task_label = str(task).strip() if task is not None else ""
    hand_type_label = str(hand_type).strip() if hand_type is not None else ""
    data_id_label = str(data_id).strip() if data_id is not None else ""
    route_identity_complete = bool(
        task_label and hand_type_label and data_id_label
    )
    if not route_identity_complete:
        errors.append("route_artifact_identity_incomplete")
    route_identity = (
        ("mano", hand_type_label, task_label, data_id_label)
        if route_identity_complete
        else None
    )

    if _is_unified_source_keypoints(source_path):
        errors.append("source_keypoints_unified_asset_forbidden")

    source_cell_root = _validate_exact_cell_source(
        source_path,
        exact_cell=exact_cell,
        artifact="source_keypoints",
        errors=errors,
    )
    _validate_relative_artifact_path(
        source_path,
        cell_root=source_cell_root,
        artifact="source_keypoints",
        required_prefix=("retargeting_outputs", "mano"),
        required_name="trajectory_keypoints.npz",
        errors=errors,
    )
    if route_identity is not None and not _contains_path_sequence(source_path, route_identity):
        errors.append(
            "source_keypoints_route_artifact_identity_mismatch:expected="
            + "/".join(route_identity)
        )

    processed_dai_root: Path | None = None
    if processed_path is None:
        errors.append("processed_keypoints_path_missing")
    else:
        processed_cells = _cell_components(processed_path)
        if processed_backend == "spider":
            resolved_processed = _resolved_path(processed_path)
            resolved_processed_cells = _cell_components(resolved_processed)
            if spider_cell not in processed_path.parts:
                found = ",".join(processed_cells) if processed_cells else "none"
                errors.append(
                    f"processed_keypoints_not_exact_spider_cell:expected={spider_cell}:found={found}"
                )
            unexpected = sorted(set(processed_cells) - {spider_cell})
            if unexpected:
                errors.append("processed_keypoints_cross_cell_path:" + ",".join(unexpected))
            if spider_cell not in resolved_processed.parts:
                found = ",".join(resolved_processed_cells) if resolved_processed_cells else "none"
                errors.append(
                    f"processed_keypoints_resolved_not_exact_spider_cell:expected={spider_cell}:found={found}"
                )
            unexpected_resolved = sorted(set(resolved_processed_cells) - {spider_cell})
            if unexpected_resolved:
                errors.append(
                    "processed_keypoints_resolved_cross_cell:"
                    + ",".join(unexpected_resolved)
                )
        elif processed_backend == "do_as_i_do":
            processed_dai_root = _validate_dai_processed_artifact(
                processed_path,
                exact_cell=exact_cell,
                source_cell_root=source_cell_root,
                artifact="processed_keypoints",
                required_prefix=(
                    "retargeting",
                    "do_as_i_do",
                    "trajectories",
                    "mano",
                ),
                required_name="trajectory_keypoints.npz",
                errors=errors,
            )
        if processed_path.name != "trajectory_keypoints.npz":
            errors.append(f"processed_keypoints_filename_invalid:{processed_path.name}")
        if route_identity is not None and not _contains_path_sequence(
            processed_path, route_identity
        ):
            errors.append(
                "processed_keypoints_route_artifact_identity_mismatch:expected="
                + "/".join(route_identity)
            )

    source_hash = _hash_file(source_path, label="source_keypoints", errors=errors)
    processed_hash = (
        _hash_file(processed_path, label="processed_keypoints", errors=errors)
        if processed_path is not None
        else None
    )

    hashes_match = bool(source_hash and processed_hash and source_hash == processed_hash)
    if source_hash is not None and processed_hash is not None and not hashes_match:
        errors.append("source_processed_sha256_mismatch")

    manifest, manifest_path = _load_manifest(adapter_manifest, errors)
    manifest_cell_root: Path | None = None
    manifest_resolved: str | None = None
    if manifest_path is not None:
        manifest_file = Path(manifest_path)
        manifest_cell_root = _validate_exact_cell_adapter_manifest(
            manifest_file,
            exact_cell=exact_cell,
            errors=errors,
        )
        manifest_resolved = _path_text(_resolved_path(manifest_file))

    mesh_validation_requested = True
    source_mesh_hash: str | None = None
    processed_mesh_hash: str | None = None
    mesh_hashes_match = False
    source_mesh_cell_root: Path | None = None
    processed_spider_root: Path | None = None
    processed_dai_mesh_root: Path | None = None
    if mesh_validation_requested:
        if source_mesh_path is None:
            errors.append("source_object_mesh_path_missing")
        else:
            source_mesh_cell_root = _validate_exact_cell_source(
                source_mesh_path,
                exact_cell=exact_cell,
                artifact="source_object_mesh",
                errors=errors,
            )
            _validate_relative_artifact_path(
                source_mesh_path,
                cell_root=source_mesh_cell_root,
                artifact="source_object_mesh",
                required_prefix=("retargeting_outputs", "assets", "objects"),
                required_name="visual.obj",
                errors=errors,
            )
            if route_identity is not None and not _contains_path_sequence(
                source_mesh_path, ("assets", "objects", task_label, "visual.obj")
            ):
                errors.append(
                    "source_object_mesh_route_artifact_identity_mismatch:expected_task="
                    + task_label
                )
            source_mesh_hash = _hash_file(
                source_mesh_path,
                label="source_object_mesh",
                errors=errors,
            )
        if processed_mesh_path is None:
            errors.append("processed_object_mesh_path_missing")
        else:
            processed_cells = _cell_components(processed_mesh_path)
            if processed_backend == "spider":
                resolved_processed_mesh = _resolved_path(processed_mesh_path)
                resolved_processed_cells = _cell_components(resolved_processed_mesh)
                if spider_cell not in processed_mesh_path.parts:
                    found = ",".join(processed_cells) if processed_cells else "none"
                    errors.append(
                        f"processed_object_mesh_not_exact_spider_cell:expected={spider_cell}:found={found}"
                    )
                unexpected = sorted(set(processed_cells) - {spider_cell})
                if unexpected:
                    errors.append("processed_object_mesh_cross_cell_path:" + ",".join(unexpected))
                if spider_cell not in resolved_processed_mesh.parts:
                    found = ",".join(resolved_processed_cells) if resolved_processed_cells else "none"
                    errors.append(
                        f"processed_object_mesh_resolved_not_exact_spider_cell:expected={spider_cell}:found={found}"
                    )
                unexpected_resolved = sorted(set(resolved_processed_cells) - {spider_cell})
                if unexpected_resolved:
                    errors.append(
                        "processed_object_mesh_resolved_cross_cell:"
                        + ",".join(unexpected_resolved)
                    )
                processed_spider_root = _cell_root(processed_mesh_path, spider_cell)
            elif processed_backend == "do_as_i_do":
                processed_dai_mesh_root = _validate_dai_processed_artifact(
                    processed_mesh_path,
                    exact_cell=exact_cell,
                    source_cell_root=source_cell_root,
                    artifact="processed_object_mesh",
                    required_prefix=(
                        "retargeting",
                        "do_as_i_do",
                        "assets",
                        "objects",
                        task_label,
                    ),
                    required_name="visual.obj",
                    errors=errors,
                )
            if processed_mesh_path.name != "visual.obj":
                errors.append(
                    f"processed_object_mesh_filename_invalid:{processed_mesh_path.name}"
                )
            if route_identity is not None and not _contains_path_sequence(
                processed_mesh_path, ("assets", "objects", task_label, "visual.obj")
            ):
                errors.append(
                    "processed_object_mesh_route_artifact_identity_mismatch:expected_task="
                    + task_label
                )
            processed_mesh_hash = _hash_file(
                processed_mesh_path,
                label="processed_object_mesh",
                errors=errors,
            )
        mesh_hashes_match = bool(
            source_mesh_hash
            and processed_mesh_hash
            and source_mesh_hash == processed_mesh_hash
        )
        if (
            source_mesh_hash is not None
            and processed_mesh_hash is not None
            and not mesh_hashes_match
        ):
            errors.append("source_processed_object_mesh_sha256_mismatch")

        if manifest_path is None:
            errors.append("adapter_manifest_path_required_for_mesh_provenance")
        exact_roots = [
            root
            for root in (source_cell_root, source_mesh_cell_root, manifest_cell_root)
            if root is not None
        ]
        if len(exact_roots) != 3 or len({str(root) for root in exact_roots}) != 1:
            errors.append(
                "exact_dai_cell_component_root_mismatch:keypoints="
                + str(source_cell_root)
                + ":mesh="
                + str(source_mesh_cell_root)
                + ":adapter="
                + str(manifest_cell_root)
            )

        if (
            processed_backend == "spider"
            and processed_path is not None
            and processed_mesh_path is not None
        ):
            processed_keypoint_root = _cell_root(processed_path, spider_cell)
            if (
                processed_keypoint_root is None
                or processed_spider_root is None
                or processed_keypoint_root != processed_spider_root
            ):
                errors.append(
                    "processed_spider_cell_component_root_mismatch:keypoints="
                    + str(processed_keypoint_root)
                    + ":mesh="
                    + str(processed_spider_root)
                )
        if (
            processed_backend == "do_as_i_do"
            and processed_dai_root is not None
            and processed_dai_mesh_root is not None
            and processed_dai_root != processed_dai_mesh_root
        ):
            errors.append(
                "processed_dai_cell_component_root_mismatch:keypoints="
                + str(processed_dai_root)
                + ":mesh="
                + str(processed_dai_mesh_root)
            )

    provenance = _validate_manifest_provenance(
        manifest,
        trajectory=trajectory,
        hand_source=hand_source,
        errors=errors,
    )

    return {
        "status": "invalid" if errors else "ok",
        "errors": errors,
        "trajectory": trajectory,
        "hand_source": hand_source,
        "processed_backend": processed_backend,
        "route_artifact_identity": {
            "task": task,
            "hand_type": hand_type,
            "data_id": data_id,
        },
        "expected_cell_key": exact_cell,
        "paths": {
            "adapter_manifest": manifest_path,
            "adapter_manifest_resolved": manifest_resolved,
            "source_keypoints": _path_text(source_path),
            "source_keypoints_resolved": _path_text(_resolved_path(source_path)),
            "processed_keypoints": _path_text(processed_path) if processed_path is not None else None,
            "processed_keypoints_resolved": (
                _path_text(_resolved_path(processed_path)) if processed_path is not None else None
            ),
            "source_object_mesh": (
                _path_text(source_mesh_path) if source_mesh_path is not None else None
            ),
            "source_object_mesh_resolved": (
                _path_text(_resolved_path(source_mesh_path))
                if source_mesh_path is not None
                else None
            ),
            "processed_object_mesh": (
                _path_text(processed_mesh_path) if processed_mesh_path is not None else None
            ),
            "processed_object_mesh_resolved": (
                _path_text(_resolved_path(processed_mesh_path))
                if processed_mesh_path is not None
                else None
            ),
            "exact_dai_cell_root": str(source_cell_root) if source_cell_root is not None else None,
            "processed_spider_cell_root": (
                str(processed_spider_root) if processed_spider_root is not None else None
            ),
            "processed_dai_cell_root": (
                str(processed_dai_root) if processed_dai_root is not None else None
            ),
        },
        "provenance": provenance,
        "hash": {
            "algorithm": "sha256",
            "source_keypoints": source_hash,
            "processed_keypoints": processed_hash,
            "matches": hashes_match,
            "source_object_mesh": source_mesh_hash,
            "processed_object_mesh": processed_mesh_hash,
            "object_mesh_matches": mesh_hashes_match,
            "keypoints": {
                "source": source_hash,
                "processed": processed_hash,
                "matches": hashes_match,
            },
            "object_mesh": {
                "source": source_mesh_hash,
                "processed": processed_mesh_hash,
                "matches": mesh_hashes_match,
                "validated": mesh_validation_requested,
            },
        },
    }


def validate_spider_postprocess_semantics(
    *,
    source_keypoints: PathLike,
    processed_keypoints: PathLike,
    trajectory: str,
    hand_source: str,
    task: str,
    hand_type: str,
    data_id: int | str,
    strict_atol: float = 1.0e-12,
    min_source_contact_frames: int = 3,
) -> dict[str, Any]:
    """Fail closed if SPIDER preprocessing changes core hand/object motion.

    ``detect_contact.py`` may synthesize contact annotations only when the exact
    route input has fewer than ``min_source_contact_frames`` active frames.  A
    usable source contact track is protected just like wrist, fingertip, and
    object motion: silently replacing it with a geometry/AABB heuristic changes
    the HOI semantics consumed by MJWP.  Derived or auxiliary MANO arrays may
    still be added/removed.  NPZ container hashes are recorded but deliberately
    are not required to match because rewriting an archive changes its ZIP
    metadata and field inventory.  Each core array instead receives an
    independent payload hash and strict numerical comparison.

    The caller must invoke this after contact detection and before XML
    generation, IK, or simulation.
    """

    if strict_atol < 0.0 or not np.isfinite(strict_atol):
        raise ValueError("strict_atol must be finite and non-negative")
    if min_source_contact_frames < 0:
        raise ValueError("min_source_contact_frames must be non-negative")

    exact_cell = expected_cell_key(trajectory, hand_source)
    spider_cell = f"traj_{trajectory}__hand_{hand_source}__retarget_spider"
    source_path = Path(source_keypoints).expanduser()
    processed_path = Path(processed_keypoints).expanduser()
    errors: list[str] = []

    if not task or not hand_type or data_id is None:
        errors.append("postprocess_route_artifact_identity_incomplete")
    route_sequence = ("mano", str(hand_type), str(task), str(data_id))

    if _is_unified_source_keypoints(source_path):
        errors.append("postprocess_source_keypoints_unified_asset_forbidden")
    source_cell_root = _validate_exact_cell_source(
        source_path,
        exact_cell=exact_cell,
        artifact="postprocess_source_keypoints",
        errors=errors,
    )
    _validate_relative_artifact_path(
        source_path,
        cell_root=source_cell_root,
        artifact="postprocess_source_keypoints",
        required_prefix=("retargeting_outputs", "mano"),
        required_name="trajectory_keypoints.npz",
        errors=errors,
    )
    if not _contains_path_sequence(source_path, route_sequence):
        errors.append(
            "postprocess_source_route_artifact_identity_mismatch:expected="
            + "/".join(route_sequence)
        )

    processed_cells = _cell_components(processed_path)
    resolved_processed = _resolved_path(processed_path)
    resolved_processed_cells = _cell_components(resolved_processed)
    if spider_cell not in processed_path.expanduser().absolute().parts:
        found = ",".join(processed_cells) if processed_cells else "none"
        errors.append(
            f"postprocess_processed_not_exact_spider_cell:expected={spider_cell}:found={found}"
        )
    unexpected = sorted(set(processed_cells) - {spider_cell})
    if unexpected:
        errors.append("postprocess_processed_cross_cell_path:" + ",".join(unexpected))
    if spider_cell not in resolved_processed.parts:
        found = ",".join(resolved_processed_cells) if resolved_processed_cells else "none"
        errors.append(
            "postprocess_processed_resolved_not_exact_spider_cell:"
            f"expected={spider_cell}:found={found}"
        )
    unexpected_resolved = sorted(set(resolved_processed_cells) - {spider_cell})
    if unexpected_resolved:
        errors.append(
            "postprocess_processed_resolved_cross_cell:"
            + ",".join(unexpected_resolved)
        )
    processed_spider_root = _cell_root(processed_path, spider_cell)
    if processed_spider_root is None:
        errors.append(f"postprocess_processed_spider_cell_root_unresolved:{spider_cell}")
    if processed_path.name != "trajectory_keypoints.npz":
        errors.append(
            f"postprocess_processed_keypoints_filename_invalid:{processed_path.name}"
        )
    if not _contains_path_sequence(processed_path, route_sequence):
        errors.append(
            "postprocess_processed_route_artifact_identity_mismatch:expected="
            + "/".join(route_sequence)
        )

    source_arrays = _load_npz_arrays(
        source_path, label="postprocess_source_keypoints", errors=errors
    )
    processed_arrays = _load_npz_arrays(
        processed_path, label="postprocess_processed_keypoints", errors=errors
    )
    source_hash = (
        _hash_file(source_path, label="postprocess_source_keypoints", errors=errors)
        if source_path.is_file()
        else None
    )
    processed_hash = (
        _hash_file(processed_path, label="postprocess_processed_keypoints", errors=errors)
        if processed_path.is_file()
        else None
    )

    core: dict[str, Any] = {}
    for name in SPIDER_CORE_HOI_ARRAYS:
        source = source_arrays.get(name)
        processed = processed_arrays.get(name)
        if source is None:
            errors.append(f"postprocess_core_source_field_missing:{name}")
        if processed is None:
            errors.append(f"postprocess_core_processed_field_missing:{name}")
        if source is None or processed is None:
            core[name] = {
                "status": "invalid",
                "source_present": source is not None,
                "processed_present": processed is not None,
            }
            continue

        shape_match = source.shape == processed.shape
        dtype_match = source.dtype == processed.dtype
        source_numeric = bool(np.issubdtype(source.dtype, np.number))
        processed_numeric = bool(np.issubdtype(processed.dtype, np.number))
        source_finite = bool(source_numeric and np.isfinite(source).all())
        processed_finite = bool(processed_numeric and np.isfinite(processed).all())
        source_payload_hash = _array_sha256(source)
        processed_payload_hash = _array_sha256(processed)
        payload_bytes_match = bool(
            dtype_match
            and shape_match
            and source_payload_hash == processed_payload_hash
        )

        strict_numeric_match = False
        max_abs_error: float | None = None
        if shape_match and dtype_match and source_finite and processed_finite:
            if source.size:
                max_abs_error = float(
                    np.max(np.abs(source.astype(np.complex128) - processed.astype(np.complex128)))
                )
            else:
                max_abs_error = 0.0
            if np.issubdtype(source.dtype, np.inexact):
                strict_numeric_match = bool(
                    np.allclose(
                        source,
                        processed,
                        atol=float(strict_atol),
                        rtol=0.0,
                        equal_nan=False,
                    )
                )
            else:
                strict_numeric_match = bool(np.array_equal(source, processed))

        if not shape_match:
            errors.append(
                f"postprocess_core_shape_mismatch:{name}:"
                f"source={source.shape}:processed={processed.shape}"
            )
        if not dtype_match:
            errors.append(
                f"postprocess_core_dtype_mismatch:{name}:"
                f"source={source.dtype}:processed={processed.dtype}"
            )
        if not source_numeric or not processed_numeric:
            errors.append(f"postprocess_core_non_numeric:{name}")
        if not source_finite:
            errors.append(f"postprocess_core_source_nonfinite:{name}")
        if not processed_finite:
            errors.append(f"postprocess_core_processed_nonfinite:{name}")
        if (
            shape_match
            and dtype_match
            and source_finite
            and processed_finite
            and not strict_numeric_match
        ):
            errors.append(
                f"postprocess_core_value_mismatch:{name}:"
                f"max_abs_error={max_abs_error}:atol={strict_atol}"
            )

        field_ok = bool(
            shape_match
            and dtype_match
            and source_finite
            and processed_finite
            and strict_numeric_match
        )
        core[name] = {
            "status": "ok" if field_ok else "invalid",
            "shape": {
                "source": list(source.shape),
                "processed": list(processed.shape),
                "matches": shape_match,
            },
            "dtype": {
                "source": str(source.dtype),
                "processed": str(processed.dtype),
                "matches": dtype_match,
            },
            "finite": {
                "source": source_finite,
                "processed": processed_finite,
            },
            "payload_sha256": {
                "source": source_payload_hash,
                "processed": processed_payload_hash,
                "bytes_match": payload_bytes_match,
            },
            "strict_numeric_match": strict_numeric_match,
            "strict_atol": float(strict_atol),
            "rtol": 0.0,
            "max_abs_error": max_abs_error,
        }

    source_fields = set(source_arrays)
    processed_fields = set(processed_arrays)
    selected_sides = (
        ("left", "right") if hand_type == "bimanual" else (str(hand_type),)
    )
    contact_provenance: dict[str, Any] = {}
    protected_contact_fields: set[str] = set()
    for side in selected_sides:
        contact_name = f"contact_{side}"
        source_contact = source_arrays.get(contact_name)
        active_frames = 0
        if source_contact is not None and source_contact.ndim >= 1:
            flattened = np.asarray(source_contact).reshape(source_contact.shape[0], -1)
            active_frames = int(np.any(flattened > 0.5, axis=1).sum())
        preserve_required = bool(
            source_contact is not None
            and active_frames >= int(min_source_contact_frames)
        )
        side_fields = tuple(
            name
            for name in (
                contact_name,
                f"contact_pos_{side}",
                f"contact_world_{side}",
            )
            if name in source_arrays
        )
        if preserve_required:
            protected_contact_fields.update(side_fields)
        contact_provenance[side] = {
            "source_active_frames": active_frames,
            "min_source_contact_frames": int(min_source_contact_frames),
            "policy": (
                "preserve_exact_route_source"
                if preserve_required
                else "fallback_recompute_allowed"
            ),
            "protected_fields": list(side_fields) if preserve_required else [],
        }

    noncore_fields = sorted((source_fields | processed_fields) - set(SPIDER_CORE_HOI_ARRAYS))
    allowed_changes: list[dict[str, str]] = []
    unexpected_changes: list[dict[str, str]] = []
    for name in noncore_fields:
        if name not in source_arrays:
            change = "added"
        elif name not in processed_arrays:
            change = "removed"
        elif not _arrays_payload_equal(source_arrays[name], processed_arrays[name]):
            change = "modified"
        else:
            continue
        category = _spider_mutable_field_category(name)
        record = {"field": name, "change": change, "category": category or "unclassified"}
        if name in protected_contact_fields:
            unexpected_changes.append(record)
            errors.append(f"postprocess_source_contact_modified:{name}:{change}")
        elif category is None:
            unexpected_changes.append(record)
            errors.append(f"postprocess_unexpected_noncore_change:{name}:{change}")
        else:
            allowed_changes.append(record)

    return {
        "schema_version": 2,
        "check": "spider_post_preprocessing_core_hoi_semantics",
        "stage": "after_detect_contact_before_simulation",
        "status": "invalid" if errors else "ok",
        "production_allowed": not errors,
        "errors": errors,
        "contact_provenance": contact_provenance,
        "trajectory": trajectory,
        "hand_source": hand_source,
        "route_artifact_identity": {
            "task": task,
            "hand_type": hand_type,
            "data_id": data_id,
            "expected_dai_cell_key": exact_cell,
            "expected_spider_cell_key": spider_cell,
        },
        "paths": {
            "source_keypoints": _path_text(source_path),
            "source_keypoints_resolved": _path_text(_resolved_path(source_path)),
            "processed_keypoints": _path_text(processed_path),
            "processed_keypoints_resolved": _path_text(_resolved_path(processed_path)),
            "exact_dai_cell_root": (
                str(source_cell_root) if source_cell_root is not None else None
            ),
            "processed_spider_cell_root": (
                str(processed_spider_root) if processed_spider_root is not None else None
            ),
        },
        "archive_sha256": {
            "source": source_hash,
            "processed": processed_hash,
            "matches": bool(source_hash and processed_hash and source_hash == processed_hash),
            "match_required": False,
            "reason": "detect_contact_may_rewrite_contact_and_derived_fields",
        },
        "core_arrays": core,
        "field_inventory": {
            "source": sorted(source_fields),
            "processed": sorted(processed_fields),
            "added": sorted(processed_fields - source_fields),
            "removed": sorted(source_fields - processed_fields),
        },
        "allowed_noncore_changes": allowed_changes,
        "unexpected_noncore_changes": unexpected_changes,
        "policy": {
            "required_core_arrays": list(SPIDER_CORE_HOI_ARRAYS),
            "strict_atol": float(strict_atol),
            "rtol": 0.0,
            "allowed_contact_prefixes": list(_SPIDER_ALLOWED_CONTACT_PREFIXES),
            "allowed_derived_prefixes": list(_SPIDER_ALLOWED_DERIVED_PREFIXES),
            "allowed_derived_fields": sorted(_SPIDER_ALLOWED_DERIVED_FIELDS),
        },
    }
