#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoe_retarget_lab.alignment import (  # noqa: E402
    load_json_object as load_json_optional,
    resolve_manifest_path,
    spider_alignment_manifest,
    valid_dai_alignment as _valid_dai_alignment,
    valid_spider_alignment as _valid_spider_alignment,
)
from aoe_retarget_lab.route_input_qc import validate_route_input  # noqa: E402
from aoe_retarget_lab.tracking_quality import validate_hard_tracking_admission  # noqa: E402


def cell_key(trajectory_6dof: str, hand_source: str, retargeting: str) -> str:
    return f"traj_{trajectory_6dof}__hand_{hand_source}__retarget_{retargeting}"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def path_contains(path: Path | None, needle: str) -> bool:
    if path is None:
        return False
    text = str(path)
    try:
        text = f"{text} {path.resolve()}"
    except OSError:
        pass
    return needle in text


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def obj_extent_summary(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"available": False, "reason": "missing"}
    vertices = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except Exception as exc:
        return {"available": False, "reason": f"read_error:{exc}"}
    if not vertices:
        return {"available": False, "reason": "no_vertices"}
    arr = np.asarray(vertices, dtype=np.float64)
    extent = np.ptp(arr, axis=0)
    return {
        "available": True,
        "path": str(path),
        "vertices": int(arr.shape[0]),
        "extent": [float(x) for x in extent],
        "diag": float(np.linalg.norm(extent)),
        "center": [float(x) for x in arr.mean(axis=0)],
    }


def link_or_copy(src: Path | None, dst: Path, mode: str, root: Path) -> str | None:
    if src is None or not src.exists():
        return None
    if dst.exists() or dst.is_symlink():
        try:
            if src.resolve() == dst.resolve():
                return rel(dst, root)
        except FileNotFoundError:
            pass
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    return rel(dst, root)


def first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            if path.is_file() and path.suffix.lower() == ".mp4" and path.stat().st_size < 1024:
                continue
            return path
    return None


def valid_dai_alignment(robot_root: Path) -> bool:
    return _valid_dai_alignment(robot_root, REPO_ROOT)


def valid_spider_alignment(root: Path) -> bool:
    return _valid_spider_alignment(root, REPO_ROOT)


def bound_spider_artifact(root: Path, field: str) -> Path | None:
    """Resolve the exact canonical artifact already authenticated by alignment QC."""
    alignment = spider_alignment_manifest(root)
    if not isinstance(alignment, dict):
        return None
    return resolve_manifest_path(
        str(alignment.get(field) or ""),
        root,
        REPO_ROOT,
    )


def object_id_from_task(task: str) -> str:
    for marker in ["_bimanual", "_right", "_left"]:
        if marker in task:
            return task.split(marker, 1)[0]
    return task


def write_do_as_i_do_object_6dof(src: Path | None, dst: Path) -> str | None:
    if src is None or not src.exists():
        return None
    data = np.load(src, allow_pickle=False)
    payload = {}
    for key in ["qpos_obj_right", "qpos_obj_left", "qpos_obj"]:
        if key in data:
            payload[key] = data[key]
    if not payload:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **payload, source=str(src))
    return str(dst)


def do_as_i_do_roots(exp: Path, task: str, hand_type: str, retargeting: str, key: str) -> dict[str, Path | None]:
    shared = exp / "intermediates" / "do_as_i_do" / "retargeting_outputs"
    cell_root = exp / "intermediates" / "retargeting" / retargeting / key / "retargeting_outputs"
    root = cell_root if cell_root.exists() else shared
    mano = root / "mano" / hand_type / task
    robot_type = "sharpa"
    robot = root / robot_type / hand_type / task / "0"
    obj = root / "assets" / "objects" / task
    raw_dir = root.parent / "raw_dir"
    return {
        "root": root,
        "raw_dir": raw_dir,
        "mano": mano,
        "robot": robot,
        "object": obj,
        "trajectory_keypoints": mano / "0" / "trajectory_keypoints.npz",
        "task_info": mano / "task_info.json",
        "object_visual": obj / "visual.obj",
        "object_convex": obj / "convex",
        "robot_type": Path(robot_type),
    }


def egoinfinity_roots(exp: Path, robot: str = "g1") -> dict[str, Path]:
    clip = exp / "intermediates" / "egoinfinity" / "clip"
    return {
        "clip": clip,
        "retarget": clip / "retarget" / robot,
        "retarget_legacy": clip / f"retarget_{robot}",
        "samples": clip / "retarget_samples",
    }


def spider_roots(exp: Path, key: str) -> dict[str, Path]:
    root = exp / "intermediates" / "retargeting" / "spider" / key
    return {
        "root": root,
        "processed": root / "processed",
        "robot": root / "robot",
    }


def dai_indexed_keypoint_paths(
    retargeting_dir: Path,
    *,
    hand_type: str,
    task: str,
    data_id: int | str = 0,
) -> tuple[Path, Path]:
    """Return the route-canonical keypoint path and legacy generic alias.

    The canonical path carries the MANO hand/task/data identity even in copy
    mode, where resolving a generic alias cannot recover its source route.
    The flat alias remains available for existing asset consumers but is never
    used as the processed input to route QC.
    """

    identity = tuple(str(value).strip() for value in (hand_type, task, data_id))
    if any(not value or "/" in value for value in identity):
        raise ValueError("hand_type, task, and data_id must be non-empty path components")
    hand_label, task_label, data_label = identity
    trajectories = retargeting_dir / "trajectories"
    canonical = (
        trajectories
        / "mano"
        / hand_label
        / task_label
        / data_label
        / "trajectory_keypoints.npz"
    )
    return canonical, trajectories / "trajectory_keypoints.npz"


def dai_indexed_visual_mesh_path(retargeting_dir: Path, *, task: str) -> Path:
    task = str(task).strip()
    if not task or "/" in task:
        raise ValueError("task must be a non-empty path component")
    return retargeting_dir / "assets" / "objects" / task / "visual.obj"


ADAPTER_PROVENANCE_FIELDS = (
    "hand_source",
    "object_track_source",
    "object_mesh_source",
    "retarget_object_source",
    "canonical_transform",
    "hoi_refinement",
    "hoi_contact_alignment",
    "adapter_rigid_invariance",
)

ROUTE_PROVENANCE_FIELDS = (
    *ADAPTER_PROVENANCE_FIELDS,
    "route_input_qc",
    "post_preprocess_semantic_qc",
)


def _manifest_path(value: object, *, relative_to: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    for path in (relative_to / candidate, REPO_ROOT / candidate):
        if path.exists():
            return path
    return relative_to / candidate


def _expected_raw_hoi_inputs(adapter_manifest: Path) -> tuple[Path | None, Path | None]:
    """Resolve the same exact raw hand/layout inputs used by the validator."""

    adapter = load_json_optional(adapter_manifest)
    if adapter is None:
        return None, None
    raw_dir = adapter_manifest.parent.resolve()
    raw_hand = first_existing(
        [raw_dir / "raw" / "all_hand_meshes.npz", raw_dir / "all_hand_meshes.npz"]
    )
    candidates = sorted(
        raw_dir.glob(
            "obj_tracking_out/*/combined_visualization/"
            "layout_camera_frame_optimized.json"
        )
    )
    config_path = raw_dir / "config.json"
    if config_path.is_file():
        config = load_json_optional(config_path)
        object_names = config.get("object_names") if isinstance(config, dict) else None
        if isinstance(object_names, list) and object_names:
            config_layout = (
                raw_dir
                / "obj_tracking_out"
                / str(object_names[0])
                / "combined_visualization"
                / "layout_camera_frame_optimized.json"
            )
            if config_layout.is_file():
                candidates = [config_layout]
    layout = None
    recorded = adapter.get("layout")
    if isinstance(recorded, str) and recorded.strip():
        candidate = Path(recorded).expanduser()
        if not candidate.is_absolute():
            candidate = raw_dir / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(raw_dir)
            contained = True
        except ValueError:
            contained = False
        if (
            contained
            and candidate.is_file()
            and (not candidates or candidate in {path.resolve() for path in candidates})
        ):
            layout = candidate
    if layout is None:
        if len(candidates) == 1:
            layout = candidates[0].resolve()
    return raw_hand, layout


def validate_raw_to_processed_hoi_binding(
    report: object,
    *,
    adapter_manifest: Path,
    processed_keypoints: Path,
    trajectory: str,
    hand_source: str,
    hand_type: str,
) -> list[str]:
    """Verify a raw-to-processed report against the exact DAI cell files.

    Status flags alone are not admissible: every source/processed artifact is
    path- and SHA-bound, and the route/hand identity must match the cell being
    indexed.  This also prevents SPIDER from accepting a report copied from a
    different DAI hand-source route.
    """

    errors: list[str] = []
    if not isinstance(report, dict):
        return ["missing raw-to-processed HOI invariance report"]
    if report.get("schema_version") != 3:
        errors.append("raw-to-processed HOI invariance schema_version is not 3")
    if report.get("full_relative_se3_checked") is not True:
        errors.append("raw-to-processed HOI invariance omitted full relative SE(3)")

    expected_sides = ["left", "right"] if hand_type == "bimanual" else [hand_type]
    expected_identity = {
        "trajectory_6dof": trajectory,
        "hand_source": hand_source,
        "hand_type": hand_type,
    }
    binding = report.get("route_binding")
    if not isinstance(binding, dict):
        errors.append("raw-to-processed HOI invariance has no route_binding")
        binding = {}
    for field, expected in expected_identity.items():
        if report.get(field) != expected:
            errors.append(
                f"raw-to-processed report {field} mismatch: "
                f"expected={expected} actual={report.get(field)}"
            )
        if binding.get(field) != expected:
            errors.append(
                f"raw-to-processed binding {field} mismatch: "
                f"expected={expected} actual={binding.get(field)}"
            )
    if binding.get("requested_sides") != expected_sides:
        errors.append("raw-to-processed binding requested_sides does not match hand_type")
    if report.get("joint_fit_components") != ["object", *expected_sides]:
        errors.append("raw-to-processed report did not jointly fit object and all hands")

    raw_hand, raw_layout = _expected_raw_hoi_inputs(adapter_manifest)
    raw_inputs = binding.get("raw_inputs")
    if not isinstance(raw_inputs, dict):
        errors.append("raw-to-processed binding has no raw_inputs")
        raw_inputs = {}
    expected_files = {
        "adapter_manifest": (binding.get("adapter_manifest"), adapter_manifest),
        "raw hand NPZ": (raw_inputs.get("hand_npz"), raw_hand),
        "raw object layout": (raw_inputs.get("object_layout"), raw_layout),
        "processed keypoints": (
            binding.get("processed_keypoints"),
            processed_keypoints,
        ),
    }
    for label, (record, expected_path) in expected_files.items():
        if expected_path is None or not expected_path.is_file():
            errors.append(f"exact-cell {label} is missing")
            continue
        if not isinstance(record, dict):
            errors.append(f"raw-to-processed binding is missing {label}")
            continue
        bound_path_value = record.get("path")
        bound_sha = record.get("sha256")
        if not isinstance(bound_path_value, str) or not bound_path_value.strip():
            errors.append(f"raw-to-processed binding {label} path is missing")
            continue
        bound_path = Path(bound_path_value).expanduser()
        try:
            same_path = bound_path.resolve() == expected_path.resolve()
        except OSError:
            same_path = False
        if not same_path:
            errors.append(f"raw-to-processed binding {label} is stale or cross-route")
            continue
        current_sha = file_sha256(expected_path)
        if bound_sha != current_sha:
            errors.append(f"raw-to-processed binding {label} SHA256 mismatch")

    # Retain the flat fields for human-readable reports, but require them to
    # agree with the authoritative nested binding so contradictions fail shut.
    flat_fields = {
        "adapter_manifest": ("adapter_manifest_sha256", adapter_manifest),
        "raw_hand_npz": ("raw_hand_sha256", raw_hand),
        "raw_object_layout": ("raw_object_layout_sha256", raw_layout),
        "processed_keypoints": ("processed_keypoints_sha256", processed_keypoints),
    }
    for path_field, (sha_field, expected_path) in flat_fields.items():
        if expected_path is None or not expected_path.is_file():
            continue
        value = report.get(path_field)
        try:
            same_path = (
                isinstance(value, str)
                and Path(value).expanduser().resolve() == expected_path.resolve()
            )
        except OSError:
            same_path = False
        if not same_path:
            errors.append(f"raw-to-processed flat {path_field} does not match exact cell")
        if report.get(sha_field) != file_sha256(expected_path):
            errors.append(f"raw-to-processed flat {sha_field} mismatch")

    if hand_type == "bimanual":
        consistency = report.get("processed_object_track_consistency")
        if not isinstance(consistency, dict) or consistency.get("required") is not True:
            errors.append("bimanual processed object-track consistency was not required")
        elif consistency.get("sides") != expected_sides:
            errors.append("bimanual processed object-track consistency sides mismatch")
    return errors


def exact_route_provenance(
    exp: Path,
    args: argparse.Namespace,
    key: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """Load provenance only from the exact cell's consumed route artifacts.

    DAI cells are sourced from their exact ``raw_dir/adapter_manifest.json``.
    SPIDER cells use their input/output manifests and the adapter path recorded
    there.  This intentionally never consults trajectory-level convenience
    manifests or ``source_trajectory_keypoints.npz`` aliases.
    """

    provenance: dict[str, object] = {field: None for field in ROUTE_PROVENANCE_FIELDS}
    evidence: dict[str, object] = {}
    if args.retargeting == "do_as_i_do":
        adapter_path = (
            exp
            / "intermediates"
            / "retargeting"
            / "do_as_i_do"
            / key
            / "raw_dir"
            / "adapter_manifest.json"
        )
        adapter = load_json_optional(adapter_path)
        evidence["adapter_manifest"] = str(adapter_path) if adapter_path.exists() else None
        evidence["adapter"] = adapter
        if adapter is not None:
            for field in ADAPTER_PROVENANCE_FIELDS:
                provenance[field] = adapter.get(field)
        route_qc_path = (
            exp
            / "assets"
            / "cells"
            / key
            / "retargeting"
            / "do_as_i_do"
            / "metadata"
            / "route_input_qc.json"
        )
        route_qc = load_json_optional(route_qc_path)
        evidence["route_input_qc_path"] = str(route_qc_path) if route_qc_path.exists() else None
        if route_qc is not None:
            provenance["route_input_qc"] = route_qc
        return provenance, evidence

    if args.retargeting == "spider":
        roots = spider_roots(exp, key)
        input_path = roots["root"] / "spider_input_manifest.json"
        output_path = roots["root"] / "spider_output_manifest.json"
        postprocess_qc_path = roots["root"] / "spider_postprocess_semantic_qc.json"
        input_manifest = load_json_optional(input_path)
        output_manifest = load_json_optional(output_path)
        postprocess_qc = load_json_optional(postprocess_qc_path)
        evidence.update({
            "spider_input_manifest_path": str(input_path) if input_path.exists() else None,
            "spider_output_manifest_path": str(output_path) if output_path.exists() else None,
            "spider_input_manifest": input_manifest,
            "spider_output_manifest": output_manifest,
            "post_preprocess_semantic_qc_path": (
                str(postprocess_qc_path) if postprocess_qc_path.exists() else None
            ),
            "post_preprocess_semantic_qc": postprocess_qc,
        })
        adapter_ref = None
        for manifest in (output_manifest, input_manifest):
            if isinstance(manifest, dict) and manifest.get("adapter_manifest"):
                adapter_ref = manifest.get("adapter_manifest")
                break
        adapter_path = _manifest_path(adapter_ref, relative_to=roots["root"])
        adapter = load_json_optional(adapter_path) if adapter_path is not None else None
        evidence["adapter_manifest"] = (
            str(adapter_path) if adapter_path is not None and adapter_path.exists() else None
        )
        evidence["adapter"] = adapter

        # The output/input manifests describe what SPIDER actually consumed;
        # the exact source adapter is the final fallback, never a generic
        # trajectory-level manifest.
        for field in ROUTE_PROVENANCE_FIELDS:
            for manifest in (output_manifest, input_manifest, adapter):
                if isinstance(manifest, dict) and manifest.get(field) is not None:
                    provenance[field] = manifest.get(field)
                    break
        if provenance["route_input_qc"] is None:
            route_qc_path = roots["root"] / "route_input_qc.json"
            route_qc = load_json_optional(route_qc_path)
            evidence["route_input_qc_path"] = (
                str(route_qc_path) if route_qc_path.exists() else None
            )
            provenance["route_input_qc"] = route_qc
        # The standalone report is the indexed quality artifact and therefore
        # the authoritative value.  Never let stale embedded manifest payloads
        # hide a missing or invalid standalone report; validation below also
        # requires both manifests to match this exact payload.
        provenance["post_preprocess_semantic_qc"] = postprocess_qc
        return provenance, evidence

    # Native Ego retargeting has no DAI source adapter.  Keep that distinction
    # explicit instead of fabricating adapter-derived metadata.
    provenance.update({
        "hand_source": args.hand_source,
        "object_track_source": "egoinfinity" if args.trajectory_6dof == "egoinfinity" else None,
        "object_mesh_source": "egoinfinity" if args.trajectory_6dof == "egoinfinity" else None,
        "retarget_object_source": "egoinfinity" if args.trajectory_6dof == "egoinfinity" else None,
    })
    evidence["adapter_manifest"] = None
    evidence["source_kind"] = "native_retarget_without_dai_adapter"
    return provenance, evidence


def validate_exact_route_provenance(
    provenance: dict[str, object],
    *,
    trajectory: str,
    hand_source: str,
    require_route_input_qc: bool = True,
    require_post_preprocess_semantic_qc: bool = False,
    task: str | None = None,
    hand_type: str | None = None,
    data_id: int | str | None = None,
    processed_backend: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if provenance.get("hand_source") != hand_source:
        errors.append(
            f"route hand_source mismatch: expected={hand_source} actual={provenance.get('hand_source')}"
        )
    expected_object_source = "egoinfinity" if trajectory == "egoinfinity" else "dai_native"
    for field in ("object_track_source", "object_mesh_source", "retarget_object_source"):
        value = provenance.get(field)
        if value != expected_object_source:
            errors.append(
                f"route {field} mismatch: expected={expected_object_source} actual={value}"
            )

    canonical = provenance.get("canonical_transform")
    if not isinstance(canonical, dict):
        errors.append("missing canonical_transform provenance")
    else:
        applied = canonical.get("applied")
        kind = canonical.get("kind")
        try:
            matrix = np.asarray(canonical.get("matrix"), dtype=np.float64)
        except (TypeError, ValueError):
            matrix = np.empty((0, 0), dtype=np.float64)
        if not isinstance(applied, bool):
            errors.append("canonical_transform applied flag is not boolean")
        if kind not in {"identity", "shared_se3"}:
            errors.append(f"canonical_transform kind is invalid: {kind}")
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            errors.append("canonical_transform matrix is missing, non-finite, or not 4x4")
        else:
            rotation = matrix[:3, :3]
            if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0):
                errors.append("canonical_transform matrix has an invalid homogeneous row")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0.0):
                errors.append("canonical_transform rotation is not orthonormal")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0.0):
                errors.append("canonical_transform rotation determinant is not +1")
            if kind == "identity" and not np.allclose(
                matrix, np.eye(4), atol=1e-8, rtol=0.0
            ):
                errors.append("identity canonical_transform matrix is not identity")
        if applied is False and kind != "identity":
            errors.append("unapplied canonical_transform is not identity")
        if applied is True and kind != "shared_se3":
            errors.append("applied canonical_transform is not shared_se3")
        if applied is True:
            required_targets = {
                "hand_joints",
                "hand_vertices",
                "object_pose",
                "contact_points_generated_post_transform",
                "object_mesh_world_via_object_pose",
            }
            targets = canonical.get("applies_to")
            actual_targets = set(targets) if isinstance(targets, list) else set()
            missing_targets = sorted(required_targets - actual_targets)
            if missing_targets:
                errors.append(
                    "shared canonical_transform omits HOI targets: "
                    + ",".join(missing_targets)
                )
    refinement = provenance.get("hoi_refinement")
    if not isinstance(refinement, dict):
        errors.append("missing hoi_refinement provenance")
    else:
        if refinement.get("applied") is not False:
            errors.append("HOI refinement was applied or is not explicitly disabled")
        applied_components = sorted(
            str(field)
            for field, value in refinement.items()
            if str(field).endswith("_applied") and value is True
        )
        if applied_components:
            errors.append(
                "HOI refinement component applied: " + ",".join(applied_components)
            )
    alignment = provenance.get("hoi_contact_alignment")
    if not isinstance(alignment, dict):
        errors.append("missing hoi_contact_alignment provenance")
    elif (
        alignment.get("applied") is not False
        or alignment.get("transforms_modified") is True
        or alignment.get("layout_modified") is True
    ):
        errors.append("HOI contact alignment modified production transforms")
    invariance = provenance.get("adapter_rigid_invariance")
    if not isinstance(invariance, dict):
        errors.append("missing adapter_rigid_invariance provenance")
    elif invariance.get("status") != "ok":
        errors.append("adapter_rigid_invariance is not ok")
    elif invariance.get("true_pre_post_hand_comparison") is not True:
        errors.append("adapter_rigid_invariance is not a true pre/post hand comparison")
    if require_route_input_qc:
        route_qc = provenance.get("route_input_qc")
        if not isinstance(route_qc, dict):
            errors.append("missing route_input_qc provenance")
        else:
            if route_qc.get("status") != "ok":
                errors.append("route_input_qc is not ok")
            if route_qc.get("hand_source") != hand_source:
                errors.append("route_input_qc hand_source does not match cell")
            if route_qc.get("expected_cell_key") != cell_key(trajectory, hand_source, "do_as_i_do"):
                errors.append("route_input_qc does not name the exact DAI source cell")
            if processed_backend is not None and route_qc.get("processed_backend") != processed_backend:
                errors.append("route_input_qc processed_backend does not match cell")
            if any(value is not None for value in (task, hand_type, data_id)):
                expected_identity = {
                    "task": task,
                    "hand_type": hand_type,
                    "data_id": data_id,
                }
                if route_qc.get("route_artifact_identity") != expected_identity:
                    errors.append("route_input_qc artifact identity does not match cell")
    if require_post_preprocess_semantic_qc:
        postprocess_qc = provenance.get("post_preprocess_semantic_qc")
        if not isinstance(postprocess_qc, dict):
            errors.append("missing post_preprocess_semantic_qc provenance")
        else:
            if postprocess_qc.get("status") != "ok":
                errors.append("post_preprocess_semantic_qc is not ok")
            if postprocess_qc.get("trajectory") != trajectory:
                errors.append("post_preprocess_semantic_qc trajectory does not match cell")
            if postprocess_qc.get("hand_source") != hand_source:
                errors.append("post_preprocess_semantic_qc hand_source does not match cell")
            if any(value is not None for value in (task, hand_type, data_id)):
                identity = postprocess_qc.get("route_artifact_identity")
                if not isinstance(identity, dict) or any(
                    identity.get(field) != expected
                    for field, expected in (
                        ("task", task),
                        ("hand_type", hand_type),
                        ("data_id", data_id),
                    )
                ):
                    errors.append(
                        "post_preprocess_semantic_qc artifact identity does not match cell"
                    )
    return errors


def validate_dai_raw_dir_preflight(
    raw_dir: Path,
    *,
    trajectory: str,
    hand_source: str,
) -> dict[str, object]:
    """Fail-closed preflight for the exact adapter consumed by official DAI.

    The generated DAI keypoints do not exist until ``launch.py`` runs, so this
    preflight intentionally validates the adapter's route identity, complete
    HOI provenance, and adapter-level input QC.  The source/processed keypoint
    hash comparison remains a post-launch gate in :func:`validate_route_input`.
    """

    raw_dir = raw_dir.expanduser()
    adapter_path = raw_dir / "adapter_manifest.json"
    adapter = load_json_optional(adapter_path)
    errors: list[str] = []
    if adapter is None:
        errors.append(f"missing or invalid exact raw-dir adapter manifest: {adapter_path}")
        adapter = {}

    route_provenance = {
        field: adapter.get(field)
        for field in ADAPTER_PROVENANCE_FIELDS
    }
    errors.extend(
        validate_exact_route_provenance(
            route_provenance,
            trajectory=trajectory,
            hand_source=hand_source,
            require_route_input_qc=False,
        )
    )

    schema_version = adapter.get("adapter_schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 3:
        errors.append(f"adapter_schema_version must be >=3: {schema_version}")
    if adapter.get("production_hoi_policy") != "preserve_source_relative_transform":
        errors.append("adapter production_hoi_policy does not preserve source relative transform")
    adapter_qc = adapter.get("retarget_input_qc")
    if not isinstance(adapter_qc, dict):
        errors.append("missing adapter retarget_input_qc")
    elif adapter_qc.get("status") != "ok":
        errors.append(f"adapter retarget_input_qc is not ok: {adapter_qc.get('status')}")

    return {
        "status": "invalid" if errors else "ok",
        "errors": errors,
        "trajectory_6dof": trajectory,
        "hand_source": hand_source,
        "expected_cell_key": cell_key(trajectory, hand_source, "do_as_i_do"),
        "raw_dir": str(raw_dir),
        "adapter_manifest": str(adapter_path),
        "adapter_retarget_input_qc": adapter_qc if isinstance(adapter_qc, dict) else None,
        "provenance": route_provenance,
    }


def manifest_admission_errors(manifest: dict, *, retargeting: str) -> list[str]:
    """Return hard indexing failures after the manifest has been written."""

    errors: list[str] = []
    missing = manifest.get("missing")
    if not isinstance(missing, list):
        errors.append("missing required-asset inventory")
    elif missing:
        errors.append("missing required indexed assets: " + ", ".join(map(str, missing)))

    validation = manifest.get("provenance_validation")
    if not isinstance(validation, dict):
        errors.append("missing provenance_validation report")
    elif validation.get("status") != "ok":
        details = validation.get("errors") or [validation.get("status")]
        errors.append("provenance_validation is not ok: " + "; ".join(map(str, details)))

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("missing manifest provenance")
        provenance = {}
    elif provenance.get("status") == "invalid":
        details = provenance.get("invalid_reasons") or ["unspecified"]
        errors.append("manifest provenance is invalid: " + "; ".join(map(str, details)))

    if retargeting in {"do_as_i_do", "spider"}:
        route_qc = provenance.get("route_input_qc")
        if not isinstance(route_qc, dict):
            errors.append("missing route_input_qc in exact route provenance")
        elif route_qc.get("status") != "ok":
            details = route_qc.get("errors") or [route_qc.get("status")]
            errors.append("route_input_qc is not ok: " + "; ".join(map(str, details)))
    if retargeting == "spider":
        postprocess_qc = provenance.get("post_preprocess_semantic_qc")
        if not isinstance(postprocess_qc, dict):
            errors.append("missing post_preprocess_semantic_qc in exact SPIDER provenance")
        elif postprocess_qc.get("status") != "ok":
            details = postprocess_qc.get("errors") or [postprocess_qc.get("status")]
            errors.append(
                "post_preprocess_semantic_qc is not ok: "
                + "; ".join(map(str, details))
            )
    return errors


def aoe_hand_candidates(
    exp: Path,
    task: str,
    hand_type: str,
    retargeting: str,
    key: str,
) -> list[Path]:
    candidates = [
        exp / "intermediates" / "egoinfinity" / "clip" / "aoe_hands.npz",
        exp
        / "intermediates"
        / "trajectory_6dof"
        / "do_as_i_do"
        / "reconstruction"
        / "clip"
        / "raw"
        / "all_hand_meshes.npz",
        exp
        / "intermediates"
        / "trajectory_6dof"
        / "egoinfinity"
        / "do_as_i_do_raw_dir_hand_aoe"
        / "raw"
        / "all_hand_meshes.npz",
        exp
        / "assets"
        / "trajectory_6dof"
        / "do_as_i_do"
        / task
        / "hand_meshes"
        / "all_hand_meshes.npz",
    ]
    if retargeting == "do_as_i_do":
        roots = do_as_i_do_roots(exp, task, hand_type, retargeting, key)
        candidates.insert(0, roots["raw_dir"] / "raw" / "all_hand_meshes.npz")
        candidates.insert(1, roots["raw_dir"] / "all_hand_meshes.npz")
    return candidates


def validate_index_provenance(exp: Path, args: argparse.Namespace, key: str) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    evidence: dict = {
        "trajectory_6dof": args.trajectory_6dof,
        "retargeting": args.retargeting,
    }
    if args.retargeting == "do_as_i_do":
        roots = do_as_i_do_roots(exp, args.task, args.hand_type, args.retargeting, key)
        adapter_path = (
            exp
            / "intermediates"
            / "retargeting"
            / "do_as_i_do"
            / key
            / "raw_dir"
            / "adapter_manifest.json"
        )
        exact_processed_keypoints = (
            exp
            / "intermediates"
            / "retargeting"
            / "do_as_i_do"
            / key
            / "retargeting_outputs"
            / "mano"
            / args.hand_type
            / args.task
            / "0"
            / "trajectory_keypoints.npz"
        )
        hand_selection_path = (
            exp
            / "logs"
            / f"do_as_i_do_hand_selection__{args.trajectory_6dof}__{args.hand_source}.json"
        )
        hand_selection = load_json_optional(hand_selection_path)
        object_mesh = roots["object_visual"]
        evidence["robot_root"] = str(roots["robot"])
        alignment_path = roots["robot"] / "mjwp_alignment_manifest.json"
        evidence["mjwp_alignment_manifest"] = str(alignment_path)
        evidence["mjwp_alignment"] = load_json_optional(alignment_path)
        evidence["mjwp_alignment_valid"] = valid_dai_alignment(roots["robot"])
        tracking_quality_path = roots["robot"] / "mjwp_object_tracking_quality.json"
        tracking_quality = load_json_optional(tracking_quality_path)
        hard_tracking_admission = validate_hard_tracking_admission(tracking_quality)
        preprocess_invariance_path = roots["trajectory_keypoints"].with_name(
            "hoi_preprocess_invariance.json"
        )
        preprocess_invariance = load_json_optional(preprocess_invariance_path)
        raw_to_processed_hoi_path = exact_processed_keypoints.with_name(
            "raw_to_processed_hoi_invariance.json"
        )
        raw_to_processed_hoi = load_json_optional(raw_to_processed_hoi_path)
        evidence["mjwp_object_tracking_quality"] = tracking_quality
        evidence["raw_to_processed_hoi_invariance"] = raw_to_processed_hoi
        evidence["hard_tracking_admission"] = hard_tracking_admission
        evidence["object_mesh"] = obj_extent_summary(object_mesh)
        evidence["retarget_hand_selection_report"] = (
            str(hand_selection_path) if hand_selection_path.exists() else None
        )
        evidence["retarget_hand_selection"] = hand_selection
        if not evidence["mjwp_alignment_valid"]:
            errors.append("missing or invalid Do-as-I-Do MJWP alignment manifest for retarget robot")
        if hard_tracking_admission.get("status") != "ok":
            errors.extend(
                "Do-as-I-Do hard tracking admission: " + str(error)
                for error in hard_tracking_admission.get("errors") or ["invalid"]
            )
        if not isinstance(raw_to_processed_hoi, dict):
            errors.append("missing exact-route raw-to-processed DAI HOI invariance")
        elif (
            raw_to_processed_hoi.get("status") != "ok"
            or raw_to_processed_hoi.get("production_eligible") is not True
            or raw_to_processed_hoi.get("cross_hand_source_comparison") is not False
        ):
            errors.append("exact-route raw-to-processed DAI HOI invariance is invalid")
        if isinstance(raw_to_processed_hoi, dict):
            errors.extend(
                "exact-route raw-to-processed DAI binding: " + error
                for error in validate_raw_to_processed_hoi_binding(
                    raw_to_processed_hoi,
                    adapter_manifest=adapter_path,
                    processed_keypoints=exact_processed_keypoints,
                    trajectory=args.trajectory_6dof,
                    hand_source=args.hand_source,
                    hand_type=args.hand_type,
                )
            )
        if hand_selection is None:
            warnings.append("missing trajectory-specific Do-as-I-Do hand selection report")
        elif hand_selection.get("selected_hand_type") != args.hand_type:
            errors.append(
                "indexed Do-as-I-Do hand type does not match the trajectory-specific selection report"
            )
        route_provenance, route_evidence = exact_route_provenance(exp, args, key)
        evidence["exact_route_provenance"] = route_provenance
        evidence["exact_route_sources"] = route_evidence
        errors.extend(
            validate_exact_route_provenance(
                route_provenance,
                trajectory=args.trajectory_6dof,
                hand_source=args.hand_source,
                task=args.task,
                hand_type=args.hand_type,
                data_id=0,
                processed_backend="do_as_i_do",
            )
        )
        adapter = load_json_optional(adapter_path)
        if args.trajectory_6dof == "egoinfinity":
            evidence["adapter_manifest"] = str(adapter_path) if adapter_path.exists() else None
            if adapter is None:
                errors.append("missing EgoInfinity adapter_manifest.json for Ego->DAI retarget raw_dir")
            else:
                schema_version = adapter.get("adapter_schema_version")
                if (
                    not isinstance(schema_version, int)
                    or isinstance(schema_version, bool)
                    or schema_version < 3
                ):
                    errors.append(f"Ego->DAI adapter_schema_version must be >=3: {schema_version}")
                replacement = adapter.get("object_mesh_replacement") or {}
                evidence["adapter"] = {
                    "ego_object_id": adapter.get("ego_object_id"),
                    "ego_prompt": adapter.get("ego_prompt"),
                    "mesh_scale": adapter.get("mesh_scale"),
                    "mesh_scale_source": adapter.get("mesh_scale_source"),
                    "object_mesh_replacement": {
                        "ego_mesh": replacement.get("ego_mesh") if isinstance(replacement, dict) else None,
                        "targets": replacement.get("targets") if isinstance(replacement, dict) else None,
                        "boxlike_fallback": replacement.get("boxlike_fallback") if isinstance(replacement, dict) else None,
                    },
                    "geometry_interaction_hand": adapter.get("geometry_interaction_hand"),
                    "visual_interaction_hand": adapter.get("visual_interaction_hand"),
                }
                if not isinstance(replacement, dict) or not replacement.get("ego_mesh") or not replacement.get("targets"):
                    errors.append("Ego->DAI adapter did not record an Ego object mesh replacement")
                geometry = adapter.get("geometry_interaction_hand") or {}
                if isinstance(geometry, dict) and geometry.get("available") and not geometry.get("selected_hand"):
                    warnings.append("Ego object track has no geometric hand-object contact; treat this scene as weak display material")
        else:
            evidence["adapter_manifest"] = str(adapter_path) if adapter_path.exists() else None
            if adapter is None:
                errors.append("missing pure DAI adapter_manifest.json for exact DAI retarget raw_dir")
            else:
                schema_version = adapter.get("adapter_schema_version")
                if (
                    not isinstance(schema_version, int)
                    or isinstance(schema_version, bool)
                    or schema_version < 3
                ):
                    errors.append(f"pure DAI adapter_schema_version must be >=3: {schema_version}")
                evidence["adapter"] = {
                    "hand_source": adapter.get("hand_source"),
                    "object_track_source": adapter.get("object_track_source"),
                    "object_mesh_source": adapter.get("object_mesh_source"),
                    "retarget_object_source": adapter.get("retarget_object_source"),
                    "canonical_transform": adapter.get("canonical_transform"),
                    "hoi_refinement": adapter.get("hoi_refinement"),
                    "hoi_contact_alignment": adapter.get("hoi_contact_alignment"),
                    "adapter_rigid_invariance": adapter.get("adapter_rigid_invariance"),
                }
                if adapter.get("object_track_source") != "dai_native" or adapter.get("object_mesh_source") != "dai_native":
                    errors.append("pure DAI adapter provenance is not dai_native")
            if path_contains(object_mesh, "egoinfinity") or path_contains(roots["trajectory_keypoints"], "egoinfinity"):
                errors.append("pure DAI indexed assets resolve to EgoInfinity paths")
    elif args.retargeting == "spider":
        roots = spider_roots(exp, key)
        route_provenance, route_evidence = exact_route_provenance(exp, args, key)
        evidence["exact_route_provenance"] = route_provenance
        evidence["exact_route_sources"] = route_evidence
        if not valid_spider_alignment(roots["root"]):
            errors.append("missing or invalid SPIDER MJWP alignment manifest")
        input_manifest = route_evidence.get("spider_input_manifest")
        output_manifest = route_evidence.get("spider_output_manifest")
        if not isinstance(input_manifest, dict):
            errors.append("missing SPIDER input manifest")
        if not isinstance(output_manifest, dict):
            errors.append("missing SPIDER output manifest")
        dai_preprocess = (
            input_manifest.get("dai_preprocess_invariance")
            if isinstance(input_manifest, dict)
            else None
        )
        evidence["dai_preprocess_invariance"] = dai_preprocess
        if not isinstance(dai_preprocess, dict):
            errors.append("missing exact-route DAI preprocess invariance for SPIDER")
        elif (
            dai_preprocess.get("status") != "pass"
            or dai_preprocess.get("production_eligible") is not True
        ):
            errors.append("exact-route DAI preprocess invariance is production-ineligible")
        embedded_raw_to_processed_hoi = (
            input_manifest.get("raw_to_processed_hoi_invariance")
            if isinstance(input_manifest, dict)
            else None
        )
        expected_dai_cell = cell_key(
            args.trajectory_6dof, args.hand_source, "do_as_i_do"
        )
        expected_dai_adapter = (
            exp
            / "intermediates"
            / "retargeting"
            / "do_as_i_do"
            / expected_dai_cell
            / "raw_dir"
            / "adapter_manifest.json"
        )
        expected_dai_keypoints = (
            exp
            / "intermediates"
            / "retargeting"
            / "do_as_i_do"
            / expected_dai_cell
            / "retargeting_outputs"
            / "mano"
            / args.hand_type
            / args.task
            / "0"
            / "trajectory_keypoints.npz"
        )
        expected_dai_invariance_path = expected_dai_keypoints.with_name(
            "raw_to_processed_hoi_invariance.json"
        )
        raw_to_processed_hoi = load_json_optional(expected_dai_invariance_path)
        evidence["raw_to_processed_hoi_invariance"] = raw_to_processed_hoi
        evidence["raw_to_processed_hoi_invariance_path"] = str(
            expected_dai_invariance_path
        )
        if not isinstance(raw_to_processed_hoi, dict):
            errors.append("missing exact-route raw-to-processed DAI HOI invariance for SPIDER")
        elif (
            raw_to_processed_hoi.get("status") != "ok"
            or raw_to_processed_hoi.get("production_eligible") is not True
            or raw_to_processed_hoi.get("cross_hand_source_comparison") is not False
        ):
            errors.append("SPIDER raw-to-processed DAI HOI invariance is invalid")
        if isinstance(raw_to_processed_hoi, dict):
            errors.extend(
                "SPIDER raw-to-processed DAI binding: " + error
                for error in validate_raw_to_processed_hoi_binding(
                    raw_to_processed_hoi,
                    adapter_manifest=expected_dai_adapter,
                    processed_keypoints=expected_dai_keypoints,
                    trajectory=args.trajectory_6dof,
                    hand_source=args.hand_source,
                    hand_type=args.hand_type,
                )
            )
        if embedded_raw_to_processed_hoi != raw_to_processed_hoi:
            errors.append(
                "SPIDER embedded raw-to-processed DAI invariance does not match "
                "the exact source-cell report"
            )
        rest_support = (
            output_manifest.get("rest_support_provenance")
            if isinstance(output_manifest, dict)
            else None
        )
        evidence["spider_rest_support_provenance"] = rest_support
        if not isinstance(rest_support, dict):
            errors.append("missing SPIDER rest-support provenance")
        else:
            if rest_support.get("schema_version") != 1:
                errors.append("SPIDER rest-support schema_version is not 1")
            if rest_support.get("status") != "ok":
                errors.append("SPIDER rest-support provenance is not ok")
            policy = rest_support.get("policy") or {}
            integrity = rest_support.get("trajectory_integrity") or {}
            binding = rest_support.get("route_binding") or {}
            endpoint_accounting = rest_support.get("endpoint_accounting") or {}
            if policy.get("trajectory_write_allowed") is not False:
                errors.append("SPIDER rest-support allowed trajectory mutation")
            if policy.get("support_only_when_endpoint_not_in_hand") is not True:
                errors.append("SPIDER rest-support endpoint policy is invalid")
            if policy.get("endpoints") != ["start", "end"]:
                errors.append("SPIDER rest-support did not audit both endpoints")
            if policy.get("support_application_mode") != "audit_only":
                errors.append("SPIDER production scene used applied rest-support geometry")
            if policy.get("diagnostic_only") is not True:
                errors.append("SPIDER rest-support evidence is not audit-only")
            if policy.get("production_scene_geometry_modified") is not False:
                errors.append("SPIDER rest-support modified the production scene")
            if rest_support.get("supports_added") != 0:
                errors.append("SPIDER production scene contains generated rest supports")
            if integrity.get("unchanged") is not True:
                errors.append("SPIDER rest-support changed the IK trajectory")
            if endpoint_accounting.get("complete") is not True:
                errors.append("SPIDER rest-support endpoint accounting is incomplete")
            expected_binding = {
                "cell_key": key,
                "trajectory_6dof": args.trajectory_6dof,
                "hand_source": args.hand_source,
                "task": args.task,
                "hand_type": args.hand_type,
                "data_id": 0,
            }
            for field, expected in expected_binding.items():
                if binding.get(field) != expected:
                    errors.append(
                        f"SPIDER rest-support route binding mismatch for {field}: "
                        f"expected={expected} actual={binding.get(field)}"
                    )
            for field in ("adapter_manifest", "source_keypoints"):
                bound_path_value = binding.get(field)
                bound_sha = binding.get(f"{field}_sha256")
                expected_path_value = (
                    input_manifest.get(field) if isinstance(input_manifest, dict) else None
                )
                if not bound_path_value or not expected_path_value:
                    errors.append(f"SPIDER rest-support missing bound {field}")
                    continue
                bound_path = Path(str(bound_path_value))
                expected_path = Path(str(expected_path_value))
                try:
                    same_path = bound_path.resolve() == expected_path.resolve()
                except OSError:
                    same_path = False
                if not same_path or not bound_path.is_file():
                    errors.append(f"SPIDER rest-support stale/cross-route {field}")
                elif bound_sha != file_sha256(bound_path):
                    errors.append(f"SPIDER rest-support {field} hash mismatch")
            outputs = rest_support.get("outputs") or {}
            consumed_scene = Path(str(outputs.get("scene_act_xml") or ""))
            consumed_scene_sha = outputs.get("scene_act_sha256_after")
            if not consumed_scene.is_file():
                errors.append("SPIDER rest-support consumed scene is missing")
            elif consumed_scene_sha != file_sha256(consumed_scene):
                errors.append("SPIDER rest-support consumed scene hash mismatch")
        spider_tracking = (
            output_manifest.get("object_tracking_quality")
            if isinstance(output_manifest, dict)
            else None
        )
        hard_tracking_admission = validate_hard_tracking_admission(spider_tracking)
        evidence["hard_tracking_admission"] = hard_tracking_admission
        if hard_tracking_admission.get("status") != "ok":
            errors.extend(
                "SPIDER hard tracking admission: " + str(error)
                for error in hard_tracking_admission.get("errors") or ["invalid"]
            )
        for label, source_manifest in (
            ("SPIDER input", input_manifest),
            ("SPIDER output", output_manifest),
        ):
            if not isinstance(source_manifest, dict):
                continue
            if source_manifest.get("trajectory_6dof") != args.trajectory_6dof:
                errors.append(f"{label} trajectory_6dof does not match cell")
            if source_manifest.get("hand_source") != args.hand_source:
                errors.append(f"{label} hand_source does not match cell")
            if source_manifest.get("task") != args.task:
                errors.append(f"{label} task does not match cell")
            if source_manifest.get("data_id") != 0:
                errors.append(f"{label} data_id does not match cell")
            manifest_hand_type = (
                source_manifest.get("validated_hand_type")
                or source_manifest.get("resolved_hand_type")
                or source_manifest.get("hand_type")
            )
            if manifest_hand_type != args.hand_type:
                errors.append(f"{label} hand_type does not match cell")
            if source_manifest.get("raw_to_processed_hoi_invariance") != raw_to_processed_hoi:
                errors.append(
                    f"{label} raw-to-processed DAI invariance conflicts with exact source cell"
                )
            for field in ADAPTER_PROVENANCE_FIELDS:
                value = source_manifest.get(field)
                if value is not None and value != route_provenance.get(field):
                    errors.append(f"{label} {field} conflicts with exact route provenance")
            postprocess_qc = source_manifest.get("post_preprocess_semantic_qc")
            if not isinstance(postprocess_qc, dict):
                errors.append(f"{label} missing post_preprocess_semantic_qc")
            elif postprocess_qc.get("status") != "ok":
                errors.append(f"{label} post_preprocess_semantic_qc is not ok")
            elif postprocess_qc != route_provenance.get("post_preprocess_semantic_qc"):
                errors.append(
                    f"{label} post_preprocess_semantic_qc conflicts with exact route provenance"
                )
        if isinstance(input_manifest, dict) and input_manifest.get("expected_dai_cell_key") != expected_dai_cell:
            errors.append("SPIDER input manifest does not name the exact DAI source cell")
        adapter = route_evidence.get("adapter")
        adapter_path = route_evidence.get("adapter_manifest")
        if not isinstance(adapter, dict):
            errors.append("missing exact DAI source adapter for SPIDER cell")
        else:
            schema_version = adapter.get("adapter_schema_version")
            if (
                not isinstance(schema_version, int)
                or isinstance(schema_version, bool)
                or schema_version < 3
            ):
                errors.append(f"SPIDER source adapter_schema_version must be >=3: {schema_version}")
            for field in ADAPTER_PROVENANCE_FIELDS:
                if adapter.get(field) != route_provenance.get(field):
                    errors.append(f"SPIDER {field} does not match exact DAI source adapter")
        if not adapter_path or expected_dai_cell not in str(adapter_path):
            errors.append("SPIDER source adapter path is not from the exact DAI source cell")
        errors.extend(
            validate_exact_route_provenance(
                route_provenance,
                trajectory=args.trajectory_6dof,
                hand_source=args.hand_source,
                require_post_preprocess_semantic_qc=True,
                task=args.task,
                hand_type=args.hand_type,
                data_id=0,
                processed_backend="spider",
            )
        )
    return {
        "status": "invalid" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create canonical experiment asset indexes for one 12-way demo cell.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--trajectory-6dof", required=True, choices=["egoinfinity", "do_as_i_do"])
    parser.add_argument("--hand-source", required=True, choices=["aoe", "estimated"])
    parser.add_argument("--retargeting", required=True, choices=["egoinfinity", "do_as_i_do", "spider"])
    parser.add_argument("--task", default="foundation_jar_bimanual_leftscale")
    parser.add_argument("--hand-type", default="bimanual")
    parser.add_argument("--robot", default=None)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument(
        "--preflight-raw-dir",
        type=Path,
        default=None,
        help="Validate only this exact DAI raw-dir adapter before official launch.py.",
    )
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=None,
        help="Optional JSON path for --preflight-raw-dir results.",
    )
    args = parser.parse_args()

    if args.preflight_raw_dir is not None:
        if args.retargeting != "do_as_i_do":
            parser.error("--preflight-raw-dir is only valid with --retargeting do_as_i_do")
        preflight = validate_dai_raw_dir_preflight(
            args.preflight_raw_dir,
            trajectory=args.trajectory_6dof,
            hand_source=args.hand_source,
        )
        if args.preflight_output is not None:
            args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_output.write_text(
                json.dumps(preflight, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return 0 if preflight.get("status") == "ok" else 6

    exp = REPO_ROOT / "experiments" / args.run_name
    key = cell_key(args.trajectory_6dof, args.hand_source, args.retargeting)
    hand_selection = None
    route_provenance, route_provenance_evidence = exact_route_provenance(exp, args, key)
    if args.retargeting == "do_as_i_do":
        hand_selection = load_json_optional(
            exp
            / "logs"
            / f"do_as_i_do_hand_selection__{args.trajectory_6dof}__{args.hand_source}.json"
        )
    traj_assets = ensure_dir(exp / "assets" / "trajectory_6dof" / args.trajectory_6dof / args.task)
    cell_assets = ensure_dir(exp / "assets" / "cells" / key)
    cell_bundle = ensure_dir(exp / "cells" / key)
    manifest = {
        "cell_key": key,
        "settings": {
            "trajectory_6dof": args.trajectory_6dof,
            "hand_source": args.hand_source,
            "retargeting": args.retargeting,
            "task": args.task,
            "hand_type": args.hand_type,
            "robot": args.robot,
        },
        "experiment_root": str(exp),
        "provenance": {
            **route_provenance,
            "route_provenance_sources": route_provenance_evidence,
            "recommended_route": args.trajectory_6dof == "egoinfinity",
            "baseline_only": args.trajectory_6dof == "do_as_i_do",
            "retarget_hand_type": args.hand_type,
            "retarget_hand_selection_source": (
                hand_selection.get("selection_source") if hand_selection else None
            ),
            "retarget_hand_selection_reason": (
                hand_selection.get("selection_reason") if hand_selection else None
            ),
        },
        "assets": {},
        "missing": [],
    }

    def record(name: str, value: str | None) -> None:
        manifest["assets"][name] = value
        if value is None:
            manifest["missing"].append(name)

    def record_optional(name: str, value: str | None) -> None:
        manifest["assets"][name] = value

    # Trajectory-6DoF assets: object trajectory, object mesh, overlay, depth.
    if args.trajectory_6dof == "do_as_i_do":
        roots = do_as_i_do_roots(exp, args.task, args.hand_type, args.retargeting, key)
        object_6dof_path = traj_assets / "object_6dof.npz"
        made = write_do_as_i_do_object_6dof(roots["trajectory_keypoints"], object_6dof_path)
        if not made and object_6dof_path.exists():
            made = str(object_6dof_path)
        record("trajectory_6dof.object_6dof_npz", rel(object_6dof_path, exp) if made else None)
        record("trajectory_6dof.object_visual_mesh", link_or_copy(first_existing([roots["object_visual"], traj_assets / "object_meshes" / "visual.obj"]), traj_assets / "object_meshes" / "visual.obj", args.mode, exp))
        record("trajectory_6dof.object_convex_meshes", link_or_copy(first_existing([roots["object_convex"], traj_assets / "object_meshes" / "convex"]), traj_assets / "object_meshes" / "convex", args.mode, exp))
        record("trajectory_6dof.task_info", link_or_copy(first_existing([roots["task_info"], traj_assets / "task_info.json"]), traj_assets / "task_info.json", args.mode, exp))
        # Hand-bearing keypoints are exact-cell assets. Never write them to a
        # trajectory-level last-writer-wins alias that can mix aoe/estimated.
        object_id = object_id_from_task(args.task)
        did_clip = first_existing(
            [
                exp / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction",
                exp / "intermediates" / "do_as_i_do" / "reconstruction",
            ]
        ) or (exp / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction")
        raw_source = first_existing([did_clip / "raw_dir"]) or did_clip
        # Prefer lab-rendered reconstruction mesh overlays over raw Do-as-I-Do
        # TAPIR/debug videos.  raw_dir normally points back to the official clip
        # directory, where output_tapir_*.mp4 exists but does not contain the
        # aligned hand+object mesh overlay expected by the 12-demo triptychs.
        record("trajectory_6dof.raw_dir", link_or_copy(raw_source, traj_assets / "inputs" / "raw_dir", args.mode, exp))
        overlay_candidates = [
            did_clip / "mesh_overlay.mp4",
            did_clip / "overlay.mp4",
            raw_source / "mesh_overlay.mp4",
            raw_source / "overlay.mp4",
            did_clip / "obj_tracking_out" / object_id / "combined_visualization" / "projected_hand_object_optimized" / "video.mp4",
            raw_source / "obj_tracking_out" / object_id / "combined_visualization" / "projected_hand_object_optimized" / "video.mp4",
            did_clip / "obj_tracking_out" / object_id / "combined_visualization" / "projected" / "video.mp4",
            raw_source / "obj_tracking_out" / object_id / "combined_visualization" / "projected" / "video.mp4",
            did_clip / f"output_tapir_{object_id}_overlay.mp4",
            raw_source / f"output_tapir_{object_id}_overlay.mp4",
            did_clip / f"output_tapir_{object_id}.mp4",
            raw_source / f"output_tapir_{object_id}.mp4",
            did_clip / "raw.mp4",
            raw_source / "raw.mp4",
        ]
        depth_candidates = [
            did_clip / "depth.mp4",
            raw_source / "depth.mp4",
            did_clip / "moge_depth.mp4",
            raw_source / "moge_depth.mp4",
            did_clip / "pointmap_depth.mp4",
            raw_source / "pointmap_depth.mp4",
            did_clip / "raw.mp4",
            raw_source / "raw.mp4",
        ]
        record("trajectory_6dof.overlay_video", link_or_copy(first_existing(overlay_candidates), traj_assets / "overlay.mp4", args.mode, exp))
        record("trajectory_6dof.depth_video", link_or_copy(first_existing(depth_candidates), traj_assets / "depth.mp4", args.mode, exp))
    else:
        roots = egoinfinity_roots(exp)
        record("trajectory_6dof.raw_clip", link_or_copy(roots["clip"], traj_assets / "inputs" / "clip", args.mode, exp))
        record("trajectory_6dof.object_6dof_native", link_or_copy(first_existing([roots["clip"] / "pipeline_result.pkl.gz", roots["clip"] / "object_poses.npz"]), traj_assets / "object_6dof_native", args.mode, exp))
        record(
            "trajectory_6dof.object_visual_meshes",
            link_or_copy(
                first_existing([roots["clip"] / "sam3_meshes", roots["clip"] / "sam3d_objects", roots["clip"] / "objects"]),
                traj_assets / "object_meshes",
                args.mode,
                exp,
            ),
        )
        record("trajectory_6dof.overlay_video", link_or_copy(first_existing([
            roots["clip"] / "rgb_mesh_overlay.mp4",
        ]), traj_assets / "overlay.mp4", args.mode, exp))
        record("trajectory_6dof.depth_video", link_or_copy(first_existing([roots["samples"] / "depth.mp4", roots["clip"] / "depth.mp4"]), traj_assets / "depth.mp4", args.mode, exp))

    # Hand source assets.
    hand_dir = ensure_dir(cell_assets / "hand_source")
    if args.hand_source == "aoe":
        record(
            "hand_source.aoe_hands",
            link_or_copy(
                first_existing(
                    aoe_hand_candidates(
                        exp,
                        args.task,
                        args.hand_type,
                        args.retargeting,
                        key,
                    )
                ),
                hand_dir / "aoe_hands.npz",
                args.mode,
                exp,
            ),
        )
    else:
        if args.trajectory_6dof == "do_as_i_do":
            roots = do_as_i_do_roots(exp, args.task, args.hand_type, args.retargeting, key)
            record("hand_source.estimated_hands", link_or_copy(roots["trajectory_keypoints"], hand_dir / "estimated_hands_and_keypoints.npz", args.mode, exp))
        else:
            roots = egoinfinity_roots(exp)
            record("hand_source.estimated_hands", link_or_copy(first_existing([roots["samples"] / "hand_joints.bin", roots["clip"] / "hand_joints.bin"]), hand_dir / "estimated_hand_joints", args.mode, exp))

    # Retargeting assets.
    ret_dir = ensure_dir(cell_assets / "retargeting" / args.retargeting)
    if args.retargeting == "do_as_i_do":
        roots = do_as_i_do_roots(exp, args.task, args.hand_type, args.retargeting, key)
        exact_adapter_manifest = (
            exp
            / "intermediates"
            / "retargeting"
            / "do_as_i_do"
            / key
            / "raw_dir"
            / "adapter_manifest.json"
        )
        exact_processed_keypoints = (
            exp
            / "intermediates"
            / "retargeting"
            / "do_as_i_do"
            / key
            / "retargeting_outputs"
            / "mano"
            / args.hand_type
            / args.task
            / "0"
            / "trajectory_keypoints.npz"
        )
        preprocess_invariance_path = roots["trajectory_keypoints"].with_name(
            "hoi_preprocess_invariance.json"
        )
        preprocess_invariance = load_json_optional(preprocess_invariance_path)
        raw_to_processed_hoi_path = exact_processed_keypoints.with_name(
            "raw_to_processed_hoi_invariance.json"
        )
        raw_to_processed_hoi = load_json_optional(raw_to_processed_hoi_path)
        dai_alignment_valid = valid_dai_alignment(roots["robot"])
        alignment_manifest = roots["robot"] / "mjwp_alignment_manifest.json"
        tracking_quality_path = roots["robot"] / "mjwp_object_tracking_quality.json"
        tracking_quality = load_json_optional(tracking_quality_path)
        hard_tracking_admission = validate_hard_tracking_admission(tracking_quality)
        hand_provenance_path = (
            exp
            / "intermediates"
            / "retargeting"
            / "do_as_i_do"
            / key
            / "input_hand_provenance.json"
        )
        hand_provenance = load_json_optional(hand_provenance_path)
        manifest["provenance"]["retarget_quality_status"] = (
            tracking_quality.get("status") if tracking_quality else "missing"
        )
        manifest["provenance"]["retarget_quality"] = tracking_quality
        manifest["provenance"]["hard_tracking_admission"] = hard_tracking_admission
        manifest["provenance"]["input_hand"] = hand_provenance
        manifest["provenance"]["dai_preprocess_invariance"] = preprocess_invariance
        manifest["provenance"]["raw_to_processed_hoi_invariance"] = raw_to_processed_hoi
        invalid_reasons = []
        if not hand_provenance:
            invalid_reasons.append("missing input hand provenance report")
        elif hand_provenance.get("status") != "ok":
            invalid_reasons.extend(hand_provenance.get("errors") or ["invalid input hand provenance"])
        if not isinstance(preprocess_invariance, dict):
            invalid_reasons.append("missing DAI HOI preprocess invariance report")
        elif (
            preprocess_invariance.get("status") != "pass"
            or preprocess_invariance.get("production_eligible") is not True
        ):
            invalid_reasons.append("DAI HOI preprocessing is production-ineligible")
        if not isinstance(raw_to_processed_hoi, dict):
            invalid_reasons.append("missing exact-route raw-to-processed DAI HOI invariance")
        elif (
            raw_to_processed_hoi.get("status") != "ok"
            or raw_to_processed_hoi.get("production_eligible") is not True
            or raw_to_processed_hoi.get("cross_hand_source_comparison") is not False
        ):
            invalid_reasons.append("exact-route raw-to-processed DAI HOI invariance is invalid")
        if isinstance(raw_to_processed_hoi, dict):
            invalid_reasons.extend(
                "raw-to-processed DAI binding: " + error
                for error in validate_raw_to_processed_hoi_binding(
                    raw_to_processed_hoi,
                    adapter_manifest=exact_adapter_manifest,
                    processed_keypoints=exact_processed_keypoints,
                    trajectory=args.trajectory_6dof,
                    hand_source=args.hand_source,
                    hand_type=args.hand_type,
                )
            )
        if hard_tracking_admission.get("status") != "ok":
            invalid_reasons.extend(
                hard_tracking_admission.get("errors")
                or ["hard MJWP object/HOI tracking admission invalid"]
            )
        if not invalid_reasons:
            manifest["provenance"]["status"] = "ok"
        else:
            manifest["provenance"]["status"] = "invalid"
            manifest["provenance"]["invalid_reasons"] = invalid_reasons
        record(
            "retargeting.input_hand_provenance",
            link_or_copy(
                hand_provenance_path if hand_provenance_path.exists() else None,
                ret_dir / "metadata" / "input_hand_provenance.json",
                args.mode,
                exp,
            ),
        )
        record(
            "retargeting.quality.dai_preprocess_invariance",
            link_or_copy(
                preprocess_invariance_path
                if preprocess_invariance_path.exists()
                else None,
                ret_dir / "metadata" / "dai_preprocess_invariance.json",
                args.mode,
                exp,
            ),
        )
        record(
            "retargeting.quality.raw_to_processed_hoi_invariance",
            link_or_copy(
                raw_to_processed_hoi_path
                if raw_to_processed_hoi_path.exists()
                else None,
                ret_dir / "metadata" / "raw_to_processed_hoi_invariance.json",
                args.mode,
                exp,
            ),
        )
        record(
            "retargeting.alignment.mjwp_manifest",
            link_or_copy(
                alignment_manifest if dai_alignment_valid else None,
                ret_dir / "metadata" / "mjwp_alignment_manifest.json",
                args.mode,
                exp,
            ),
        )
        record_optional(
            "retargeting.quality.mjwp_object_tracking",
            link_or_copy(
                tracking_quality_path if tracking_quality_path.exists() else None,
                ret_dir / "metadata" / "mjwp_object_tracking_quality.json",
                args.mode,
                exp,
            ),
        )
        for name in ["scene.xml", "scene_ik.xml", "scene_eq.xml"]:
            record(f"retargeting.scene.{name}", link_or_copy(roots["robot"] / name, ret_dir / "scenes" / name, args.mode, exp))
        for name in ["trajectory_kinematic.npz", "trajectory_ikrollout.npz"]:
            record(f"retargeting.trajectory.{name}", link_or_copy(roots["robot"] / name, ret_dir / "trajectories" / name, args.mode, exp))
        record(
            "retargeting.trajectory.config.yaml",
            link_or_copy(
                first_existing([
                    roots["robot"] / "config.yaml",
                    roots["robot"] / "config_act.yaml",
                ]),
                ret_dir / "trajectories" / "config.yaml",
                args.mode,
                exp,
            ),
        )
        record(
            "retargeting.trajectory.trajectory_mjwp.npz",
            link_or_copy(
            first_existing([
                roots["robot"] / "trajectory_mjwp_act_aligned.npz",
                roots["robot"] / "trajectory_mjwp_aligned.npz",
                ]),
                ret_dir / "trajectories" / "trajectory_mjwp.npz",
                args.mode,
                exp,
            ),
        )
        processed_keypoints, generic_keypoints_alias = dai_indexed_keypoint_paths(
            ret_dir,
            hand_type=args.hand_type,
            task=args.task,
            data_id=0,
        )
        canonical_keypoints_asset = link_or_copy(
            roots["trajectory_keypoints"], processed_keypoints, args.mode, exp
        )
        record_optional(
            "retargeting.trajectory.trajectory_keypoints.route_canonical",
            canonical_keypoints_asset,
        )
        record(
            "retargeting.trajectory.trajectory_keypoints.npz",
            link_or_copy(
                processed_keypoints if processed_keypoints.exists() else None,
                generic_keypoints_alias,
                args.mode,
                exp,
            ),
        )
        processed_object_mesh = dai_indexed_visual_mesh_path(
            ret_dir, task=args.task
        )
        record(
            "retargeting.object_mesh.visual.route_canonical",
            link_or_copy(
                roots["object_visual"] if roots["object_visual"].is_file() else None,
                processed_object_mesh,
                args.mode,
                exp,
            ),
        )
        adapter_manifest_path = roots["raw_dir"] / "adapter_manifest.json"
        route_input_qc = validate_route_input(
            adapter_manifest=adapter_manifest_path,
            source_keypoints=roots["trajectory_keypoints"],
            processed_keypoints=processed_keypoints,
            processed_backend="do_as_i_do",
            source_object_mesh=roots["object_visual"],
            processed_object_mesh=processed_object_mesh,
            trajectory=args.trajectory_6dof,
            hand_source=args.hand_source,
            task=args.task,
            hand_type=args.hand_type,
            data_id=0,
        )
        route_input_qc_path = ret_dir / "metadata" / "route_input_qc.json"
        route_input_qc_path.parent.mkdir(parents=True, exist_ok=True)
        route_input_qc_path.write_text(json.dumps(route_input_qc, indent=2), encoding="utf-8")
        record_optional(
            "retargeting.quality.route_input_qc",
            rel(route_input_qc_path, exp),
        )
        manifest["provenance"]["route_input_qc"] = route_input_qc
        if route_input_qc.get("status") != "ok":
            manifest["provenance"]["status"] = "invalid"
            manifest["provenance"].setdefault("invalid_reasons", []).extend(
                route_input_qc.get("errors") or ["route input QC invalid"]
            )
        record_optional("retargeting.video.visualization_ik.mp4", link_or_copy(roots["robot"] / "visualization_ik.mp4", ret_dir / "videos" / "visualization_ik.mp4", args.mode, exp))
        mjwp_plain = (
            first_existing([
                roots["robot"] / "visualization_mjwp_act_aligned.mp4",
            ])
            if dai_alignment_valid
            else None
        )
        record("retargeting.video.visualization_mjwp.mp4", link_or_copy(mjwp_plain, ret_dir / "videos" / "visualization_mjwp.mp4", args.mode, exp))
        record_optional("retargeting.video.visualization_mjwp_front.mp4", link_or_copy(mjwp_plain, ret_dir / "videos" / "visualization_mjwp_front.mp4", args.mode, exp))
        robot_video = ret_dir / "videos" / "visualization_mjwp.mp4"
    elif args.retargeting == "egoinfinity":
        robot = args.robot or "g1"
        roots = egoinfinity_roots(exp, robot)
        rr = roots["retarget"] if roots["retarget"].exists() else roots["retarget_legacy"]
        record("retargeting.trajectory.trajectory_npz", link_or_copy(rr / "trajectory.npz", ret_dir / "trajectories" / "trajectory.npz", args.mode, exp))
        record("retargeting.video.input_viz", link_or_copy(rr / "input_viz.mp4", ret_dir / "videos" / "input_viz.mp4", args.mode, exp))
        record("retargeting.video.robot_sim", link_or_copy(rr / "robot_sim.mp4", ret_dir / "videos" / "robot_sim.mp4", args.mode, exp))
        robot_video = ret_dir / "videos" / "robot_sim.mp4"
    else:
        roots = spider_roots(exp, key)
        spider_alignment_valid = valid_spider_alignment(roots["root"])
        spider_trajectory = (
            bound_spider_artifact(roots["root"], "aligned_trajectory")
            if spider_alignment_valid
            else None
        )
        spider_video = (
            bound_spider_artifact(roots["root"], "aligned_video")
            if spider_alignment_valid
            else None
        )
        spider_scene = (
            bound_spider_artifact(roots["root"], "render_scene")
            if spider_alignment_valid
            else None
        )
        postprocess_qc_path = roots["root"] / "spider_postprocess_semantic_qc.json"
        record(
            "retargeting.quality.post_preprocess_semantic_qc",
            link_or_copy(
                postprocess_qc_path if postprocess_qc_path.exists() else None,
                ret_dir / "metadata" / "spider_postprocess_semantic_qc.json",
                args.mode,
                exp,
            ),
        )
        record(
            "retargeting.trajectory.trajectory_mjwp",
            link_or_copy(
                spider_trajectory,
                ret_dir / "trajectories" / "trajectory_mjwp.npz",
                args.mode,
                exp,
            ),
        )
        record(
            "retargeting.scene.scene_xml",
            link_or_copy(
                spider_scene,
                ret_dir / "scenes" / "scene.xml",
                args.mode,
                exp,
            ),
        )
        record(
            "retargeting.video.visualization_mjwp",
            link_or_copy(
                spider_video,
                ret_dir / "videos" / "visualization_mjwp.mp4",
                args.mode,
                exp,
            ),
        )
        robot_video = ret_dir / "videos" / "visualization_mjwp.mp4"

    # Per-cell convenience bundle used by compose scripts.
    overlay = traj_assets / "overlay.mp4"
    depth = traj_assets / "depth.mp4"
    record("cell.overlay", link_or_copy(overlay if overlay.exists() else None, cell_bundle / "overlay.mp4", args.mode, exp))
    record("cell.depth", link_or_copy(depth if depth.exists() else None, cell_bundle / "depth.mp4", args.mode, exp))
    record("cell.robot", link_or_copy(robot_video if robot_video.exists() else None, cell_bundle / "robot.mp4", args.mode, exp))
    manifest["provenance_validation"] = validate_index_provenance(exp, args, key)
    admission_errors = manifest_admission_errors(manifest, retargeting=args.retargeting)
    manifest["admission"] = {
        "status": "invalid" if admission_errors else "ok",
        "errors": admission_errors,
        "policy": "route_and_spider_postprocess_qc_fail_closed_v2",
    }

    manifest_path = cell_assets / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (cell_bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "cell_key": key,
        "manifest": str(manifest_path),
        "missing": manifest["missing"],
        "admission": manifest["admission"],
    }, indent=2))
    return 0 if not admission_errors else 5


if __name__ == "__main__":
    raise SystemExit(main())
