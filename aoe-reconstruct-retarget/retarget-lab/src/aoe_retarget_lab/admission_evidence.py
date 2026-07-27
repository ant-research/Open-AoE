from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def resolve_recorded_path(
    root: Path,
    value: object,
    *,
    repo_root: Path | None = None,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [root / path]
    if repo_root is not None:
        candidates.append(repo_root / path)
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_to_processed_invariance_errors(
    payload: object,
    *,
    label: str = "raw_to_processed_hoi_invariance",
    trajectory_6dof: str | None = None,
    hand_source: str | None = None,
    hand_type: str | None = None,
    adapter_manifest: Path | None = None,
    processed_keypoints: Path | None = None,
) -> list[str]:
    if not isinstance(payload, Mapping):
        return [f"missing {label}"]
    errors: list[str] = []
    if payload.get("status") != "ok":
        errors.append(f"{label} status is not ok")
    if payload.get("production_eligible") is not True:
        errors.append(f"{label} is not production eligible")
    if payload.get("cross_hand_source_comparison") is not False:
        errors.append(f"{label} did not explicitly prohibit cross-hand-source comparison")
    if payload.get("schema_version") != 3:
        errors.append(f"{label} schema_version is not 3")
    if payload.get("full_relative_se3_checked") is not True:
        errors.append(f"{label} did not verify full relative SE(3)")
    if payload.get("comparison_semantics") != "exact_route_exact_hand_raw_adapter_to_dai_processed":
        errors.append(f"{label} comparison semantics are not exact-route/exact-hand")

    binding = payload.get("route_binding")
    if not isinstance(binding, Mapping):
        errors.append(f"{label} route_binding is missing")
        binding = {}
    expected_identity = {
        "trajectory_6dof": trajectory_6dof,
        "hand_source": hand_source,
        "hand_type": hand_type,
    }
    for field, expected in expected_identity.items():
        report_value = payload.get(field)
        bound_value = binding.get(field)
        if report_value != bound_value:
            errors.append(f"{label} {field} contradicts route_binding")
        if expected is not None and (report_value != expected or bound_value != expected):
            errors.append(
                f"{label} {field} does not match exact route: "
                f"expected={expected} report={report_value} binding={bound_value}"
            )
    effective_hand_type = hand_type or payload.get("hand_type")
    expected_sides = (
        ["left", "right"]
        if effective_hand_type == "bimanual"
        else [effective_hand_type]
        if effective_hand_type in {"left", "right"}
        else None
    )
    if expected_sides is None:
        errors.append(f"{label} hand_type is invalid")
    else:
        if binding.get("requested_sides") != expected_sides:
            errors.append(f"{label} requested_sides do not match hand_type")
        if payload.get("joint_fit_components") != ["object", *expected_sides]:
            errors.append(f"{label} did not jointly fit object and all requested hands")

    raw_inputs = binding.get("raw_inputs")
    if not isinstance(raw_inputs, Mapping):
        errors.append(f"{label} raw_inputs binding is missing")
        raw_inputs = {}
    records = {
        "adapter_manifest": binding.get("adapter_manifest"),
        "raw_hand_npz": raw_inputs.get("hand_npz"),
        "raw_object_layout": raw_inputs.get("object_layout"),
        "processed_keypoints": binding.get("processed_keypoints"),
    }
    resolved: dict[str, Path] = {}
    for field, record in records.items():
        if not isinstance(record, Mapping):
            errors.append(f"{label} binding is missing {field}")
            continue
        path_value = record.get("path")
        path = Path(str(path_value)).expanduser() if path_value else None
        if path is None or not path.is_file():
            errors.append(f"{label} bound {field} is missing")
            continue
        resolved[field] = path
        try:
            actual_sha = file_sha256(path)
        except OSError:
            actual_sha = None
        if record.get("sha256") != actual_sha:
            errors.append(f"{label} bound {field} hash mismatch")

    expected_files = {
        "adapter_manifest": adapter_manifest,
        "processed_keypoints": processed_keypoints,
    }
    for field, expected in expected_files.items():
        actual = resolved.get(field)
        if expected is None:
            continue
        try:
            matches = actual is not None and actual.resolve() == expected.resolve()
        except OSError:
            matches = False
        if not matches:
            errors.append(f"{label} bound {field} is stale or cross-route")

    if adapter_manifest is not None:
        try:
            exact_raw_root = adapter_manifest.expanduser().resolve().parent
        except OSError:
            exact_raw_root = None
        if exact_raw_root is not None:
            for field in ("raw_hand_npz", "raw_object_layout"):
                actual = resolved.get(field)
                if actual is None:
                    continue
                try:
                    actual.resolve().relative_to(exact_raw_root)
                except (OSError, ValueError):
                    errors.append(
                        f"{label} bound {field} is outside exact raw route"
                    )
            config = load_json_object(exact_raw_root / "config.json")
            object_names = config.get("object_names") if isinstance(config, dict) else None
            if isinstance(object_names, list) and object_names:
                expected_layout = (
                    exact_raw_root
                    / "obj_tracking_out"
                    / str(object_names[0])
                    / "combined_visualization"
                    / "layout_camera_frame_optimized.json"
                )
                actual_layout = resolved.get("raw_object_layout")
                try:
                    exact_layout_matches = (
                        actual_layout is not None
                        and actual_layout.resolve() == expected_layout.resolve()
                    )
                except OSError:
                    exact_layout_matches = False
                if not exact_layout_matches:
                    errors.append(
                        f"{label} bound raw_object_layout is not config-selected exact route"
                    )

    flat_bindings = {
        "adapter_manifest": ("adapter_manifest_sha256", "adapter_manifest"),
        "raw_hand_npz": ("raw_hand_sha256", "raw_hand_npz"),
        "raw_object_layout": ("raw_object_layout_sha256", "raw_object_layout"),
        "processed_keypoints": ("processed_keypoints_sha256", "processed_keypoints"),
    }
    for field, (sha_field, path_field) in flat_bindings.items():
        nested = resolved.get(field)
        flat_value = payload.get(path_field)
        flat_path = Path(str(flat_value)).expanduser() if flat_value else None
        try:
            paths_match = (
                nested is not None
                and flat_path is not None
                and nested.resolve() == flat_path.resolve()
            )
        except OSError:
            paths_match = False
        if not paths_match:
            errors.append(f"{label} flat {path_field} contradicts route_binding")
        elif payload.get(sha_field) != file_sha256(nested):
            errors.append(f"{label} flat {sha_field} mismatch")
    return errors


def spider_rest_support_errors(
    payload: object,
    *,
    cell_key: str,
    trajectory_6dof: str,
    hand_source: str,
    task: str,
    hand_type: str,
    data_id: int,
    input_manifest: Mapping[str, Any],
    root: Path,
    repo_root: Path | None = None,
) -> list[str]:
    label = "SPIDER rest_support_provenance"
    if not isinstance(payload, Mapping):
        return [f"missing {label}"]
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append(f"{label} schema_version is not 1")
    if payload.get("status") != "ok":
        errors.append(f"{label} status is not ok")

    policy = payload.get("policy")
    if not isinstance(policy, Mapping):
        errors.append(f"{label} policy is missing")
    else:
        if policy.get("trajectory_write_allowed") is not False:
            errors.append(f"{label} allowed trajectory mutation")
        if policy.get("support_only_when_endpoint_not_in_hand") is not True:
            errors.append(f"{label} endpoint support policy is invalid")
        if policy.get("endpoints") != ["start", "end"]:
            errors.append(f"{label} did not audit both endpoints")
        if policy.get("support_application_mode") != "audit_only":
            errors.append(f"{label} is not audit-only")
        if policy.get("diagnostic_only") is not True:
            errors.append(f"{label} is not explicitly diagnostic-only")
        if policy.get("production_scene_geometry_modified") is not False:
            errors.append(f"{label} modified production scene geometry")
    if payload.get("supports_added") != 0:
        errors.append(f"{label} added artificial support geometry")

    integrity = payload.get("trajectory_integrity")
    if not isinstance(integrity, Mapping) or integrity.get("unchanged") is not True:
        errors.append(f"{label} did not prove the IK trajectory unchanged")
    accounting = payload.get("endpoint_accounting")
    if not isinstance(accounting, Mapping) or accounting.get("complete") is not True:
        errors.append(f"{label} endpoint accounting is incomplete")

    binding = payload.get("route_binding")
    if not isinstance(binding, Mapping):
        errors.append(f"{label} route_binding is missing")
        binding = {}
    expected_binding = {
        "cell_key": cell_key,
        "trajectory_6dof": trajectory_6dof,
        "hand_source": hand_source,
        "task": task,
        "hand_type": hand_type,
        "data_id": data_id,
    }
    for field, expected in expected_binding.items():
        if binding.get(field) != expected:
            errors.append(
                f"{label} route binding mismatch for {field}: "
                f"expected={expected} actual={binding.get(field)}"
            )

    for field in ("adapter_manifest", "source_keypoints"):
        bound_path = resolve_recorded_path(
            root, binding.get(field), repo_root=repo_root
        )
        expected_path = resolve_recorded_path(
            root, input_manifest.get(field), repo_root=repo_root
        )
        if bound_path is None or expected_path is None:
            errors.append(f"{label} missing bound {field}")
            continue
        try:
            paths_match = bound_path.resolve() == expected_path.resolve()
        except OSError:
            paths_match = False
        if not paths_match or not bound_path.is_file():
            errors.append(f"{label} has stale/cross-route {field}")
            continue
        try:
            actual_sha = file_sha256(bound_path)
        except OSError:
            actual_sha = None
        if binding.get(f"{field}_sha256") != actual_sha:
            errors.append(f"{label} {field} hash mismatch")

    outputs = payload.get("outputs")
    consumed_scene = None
    consumed_scene_sha = None
    if isinstance(outputs, Mapping):
        consumed_scene = resolve_recorded_path(
            root, outputs.get("scene_act_xml"), repo_root=repo_root
        )
        consumed_scene_sha = outputs.get("scene_act_sha256_after")
    if consumed_scene is None or not consumed_scene.is_file():
        errors.append(f"{label} consumed scene is missing")
    else:
        try:
            actual_scene_sha = file_sha256(consumed_scene)
        except OSError:
            actual_scene_sha = None
        if consumed_scene_sha != actual_scene_sha:
            errors.append(f"{label} consumed scene hash mismatch")
    return errors


def authoritative_spider_aligned_tracking(
    output_manifest: Mapping[str, Any],
    *,
    root: Path,
    cell_key: str,
    repo_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Load the standalone aligned-QC report and prove what it evaluated.

    The report named by ``object_tracking_quality_file`` is authoritative.  An
    embedded payload alone is deliberately insufficient because it can be
    stale while the standalone post-alignment report has changed.
    """

    errors: list[str] = []
    evidence: dict[str, Any] = {
        "source": "spider_output_manifest.object_tracking_quality_file",
        "path": None,
        "aligned_trajectory": None,
    }
    report_path = resolve_recorded_path(
        root,
        output_manifest.get("object_tracking_quality_file"),
        repo_root=repo_root,
    )
    if report_path is None:
        return {}, evidence, ["missing SPIDER authoritative aligned object_tracking_quality_file"]
    evidence["path"] = str(report_path)
    if cell_key not in report_path.parts:
        errors.append(
            "SPIDER authoritative aligned object_tracking_quality_file does not "
            f"belong to exact cell {cell_key}: {report_path}"
        )
    report = load_json_object(report_path)
    if report is None:
        errors.append(
            f"SPIDER authoritative aligned object tracking report is missing/invalid: {report_path}"
        )
        report = {}

    embedded = output_manifest.get("object_tracking_quality")
    if not isinstance(embedded, Mapping):
        errors.append("missing embedded SPIDER object_tracking_quality")
    elif report and dict(embedded) != report:
        errors.append(
            "SPIDER embedded object_tracking_quality does not match the authoritative standalone report"
        )

    alignment = output_manifest.get("alignment")
    if not isinstance(alignment, Mapping) or alignment.get("applied") is not True:
        errors.append("SPIDER authoritative aligned QC has no applied alignment")
        alignment = {}
    aligned_path = resolve_recorded_path(
        root, alignment.get("aligned_trajectory"), repo_root=repo_root
    )
    evidence["aligned_trajectory"] = str(aligned_path) if aligned_path else None
    if aligned_path is None or not aligned_path.is_file():
        errors.append("SPIDER authoritative aligned QC trajectory is missing")
    else:
        if cell_key not in aligned_path.parts:
            errors.append(
                "SPIDER aligned QC trajectory does not belong to exact cell "
                f"{cell_key}: {aligned_path}"
            )
        if "aligned" not in aligned_path.name:
            errors.append(
                f"SPIDER authoritative QC trajectory is not aligned: {aligned_path}"
            )

    tracked_path = resolve_recorded_path(
        root, report.get("mjwp_trajectory"), repo_root=repo_root
    )
    evidence["reported_mjwp_trajectory"] = str(tracked_path) if tracked_path else None
    if tracked_path is None or not tracked_path.is_file():
        errors.append("SPIDER authoritative QC report has no existing mjwp_trajectory")
    elif aligned_path is not None:
        try:
            paths_match = tracked_path.resolve() == aligned_path.resolve()
        except OSError:
            paths_match = False
        if not paths_match:
            errors.append(
                "SPIDER authoritative QC did not evaluate alignment.aligned_trajectory"
            )

    for field in ("scene", "reference_trajectory"):
        path = resolve_recorded_path(root, report.get(field), repo_root=repo_root)
        evidence[field] = str(path) if path else None
        if path is None or not path.is_file():
            errors.append(f"SPIDER authoritative QC report has no existing {field}")
    return report, evidence, errors
