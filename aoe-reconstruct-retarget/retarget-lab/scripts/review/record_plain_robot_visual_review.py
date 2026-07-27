#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoe_retarget_lab.artifact_policy import (  # noqa: E402
    EXCLUDED_VISUAL_EVIDENCE,
    PLAIN_ALIGNED_ROBOT_NAMES,
)
from aoe_retarget_lab.admission_evidence import (  # noqa: E402
    authoritative_spider_aligned_tracking,
    raw_to_processed_invariance_errors,
    spider_rest_support_errors,
)
from aoe_retarget_lab.tracking_quality import (  # noqa: E402
    validate_hard_tracking_admission,
)


REVIEW_POLICY = "retarget_only_plain_mujoco_numerical_and_visual_v3"


def load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_artifact_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    candidates = [root / path, REPO_ROOT / path]
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def exact_cell_artifact(path: Path, cell: str) -> bool:
    return cell in path.parts


def exact_dai_route_binding_from_robot(
    source_robot: Path | None,
) -> tuple[Path | None, Path | None, list[str]]:
    """Resolve the adapter/keypoints for the exact DAI task rendered by robot.

    Cell keys intentionally do not encode ``task`` or ``data_id``.  Deriving
    these paths from the authoritative plain robot artifact prevents an inline
    report from another task in the same trajectory/hand cell from satisfying
    numerical admission.
    """

    label = "exact DAI robot route binding"
    if source_robot is None:
        return None, None, [f"{label}: missing source.robot"]
    try:
        robot = source_robot.resolve()
    except OSError:
        robot = source_robot
    parts = robot.parts
    indices = [index for index, part in enumerate(parts) if part == "retargeting_outputs"]
    if not indices:
        return None, None, [f"{label}: source.robot is outside retargeting_outputs: {robot}"]
    index = indices[-1]
    suffix = parts[index + 1 :]
    if len(suffix) < 5:
        return None, None, [f"{label}: malformed source.robot path: {robot}"]
    _, hand_type, task, data_id = suffix[:4]
    if hand_type not in {"left", "right", "bimanual"} or not task or not data_id:
        return None, None, [f"{label}: invalid hand/task/data_id path: {robot}"]
    route_root = Path(*parts[:index])
    adapter_manifest = route_root / "raw_dir" / "adapter_manifest.json"
    processed_keypoints = (
        route_root
        / "retargeting_outputs"
        / "mano"
        / hand_type
        / task
        / data_id
        / "trajectory_keypoints.npz"
    )
    errors = []
    if not adapter_manifest.is_file():
        errors.append(f"{label}: missing adapter manifest: {adapter_manifest}")
    if not processed_keypoints.is_file():
        errors.append(f"{label}: missing processed keypoints: {processed_keypoints}")
    return adapter_manifest, processed_keypoints, errors


def cell_artifact_manifests(
    matrix_run: Path,
    cell: str,
    manifest: dict[str, Any],
) -> list[tuple[dict[str, Any], Path, str]]:
    sources: list[tuple[dict[str, Any], Path, str]] = [
        (manifest, matrix_run, "matrix_cell_manifest")
    ]
    candidate_roots = [matrix_run]
    matrix_manifest = load_json(matrix_run / "reuse_12_demo_manifest.json")
    source_run = (
        (manifest.get("reuse_policy") or {}).get("source_run")
        or matrix_manifest.get("source_run")
    )
    if source_run:
        source_path = Path(str(source_run)).expanduser()
        candidate_roots.append(
            source_path if source_path.is_absolute() else matrix_run.parent / source_path
        )

    source_robot = resolve_artifact_path(
        matrix_run, str((manifest.get("assets") or {}).get("source.robot") or "")
    )
    if source_robot is not None:
        for parent in source_robot.parents:
            if (parent / "assets" / "cells" / cell).is_dir():
                candidate_roots.append(parent)
                break

    seen: set[Path] = set()
    for root in candidate_roots:
        asset_manifest_path = root / "assets" / "cells" / cell / "asset_manifest.json"
        if asset_manifest_path in seen:
            continue
        seen.add(asset_manifest_path)
        asset_manifest = load_json(asset_manifest_path)
        if asset_manifest:
            sources.append((asset_manifest, root, str(asset_manifest_path)))
    return sources


def find_dai_artifact(
    *,
    cell: str,
    sources: list[tuple[dict[str, Any], Path, str]],
    inline_key: str,
    asset_key: str,
    fallback_paths: list[Path],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    resolution_errors: list[str] = []
    for source_manifest, _, source_label in sources:
        payload = (source_manifest.get("provenance") or {}).get(inline_key)
        if isinstance(payload, dict) and payload:
            return payload, {"source": f"{source_label}:provenance.{inline_key}"}, resolution_errors

    for source_manifest, root, source_label in sources:
        value = (source_manifest.get("assets") or {}).get(asset_key)
        path = resolve_artifact_path(root, str(value or ""))
        if path is None:
            continue
        if not exact_cell_artifact(path, cell):
            resolution_errors.append(
                f"{asset_key} does not belong to exact cell {cell}: {path}"
            )
            continue
        payload = load_json(path)
        if payload:
            return payload, {"source": source_label, "path": str(path)}, resolution_errors

    seen: set[Path] = set()
    for path in fallback_paths:
        if path in seen:
            continue
        seen.add(path)
        if not exact_cell_artifact(path, cell):
            resolution_errors.append(
                f"{asset_key} fallback does not belong to exact cell {cell}: {path}"
            )
            continue
        payload = load_json(path)
        if payload:
            return payload, {"source": "exact_cell_path", "path": str(path)}, resolution_errors
    return {}, {"source": "missing", "path": None}, resolution_errors


def artifact_evidence(
    payload: dict[str, Any],
    location: dict[str, Any],
) -> dict[str, Any]:
    return {
        **location,
        "status": payload.get("status") if payload else "missing",
        "errors": payload.get("errors") if payload else ["artifact missing"],
        "artifact": payload,
    }


def require_ok_status(label: str, payload: dict[str, Any]) -> list[str]:
    if not payload:
        return [f"missing {label}"]
    if payload.get("status") != "ok":
        return [
            f"{label} status={payload.get('status', 'missing')}: "
            f"errors={payload.get('errors')}"
        ]
    return []


def route_identity_errors(
    cell: str,
    manifest: dict[str, Any],
    route_input_qc: dict[str, Any],
) -> list[str]:
    settings = manifest.get("settings") or {}
    trajectory = str(settings.get("trajectory_6dof") or "")
    hand_source = str(settings.get("hand_source") or "")
    retargeter = str(settings.get("retargeting") or "")
    expected_dai_cell = (
        cell
        if retargeter == "do_as_i_do"
        else cell.rsplit("__retarget_", 1)[0] + "__retarget_do_as_i_do"
    )
    expected = {
        "expected_cell_key": expected_dai_cell,
        "trajectory": trajectory,
        "hand_source": hand_source,
    }
    errors: list[str] = []
    for key, value in expected.items():
        if route_input_qc.get(key) != value:
            errors.append(
                f"route_input_qc {key}={route_input_qc.get(key)}, expected {value}"
            )
    return errors


def interaction_quality_errors(tracking: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    interaction = tracking.get("interaction_quality")
    if not isinstance(interaction, dict):
        return {}, ["missing object_tracking_quality.interaction_quality"]
    errors = require_ok_status("interaction_quality", interaction)
    sides = interaction.get("sides") or {}
    active_sides = interaction.get("active_sides") or []
    for side in active_sides:
        quality = sides.get(side) if isinstance(sides, dict) else None
        if not isinstance(quality, dict):
            errors.append(f"interaction_quality active side={side} is missing")
        elif quality.get("status") != "ok":
            errors.append(
                f"interaction_quality side={side} status={quality.get('status', 'missing')}: "
                f"errors={quality.get('errors')}"
            )
    return interaction, errors


def dai_numerical_validation(
    matrix_run: Path,
    cell: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    sources = cell_artifact_manifests(matrix_run, cell, manifest)
    source_identity_errors = [
        f"artifact manifest cell_key={source_manifest.get('cell_key')}, expected {cell}: "
        f"{source_label}"
        for source_manifest, _, source_label in sources
        if source_label != "matrix_cell_manifest"
        and source_manifest.get("cell_key") != cell
    ]
    source_robot = resolve_artifact_path(
        matrix_run, str((manifest.get("assets") or {}).get("source.robot") or "")
    )
    expected_adapter, expected_keypoints, exact_binding_errors = (
        exact_dai_route_binding_from_robot(source_robot)
    )
    metadata_roots = [
        root / "assets" / "cells" / cell / "retargeting" / "do_as_i_do" / "metadata"
        for _, root, _ in sources
    ]
    route_input_qc, route_location, route_resolution_errors = find_dai_artifact(
        cell=cell,
        sources=sources,
        inline_key="route_input_qc",
        asset_key="retargeting.quality.route_input_qc",
        fallback_paths=[root / "route_input_qc.json" for root in metadata_roots],
    )
    raw_to_processed, raw_location, raw_resolution_errors = find_dai_artifact(
        cell=cell,
        sources=sources,
        inline_key="raw_to_processed_hoi_invariance",
        asset_key="retargeting.quality.raw_to_processed_hoi_invariance",
        fallback_paths=[
            root / "raw_to_processed_hoi_invariance.json" for root in metadata_roots
        ],
    )
    tracking_fallbacks = [
        root / "mjwp_object_tracking_quality.json" for root in metadata_roots
    ]
    if source_robot is not None:
        tracking_fallbacks.insert(0, source_robot.parent / "mjwp_object_tracking_quality.json")
    tracking, tracking_location, tracking_resolution_errors = find_dai_artifact(
        cell=cell,
        sources=sources,
        inline_key="retarget_quality",
        asset_key="retargeting.quality.mjwp_object_tracking",
        fallback_paths=tracking_fallbacks,
    )
    interaction, interaction_errors = interaction_quality_errors(tracking)
    hard_tracking_admission = validate_hard_tracking_admission(tracking)
    errors = [
        *source_identity_errors,
        *route_resolution_errors,
        *raw_resolution_errors,
        *tracking_resolution_errors,
        *exact_binding_errors,
        *require_ok_status("route_input_qc", route_input_qc),
        *route_identity_errors(cell, manifest, route_input_qc),
        *raw_to_processed_invariance_errors(
            raw_to_processed,
            label="exact-route raw_to_processed_hoi_invariance",
            trajectory_6dof=str((manifest.get("settings") or {}).get("trajectory_6dof") or ""),
            hand_source=str((manifest.get("settings") or {}).get("hand_source") or ""),
            hand_type=(manifest.get("settings") or {}).get("dai_hand_type")
            or (manifest.get("settings") or {}).get("hand_type"),
            adapter_manifest=expected_adapter,
            processed_keypoints=expected_keypoints,
        ),
        *require_ok_status("object_tracking_quality", tracking),
        *interaction_errors,
        *(
            f"hard_tracking_admission: {error}"
            for error in hard_tracking_admission.get("errors") or []
        ),
    ]
    return {
        "status": "invalid" if errors else "ok",
        "errors": errors,
        "evidence": {
            "route_input_qc": artifact_evidence(route_input_qc, route_location),
            "raw_to_processed_hoi_invariance": artifact_evidence(
                raw_to_processed, raw_location
            ),
            "object_tracking_quality": artifact_evidence(tracking, tracking_location),
            "interaction_quality": {
                "source": "object_tracking_quality.interaction_quality",
                "status": interaction.get("status") if interaction else "missing",
                "errors": interaction.get("errors") if interaction else ["artifact missing"],
                "artifact": interaction,
            },
            "hard_tracking_admission": hard_tracking_admission,
        },
    }


def spider_numerical_validation(
    matrix_run: Path,
    cell: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    root = matrix_run / "intermediates" / "retargeting" / "spider" / cell
    input_path = root / "spider_input_manifest.json"
    output_path = root / "spider_output_manifest.json"
    input_manifest = load_json(input_path)
    output_manifest = load_json(output_path)
    route_input_qc = input_manifest.get("route_input_qc")
    route_location: dict[str, Any] = {
        "source": "spider_input_manifest.route_input_qc",
        "path": str(input_path),
    }
    if not isinstance(route_input_qc, dict) or not route_input_qc:
        route_input_qc = output_manifest.get("route_input_qc")
        route_location = {
            "source": "spider_output_manifest.route_input_qc",
            "path": str(output_path),
        }
    if not isinstance(route_input_qc, dict) or not route_input_qc:
        route_path = root / "route_input_qc.json"
        route_input_qc = load_json(route_path)
        route_location = {"source": "exact_cell_path", "path": str(route_path)}

    tracking, tracking_location, tracking_resolution_errors = (
        authoritative_spider_aligned_tracking(
            output_manifest,
            root=root,
            cell_key=cell,
            repo_root=REPO_ROOT,
        )
    )

    raw_to_processed = input_manifest.get("raw_to_processed_hoi_invariance")
    raw_location: dict[str, Any] = {
        "source": "spider_input_manifest.raw_to_processed_hoi_invariance",
        "path": str(input_path),
    }
    if not isinstance(raw_to_processed, dict):
        raw_to_processed = output_manifest.get("raw_to_processed_hoi_invariance")
        raw_location = {
            "source": "spider_output_manifest.raw_to_processed_hoi_invariance",
            "path": str(output_path),
        }
    rest_support = output_manifest.get("rest_support_provenance")
    rest_support_validation_errors = spider_rest_support_errors(
        rest_support,
        cell_key=cell,
        trajectory_6dof=str(output_manifest.get("trajectory_6dof") or ""),
        hand_source=str(output_manifest.get("hand_source") or ""),
        task=str(output_manifest.get("task") or ""),
        hand_type=str(output_manifest.get("resolved_hand_type") or ""),
        data_id=output_manifest.get("data_id"),
        input_manifest=input_manifest,
        root=root,
        repo_root=REPO_ROOT,
    )

    interaction, interaction_errors = interaction_quality_errors(tracking)
    hard_tracking_admission = validate_hard_tracking_admission(tracking)
    settings = manifest.get("settings") or {}
    expected_dai_cell = cell.rsplit("__retarget_", 1)[0] + "__retarget_do_as_i_do"
    errors = [
        *require_ok_status("SPIDER input manifest", input_manifest),
        *tracking_resolution_errors,
        *raw_to_processed_invariance_errors(
            raw_to_processed,
            label="exact-route raw_to_processed_hoi_invariance",
            trajectory_6dof=str(settings.get("trajectory_6dof") or ""),
            hand_source=str(settings.get("hand_source") or ""),
            hand_type=str(output_manifest.get("resolved_hand_type") or ""),
            adapter_manifest=resolve_artifact_path(
                root, str(input_manifest.get("adapter_manifest") or "")
            ),
            processed_keypoints=resolve_artifact_path(
                root, str(input_manifest.get("source_keypoints") or "")
            ),
        ),
        *rest_support_validation_errors,
        *require_ok_status("route_input_qc", route_input_qc),
        *route_identity_errors(cell, manifest, route_input_qc),
        *require_ok_status("object_tracking_quality", tracking),
        *interaction_errors,
        *(
            f"hard_tracking_admission: {error}"
            for error in hard_tracking_admission.get("errors") or []
        ),
    ]
    if not output_manifest:
        errors.append("missing SPIDER output manifest")
    expected_input_fields = {
        "expected_dai_cell_key": expected_dai_cell,
        "trajectory_6dof": settings.get("trajectory_6dof"),
        "hand_source": settings.get("hand_source"),
    }
    for key, value in expected_input_fields.items():
        if input_manifest.get(key) != value:
            errors.append(
                f"SPIDER input manifest {key}={input_manifest.get(key)}, expected {value}"
            )
    fallback = output_manifest.get("mjwp_fallback") or {}
    if fallback.get("applied"):
        errors.append(f"SPIDER mjwp_fallback applied: {fallback.get('reason')}")
    if output_manifest.get("cell_key") != cell:
        errors.append(
            f"SPIDER output manifest cell_key={output_manifest.get('cell_key')}, expected {cell}"
        )
    return {
        "status": "invalid" if errors else "ok",
        "errors": errors,
        "evidence": {
            "spider_input_manifest": str(input_path),
            "spider_output_manifest": str(output_path),
            "route_input_qc": artifact_evidence(route_input_qc, route_location),
            "raw_to_processed_hoi_invariance": artifact_evidence(
                raw_to_processed if isinstance(raw_to_processed, dict) else {},
                raw_location,
            ),
            "rest_support_provenance": {
                "source": "spider_output_manifest.rest_support_provenance",
                "path": str(output_path),
                "status": (
                    rest_support.get("status")
                    if isinstance(rest_support, dict)
                    else "missing"
                ),
                "errors": rest_support_validation_errors,
                "artifact": rest_support if isinstance(rest_support, dict) else {},
            },
            "object_tracking_quality": artifact_evidence(tracking, tracking_location),
            "interaction_quality": {
                "source": "object_tracking_quality.interaction_quality",
                "status": interaction.get("status") if interaction else "missing",
                "errors": interaction.get("errors") if interaction else ["artifact missing"],
                "artifact": interaction,
            },
            "hard_tracking_admission": hard_tracking_admission,
        },
    }


def cell_numerical_validation(
    matrix_run: Path,
    cell: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    retargeter = str((manifest.get("settings") or {}).get("retargeting") or "")
    if retargeter == "do_as_i_do":
        return dai_numerical_validation(matrix_run, cell, manifest)
    if retargeter == "spider":
        return spider_numerical_validation(matrix_run, cell, manifest)
    return {
        "status": "invalid",
        "errors": [f"unsupported retargeter: {retargeter}"],
        "evidence": {},
    }


def collect_retarget_cells(matrix_run: Path) -> tuple[list[dict[str, Any]], list[str]]:
    reviewed: list[dict[str, Any]] = []
    errors: list[str] = []
    for cell_dir in sorted((matrix_run / "cells").glob("*")):
        if not cell_dir.is_dir():
            continue
        manifest = load_json(cell_dir / "manifest.json")
        retargeter = str((manifest.get("settings") or {}).get("retargeting") or "")
        if retargeter not in {"do_as_i_do", "spider"}:
            continue
        source_robot = str((manifest.get("assets") or {}).get("source.robot") or "")
        if Path(source_robot).name not in PLAIN_ALIGNED_ROBOT_NAMES:
            errors.append(f"{cell_dir.name}: source.robot is not a plain aligned MuJoCo video")
        if not (cell_dir / "robot.mp4").is_file():
            errors.append(f"{cell_dir.name}: robot.mp4 is missing")
        provenance_validation = manifest.get("provenance_validation") or {}
        provenance_status = str(provenance_validation.get("status") or "")
        provenance_errors = provenance_validation.get("errors") or []
        if provenance_status != "ok" or provenance_errors:
            errors.append(
                f"{cell_dir.name}: provenance validation is not ok: "
                f"status={provenance_status or 'missing'} errors={provenance_errors}"
            )
        numerical_validation = cell_numerical_validation(
            matrix_run, cell_dir.name, manifest
        )
        errors.extend(
            f"{cell_dir.name}: numerical validation failed: {error}"
            for error in numerical_validation.get("errors") or []
        )
        reviewed.append(
            {
                "cell": cell_dir.name,
                "retargeter": retargeter,
                "source_robot": source_robot,
                "provenance": manifest.get("provenance") or {},
                "provenance_validation": provenance_validation,
                "numerical_validation": numerical_validation,
            }
        )
    return reviewed, errors


def completeness_errors(reviewed: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for name in ("do_as_i_do", "spider"):
        count = sum(item["retargeter"] == name for item in reviewed)
        if count != 4:
            errors.append(f"expected 4 {name} cells, found {count}")
    return errors


def matrix_completeness_errors(
    matrix_run: Path,
    reviewed: list[dict[str, Any]],
    cell_errors: list[str],
) -> list[str]:
    errors = completeness_errors(reviewed)
    manifest = load_json(matrix_run / "reuse_12_demo_manifest.json")
    if manifest:
        if manifest.get("cells") != 12:
            errors.append(f"manifest cells={manifest.get('cells')}, expected 12")
        if manifest.get("composed") != 12:
            errors.append(f"manifest composed={manifest.get('composed')}, expected 12")
        missing = manifest.get("missing_cells")
        if missing:
            errors.append(f"manifest missing_cells is not empty: {missing}")
    else:
        errors.extend(
            error
            for error in cell_errors
            if "source.robot is not a plain aligned" in error
            or "robot.mp4 is missing" in error
        )
    return list(dict.fromkeys(errors))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record temporal visual review of aligned plain DAI/SPIDER MuJoCo videos."
    )
    parser.add_argument("--matrix-run", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--passed-cell", action="append", default=[])
    parser.add_argument("--dai-status", choices=("pass", "fail"))
    parser.add_argument("--spider-status", choices=("pass", "fail"))
    parser.add_argument("--failed-cell", action="append", default=[])
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    matrix_run = args.matrix_run.expanduser().resolve()
    reviewed, errors = collect_retarget_cells(matrix_run)
    by_name = {item["cell"]: item for item in reviewed}
    failed_cells = sorted(set(args.failed_cell))
    visually_passed_cells = sorted(set(args.passed_cell))
    if not visually_passed_cells:
        legacy_passed_retargeters = {
            name
            for name, status in {
                "do_as_i_do": args.dai_status,
                "spider": args.spider_status,
            }.items()
            if status == "pass"
        }
        visually_passed_cells = sorted(
            item["cell"]
            for item in reviewed
            if item["retargeter"] in legacy_passed_retargeters
            and item["cell"] not in failed_cells
        )
    unknown_successes = sorted(set(visually_passed_cells) - set(by_name))
    conflicting_cells = sorted(set(visually_passed_cells) & set(failed_cells))
    invalid_successes = [
        error
        for error in errors
        if any(error.startswith(f"{cell}:") for cell in visually_passed_cells)
    ]
    validation_errors = [
        *(f"unknown successful cell: {cell}" for cell in unknown_successes),
        *(f"cell marked both successful and failed: {cell}" for cell in conflicting_cells),
        *invalid_successes,
    ]
    rejected_successes = {
        *unknown_successes,
        *conflicting_cells,
        *(
            cell
            for cell in visually_passed_cells
            if any(error.startswith(f"{cell}:") for error in invalid_successes)
        ),
    }
    successful_cells = sorted(set(visually_passed_cells) - rejected_successes)
    numerically_rejected_passed_cells = sorted(
        cell
        for cell in visually_passed_cells
        if cell in by_name
        and (by_name[cell].get("numerical_validation") or {}).get("status") != "ok"
    )
    successful_retargeters = {
        by_name[cell]["retargeter"]
        for cell in successful_cells
        if cell in by_name
    }
    visually_passed_retargeters = {
        by_name[cell]["retargeter"]
        for cell in visually_passed_cells
        if cell in by_name
    }
    statuses = {
        "do_as_i_do": "pass" if "do_as_i_do" in successful_retargeters else "fail",
        "spider": "pass" if "spider" in successful_retargeters else "fail",
    }
    visual_statuses = {
        "do_as_i_do": (
            "pass"
            if "do_as_i_do" in visually_passed_retargeters
            else (args.dai_status or "fail")
        ),
        "spider": (
            "pass"
            if "spider" in visually_passed_retargeters
            else (args.spider_status or "fail")
        ),
    }
    passed = bool(successful_cells) and not validation_errors
    matrix_errors = matrix_completeness_errors(matrix_run, reviewed, errors)
    payload = {
        "policy": REVIEW_POLICY,
        "evidence_kind": "plain_aligned_mujoco",
        "status": "pass" if passed else "fail",
        "reviewer": args.reviewer,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "matrix_run": matrix_run.name,
        "retargeters": statuses,
        "visual_review": {
            "status": "pass" if visually_passed_cells else "fail",
            "retargeters": visual_statuses,
            "passed_cells": visually_passed_cells,
            "failed_cells": failed_cells,
        },
        "reviewed_cell_count": len(reviewed),
        "reviewed_cells": reviewed,
        "successful_cell_count": len(successful_cells),
        "successful_cells": successful_cells,
        "visually_passed_cells": visually_passed_cells,
        "numerically_rejected_passed_cells": numerically_rejected_passed_cells,
        "failed_cells": failed_cells,
        "validation_errors": validation_errors,
        "matrix_complete": not matrix_errors,
        "matrix_completeness_errors": matrix_errors,
        "nonblocking_cell_errors": errors,
        "notes": args.notes,
        "excluded_evidence": list(EXCLUDED_VISUAL_EVIDENCE),
    }
    output = args.output_json or matrix_run / "plain_robot_visual_review.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": payload["status"], "errors": validation_errors}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
