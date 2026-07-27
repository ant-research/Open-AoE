#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from fractions import Fraction
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoe_retarget_lab.artifact_policy import (  # noqa: E402
    PLAIN_ALIGNED_ROBOT_NAMES,
    forbidden_robot_rgb_artifacts,
)
from aoe_retarget_lab.admission_evidence import (  # noqa: E402
    authoritative_spider_aligned_tracking,
    raw_to_processed_invariance_errors,
    spider_rest_support_errors,
)
from aoe_retarget_lab.tracking_quality import (  # noqa: E402
    validate_hard_tracking_admission,
)
from aoe_retarget_lab.alignment import (  # noqa: E402
    valid_dai_alignment,
    valid_spider_alignment,
)


RETARGET_BACKENDS = {"do_as_i_do", "spider"}
EXPLICIT_INPUT_UNAVAILABLE_STATUSES = {
    "disabled_by_input_qc",
    "failed_input_preflight",
    "input_unavailable",
    "invalid_input",
    "invalid_route_input",
    "preflight_failed",
    "skipped_invalid_input",
    "unavailable",
    "unavailable_input",
}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def is_exact_retarget_route(cell: str) -> bool:
    return any(cell.endswith(f"__retarget_{backend}") for backend in RETARGET_BACKENDS)


def route_backend(cell: str) -> str | None:
    marker = "__retarget_"
    if marker not in cell:
        return None
    backend = cell.rsplit(marker, 1)[1]
    return backend if backend in RETARGET_BACKENDS else None


def route_identity(cell: str) -> dict[str, str] | None:
    prefix = "traj_"
    hand_marker = "__hand_"
    retarget_marker = "__retarget_"
    if not cell.startswith(prefix) or hand_marker not in cell or retarget_marker not in cell:
        return None
    trajectory, remainder = cell[len(prefix) :].split(hand_marker, 1)
    hand_source, backend = remainder.rsplit(retarget_marker, 1)
    if not trajectory or not hand_source or backend not in RETARGET_BACKENDS:
        return None
    return {
        "route": cell,
        "trajectory_6dof": trajectory,
        "hand_source": hand_source,
        "retargeting": backend,
    }


def _exact_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        integer = int(value)
        if float(value) != float(integer):
            return None
        return integer
    except (TypeError, ValueError, OverflowError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_input_key(cell: str) -> str:
    """Return the input/preflight key shared by an exact DAI/SPIDER route."""
    if cell.endswith("__retarget_spider"):
        return cell.rsplit("__retarget_", 1)[0] + "__retarget_do_as_i_do"
    return cell


def _evidence_path(reference: dict[str, Any], evidence_root: Path) -> Path | None:
    raw = reference.get("path")
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else evidence_root / path


def validate_route_preflight_evidence(
    cell: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Validate independent, byte-bound evidence for a route input claim."""

    errors: list[str] = []
    identity = route_identity(cell)
    if identity is None:
        return {"status": "invalid", "errors": ["invalid exact route identity"]}
    input_rc = _exact_int(entry.get("input_rc"))
    if input_rc is None:
        errors.append("route availability input_rc is missing or non-integral")

    reference = entry.get("preflight_evidence")
    if not isinstance(reference, dict):
        return {
            "status": "invalid",
            "errors": errors + ["missing preflight_evidence binding"],
        }
    evidence_root_raw = entry.get("_evidence_root")
    evidence_root = Path(str(evidence_root_raw)) if evidence_root_raw else None
    if evidence_root is None:
        errors.append("missing preflight evidence root")
        evidence_path = None
    else:
        evidence_path = _evidence_path(reference, evidence_root)
    expected_sha = str(reference.get("sha256") or "")
    if evidence_path is None:
        errors.append("preflight_evidence.path is missing")
    elif not evidence_path.is_file():
        errors.append(f"preflight evidence file is missing: {evidence_path}")
    if len(expected_sha) != 64:
        errors.append("preflight_evidence.sha256 is missing or invalid")

    payload: dict[str, Any] | None = None
    actual_sha: str | None = None
    if evidence_path is not None and evidence_path.is_file():
        actual_sha = _sha256(evidence_path)
        if expected_sha and actual_sha != expected_sha:
            errors.append(
                f"preflight evidence sha256 mismatch: {actual_sha} != {expected_sha}"
            )
        try:
            payload = load_json(evidence_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if not isinstance(payload, dict):
            errors.append("preflight evidence is not valid JSON object")

    if isinstance(payload, dict):
        for key, expected in identity.items():
            if payload.get(key) != expected:
                errors.append(
                    f"preflight evidence {key}={payload.get(key)!r}, expected {expected!r}"
                )
        payload_rc = _exact_int(payload.get("input_rc"))
        if input_rc is None or payload_rc != input_rc:
            errors.append(
                f"preflight evidence input_rc={payload.get('input_rc')!r} "
                f"does not match route input_rc={entry.get('input_rc')!r}"
            )
        for key in ("trajectory_6dof", "hand_source", "status", "input_rc"):
            if key not in reference:
                errors.append(f"preflight_evidence binding missing {key}")
            elif reference.get(key) != payload.get(key):
                errors.append(
                    f"preflight_evidence binding {key}={reference.get(key)!r} "
                    f"does not match payload {payload.get(key)!r}"
                )
        validation = payload.get("evidence_validation")
        if not isinstance(validation, dict) or validation.get("status") != "ok":
            errors.append("preflight payload evidence_validation is not ok")
        elif validation.get("errors"):
            errors.append(
                f"preflight payload evidence_validation has errors={validation.get('errors')}"
            )

        adapter_binding = payload.get("underlying_adapter_manifest")
        if not isinstance(adapter_binding, dict):
            errors.append("preflight payload missing underlying_adapter_manifest")
        elif evidence_root is not None:
            adapter_path = _evidence_path(adapter_binding, evidence_root)
            recorded_exists = adapter_binding.get("exists")
            actual_exists = bool(adapter_path is not None and adapter_path.is_file())
            if recorded_exists is not actual_exists:
                errors.append(
                    "underlying adapter exists flag does not match filesystem: "
                    f"recorded={recorded_exists!r}, actual={actual_exists}"
                )
            if actual_exists and adapter_path is not None:
                adapter_sha = str(adapter_binding.get("sha256") or "")
                actual_adapter_sha = _sha256(adapter_path)
                if len(adapter_sha) != 64 or adapter_sha != actual_adapter_sha:
                    errors.append("underlying adapter sha256 is missing or mismatched")
                try:
                    adapter = load_json(adapter_path)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    adapter = None
                if not isinstance(adapter, dict):
                    errors.append("underlying adapter is not valid JSON object")
                else:
                    actual_hand = adapter.get("hand_source")
                    if actual_hand != identity["hand_source"]:
                        errors.append(
                            f"underlying adapter hand_source={actual_hand!r}, "
                            f"expected {identity['hand_source']!r}"
                        )
                    if adapter_binding.get("hand_source") != actual_hand:
                        errors.append("underlying adapter recorded hand_source is inconsistent")
                    adapter_trajectory = adapter.get("trajectory_6dof")
                    if (
                        adapter_trajectory is not None
                        and adapter_trajectory != identity["trajectory_6dof"]
                    ):
                        errors.append(
                            "underlying adapter trajectory_6dof conflicts with exact route"
                        )
                    for recorded_key, adapter_key in (
                        ("retarget_input_qc_status", "retarget_input_qc"),
                        ("adapter_rigid_invariance_status", "adapter_rigid_invariance"),
                    ):
                        actual_status = (adapter.get(adapter_key) or {}).get("status")
                        if adapter_binding.get(recorded_key) != actual_status:
                            errors.append(
                                f"underlying adapter {recorded_key} is inconsistent"
                            )
                    if input_rc == 0:
                        if (adapter.get("retarget_input_qc") or {}).get("status") != "ok":
                            errors.append("available input adapter retarget_input_qc is not ok")
                        if (
                            (adapter.get("adapter_rigid_invariance") or {}).get("status")
                            != "ok"
                        ):
                            errors.append(
                                "available input adapter adapter_rigid_invariance is not ok"
                            )
            else:
                if input_rc == 0:
                    errors.append("available route has no underlying adapter manifest")
                for key in (
                    "path",
                    "sha256",
                    "hand_source",
                    "retarget_input_qc_status",
                    "adapter_rigid_invariance_status",
                ):
                    if adapter_binding.get(key) is not None:
                        errors.append(
                            f"absent underlying adapter has inconsistent {key}"
                        )

    return {
        "status": "ok" if not errors else "invalid",
        "errors": errors,
        "path": str(evidence_path) if evidence_path is not None else None,
        "sha256": actual_sha,
        "payload": payload,
    }


def _post_retarget_invalid(entry: dict[str, Any] | None) -> tuple[bool, str | None]:
    if not isinstance(entry, dict) or _exact_int(entry.get("input_rc")) != 0:
        return False, None
    post_rc = _exact_int(entry.get("post_retarget_rc"))
    status = str(entry.get("status") or "").strip().lower()
    failed_status = status in {
        "failed",
        "invalid",
        "post_retarget_invalid",
        "post_retarget_failed",
    }
    if post_rc not in (None, 0):
        return True, f"post_retarget_rc={post_rc}"
    if failed_status:
        return True, f"status={status}"
    if entry.get("available") is False:
        return True, "available=false after successful input preflight"
    return False, None


def route_availability_map(
    matrix_root: Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Load exact-route input availability without guessing from missing outputs.

    The fresh full-run records preflight return codes in its reuse manifest.  A
    matrix may materialize that source as ``reuse/source_run`` or point to it by
    name.  A top-level matrix declaration, when present, is authoritative.
    """

    merged: dict[str, dict[str, Any]] = {}
    source_run = str(manifest.get("source_run") or "")
    candidates: list[Path] = []
    if source_run:
        candidates.extend(
            [
                matrix_root.parent / source_run / "reuse" / "reuse_manifest.json",
                matrix_root / "reuse" / "source_run" / "reuse" / "reuse_manifest.json",
            ]
        )
    candidates.append(matrix_root / "reuse" / "reuse_manifest.json")
    for path in candidates:
        payload = load_json(path) or {}
        availability = payload.get("route_availability") or {}
        if not isinstance(availability, dict):
            continue
        for cell, entry in availability.items():
            if isinstance(entry, dict):
                merged[str(cell)] = {
                    **entry,
                    "availability_manifest": str(path),
                    "_evidence_root": str(path.parent.parent),
                }

    spider_items = manifest.get("spider") or []
    if isinstance(spider_items, list):
        for item in spider_items:
            if not isinstance(item, dict) or not item.get("cell"):
                continue
            item_entry = {
                "status": item.get("status"),
                "input_status": item.get("input_status"),
                "input_rc": item.get("input_rc"),
                "post_retarget_rc": item.get("returncode"),
                "available": bool(item.get("robot")),
                "preflight_evidence": item.get("preflight_evidence")
                or item.get("input_preflight"),
            }
            claimed_unavailable, _ = explicit_input_unavailability(item_entry)
            if claimed_unavailable or item_entry["preflight_evidence"] is not None:
                merged[str(item["cell"])] = {
                    **item_entry,
                    "availability_manifest": "matrix_spider_item",
                    "_evidence_root": str(matrix_root),
                }

    top_level = manifest.get("route_availability") or {}
    if isinstance(top_level, dict):
        for cell, entry in top_level.items():
            if isinstance(entry, dict):
                backend = route_backend(str(cell))
                source_root = (
                    matrix_root
                    if backend == "spider" or not source_run
                    else matrix_root.parent / source_run
                )
                merged[str(cell)] = {
                    **entry,
                    "availability_manifest": "matrix_manifest",
                    "_evidence_root": str(source_root),
                }
    return merged


def explicit_input_unavailability(entry: dict[str, Any] | None) -> tuple[bool, str | None]:
    if not isinstance(entry, dict):
        return False, None
    for key in ("input_rc", "preflight_rc", "route_input_rc"):
        if key not in entry or entry[key] is None:
            continue
        try:
            return (int(entry[key]) != 0, f"{key}={entry[key]}")
        except (TypeError, ValueError):
            continue
    status = str(entry.get("status") or entry.get("input_status") or "").strip().lower()
    if status in EXPLICIT_INPUT_UNAVAILABLE_STATUSES:
        return True, f"status={status}"
    if entry.get("available") is False and (
        entry.get("input_errors")
        or entry.get("input_qc")
        or "input" in str(entry.get("reason") or "").lower()
        or "preflight" in str(entry.get("reason") or "").lower()
    ):
        return True, str(entry.get("reason") or "available=false")
    return False, None


def route_availability(
    cell: str,
    availability: dict[str, dict[str, Any]],
    *,
    item: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_key = route_input_key(cell)
    # Availability is exact-backend local.  SPIDER may consume a DAI-formatted
    # keypoint bundle, but a disabled DAI simulation route does not by itself
    # prove that the independent SPIDER route input is unavailable.
    entry = availability.get(cell)
    if entry is None and isinstance(item, dict):
        item_entry = {
            "status": item.get("status"),
            "input_status": item.get("input_status"),
            "input_rc": item.get("input_rc"),
            "post_retarget_rc": item.get("returncode"),
            "available": bool(item.get("robot")),
            "preflight_evidence": item.get("preflight_evidence")
            or item.get("input_preflight"),
        }
        claimed, _ = explicit_input_unavailability(item_entry)
        if claimed or item_entry["preflight_evidence"] is not None:
            entry = {**item_entry, "availability_manifest": "route_item"}

    claimed_unavailable, claim_reason = explicit_input_unavailability(entry)
    post_invalid, post_reason = _post_retarget_invalid(entry)
    evidence_validation = (
        validate_route_preflight_evidence(cell, entry)
        if isinstance(entry, dict)
        else {"status": "missing", "errors": []}
    )
    trusted = evidence_validation.get("status") == "ok"
    if claimed_unavailable and trusted:
        status = "unavailable"
        reason = claim_reason
    elif post_invalid and trusted:
        status = "post_retarget_invalid"
        reason = post_reason
    elif isinstance(entry, dict) and not trusted:
        status = "invalid_evidence"
        reason = "preflight evidence is missing, stale, or inconsistent"
    elif isinstance(entry, dict):
        status = "available"
        reason = None
    else:
        status = "unknown"
        reason = None
    return {
        "status": status,
        "explicit_input_unavailable": status == "unavailable",
        "post_retarget_invalid": status == "post_retarget_invalid",
        "claimed_input_unavailable": claimed_unavailable,
        "reason": reason,
        "route_input_key": input_key,
        "evidence": entry,
        "preflight_evidence_validation": evidence_validation,
        "errors": list(evidence_validation.get("errors") or []),
    }


def rel_or_abs(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def resolve_manifest_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [root / path, REPO_ROOT / path]
    for parent in (root, *root.parents):
        candidates.append(parent / path)
        if parent.name == "experiments":
            candidates.append(parent.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else root / path


def video_probe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,duration,r_frame_rate,width,height",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(proc.stdout)
        streams = data.get("streams") or []
        stream = streams[0] if streams else {}
        return {"exists": True, **stream}
    except Exception as exc:
        return {"exists": True, "probe_error": str(exc)}


def frame_count(probe: dict[str, Any]) -> int | None:
    value = probe.get("nb_read_frames")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fps_value(probe: dict[str, Any]) -> float | None:
    value = probe.get("r_frame_rate")
    if not value or value == "0/0":
        return None
    try:
        fps = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None
    return fps if fps > 0 else None


def duration_value(probe: dict[str, Any]) -> float | None:
    value = probe.get("duration")
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def probe_errors_for_aligned_video(
    *,
    label: str,
    probe: dict[str, Any],
    target_frames: int | None,
    target_fps: float | None,
) -> list[str]:
    errors: list[str] = []
    if not probe.get("exists"):
        errors.append(f"{label}: video missing")
        return errors
    if probe.get("probe_error"):
        errors.append(f"{label}: ffprobe failed: {probe.get('probe_error')}")
        return errors
    frames = frame_count(probe)
    fps = fps_value(probe)
    duration = duration_value(probe)
    if target_frames and frames != target_frames:
        errors.append(f"{label}: frame_count={frames}, expected {target_frames}")
    if target_fps:
        if fps is None or abs(fps - target_fps) > 0.05:
            errors.append(f"{label}: fps={fps}, expected {target_fps:.4f}")
        expected_duration = (target_frames / target_fps) if target_frames else None
        if expected_duration and duration is not None and abs(duration - expected_duration) > max(0.08, expected_duration * 0.03):
            errors.append(f"{label}: duration={duration:.4f}, expected {expected_duration:.4f}")
    return errors


def nearest_raw_video(path: Path) -> Path | None:
    for parent in path.parents:
        candidate = parent / "raw_dir" / "raw.mp4"
        if candidate.exists():
            return candidate
    return None


def adapter_manifest_for_dai_alignment(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name.startswith("traj_") and parent.name.endswith("__retarget_do_as_i_do"):
            return parent / "raw_dir" / "adapter_manifest.json"
    return None


def raw_to_processed_hoi_for_dai_alignment(path: Path) -> Path | None:
    output_root = next(
        (parent for parent in path.parents if parent.name == "retargeting_outputs"),
        None,
    )
    if output_root is None:
        return None
    robot_dir = path.parent
    try:
        task = robot_dir.parent.name
        hand_type = robot_dir.parent.parent.name
    except IndexError:
        return None
    return (
        output_root
        / "mano"
        / hand_type
        / task
        / "0"
        / "raw_to_processed_hoi_invariance.json"
    )


def dai_raw_to_processed_findings(
    alignment_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    report_path = raw_to_processed_hoi_for_dai_alignment(alignment_path)
    report = load_json(report_path) if report_path is not None else None
    cell = next(
        (
            parent.name
            for parent in alignment_path.parents
            if parent.name.startswith("traj_")
            and "__retarget_do_as_i_do" in parent.name
        ),
        "",
    )
    route = cell[5:] if cell.startswith("traj_") else cell
    trajectory, hand_source = None, None
    if "__hand_" in route:
        trajectory, remainder = route.split("__hand_", 1)
        hand_source = remainder.split("__retarget_", 1)[0]
    hand_type = alignment_path.parent.parent.parent.name
    processed_keypoints = (
        report_path.with_name("trajectory_keypoints.npz")
        if report_path is not None
        else None
    )
    errors = raw_to_processed_invariance_errors(
        report,
        label="exact-route raw_to_processed_hoi_invariance",
        trajectory_6dof=trajectory,
        hand_source=hand_source,
        hand_type=hand_type,
        adapter_manifest=adapter_manifest_for_dai_alignment(alignment_path),
        processed_keypoints=processed_keypoints,
    )
    return errors, {
        "path": str(report_path) if report_path is not None else None,
        "report": report,
    }


def ego_retarget_input_qc_findings(
    alignment_path: Path,
    allow_legacy_unaligned: bool,
) -> tuple[list[str], dict[str, Any]]:
    if "traj_egoinfinity__" not in str(alignment_path):
        return [], {}
    adapter_path = adapter_manifest_for_dai_alignment(alignment_path)
    evidence: dict[str, Any] = {
        "adapter_manifest": str(adapter_path) if adapter_path else None,
        "retarget_input_qc": None,
    }
    if allow_legacy_unaligned:
        return [], evidence
    if adapter_path is None:
        return [f"Ego->DAI alignment has no resolvable adapter cell: {alignment_path}"], evidence
    adapter = load_json(adapter_path)
    if adapter is None:
        return [f"Ego->DAI alignment is missing adapter manifest: {adapter_path}"], evidence
    qc = adapter.get("retarget_input_qc") or {}
    evidence["retarget_input_qc"] = qc
    if qc.get("status") != "ok":
        return [
            f"Ego->DAI alignment has invalid retarget_input_qc at {adapter_path}: "
            f"status={qc.get('status')}, errors={qc.get('errors')}"
        ], evidence
    return [], evidence


def is_aligned_reference(value: str | None) -> bool:
    return bool(value) and "aligned" in Path(str(value)).name


def dai_alignment_errors(manifest_path: Path, allow_legacy_unaligned: bool) -> list[str]:
    if allow_legacy_unaligned:
        return []
    if not manifest_path.is_file():
        return [f"missing DAI alignment manifest: {manifest_path}"]
    if not valid_dai_alignment(manifest_path.parent, REPO_ROOT):
        return [
            f"DAI alignment {manifest_path}: strict canonical artifact/source-video "
            "binding or exact timing grid is invalid"
        ]
    return []


def dai_tracking_quality_findings(
    manifest_path: Path,
    data: dict[str, Any],
    allow_legacy_unaligned: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    tracking_quality = data.get("object_tracking_quality")
    pure_dai_baseline = "traj_do_as_i_do__" in str(manifest_path)
    admission = validate_hard_tracking_admission(
        tracking_quality if isinstance(tracking_quality, dict) else None
    )
    if admission.get("status") != "ok":
        status = (
            tracking_quality.get("status", "missing")
            if isinstance(tracking_quality, dict)
            else "missing"
        )
        message = (
            f"DAI MJWP object tracking status={status}; hard tracking admission invalid: "
            f"{manifest_path}; errors={admission.get('errors')}"
        )
        route = "pure DAI baseline" if pure_dai_baseline else "EgoInfinity-to-DAI route"
        # Alignment compatibility may never downgrade a numerical failure.
        errors.append(f"{route} numerical tracking error: {message}")
    return errors, warnings


def add_file_check(errors: list[str], root: Path, label: str, value: str | None) -> Path | None:
    path = rel_or_abs(root, value)
    if path is None:
        errors.append(f"{label}: missing path in manifest")
        return None
    if not path.exists():
        errors.append(f"{label}: file does not exist: {path}")
    return path


def audit_spider(matrix_root: Path, manifest: dict[str, Any], allow_legacy_unaligned: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    items = manifest.get("spider") or []
    if not isinstance(items, list):
        errors.append("reuse_12_demo_manifest.spider is not a list")
        items = []
    report_items: list[dict[str, Any]] = []
    admitted_routes: list[str] = []
    unavailable_route_details: list[dict[str, Any]] = []
    invalid_route_details: list[dict[str, Any]] = []
    availability = route_availability_map(matrix_root, manifest)
    if not items:
        warnings.append("reuse_12_demo_manifest.spider is empty; matrix completeness is diagnostic")

    for item in items:
        error_start = len(errors)
        warning_start = len(warnings)
        cell = str(item.get("cell") or "")
        status = item.get("status")
        if route_backend(cell) != "spider":
            errors.append(
                f"SPIDER item cell={cell!r} is not an exact __retarget_spider route"
            )
        item_availability = route_availability(cell, availability, item=item)
        errors.extend(
            f"SPIDER {cell}: untrusted route availability evidence: {error}"
            for error in item_availability.get("errors", [])
        )
        robot_value = item.get("robot")
        robot_candidate = rel_or_abs(matrix_root, robot_value)
        robot_exists = bool(robot_candidate is not None and robot_candidate.exists())
        if (
            item_availability["explicit_input_unavailable"]
            and not robot_exists
            and len(errors) == error_start
        ):
            warnings.append(
                f"SPIDER {cell}: exact-route input unavailable ({item_availability.get('reason')}); "
                "robot/triptych output is not required"
            )
            unavailable_route_details.append(
                {
                    "cell": cell,
                    "backend": "spider",
                    "reason": item_availability.get("reason"),
                    "availability": item_availability,
                }
            )
            report_items.append(
                {
                    "cell": cell,
                    "status": status,
                    "admission_status": "unavailable",
                    "availability": item_availability,
                    "robot": None,
                    "errors": [],
                    "warnings": warnings[warning_start:],
                }
            )
            continue
        if item_availability["explicit_input_unavailable"] and robot_exists:
            errors.append(
                f"SPIDER {cell}: route declares unavailable input but exposes a robot output; "
                "the output cannot masquerade as an admitted demo"
            )
        if status != "ok":
            if str(status).startswith("reviewable_") and "invalid_tracking" not in str(status):
                warnings.append(f"SPIDER {cell}: status={status}")
            else:
                errors.append(f"SPIDER {cell}: status={status}")
        robot = add_file_check(errors, matrix_root, f"SPIDER {cell} robot", item.get("robot"))
        if robot is not None and robot.name not in PLAIN_ALIGNED_ROBOT_NAMES:
            errors.append(f"SPIDER {cell}: robot is not a plain aligned MuJoCo render: {robot}")
        cell_root = matrix_root / "intermediates" / "retargeting" / "spider" / cell
        input_manifest = load_json(cell_root / "spider_input_manifest.json") or {}
        output_manifest = load_json(cell_root / "spider_output_manifest.json") or {}
        if not valid_spider_alignment(cell_root, REPO_ROOT, output_manifest):
            errors.append(
                f"SPIDER {cell}: alignment artifacts are stale, cross-cell, "
                "or not bound to the exact source timing grid"
            )
        expected_dai_cell = cell.rsplit("__retarget_", 1)[0] + "__retarget_do_as_i_do"
        source_keypoints = str(input_manifest.get("source_keypoints") or "")
        if input_manifest.get("expected_dai_cell_key") != expected_dai_cell:
            errors.append(
                f"SPIDER {cell}: expected_dai_cell_key={input_manifest.get('expected_dai_cell_key')}, "
                f"expected {expected_dai_cell}"
            )
        if expected_dai_cell not in source_keypoints:
            errors.append(
                f"SPIDER {cell}: source keypoints do not come from the exact hand-source cell: {source_keypoints}"
            )
        route_input_qc = input_manifest.get("route_input_qc") or {}
        if route_input_qc.get("status") != "ok":
            errors.append(
                f"SPIDER {cell}: route-specific source/processed input QC invalid: "
                f"{route_input_qc.get('errors')}"
            )
        if (input_manifest.get("hoi_contact_alignment") or {}).get("applied"):
            errors.append(f"SPIDER {cell}: object-only HOI contact translation was applied")
        if (input_manifest.get("hoi_refinement") or {}).get("hand_contact_translation_applied"):
            errors.append(f"SPIDER {cell}: forbidden HOI hand-contact refinement was applied")
        if (input_manifest.get("adapter_rigid_invariance") or {}).get("status") != "ok":
            errors.append(f"SPIDER {cell}: adapter rigid-invariance QC is missing or invalid")
        if input_manifest.get("status") in {"invalid_weak_interaction", "warning_weak_interaction"}:
            warnings.append(f"SPIDER {cell}: {input_manifest.get('status')}")
        alignment = output_manifest.get("alignment") or {}
        fallback = output_manifest.get("mjwp_fallback") or {}
        tracking, aligned_tracking_evidence, aligned_tracking_errors = (
            authoritative_spider_aligned_tracking(
                output_manifest,
                root=cell_root,
                cell_key=cell,
                repo_root=REPO_ROOT,
            )
        )
        errors.extend(f"SPIDER {cell}: {error}" for error in aligned_tracking_errors)
        raw_to_processed = input_manifest.get("raw_to_processed_hoi_invariance")
        errors.extend(
            f"SPIDER {cell}: {error}"
            for error in raw_to_processed_invariance_errors(
                raw_to_processed,
                label="exact-route raw_to_processed_hoi_invariance",
                trajectory_6dof=str(input_manifest.get("trajectory_6dof") or ""),
                hand_source=str(input_manifest.get("hand_source") or ""),
                hand_type=str(output_manifest.get("resolved_hand_type") or ""),
                adapter_manifest=resolve_manifest_path(
                    cell_root, input_manifest.get("adapter_manifest")
                ),
                processed_keypoints=resolve_manifest_path(
                    cell_root, input_manifest.get("source_keypoints")
                ),
            )
        )
        rest_support = output_manifest.get("rest_support_provenance")
        rest_errors = spider_rest_support_errors(
            rest_support,
            cell_key=cell,
            trajectory_6dof=str(output_manifest.get("trajectory_6dof") or ""),
            hand_source=str(output_manifest.get("hand_source") or ""),
            task=str(output_manifest.get("task") or ""),
            hand_type=str(output_manifest.get("resolved_hand_type") or ""),
            data_id=output_manifest.get("data_id"),
            input_manifest=input_manifest,
            root=cell_root,
            repo_root=REPO_ROOT,
        )
        errors.extend(f"SPIDER {cell}: {error}" for error in rest_errors)
        if fallback.get("applied"):
            errors.append(f"SPIDER {cell}: mjwp_fallback applied: {fallback.get('reason')}")
        hard_tracking_admission = validate_hard_tracking_admission(tracking)
        if hard_tracking_admission.get("status") != "ok":
            metrics = tracking.get("metrics") or {}
            errors.append(
                f"SPIDER {cell}: object tracking quality is not valid "
                f"(status={tracking.get('status')}, pos_median={metrics.get('pos_median_m', tracking.get('pos_median'))}, "
                f"pos_p95={metrics.get('pos_p95_m', tracking.get('pos_p95'))}, "
                f"rot_median={metrics.get('rot_median_rad', tracking.get('rot_median'))}, "
                f"lost_fraction={metrics.get('lost_frame_fraction', tracking.get('lost_frame_fraction'))}); "
                f"hard_admission_errors={hard_tracking_admission.get('errors')}"
            )
        if not allow_legacy_unaligned and not alignment.get("applied"):
            errors.append(f"SPIDER {cell}: missing applied alignment")
        if not allow_legacy_unaligned:
            if alignment.get("alignment_kind") != "sample_exact_source_timestamps":
                errors.append(
                    f"SPIDER {cell}: unexpected alignment_kind={alignment.get('alignment_kind')}, "
                    "expected sample_exact_source_timestamps"
                )
            if alignment.get("exact_source_timestamps") is not True:
                errors.append(f"SPIDER {cell}: exact_source_timestamps is not true")
            if alignment.get("sampling_semantics") != "exact_source_timestamp_indices":
                errors.append(f"SPIDER {cell}: invalid sampling_semantics")
            if alignment.get("source_kind") != "original_spider_mjwp":
                errors.append(
                    f"SPIDER {cell}: alignment source_kind={alignment.get('source_kind')}, "
                    "expected original_spider_mjwp"
                )
            try:
                start_index = int(alignment.get("start_index", -1))
            except (TypeError, ValueError):
                start_index = -1
            if start_index != 0:
                errors.append(f"SPIDER {cell}: start_index={start_index}, expected 0")
            try:
                warmup_steps = int(alignment.get("warmup_steps", -1))
            except (TypeError, ValueError):
                warmup_steps = -1
            if warmup_steps != 0:
                errors.append(f"SPIDER {cell}: warmup_steps={warmup_steps}, expected 0")
        aligned_video = rel_or_abs(matrix_root, alignment.get("aligned_video"))
        if alignment.get("applied") and (aligned_video is None or not aligned_video.exists()):
            errors.append(f"SPIDER {cell}: aligned_video missing: {alignment.get('aligned_video')}")
        video = video_probe(aligned_video) if aligned_video else {"exists": False}
        if alignment.get("applied"):
            target_frames = alignment.get("target_frames")
            try:
                target_frames = int(target_frames)
            except (TypeError, ValueError):
                target_frames = None
            try:
                target_fps = float(alignment.get("render_fps", 30.0))
            except (TypeError, ValueError):
                target_fps = 30.0
            errors.extend(
                probe_errors_for_aligned_video(
                    label=f"SPIDER {cell} aligned_video",
                    probe=video,
                    target_frames=target_frames,
                    target_fps=target_fps,
                )
            )
        report_items.append(
            {
                "cell": cell,
                "status": status,
                "admission_status": (
                    "admitted" if status == "ok" and len(errors) == error_start else "invalid"
                ),
                "availability": item_availability,
                "robot": str(robot) if robot else None,
                "input_status": input_manifest.get("status"),
                "alignment": alignment,
                "mjwp_fallback": fallback,
                "raw_to_processed_hoi_invariance": raw_to_processed,
                "rest_support_provenance": rest_support,
                "rest_support_errors": rest_errors,
                "aligned_tracking_evidence": aligned_tracking_evidence,
                "object_tracking_quality": tracking,
                "video": video_probe(robot) if robot else {"exists": False},
                "aligned_video": video,
                "errors": errors[error_start:],
                "warnings": warnings[warning_start:],
            }
        )
        if status == "ok" and len(errors) == error_start:
            admitted_routes.append(cell)
        elif len(errors) > error_start:
            invalid_route_details.append(
                {
                    "cell": cell,
                    "backend": "spider",
                    "reason": "backend_or_availability_audit_failed",
                    "availability": item_availability,
                    "errors": errors[error_start:],
                }
            )
    return {
        "errors": errors,
        "warnings": warnings,
        "items": report_items,
        "admitted_routes": sorted(set(admitted_routes)),
        "unavailable_routes": sorted(
            {str(item["cell"]) for item in unavailable_route_details}
        ),
        "unavailable_route_details": unavailable_route_details,
        "invalid_routes": sorted(
            {str(item["cell"]) for item in invalid_route_details}
        ),
        "invalid_route_details": invalid_route_details,
    }


def audit_cells(
    matrix_root: Path,
    manifest: dict[str, Any],
    allow_legacy_unaligned: bool,
    *,
    require_triptych: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    errors.extend(
        f"forbidden robot-on-RGB artifact: {path}"
        for path in forbidden_robot_rgb_artifacts(matrix_root)
    )
    cells_dir = matrix_root / "cells"
    cell_dirs = sorted(p for p in cells_dir.glob("*") if p.is_dir())
    triptychs = sorted((matrix_root / "videos").glob("*__triptych.mp4"))
    availability = route_availability_map(matrix_root, manifest)

    composed = manifest.get("composed")
    cells = manifest.get("cells")
    missing_cells = manifest.get("missing_cells")
    if cells != 12:
        warnings.append(f"matrix completeness: manifest cells={cells}, expected 12")
    if composed != 12:
        warnings.append(f"matrix completeness: manifest composed={composed}, expected 12")
    if missing_cells not in ({}, None):
        warnings.append(f"matrix completeness: manifest missing_cells is not empty: {missing_cells}")
    validation = manifest.get("trajectory_asset_validation") or {}
    if not validation and not allow_legacy_unaligned:
        errors.append("missing trajectory_asset_validation")
    elif validation:
        dai_provenance = (validation.get("object_provenance") or {}).get("do_as_i_do") or {}
        expected_native = {
            "object_track_source": "dai_native",
            "object_mesh_source": "dai_native",
            "retarget_object_source": "dai_native",
        }
        mismatched = {
            key: dai_provenance.get(key)
            for key, expected in expected_native.items()
            if dai_provenance.get(key) != expected
        }
        if manifest.get("spider_fuse_ego_object_for_dai"):
            errors.append("pure DAI matrix unexpectedly enables spider_fuse_ego_object_for_dai")
        if mismatched:
            errors.append(f"invalid DAI baseline provenance: {mismatched}")
        if dai_provenance.get("baseline_only") is not True:
            errors.append("pure DAI route must be explicitly marked baseline_only=true")
        if validation.get("do_as_i_do_equals_egoinfinity") is True:
            errors.append("Do-as-I-Do baseline keypoints unexpectedly equal EgoInfinity keypoints")
        if validation.get("do_as_i_do_quality_valid") is False:
            if dai_provenance.get("status") != "invalid":
                errors.append("invalid DAI quality must be explicitly marked status=invalid")
            explicitly_invalid_baseline = (
                not mismatched
                and dai_provenance.get("baseline_only") is True
                and dai_provenance.get("status") == "invalid"
            )
            if explicitly_invalid_baseline:
                warnings.append(
                    "DAI native baseline failed hand-object quality thresholds and is explicitly marked invalid"
                )

    if len(cell_dirs) != 12:
        warnings.append(f"matrix completeness: cells directory count={len(cell_dirs)}, expected 12")
    if len(triptychs) != 12:
        warnings.append(f"matrix completeness: triptych video count={len(triptychs)}, expected 12")

    top_provenance = manifest.get("provenance_validation") or {}
    per_cell: list[dict[str, Any]] = []
    admitted_routes: list[str] = []
    spider_candidate_routes: list[str] = []
    unavailable_route_details: list[dict[str, Any]] = []
    invalid_route_details: list[dict[str, Any]] = []
    seen_routes: set[str] = set()
    for cell_dir in cell_dirs:
        error_start = len(errors)
        warning_start = len(warnings)
        cell_manifest = load_json(cell_dir / "manifest.json") or {}
        settings = cell_manifest.get("settings") or {}
        assets = cell_manifest.get("assets") or {}
        declared_cell = str(cell_manifest.get("cell_key") or "")
        cell = cell_dir.name
        if declared_cell and declared_cell != cell:
            errors.append(
                f"{cell}: manifest cell_key={declared_cell} does not match cell directory"
            )
        suffix_backend = route_backend(cell)
        settings_backend = str(settings.get("retargeting") or "") or None
        if suffix_backend is not None:
            retargeting = suffix_backend
            if settings_backend != suffix_backend:
                errors.append(
                    f"{cell}: settings.retargeting={settings_backend}, "
                    f"but route suffix requires {suffix_backend}"
                )
        else:
            retargeting = settings_backend
        exact_route = is_exact_retarget_route(cell) and retargeting in RETARGET_BACKENDS
        if exact_route:
            seen_routes.add(cell)
        source_robot = assets.get("source.robot")
        source_robot_path = rel_or_abs(matrix_root, source_robot)
        if exact_route and source_robot_path is not None:
            source_run = str(manifest.get("source_run") or "")
            expected_route_roots = [
                matrix_root
                / "intermediates"
                / "retargeting"
                / str(retargeting)
                / cell,
            ]
            if source_run:
                expected_route_roots.append(
                    matrix_root.parent
                    / source_run
                    / "intermediates"
                    / "retargeting"
                    / str(retargeting)
                    / cell
                )
            resolved_robot = source_robot_path.expanduser().resolve()
            if not any(
                resolved_robot.is_relative_to(root.expanduser().resolve())
                for root in expected_route_roots
            ):
                errors.append(
                    f"{cell}: source.robot is not bound to the exact route root: "
                    f"{source_robot_path}"
                )
        cell_robot = cell_dir / "robot.mp4"
        triptych = matrix_root / "videos" / f"{cell}__triptych.mp4"
        output_present = bool(
            cell_robot.exists()
            or triptych.exists()
            or (source_robot_path is not None and source_robot_path.exists())
        )
        cell_availability = route_availability(cell, availability)
        if exact_route:
            errors.extend(
                f"{cell}: untrusted route availability evidence: {error}"
                for error in cell_availability.get("errors", [])
            )
        explicitly_unavailable = bool(
            exact_route and cell_availability["explicit_input_unavailable"]
        )
        post_retarget_invalid = bool(
            exact_route and cell_availability.get("post_retarget_invalid")
        )
        missing = [
            name
            for name in ("overlay.mp4", "depth.mp4", "robot.mp4", "manifest.json")
            if not (cell_dir / name).exists()
        ]

        if explicitly_unavailable and not output_present and len(errors) == error_start:
            message = (
                f"{cell}: exact-route input unavailable ({cell_availability.get('reason')}); "
                "robot/triptych output is not required"
            )
            warnings.append(message)
            unavailable_route_details.append(
                {
                    "cell": cell,
                    "backend": retargeting,
                    "reason": cell_availability.get("reason"),
                    "availability": cell_availability,
                }
            )
            per_cell.append(
                {
                    "cell": cell,
                    "missing": missing,
                    "retargeting": retargeting,
                    "source_robot": source_robot,
                    "availability": cell_availability,
                    "admission_status": "unavailable",
                    "errors": [],
                    "warnings": warnings[warning_start:],
                    "robot": video_probe(cell_robot),
                    "triptych": video_probe(triptych),
                }
            )
            continue

        if post_retarget_invalid and not output_present and len(errors) == error_start:
            message = (
                f"{cell}: input preflight passed but post-retarget failed "
                f"({cell_availability.get('reason')}); robot/triptych output is not required"
            )
            warnings.append(message)
            invalid_route_details.append(
                {
                    "cell": cell,
                    "backend": retargeting,
                    "reason": "post_retarget_invalid",
                    "availability": cell_availability,
                    "errors": [],
                }
            )
            per_cell.append(
                {
                    "cell": cell,
                    "missing": missing,
                    "retargeting": retargeting,
                    "source_robot": source_robot,
                    "availability": cell_availability,
                    "admission_status": "post_retarget_invalid",
                    "errors": [],
                    "warnings": warnings[warning_start:],
                    "robot": video_probe(cell_robot),
                    "triptych": video_probe(triptych),
                }
            )
            continue

        if exact_route and explicitly_unavailable and output_present:
            errors.append(
                f"{cell}: route declares unavailable input but exposes robot/triptych output; "
                "the output cannot masquerade as an admitted demo"
            )
        if exact_route and post_retarget_invalid and output_present:
            errors.append(
                f"{cell}: post-retarget-invalid route exposes robot/triptych output; "
                "the output cannot masquerade as an admitted demo"
            )
        if missing:
            message = f"{cell}: missing {', '.join(missing)}"
            if exact_route:
                errors.append(message)
            else:
                warnings.append(f"matrix completeness: {message}")
        if exact_route and require_triptych and not triptych.exists():
            errors.append(f"{cell}: exact-route triptych is missing: {triptych}")

        provenance_validation = cell_manifest.get("provenance_validation")
        if not isinstance(provenance_validation, dict):
            candidate = top_provenance.get(cell) if isinstance(top_provenance, dict) else None
            provenance_validation = candidate if isinstance(candidate, dict) else None
        if exact_route:
            if not isinstance(provenance_validation, dict):
                errors.append(f"{cell}: missing exact-route provenance_validation")
            elif provenance_validation.get("status") != "ok":
                errors.append(
                    f"{cell}: provenance_validation status={provenance_validation.get('status')}; "
                    f"errors={provenance_validation.get('errors')}"
                )
        if retargeting in {"do_as_i_do", "spider"} and not is_aligned_reference(source_robot):
            errors.append(f"{cell}: retarget {retargeting} source.robot is not aligned: {source_robot}")
        if retargeting in {"do_as_i_do", "spider"} and source_robot_path is not None:
            if source_robot_path.name not in PLAIN_ALIGNED_ROBOT_NAMES:
                errors.append(
                    f"{cell}: bottom panel is not a plain aligned MuJoCo render: {source_robot_path}"
                )
            if not source_robot_path.exists():
                errors.append(f"{cell}: source.robot does not exist: {source_robot_path}")
        if retargeting == "do_as_i_do" and source_robot_path is not None:
            alignment_path = source_robot_path.parent / "mjwp_alignment_manifest.json"
            errors.extend(
                f"{cell}: {error}"
                for error in dai_alignment_errors(
                    alignment_path,
                    allow_legacy_unaligned,
                )
            )
            alignment_data = load_json(alignment_path) or {}
            input_qc_errors, _ = ego_retarget_input_qc_findings(
                alignment_path, allow_legacy_unaligned
            )
            errors.extend(f"{cell}: {error}" for error in input_qc_errors)
            quality_errors, quality_warnings = dai_tracking_quality_findings(
                alignment_path, alignment_data, allow_legacy_unaligned
            )
            errors.extend(f"{cell}: {error}" for error in quality_errors)
            warnings.extend(f"{cell}: {warning}" for warning in quality_warnings)
            raw_errors, _ = dai_raw_to_processed_findings(alignment_path)
            errors.extend(f"{cell}: {error}" for error in raw_errors)

        local_errors = errors[error_start:]
        local_warnings = warnings[warning_start:]
        if exact_route and not local_errors:
            if retargeting == "do_as_i_do":
                admitted_routes.append(cell)
            elif retargeting == "spider":
                spider_candidate_routes.append(cell)
        elif exact_route:
            invalid_route_details.append(
                {
                    "cell": cell,
                    "backend": retargeting,
                    "reason": "route_audit_failed",
                    "availability": cell_availability,
                    "errors": local_errors,
                }
            )
        per_cell.append(
            {
                "cell": cell,
                "missing": missing,
                "retargeting": retargeting,
                "source_robot": source_robot,
                "availability": cell_availability,
                "provenance_validation": provenance_validation,
                "admission_status": (
                    "candidate" if exact_route and not local_errors else (
                        "invalid" if exact_route else "not_applicable"
                    )
                ),
                "errors": local_errors,
                "warnings": local_warnings,
                "robot": video_probe(cell_robot),
                "triptych": video_probe(triptych),
            }
        )

    missing_declared = missing_cells if isinstance(missing_cells, dict) else {}
    declared_routes = {
        *(
            cell
            for cell in availability
            if is_exact_retarget_route(cell)
        ),
        *(
            str(cell)
            for cell in missing_declared
            if is_exact_retarget_route(str(cell))
        ),
    }
    for cell in sorted(declared_routes - seen_routes):
        cell_availability = route_availability(cell, availability)
        if cell_availability["explicit_input_unavailable"]:
            warnings.append(
                f"{cell}: exact-route input unavailable ({cell_availability.get('reason')}); "
                "cell/robot/triptych output is not required"
            )
            unavailable_route_details.append(
                {
                    "cell": cell,
                    "backend": route_backend(cell),
                    "reason": cell_availability.get("reason"),
                    "availability": cell_availability,
                }
            )
        elif cell_availability.get("post_retarget_invalid"):
            warnings.append(
                f"{cell}: input preflight passed but post-retarget failed "
                f"({cell_availability.get('reason')}); cell/robot/triptych output is not required"
            )
            invalid_route_details.append(
                {
                    "cell": cell,
                    "backend": route_backend(cell),
                    "reason": "post_retarget_invalid",
                    "availability": cell_availability,
                    "errors": [],
                }
            )
        else:
            availability_errors = list(cell_availability.get("errors") or [])
            if availability_errors:
                errors.extend(
                    f"{cell}: untrusted route availability evidence: {error}"
                    for error in availability_errors
                )
            else:
                errors.append(f"{cell}: available exact route has no matrix cell record")
            invalid_route_details.append(
                {
                    "cell": cell,
                    "backend": route_backend(cell),
                    "reason": "missing_matrix_cell_or_untrusted_availability",
                    "availability": cell_availability,
                    "errors": availability_errors,
                }
            )

    return {
        "errors": errors,
        "warnings": warnings,
        "matrix_complete": bool(
            cells == 12
            and composed == 12
            and missing_cells in ({}, None)
            and len(cell_dirs) == 12
            and len(triptychs) == 12
        ),
        "matrix_completeness_warnings": [
            warning for warning in warnings if warning.startswith("matrix completeness:")
        ],
        "cell_count": len(cell_dirs),
        "triptych_count": len(triptychs),
        "triptych_required_for_admission": bool(require_triptych),
        "cells": per_cell,
        "admitted_routes": sorted(set(admitted_routes)),
        "spider_candidate_routes": sorted(set(spider_candidate_routes)),
        "unavailable_routes": sorted(
            {str(item["cell"]) for item in unavailable_route_details}
        ),
        "unavailable_route_details": unavailable_route_details,
        "invalid_routes": sorted(
            {str(item["cell"]) for item in invalid_route_details}
        ),
        "invalid_route_details": invalid_route_details,
    }


def audit_source(
    source_root: Path,
    allow_legacy_unaligned: bool,
    availability: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    errors.extend(
        f"forbidden robot-on-RGB artifact: {path}"
        for path in forbidden_robot_rgb_artifacts(source_root)
    )
    metadata = load_json(source_root / "fresh_run_metadata.json")
    if metadata is None:
        errors.append("missing fresh_run_metadata.json")
    else:
        git = metadata.get("git") or {}
        gpu = metadata.get("gpu_mapping") or {}
        for key in ("commit", "dirty_diff_sha256", "status_short"):
            if key not in git:
                errors.append(f"fresh_run_metadata.git missing {key}")
        for key in ("main_cuda", "dai_cuda_visible_devices", "spider_cuda_visible_devices", "spider_device"):
            if key not in gpu:
                errors.append(f"fresh_run_metadata.gpu_mapping missing {key}")

    clip_root = source_root / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction" / "clip"
    clip_metadata_path = clip_root / "metadata.json"
    clip_metadata = load_json(clip_metadata_path) or {}
    bbox_transform = clip_metadata.get("bbox_coordinate_transform") or {}
    sam3_prompt_gate_path = clip_root / "video_segmentation" / "prompt_gate.json"
    sam3_prompt_gate = load_json(sam3_prompt_gate_path) or {}
    if not allow_legacy_unaligned:
        if not clip_metadata:
            errors.append(f"missing fresh clip metadata: {clip_metadata_path}")
        if bbox_transform.get("source") != "normalized-1000":
            errors.append(
                f"fresh clip bbox source={bbox_transform.get('source')}, expected normalized-1000"
            )
        if bbox_transform.get("annotation_size") != [1000, 1000]:
            errors.append(
                f"fresh clip annotation_size={bbox_transform.get('annotation_size')}, expected [1000, 1000]"
            )
        transformed_bbox = bbox_transform.get("transformed_bbox") or []
        output_size = bbox_transform.get("output_video_size") or []
        if len(transformed_bbox) != 4 or len(output_size) != 2:
            errors.append("fresh clip bbox transform is incomplete")
        else:
            try:
                x1, y1, x2, y2 = [float(value) for value in transformed_bbox]
                width, height = [float(value) for value in output_size]
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    errors.append(
                        f"fresh clip transformed_bbox={transformed_bbox} is outside output_size={output_size}"
                    )
            except (TypeError, ValueError):
                errors.append(f"fresh clip transformed_bbox is not numeric: {transformed_bbox}")
        if not sam3_prompt_gate:
            errors.append(f"missing SAM3 production prompt gate: {sam3_prompt_gate_path}")
        else:
            if sam3_prompt_gate.get("status") != "ok":
                errors.append(
                    f"SAM3 production prompt gate status={sam3_prompt_gate.get('status')}, expected ok"
                )
            if sam3_prompt_gate.get("required_mode") != "semantic_text_with_bbox_instance_selection":
                errors.append(
                    "SAM3 production prompt mode is not semantic_text_with_bbox_instance_selection"
                )
            if sam3_prompt_gate.get("errors"):
                errors.append(f"SAM3 production prompt gate errors={sam3_prompt_gate.get('errors')}")
            warnings.extend(
                f"SAM3 production prompt gate: {message}"
                for message in (sam3_prompt_gate.get("warnings") or [])
            )

    ego_clip = source_root / "intermediates" / "egoinfinity" / "clip"
    selected_pipeline = ego_clip / "pipeline_result_selected.pkl.gz"
    selection_report_path = ego_clip / "object_selection_report.json"
    selection_report = load_json(selection_report_path) or {}
    if not allow_legacy_unaligned:
        if not selected_pipeline.exists():
            errors.append(f"missing selected EgoInfinity pipeline result: {selected_pipeline}")
        if not selection_report:
            errors.append(f"missing EgoInfinity object selection report: {selection_report_path}")
        else:
            if selection_report.get("selection_mode") != "best":
                errors.append(
                    f"EgoInfinity selection_mode={selection_report.get('selection_mode')}, expected best"
                )
            keep_oids = selection_report.get("keep_oids") or []
            if len(keep_oids) != 1:
                errors.append(f"EgoInfinity selected object count={len(keep_oids)}, expected 1")
            target_point = selection_report.get("target_point") or []
            if len(target_point) != 2:
                errors.append(f"EgoInfinity target_point is missing or invalid: {target_point}")

    dai_alignment_paths = sorted(source_root.glob("intermediates/retargeting/do_as_i_do/**/mjwp_alignment_manifest.json"))
    if not dai_alignment_paths and not allow_legacy_unaligned:
        dai_entries = {
            cell: entry
            for cell, entry in (availability or {}).items()
            if cell.endswith("__retarget_do_as_i_do")
        }
        terminal_unadmitted = {
            cell: route_availability(cell, dai_entries)
            for cell in dai_entries
        }
        all_terminal_unadmitted = bool(terminal_unadmitted) and all(
            item.get("status") in {"unavailable", "post_retarget_invalid"}
            for item in terminal_unadmitted.values()
        )
        if all_terminal_unadmitted:
            warnings.append(
                "no Do-as-I-Do alignment exists because every declared DAI route is "
                "trusted unavailable or post-retarget invalid"
            )
        else:
            errors.append("missing Do-as-I-Do mjwp_alignment_manifest.json")
    dai_alignments = []
    for path in dai_alignment_paths:
        data = load_json(path) or {}
        errors.extend(dai_alignment_errors(path, allow_legacy_unaligned))
        input_qc_errors, input_qc_evidence = ego_retarget_input_qc_findings(
            path, allow_legacy_unaligned
        )
        errors.extend(input_qc_errors)
        quality_errors, quality_warnings = dai_tracking_quality_findings(
            path, data, allow_legacy_unaligned
        )
        errors.extend(quality_errors)
        warnings.extend(quality_warnings)
        raw_invariance_errors, raw_invariance_evidence = (
            dai_raw_to_processed_findings(path)
        )
        errors.extend(
            f"DAI alignment {path}: {error}" for error in raw_invariance_errors
        )
        try:
            target_count = int(data.get("target_count", 0))
        except (TypeError, ValueError):
            target_count = 0
        if target_count <= 0:
            warnings.append(f"DAI alignment has no target_count: {path}")
        try:
            source_frame_count = int(data.get("source_frame_count", 0))
        except (TypeError, ValueError):
            source_frame_count = 0
        if not allow_legacy_unaligned and source_frame_count and target_count and source_frame_count != target_count:
            errors.append(
                f"DAI alignment {path}: source_frame_count={source_frame_count}, target_count={target_count}"
            )
        raw_video = nearest_raw_video(path)
        raw_probe = video_probe(raw_video) if raw_video else {"exists": False}
        target_fps = fps_value(raw_probe)
        aligned_videos = []
        for name in sorted(PLAIN_ALIGNED_ROBOT_NAMES):
            video_path = path.parent / name
            if not video_path.exists():
                continue
            probe = video_probe(video_path)
            errors.extend(
                probe_errors_for_aligned_video(
                    label=f"DAI aligned video {video_path}",
                    probe=probe,
                    target_frames=target_count,
                    target_fps=target_fps,
                )
            )
            aligned_videos.append({"path": str(video_path), "probe": probe})
        if not aligned_videos and not allow_legacy_unaligned:
            errors.append(f"missing DAI aligned visualization video next to {path}")
        dai_alignments.append(
            {
                "path": str(path),
                "raw_video": str(raw_video) if raw_video else None,
                "raw_video_probe": raw_probe,
                "target_fps": target_fps,
                "aligned_videos": aligned_videos,
                "retarget_input": input_qc_evidence,
                "raw_to_processed_hoi_invariance": raw_invariance_evidence,
                **data,
            }
        )
    return {
        "errors": errors,
        "warnings": warnings,
        "fresh_run_metadata": str(source_root / "fresh_run_metadata.json"),
        "fresh_clip_metadata": str(clip_metadata_path),
        "bbox_coordinate_transform": bbox_transform,
        "sam3_prompt_gate_path": str(sam3_prompt_gate_path),
        "sam3_prompt_gate": sam3_prompt_gate,
        "egoinfinity_selected_pipeline": str(selected_pipeline),
        "egoinfinity_object_selection_report": str(selection_report_path),
        "egoinfinity_object_selection": selection_report,
        "dai_alignments": dai_alignments,
    }


def summarize_route_admission(
    cell_report: dict[str, Any],
    spider_report: dict[str, Any],
) -> dict[str, Any]:
    """Combine matrix presentation checks with backend-specific hard QC.

    DAI hard QC is performed while auditing its exact matrix cell.  SPIDER must
    pass both the matrix-cell checks and the authoritative SPIDER output audit.
    An input-unavailable route is evidence, not a demo and not a global failure.
    """

    errors: list[str] = []
    dai_admitted = {
        str(cell) for cell in (cell_report.get("admitted_routes") or [])
    }
    spider_cell_candidates = {
        str(cell) for cell in (cell_report.get("spider_candidate_routes") or [])
    }
    spider_backend_admitted = {
        str(cell) for cell in (spider_report.get("admitted_routes") or [])
    }
    spider_reported = {
        str(item.get("cell"))
        for item in (spider_report.get("items") or [])
        if isinstance(item, dict) and item.get("cell")
    }
    spider_admitted = spider_cell_candidates & spider_backend_admitted

    for cell in sorted(spider_cell_candidates - spider_reported):
        errors.append(
            f"SPIDER {cell}: exact matrix output has no backend audit item"
        )

    backend_only = sorted(spider_backend_admitted - spider_cell_candidates)
    for cell in backend_only:
        errors.append(
            f"SPIDER {cell}: backend output passed but exact matrix cell/triptych/provenance did not"
        )

    unavailable_by_cell: dict[str, dict[str, Any]] = {}
    invalid_by_cell: dict[str, dict[str, Any]] = {}
    for subreport in (cell_report, spider_report):
        details = subreport.get("unavailable_route_details") or []
        if isinstance(details, list):
            for item in details:
                if isinstance(item, dict) and item.get("cell"):
                    unavailable_by_cell[str(item["cell"])] = item
        routes = subreport.get("unavailable_routes") or []
        if isinstance(routes, list):
            for cell in routes:
                if isinstance(cell, str) and cell:
                    unavailable_by_cell.setdefault(cell, {"cell": cell})
        invalid_details = subreport.get("invalid_route_details") or []
        if isinstance(invalid_details, list):
            for item in invalid_details:
                if isinstance(item, dict) and item.get("cell"):
                    invalid_by_cell[str(item["cell"])] = item
        invalid_routes = subreport.get("invalid_routes") or []
        if isinstance(invalid_routes, list):
            for cell in invalid_routes:
                if isinstance(cell, str) and cell:
                    invalid_by_cell.setdefault(cell, {"cell": cell})

    admitted_routes = sorted(dai_admitted | spider_admitted)
    conflicts = sorted(set(admitted_routes) & set(unavailable_by_cell))
    for cell in conflicts:
        errors.append(
            f"{cell}: route cannot be both admitted and explicitly input-unavailable"
        )
    admitted_routes = [cell for cell in admitted_routes if cell not in conflicts]
    invalid_conflicts = sorted(set(admitted_routes) & set(invalid_by_cell))
    for cell in invalid_conflicts:
        errors.append(f"{cell}: route cannot be both admitted and invalid")
    admitted_routes = [cell for cell in admitted_routes if cell not in invalid_conflicts]
    admitted_route_count = len(admitted_routes)
    return {
        "admitted_routes": admitted_routes,
        "unavailable_routes": sorted(unavailable_by_cell),
        "unavailable_route_details": [
            unavailable_by_cell[cell] for cell in sorted(unavailable_by_cell)
        ],
        "invalid_routes": sorted(invalid_by_cell),
        "invalid_route_details": [
            invalid_by_cell[cell] for cell in sorted(invalid_by_cell)
        ],
        "admitted_route_count": admitted_route_count,
        "demo_count": admitted_route_count,
        "scene_admitted": bool(admitted_routes),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a completed AoE retarget matrix run for 12/12 output and alignment provenance.")
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--matrix-run", required=True)
    parser.add_argument("--source-run", default="")
    parser.add_argument("--allow-legacy-unaligned", action="store_true")
    parser.add_argument(
        "--no-require-triptych",
        action="store_true",
        help=(
            "Audit exact numerical/provenance routes without requiring composed "
            "triptych videos. This never counts an admitted route as a Demo."
        ),
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--warn-only", action="store_true")
    args = parser.parse_args()

    experiments_root = args.experiments_root.expanduser().resolve()
    matrix_root = experiments_root / args.matrix_run
    errors: list[str] = []
    warnings: list[str] = []

    manifest_path = matrix_root / "reuse_12_demo_manifest.json"
    manifest = load_json(manifest_path)
    if manifest is None:
        manifest = {}
        errors.append(f"missing matrix manifest: {manifest_path}")

    manifest_source_run = str(manifest.get("source_run") or "")
    source_run_mismatch = bool(
        args.source_run
        and manifest_source_run
        and args.source_run != manifest_source_run
    )
    if source_run_mismatch:
        errors.append(
            f"--source-run={args.source_run} does not match matrix manifest "
            f"source_run={manifest_source_run}"
        )
    source_run = manifest_source_run if source_run_mismatch else (
        args.source_run or manifest_source_run
    )
    report: dict[str, Any] = {
        "matrix_run": args.matrix_run,
        "matrix_root": str(matrix_root),
        "source_run": source_run,
        "manifest": str(manifest_path),
    }

    cell_report: dict[str, Any] = {}
    spider_report: dict[str, Any] = {}

    if manifest:
        cell_report = audit_cells(
            matrix_root,
            manifest,
            args.allow_legacy_unaligned,
            require_triptych=not args.no_require_triptych,
        )
        spider_report = audit_spider(matrix_root, manifest, args.allow_legacy_unaligned)
        report["cells"] = cell_report
        report["spider"] = spider_report
        errors.extend(cell_report["errors"])
        errors.extend(spider_report["errors"])
        warnings.extend(cell_report["warnings"])
        warnings.extend(spider_report["warnings"])

    if source_run:
        source_report = audit_source(
            experiments_root / source_run,
            args.allow_legacy_unaligned,
            route_availability_map(matrix_root, manifest),
        )
        report["source"] = source_report
        errors.extend(source_report["errors"])
        warnings.extend(source_report["warnings"])
    else:
        warnings.append("source_run not provided and not found in matrix manifest")

    admission = summarize_route_admission(cell_report, spider_report)
    report["admitted_routes"] = admission["admitted_routes"]
    report["unavailable_routes"] = admission["unavailable_routes"]
    report["unavailable_route_details"] = admission["unavailable_route_details"]
    report["invalid_routes"] = admission["invalid_routes"]
    report["invalid_route_details"] = admission["invalid_route_details"]
    report["admitted_route_count"] = admission["admitted_route_count"]
    report["numerically_admitted_routes"] = admission["admitted_routes"]
    report["numerically_admitted_route_count"] = admission[
        "admitted_route_count"
    ]
    report["composition_required_for_demo"] = not args.no_require_triptych
    report["demo_count"] = (
        admission["demo_count"] if not args.no_require_triptych else 0
    )
    report["scene_admitted"] = (
        admission["scene_admitted"] if not args.no_require_triptych else False
    )
    errors.extend(admission["errors"])

    if errors and report["admitted_routes"]:
        report["candidate_routes_blocked_by_audit_errors"] = report[
            "admitted_routes"
        ]
        report["admitted_routes"] = []
        report["admitted_route_count"] = 0
        report["numerically_admitted_routes"] = []
        report["numerically_admitted_route_count"] = 0
        report["demo_count"] = 0
        report["scene_admitted"] = False

    report["status"] = "ok" if not errors else "failed"
    report["errors"] = errors
    report["warnings"] = warnings
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if errors and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
