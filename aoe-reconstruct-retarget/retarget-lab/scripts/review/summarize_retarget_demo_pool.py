#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


VISUAL_REVIEW_POLICY_V3 = "retarget_only_plain_mujoco_numerical_and_visual_v3"
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aoe_retarget_lab.tracking_quality import (  # noqa: E402
    validate_hard_tracking_admission,
)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def find_source_run(matrix_run: Path, manifest: dict[str, Any] | None) -> str | None:
    if manifest:
        source = manifest.get("source_run")
        if isinstance(source, str) and source:
            return source
    name = matrix_run.name
    if "_matrix12_" in name:
        scene, suffix = name.split("_matrix12_", 1)
        return f"{scene}_full_source_{suffix}"
    return None


def triptych_count(matrix_run: Path) -> int:
    videos_dir = matrix_run / "videos"
    if not videos_dir.exists():
        return 0
    return len(list(videos_dir.glob("*__triptych.mp4")))


def audit_error_text(audit: dict[str, Any] | None) -> list[str]:
    if not audit:
        return []
    errors = audit.get("errors")
    return [str(item) for item in errors] if isinstance(errors, list) else []


def recent_log_text(*roots: Path | None, max_bytes_per_log: int = 12000) -> str:
    chunks: list[str] = []
    seen: set[Path] = set()
    for root in roots:
        if root is None:
            continue
        logs_dir = root / "logs"
        if not logs_dir.exists():
            continue
        for path in sorted(logs_dir.glob("*.log"), key=lambda item: item.stat().st_mtime if item.exists() else 0)[-12:]:
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                with path.open("rb") as handle:
                    handle.seek(0, 2)
                    size = handle.tell()
                    handle.seek(max(0, size - max_bytes_per_log))
                    chunks.append(handle.read().decode("utf-8", errors="replace"))
            except OSError:
                continue
    return "\n".join(chunks)


def ego_retarget_preflight_failed(source_root: Path | None) -> bool:
    if source_root is None:
        return False
    path = source_root / "logs" / "stage_failures.tsv"
    if not path.exists():
        return False
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for row in rows:
        fields = row.split("\t")
        if len(fields) >= 3 and fields[1] == "egoinfinity_retarget_input_preflight":
            try:
                return int(fields[2]) != 0
            except ValueError:
                return True
    return False


def failure_tags_for_run(
    *,
    manifest: dict[str, Any] | None,
    audit: dict[str, Any] | None,
    errors: list[str],
    log_text: str,
    has_12: bool,
    dai_aligned: bool,
    spider_aligned: bool,
    spider_fallback_count: int,
    dai_baseline_invalid: bool,
    gravity_status: str,
) -> list[str]:
    tags: list[str] = []
    haystack = "\n".join(errors + [log_text])
    lower = haystack.lower()
    if "cuda out of memory" in lower or "outofmemoryerror" in lower:
        tags.append("oom")
    if "[gpu-check] waiting" in lower or "waiting 60s for sam3d gpu memory" in lower:
        tags.append("gpu_waiting")
    if manifest is None:
        tags.append("missing_manifest_or_in_progress")
    if manifest is not None and audit is None:
        tags.append("missing_audit")
    if manifest is not None and not has_12:
        tags.append("incomplete_matrix")
    if audit is not None and not dai_aligned:
        tags.append("dai_alignment_failed")
    if audit is not None and not spider_aligned:
        tags.append("spider_alignment_failed")
    if spider_fallback_count:
        tags.append("spider_fallback_applied")
    if any("invalid_weak_interaction" in error for error in errors):
        tags.append("spider_invalid_weak_interaction")
    if "mjwp_object_tracking_error_exceeded" in haystack or "object tracking error exceeds threshold" in haystack:
        tags.append("spider_object_tracking_error")
    if dai_baseline_invalid:
        tags.append("dai_baseline_invalid")
    if gravity_status == "fallback":
        tags.append("gravity_fallback")
    elif gravity_status != "estimated":
        tags.append("gravity_missing_or_unknown")
    if "provenance_validation invalid" in haystack:
        tags.append("provenance_invalid")
    if "no egoinfinity reconstructed object mesh" in lower or "no reconstructed object mesh" in lower:
        tags.append("egoinfinity_object_missing")
    if audit is not None and audit.get("status") not in {None, "ok"}:
        tags.append("audit_failed")
    if not tags and audit is not None and audit.get("status") == "ok":
        tags.append("passed_strict_audit")
    return tags


def failure_bucket(tags: list[str]) -> str:
    priority = [
        "passed_strict_audit",
        "oom",
        "gpu_waiting",
        "source_run_failed",
        "retarget_input_preflight_failed",
        "missing_manifest_or_in_progress",
        "missing_audit",
        "egoinfinity_object_missing",
        "provenance_invalid",
        "incomplete_matrix",
        "dai_alignment_failed",
        "spider_alignment_failed",
        "spider_fallback_applied",
        "spider_object_tracking_error",
        "spider_invalid_weak_interaction",
        "dai_baseline_invalid",
        "gravity_fallback",
        "gravity_missing_or_unknown",
        "audit_failed",
    ]
    for tag in priority:
        if tag in tags:
            return tag
    return tags[0] if tags else "unclassified"


def source_alignment_ok(audit: dict[str, Any] | None) -> bool:
    if not audit:
        return False
    source = audit.get("source") or {}
    if not isinstance(source, dict):
        return False
    dai_alignments = source.get("dai_alignments") or []
    if not isinstance(dai_alignments, list) or not dai_alignments:
        return False
    for alignment in dai_alignments:
        if not isinstance(alignment, dict):
            return False
        if alignment.get("applied") is not True:
            return False
        if alignment.get("alignment_kind") != "drop_warmup_and_sample_exact_source_timestamps":
            return False
        start_index = int_value(alignment.get("start_index"))
        warmup_steps = int_value(alignment.get("warmup_steps"), default=-2)
        if warmup_steps <= 0 or start_index < 0 or start_index != warmup_steps - 1:
            return False
        if alignment.get("exact_source_timestamps") is not True:
            return False
        if alignment.get("sampling_semantics") != "exact_source_timestamp_indices":
            return False
        target_count = int_value(alignment.get("target_count") or alignment.get("target_frames"), default=0)
        source_frame_count = int_value(alignment.get("source_frame_count"), default=0)
        if target_count <= 0:
            return False
        if source_frame_count and source_frame_count != target_count:
            return False
        aligned_video = alignment.get("aligned_video") or alignment.get("rendered_video")
        if not aligned_video or "aligned" not in Path(str(aligned_video)).name:
            return False
        aligned_videos = alignment.get("aligned_videos") or []
        if not isinstance(aligned_videos, list) or not aligned_videos:
            return False
        if not any(
            isinstance(item, dict)
            and isinstance(item.get("probe"), dict)
            and item["probe"].get("exists")
            and "aligned" in Path(str(item.get("path", ""))).name
            for item in aligned_videos
        ):
            return False
    return True


def dai_numerical_quality_ok(audit: dict[str, Any] | None) -> bool:
    if not audit:
        return False
    source = audit.get("source") or {}
    alignments = source.get("dai_alignments") or [] if isinstance(source, dict) else []
    if not isinstance(alignments, list) or not alignments:
        return False
    return all(
        isinstance(item, dict)
        and isinstance(item.get("object_tracking_quality"), dict)
        and validate_hard_tracking_admission(item["object_tracking_quality"]).get("status")
        == "ok"
        for item in alignments
    )


def int_value(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def spider_summary(audit: dict[str, Any] | None) -> tuple[bool, bool, int, int]:
    if not audit:
        return False, False, 0, 0
    spider = audit.get("spider") or {}
    items = spider.get("items") or [] if isinstance(spider, dict) else []
    if not isinstance(items, list):
        items = []
    fallback_count = 0
    ok_count = 0
    aligned_count = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        hard_tracking = validate_hard_tracking_admission(
            item.get("object_tracking_quality")
            if isinstance(item.get("object_tracking_quality"), dict)
            else None
        )
        if status == "ok" and hard_tracking.get("status") == "ok":
            ok_count += 1
        alignment = item.get("alignment") or {}
        aligned_video = item.get("aligned_video") or {}
        if (
            isinstance(alignment, dict)
            and alignment.get("applied")
            and alignment.get("alignment_kind") == "sample_exact_source_timestamps"
            and alignment.get("source_kind") == "original_spider_mjwp"
            and alignment.get("exact_source_timestamps") is True
            and alignment.get("sampling_semantics") == "exact_source_timestamp_indices"
            and int_value(alignment.get("start_index")) == 0
            and int_value(alignment.get("warmup_steps")) == 0
            and isinstance(aligned_video, dict)
            and aligned_video.get("exists")
        ):
            aligned_count += 1
        fallback = ((item.get("mjwp_fallback") or {}) if isinstance(item.get("mjwp_fallback"), dict) else {})
        if fallback.get("applied"):
            fallback_count += 1
    spider_ok = ok_count == 4 and fallback_count == 0
    return aligned_count == 4, spider_ok, ok_count, fallback_count


def manifest_missing_cells(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {"manifest": "missing"}
    missing = manifest.get("missing_cells")
    return missing if isinstance(missing, dict) else {"missing_cells": missing}


def plain_robot_visual_review(matrix_run: Path) -> tuple[bool, dict[str, Any]]:
    path = matrix_run / "plain_robot_visual_review.json"
    review = load_json(path) or {}
    policy = review.get("policy")
    successful_cells = {
        str(cell) for cell in (review.get("successful_cells") or [])
    }
    reviewed_cells = review.get("reviewed_cells") or []
    by_name = {
        str(item.get("cell")): item
        for item in reviewed_cells
        if isinstance(item, dict) and item.get("cell")
    }

    def hard_metrics_ok(item: dict[str, Any]) -> bool:
        numerical = item.get("numerical_validation") or {}
        evidence = numerical.get("evidence") or {}
        tracking_evidence = evidence.get("object_tracking_quality") or {}
        tracking = tracking_evidence.get("artifact")
        return bool(
            isinstance(tracking, dict)
            and validate_hard_tracking_admission(tracking).get("status") == "ok"
        )

    successful_numerical_ok = bool(
        successful_cells
        and successful_cells.issubset(by_name)
        and all(
            (by_name[cell].get("numerical_validation") or {}).get("status") == "ok"
            and not ((by_name[cell].get("numerical_validation") or {}).get("errors") or [])
            and hard_metrics_ok(by_name[cell])
            for cell in successful_cells
        )
    )
    v3_passed = bool(
        policy == VISUAL_REVIEW_POLICY_V3
        and review.get("evidence_kind") == "plain_aligned_mujoco"
        and review.get("status") == "pass"
        and int_value(review.get("successful_cell_count"), default=0) >= 1
        and isinstance(review.get("successful_cells"), list)
        and successful_numerical_ok
        and not (review.get("validation_errors") or [])
    )
    return v3_passed, {
        "path": str(path),
        **review,
        "admission_note": (
            "legacy/statusless review is diagnostic-only"
            if policy != VISUAL_REVIEW_POLICY_V3
            else "v3 review revalidated from raw hard metrics"
        ),
    }


def successful_route_provenance_ok(
    visual_review: dict[str, Any],
    matrix_provenance_ok: bool,
) -> bool:
    if visual_review.get("policy") != VISUAL_REVIEW_POLICY_V3:
        return matrix_provenance_ok
    successful_cells = {
        str(cell) for cell in (visual_review.get("successful_cells") or [])
    }
    reviewed_cells = visual_review.get("reviewed_cells") or []
    by_name = {
        str(item.get("cell")): item
        for item in reviewed_cells
        if isinstance(item, dict) and item.get("cell")
    }
    if not successful_cells or not successful_cells.issubset(by_name):
        return False
    for cell in successful_cells:
        validation = by_name[cell].get("provenance_validation") or {}
        if validation.get("status") != "ok" or validation.get("errors"):
            return False
    return True


def classify_run(matrix_run: Path, experiments_root: Path) -> dict[str, Any]:
    manifest = load_json(matrix_run / "reuse_12_demo_manifest.json")
    audit = (
        load_json(matrix_run / "audit_retarget_run.json")
        or load_json(matrix_run / "retarget_audit.json")
        or load_json(matrix_run / "run_audit.json")
    )
    source_run = find_source_run(matrix_run, manifest)
    source_root = experiments_root / source_run if source_run else None
    source_meta = load_json(source_root / "fresh_run_metadata.json") if source_root else None
    source_outcome = load_json(source_root / "fresh_run_outcome.json") if source_root else None
    source_failed = bool(
        source_outcome and source_outcome.get("status") in {"failed", "aborted"}
    )
    preflight_failed = ego_retarget_preflight_failed(source_root)
    gravity_path = (
        source_root
        / "intermediates"
        / "trajectory_6dof"
        / "do_as_i_do"
        / "reconstruction"
        / "clip"
        / "gravity.json"
        if source_root
        else None
    )
    gravity = load_json(gravity_path) if gravity_path else None
    gravity_quality = (gravity or {}).get("gravity_quality") or {}
    if gravity and (gravity.get("is_fallback") is True or gravity.get("estimation_status") == "fallback_assumption"):
        gravity_status = "fallback"
    elif gravity and gravity.get("is_fallback") is False and gravity.get("estimation_status") == "estimated":
        gravity_status = "estimated"
    else:
        gravity_status = "missing_or_unknown"
    logs = recent_log_text(source_root, matrix_run)
    videos = triptych_count(matrix_run)
    composed = manifest.get("composed") if manifest else None
    cells = manifest.get("cells") if manifest else None
    missing = manifest_missing_cells(manifest)
    has_12 = cells == 12 and composed == 12 and videos == 12 and missing == {}
    dai_aligned = source_alignment_ok(audit)
    dai_numerical_ok = dai_numerical_quality_ok(audit)
    spider_aligned, spider_ok, spider_ok_count, spider_fallback_count = spider_summary(audit)
    aligned = dai_aligned and spider_aligned
    errors = audit_error_text(audit)
    visual_success, visual_review = plain_robot_visual_review(matrix_run)
    matrix_provenance_ok = not any(
        "provenance" in error.lower() and "invalid" in error.lower()
        for error in errors
    )
    provenance_ok = successful_route_provenance_ok(
        visual_review, matrix_provenance_ok
    )

    dai_baseline_invalid = False
    if manifest:
        traj_validation = manifest.get("trajectory_asset_validation") or {}
        if isinstance(traj_validation, dict):
            dai_baseline_invalid = bool(traj_validation.get("do_as_i_do_quality_valid") is False)
    if any("do_as_i_do_quality_valid=false" in error for error in errors):
        dai_baseline_invalid = True

    numerical_quality_ready = bool(
        has_12
        and aligned
        and spider_ok
        and dai_numerical_ok
        and gravity_status == "estimated"
        and audit
        and audit.get("status") == "ok"
    )
    demo_ready = bool(numerical_quality_ready and provenance_ok and visual_success)
    successful_cells = visual_review.get("successful_cells") or []
    recommended_route_ready = bool(
        demo_ready and any("traj_egoinfinity__" in str(cell) for cell in successful_cells)
    )
    if visual_review.get("policy") == "retarget_only_plain_mujoco_v1":
        recommended_route_ready = demo_ready
    if manifest is None and source_failed:
        classification = "reject_or_stress_only"
        run_state = "failed"
    elif manifest is None:
        classification = "incomplete_or_in_progress"
        run_state = "incomplete_or_in_progress"
    elif demo_ready:
        classification = "showcase"
        run_state = "complete"
    elif has_12 and aligned:
        classification = "usable_with_caveat"
        run_state = "complete"
    else:
        classification = "reject_or_stress_only"
        run_state = "complete"
    failure_tags = failure_tags_for_run(
        manifest=manifest,
        audit=audit,
        errors=errors,
        log_text=logs,
        has_12=has_12,
        dai_aligned=dai_aligned,
        spider_aligned=spider_aligned,
        spider_fallback_count=spider_fallback_count,
        dai_baseline_invalid=dai_baseline_invalid,
        gravity_status=gravity_status,
    )
    if preflight_failed:
        failure_tags.append("retarget_input_preflight_failed")
    if source_failed:
        failure_tags.append("source_run_failed")

    return {
        "matrix_run": matrix_run.name,
        "source_run": source_run,
        "source_run_outcome": source_outcome or {},
        "source_failure_stage": (source_outcome or {}).get("stage"),
        "source_return_code": (source_outcome or {}).get("return_code"),
        "classification": classification,
        "run_state": run_state,
        "cells": cells,
        "composed": composed,
        "triptych_videos": videos,
        "missing_cells": missing,
        "matrix_complete": has_12,
        "audit_status": audit.get("status") if audit else "missing",
        "aligned": aligned,
        "dai_aligned": dai_aligned,
        "dai_numerical_ok": dai_numerical_ok,
        "spider_aligned": spider_aligned,
        "spider_ok_count": spider_ok_count,
        "spider_fallback_count": spider_fallback_count,
        "dai_baseline_invalid": dai_baseline_invalid,
        "plain_robot_visual_success": visual_success,
        "plain_robot_visual_review": visual_review,
        "provenance_ok": provenance_ok,
        "successful_route_provenance_ok": provenance_ok,
        "matrix_provenance_ok": matrix_provenance_ok,
        "numerical_quality_ready": numerical_quality_ready,
        "demo_ready": demo_ready,
        "recommended_route_ready": recommended_route_ready,
        "gravity_status": gravity_status,
        "gravity_source": (gravity or {}).get("estimation_source"),
        "gravity_roll_deg": (gravity or {}).get("roll_deg"),
        "gravity_pitch_deg": (gravity or {}).get("pitch_deg"),
        "gravity_selection_status": gravity_quality.get("status"),
        "gravity_selection_policy": gravity_quality.get("selection_policy"),
        "gravity_reference_frame": gravity_quality.get("reference_frame"),
        "gravity_selected_frame": gravity_quality.get("selected_frame"),
        "gravity_angular_error_p95_deg": gravity_quality.get("angular_error_p95_deg"),
        "geocalib_commit": (gravity or {}).get("geocalib_commit"),
        "gravity_weights_sha256": (gravity or {}).get("weights_sha256"),
        "failure_bucket": failure_bucket(failure_tags),
        "failure_tags": failure_tags,
        "error_count": len(errors),
        "first_errors": errors[:5],
        "git_commit": ((source_meta or {}).get("git") or {}).get("commit"),
        "git_dirty": ((source_meta or {}).get("git") or {}).get("dirty"),
        "dirty_diff_sha256": ((source_meta or {}).get("git") or {}).get("dirty_diff_sha256"),
        "gpu_mapping": (source_meta or {}).get("gpu_mapping"),
        "run_command": (source_meta or {}).get("run_command"),
    }


def scene_id_from_matrix_run(matrix_run: str) -> str:
    return matrix_run.split("_matrix12_", 1)[0]


def successful_retargeters(row: dict[str, Any]) -> set[str]:
    if not row.get("demo_ready"):
        return set()
    review = row.get("plain_robot_visual_review") or {}
    if review.get("policy") == "retarget_only_plain_mujoco_v1":
        retargeters = review.get("retargeters") or {}
        return {
            name
            for name in ("do_as_i_do", "spider")
            if isinstance(retargeters, dict) and retargeters.get(name) == "pass"
        }
    result: set[str] = set()
    for cell in review.get("successful_cells") or []:
        name = str(cell)
        if name.endswith("__retarget_do_as_i_do"):
            result.add("do_as_i_do")
        elif name.endswith("__retarget_spider"):
            result.add("spider")
    return result


def summarize_scenes(rows: list[dict[str, Any]]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        scene_id = scene_id_from_matrix_run(str(row.get("matrix_run") or ""))
        if scene_id:
            grouped.setdefault(scene_id, []).append(row)

    inventory: list[dict[str, Any]] = []
    for scene_id, scene_rows in sorted(grouped.items()):
        demo_rows = [row for row in scene_rows if row.get("demo_ready")]
        complete_rows = [row for row in scene_rows if row.get("matrix_complete")]
        deliverable_rows = [
            row for row in scene_rows if row.get("demo_ready") and row.get("matrix_complete")
        ]
        retargeters = set().union(*(successful_retargeters(row) for row in scene_rows))
        successful_cells = sorted(
            {
                str(cell)
                for row in demo_rows
                for cell in ((row.get("plain_robot_visual_review") or {}).get("successful_cells") or [])
            }
        )
        ranked_rows = sorted(
            scene_rows,
            key=lambda row: (
                not bool(row.get("demo_ready") and row.get("matrix_complete")),
                not bool(row.get("demo_ready")),
                not bool(row.get("recommended_route_ready")),
                not bool(row.get("matrix_complete")),
                str(row.get("matrix_run") or ""),
            ),
        )
        inventory.append(
            {
                "scene_id": scene_id,
                "run_count": len(scene_rows),
                "matrix_runs": sorted(str(row.get("matrix_run") or "") for row in scene_rows),
                "best_run": str(ranked_rows[0].get("matrix_run") or ""),
                "demo_ready": bool(demo_rows),
                "dai_success": "do_as_i_do" in retargeters,
                "spider_success": "spider" in retargeters,
                "recommended_route_ready": any(
                    bool(row.get("recommended_route_ready")) for row in scene_rows
                ),
                "matrix_complete": bool(complete_rows),
                "deliverable_12_of_12": bool(deliverable_rows),
                "demo_ready_runs": sorted(str(row.get("matrix_run") or "") for row in demo_rows),
                "deliverable_12_of_12_runs": sorted(
                    str(row.get("matrix_run") or "") for row in deliverable_rows
                ),
                "successful_cells": successful_cells,
            }
        )

    counts = {
        "tested_matrix_scenes": len(inventory),
        "demo_ready_any_route": sum(bool(item["demo_ready"]) for item in inventory),
        "dai_success_any_route": sum(bool(item["dai_success"]) for item in inventory),
        "spider_success_any_route": sum(bool(item["spider_success"]) for item in inventory),
        "recommended_route_ready": sum(
            bool(item["recommended_route_ready"]) for item in inventory
        ),
        "matrix_complete_any_quality": sum(bool(item["matrix_complete"]) for item in inventory),
        "deliverable_12_of_12": sum(
            bool(item["deliverable_12_of_12"]) for item in inventory
        ),
    }
    return counts, inventory


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "classification",
        "run_state",
        "matrix_run",
        "source_run",
        "cells",
        "composed",
        "triptych_videos",
        "audit_status",
        "aligned",
        "dai_aligned",
        "dai_numerical_ok",
        "spider_aligned",
        "spider_ok_count",
        "spider_fallback_count",
        "dai_baseline_invalid",
        "plain_robot_visual_success",
        "provenance_ok",
        "numerical_quality_ready",
        "demo_ready",
        "recommended_route_ready",
        "gravity_status",
        "gravity_source",
        "gravity_roll_deg",
        "gravity_pitch_deg",
        "gravity_selection_status",
        "gravity_selection_policy",
        "gravity_reference_frame",
        "gravity_selected_frame",
        "gravity_angular_error_p95_deg",
        "geocalib_commit",
        "gravity_weights_sha256",
        "failure_bucket",
        "failure_tags",
        "error_count",
        "git_commit",
        "git_dirty",
        "dirty_diff_sha256",
        "first_errors",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fields}
            out["first_errors"] = " | ".join(row.get("first_errors") or [])
            out["failure_tags"] = ",".join(row.get("failure_tags") or [])
            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize and classify AoE retarget matrix runs for demo-pool curation.")
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--matrix-glob", default="*matrix12*")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = args.experiments_root
    runs = sorted(path for path in root.glob(args.matrix_glob) if path.is_dir())
    rows = [classify_run(run, root) for run in runs]
    order = {
        "showcase": 0,
        "usable_with_caveat": 1,
        "incomplete_or_in_progress": 2,
        "reject_or_stress_only": 3,
    }
    rows.sort(key=lambda row: (order.get(str(row["classification"]), 99), row["matrix_run"]))
    if args.limit > 0:
        rows = rows[: args.limit]

    counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    recommended_route_ready_count = 0
    for row in rows:
        counts[str(row["classification"])] = counts.get(str(row["classification"]), 0) + 1
        if row.get("recommended_route_ready"):
            recommended_route_ready_count += 1
        bucket = str(row.get("failure_bucket") or "unclassified")
        failure_counts[bucket] = failure_counts.get(bucket, 0) + 1
    scene_counts, scene_inventory = summarize_scenes(rows)
    payload = {
        "experiments_root": str(root),
        "matrix_glob": args.matrix_glob,
        "counts": counts,
        "recommended_route_ready_count": recommended_route_ready_count,
        "failure_counts": failure_counts,
        "scene_counts": scene_counts,
        "scene_inventory": scene_inventory,
        "runs": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        write_csv(args.output_csv, rows)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv) if args.output_csv else None,
                "counts": counts,
                "recommended_route_ready_count": recommended_route_ready_count,
                "failure_counts": failure_counts,
                "scene_counts": scene_counts,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
