from __future__ import annotations

import json
import hashlib
import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_retarget_run import (  # noqa: E402
    audit_cells,
    audit_spider,
    dai_alignment_errors,
    dai_raw_to_processed_findings,
    dai_tracking_quality_findings,
    ego_retarget_input_qc_findings,
    forbidden_robot_rgb_artifacts,
    route_availability,
    summarize_route_admission,
)


def make_matrix(tmp_path: Path) -> Path:
    matrix = tmp_path / "matrix"
    for index in range(12):
        cell = matrix / "cells" / f"cell_{index:02d}"
        cell.mkdir(parents=True)
        for name in ("overlay.mp4", "depth.mp4", "robot.mp4"):
            (cell / name).touch()
        (cell / "manifest.json").write_text(
            json.dumps({"settings": {"retargeting": "none"}, "assets": {}}),
            encoding="utf-8",
        )
        video = matrix / "videos" / f"cell_{index:02d}__triptych.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.touch()
    return matrix


def base_manifest() -> dict:
    return {
        "cells": 12,
        "composed": 12,
        "missing_cells": {},
        "spider_fuse_ego_object_for_dai": False,
        "trajectory_asset_validation": {
            "do_as_i_do_quality_valid": False,
            "do_as_i_do_equals_egoinfinity": False,
            "object_provenance": {
                "do_as_i_do": {
                    "object_track_source": "dai_native",
                    "object_mesh_source": "dai_native",
                    "retarget_object_source": "dai_native",
                    "status": "invalid",
                    "baseline_only": True,
                }
            },
        },
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_preflight_evidence(
    root: Path,
    cell: str,
    *,
    input_rc: int,
    status: str = "failed_input_preflight",
) -> dict:
    route = cell[5:] if cell.startswith("traj_") else cell
    trajectory, remainder = route.split("__hand_", 1)
    hand_source, backend = remainder.rsplit("__retarget_", 1)
    adapter_path = None
    adapter_payload = None
    if input_rc == 0:
        adapter_path = root / "raw" / cell / "adapter_manifest.json"
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_payload = {
            "adapter_schema_version": 3,
            "trajectory_6dof": trajectory,
            "hand_source": hand_source,
            "retarget_input_qc": {"status": "ok"},
            "adapter_rigid_invariance": {"status": "ok"},
        }
        adapter_path.write_text(json.dumps(adapter_payload), encoding="utf-8")
    path = root / "logs" / f"route_preflight__{cell}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "route": cell,
        "trajectory_6dof": trajectory,
        "hand_source": hand_source,
        "retargeting": backend,
        "status": status,
        "input_rc": input_rc,
        "underlying_adapter_manifest": {
            "path": (
                str(adapter_path.relative_to(root)) if adapter_path is not None else None
            ),
            "path_base": "experiment_root",
            "sha256": sha256(adapter_path) if adapter_path is not None else None,
            "exists": adapter_path is not None,
            "adapter_schema_version": (
                adapter_payload.get("adapter_schema_version") if adapter_payload else None
            ),
            "hand_source": adapter_payload.get("hand_source") if adapter_payload else None,
            "retarget_input_qc_status": "ok" if adapter_payload else None,
            "adapter_rigid_invariance_status": "ok" if adapter_payload else None,
        },
        "evidence_validation": {"status": "ok", "errors": []},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "path": str(path.relative_to(root)),
        "path_base": "experiment_root",
        "sha256": sha256(path),
        "trajectory_6dof": trajectory,
        "hand_source": hand_source,
        "status": status,
        "input_rc": input_rc,
    }


def make_exact_route_cell(
    matrix: Path,
    cell: str,
    *,
    retargeting: str,
    source_robot: str | None,
    provenance_status: str,
) -> Path:
    original = matrix / "cells" / "cell_00"
    route_dir = matrix / "cells" / cell
    original.rename(route_dir)
    original_triptych = matrix / "videos" / "cell_00__triptych.mp4"
    original_triptych.rename(matrix / "videos" / f"{cell}__triptych.mp4")
    route = cell[5:] if cell.startswith("traj_") else cell
    trajectory, remainder = route.split("__hand_", 1)
    hand_source = remainder.split("__retarget_", 1)[0]
    (route_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cell_key": cell,
                "settings": {
                    "trajectory_6dof": trajectory,
                    "hand_source": hand_source,
                    "retargeting": retargeting,
                },
                "assets": {"source.robot": source_robot},
                "provenance_validation": {
                    "status": provenance_status,
                    "errors": [] if provenance_status == "ok" else ["input route unavailable"],
                },
            }
        ),
        encoding="utf-8",
    )
    return route_dir


def test_explicit_invalid_dai_baseline_is_warning(tmp_path: Path) -> None:
    report = audit_cells(make_matrix(tmp_path), base_manifest(), allow_legacy_unaligned=False)
    assert report["errors"] == []
    assert any("baseline" in warning for warning in report["warnings"])


def test_matrix_completeness_is_diagnostic_not_admission_error(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)
    (matrix / "cells" / "cell_11").rename(matrix / "omitted_cell_11")
    (matrix / "videos" / "cell_11__triptych.mp4").unlink()
    manifest = base_manifest()
    manifest.update(
        {
            "cells": 11,
            "composed": 11,
            "missing_cells": {"cell_11": ["robot"]},
        }
    )

    report = audit_cells(matrix, manifest, allow_legacy_unaligned=False)

    assert report["errors"] == []
    assert report["matrix_complete"] is False
    assert report["matrix_completeness_warnings"]
    assert any("matrix completeness" in warning for warning in report["warnings"])


def test_plain_only_route_audit_does_not_require_triptych(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)
    cell = "traj_egoinfinity__hand_estimated__retarget_spider"
    robot = (
        matrix
        / "intermediates"
        / "retargeting"
        / "spider"
        / cell
        / "robot"
        / "visualization_mjwp_act_aligned.mp4"
    )
    robot.parent.mkdir(parents=True)
    robot.touch()
    make_exact_route_cell(
        matrix,
        cell,
        retargeting="spider",
        source_robot=str(robot),
        provenance_status="ok",
    )
    triptych = matrix / "videos" / f"{cell}__triptych.mp4"
    triptych.unlink()

    composed = audit_cells(
        matrix,
        base_manifest(),
        allow_legacy_unaligned=False,
    )
    plain_only = audit_cells(
        matrix,
        base_manifest(),
        allow_legacy_unaligned=False,
        require_triptych=False,
    )

    assert any("exact-route triptych is missing" in error for error in composed["errors"])
    assert plain_only["errors"] == []
    assert plain_only["triptych_required_for_admission"] is False
    assert cell in plain_only["spider_candidate_routes"]


def test_explicit_input_unavailable_route_may_omit_robot_and_triptych(
    tmp_path: Path,
) -> None:
    matrix = make_matrix(tmp_path)
    cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    route_dir = make_exact_route_cell(
        matrix,
        cell,
        retargeting="do_as_i_do",
        source_robot=None,
        provenance_status="invalid",
    )
    (route_dir / "robot.mp4").unlink()
    (matrix / "videos" / f"{cell}__triptych.mp4").unlink()
    manifest = base_manifest()
    manifest.update(
        {
            "composed": 11,
            "missing_cells": {cell: ["robot"]},
            "route_availability": {
                cell: {
                    "input_rc": 4,
                    "preflight_evidence": make_preflight_evidence(
                        matrix, cell, input_rc=4
                    ),
                }
            },
            "provenance_validation": {
                cell: {"status": "invalid", "errors": ["input route unavailable"]}
            },
        }
    )

    report = audit_cells(matrix, manifest, allow_legacy_unaligned=False)

    assert report["errors"] == []
    assert report["unavailable_routes"] == [cell]
    route = next(item for item in report["cells"] if item["cell"] == cell)
    assert route["admission_status"] == "unavailable"


def test_unavailable_route_with_robot_output_fails_closed(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)
    cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    make_exact_route_cell(
        matrix,
        cell,
        retargeting="do_as_i_do",
        source_robot=None,
        provenance_status="invalid",
    )
    manifest = base_manifest()
    manifest["route_availability"] = {
        cell: {
            "input_rc": 4,
            "preflight_evidence": make_preflight_evidence(matrix, cell, input_rc=4),
        }
    }

    report = audit_cells(matrix, manifest, allow_legacy_unaligned=False)

    assert any("cannot masquerade as an admitted demo" in error for error in report["errors"])
    assert cell not in report["admitted_routes"]


def test_input_rc_cannot_self_prove_unavailability(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)
    cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    route_dir = make_exact_route_cell(
        matrix,
        cell,
        retargeting="do_as_i_do",
        source_robot=None,
        provenance_status="invalid",
    )
    (route_dir / "robot.mp4").unlink()
    (matrix / "videos" / f"{cell}__triptych.mp4").unlink()
    manifest = base_manifest()
    manifest["route_availability"] = {cell: {"input_rc": 4}}

    report = audit_cells(matrix, manifest, allow_legacy_unaligned=False)

    assert cell not in report["unavailable_routes"]
    assert cell in report["invalid_routes"]
    assert any("missing preflight_evidence" in error for error in report["errors"])


def test_preflight_sha_mismatch_is_not_trusted(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)
    cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    route_dir = make_exact_route_cell(
        matrix,
        cell,
        retargeting="do_as_i_do",
        source_robot=None,
        provenance_status="invalid",
    )
    (route_dir / "robot.mp4").unlink()
    (matrix / "videos" / f"{cell}__triptych.mp4").unlink()
    evidence = make_preflight_evidence(matrix, cell, input_rc=4)
    evidence["sha256"] = "0" * 64
    manifest = base_manifest()
    manifest["route_availability"] = {
        cell: {"input_rc": 4, "preflight_evidence": evidence}
    }

    report = audit_cells(matrix, manifest, allow_legacy_unaligned=False)

    assert cell not in report["unavailable_routes"]
    assert any("sha256 mismatch" in error for error in report["errors"])


def test_underlying_adapter_sha_and_qc_are_revalidated(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)
    cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    route_dir = make_exact_route_cell(
        matrix,
        cell,
        retargeting="do_as_i_do",
        source_robot=None,
        provenance_status="invalid",
    )
    (route_dir / "robot.mp4").unlink()
    (matrix / "videos" / f"{cell}__triptych.mp4").unlink()
    evidence = make_preflight_evidence(matrix, cell, input_rc=0, status="ok")
    evidence_payload = json.loads(
        (matrix / evidence["path"]).read_text(encoding="utf-8")
    )
    adapter = matrix / evidence_payload["underlying_adapter_manifest"]["path"]
    adapter_payload = json.loads(adapter.read_text(encoding="utf-8"))
    adapter_payload["retarget_input_qc"]["status"] = "invalid"
    adapter.write_text(json.dumps(adapter_payload), encoding="utf-8")
    manifest = base_manifest()
    manifest["route_availability"] = {
        cell: {
            "input_rc": 0,
            "post_retarget_rc": 7,
            "status": "failed",
            "available": False,
            "preflight_evidence": evidence,
        }
    }

    report = audit_cells(matrix, manifest, allow_legacy_unaligned=False)

    assert cell in report["invalid_routes"]
    assert any("adapter sha256" in error for error in report["errors"])
    assert any("retarget_input_qc" in error for error in report["errors"])


def test_post_retarget_failure_is_route_local_invalid_without_outputs(
    tmp_path: Path,
) -> None:
    matrix = make_matrix(tmp_path)
    cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    route_dir = make_exact_route_cell(
        matrix,
        cell,
        retargeting="do_as_i_do",
        source_robot=None,
        provenance_status="invalid",
    )
    (route_dir / "robot.mp4").unlink()
    (matrix / "videos" / f"{cell}__triptych.mp4").unlink()
    manifest = base_manifest()
    manifest["route_availability"] = {
        cell: {
            "input_rc": 0,
            "post_retarget_rc": 7,
            "status": "failed",
            "available": False,
            "preflight_evidence": make_preflight_evidence(
                matrix, cell, input_rc=0, status="ok"
            ),
        }
    }

    report = audit_cells(matrix, manifest, allow_legacy_unaligned=False)

    assert report["errors"] == []
    assert report["unavailable_routes"] == []
    assert report["invalid_routes"] == [cell]
    route = next(item for item in report["cells"] if item["cell"] == cell)
    assert route["admission_status"] == "post_retarget_invalid"


def test_post_retarget_failure_with_output_fails_closed(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)
    cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    make_exact_route_cell(
        matrix,
        cell,
        retargeting="do_as_i_do",
        source_robot=None,
        provenance_status="invalid",
    )
    manifest = base_manifest()
    manifest["route_availability"] = {
        cell: {
            "input_rc": 0,
            "post_retarget_rc": 7,
            "status": "failed",
            "available": False,
            "preflight_evidence": make_preflight_evidence(
                matrix, cell, input_rc=0, status="ok"
            ),
        }
    }

    report = audit_cells(matrix, manifest, allow_legacy_unaligned=False)

    assert any("post-retarget-invalid route exposes" in error for error in report["errors"])
    assert cell in report["invalid_routes"]


def test_route_suffix_backend_cannot_be_spoofed_by_settings(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)
    cell = "traj_egoinfinity__hand_aoe__retarget_spider"
    route_dir = make_exact_route_cell(
        matrix,
        cell,
        retargeting="do_as_i_do",
        source_robot=None,
        provenance_status="invalid",
    )
    (route_dir / "robot.mp4").unlink()
    (matrix / "videos" / f"{cell}__triptych.mp4").unlink()
    manifest = base_manifest()
    manifest["route_availability"] = {
        cell: {
            "input_rc": 4,
            "preflight_evidence": make_preflight_evidence(matrix, cell, input_rc=4),
        }
    }

    report = audit_cells(matrix, manifest, allow_legacy_unaligned=False)

    assert any("route suffix requires spider" in error for error in report["errors"])
    assert cell not in report["admitted_routes"]


def test_manifest_cell_key_must_match_directory(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)
    cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    route_dir = make_exact_route_cell(
        matrix,
        cell,
        retargeting="do_as_i_do",
        source_robot=None,
        provenance_status="invalid",
    )
    payload = json.loads((route_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["cell_key"] = "traj_other__hand_estimated__retarget_do_as_i_do"
    (route_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    (route_dir / "robot.mp4").unlink()
    (matrix / "videos" / f"{cell}__triptych.mp4").unlink()
    manifest = base_manifest()
    manifest["route_availability"] = {
        cell: {
            "input_rc": 4,
            "preflight_evidence": make_preflight_evidence(matrix, cell, input_rc=4),
        }
    }

    report = audit_cells(matrix, manifest, allow_legacy_unaligned=False)

    assert any("does not match cell directory" in error for error in report["errors"])


def test_source_robot_must_be_inside_exact_route_root(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)
    cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    other_cell = "traj_egoinfinity__hand_estimated__retarget_do_as_i_do"
    wrong_robot = (
        matrix
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / other_cell
        / "retargeting_outputs"
        / "sharpa"
        / "visualization_mjwp_act_aligned.mp4"
    )
    wrong_robot.parent.mkdir(parents=True)
    wrong_robot.touch()
    make_exact_route_cell(
        matrix,
        cell,
        retargeting="do_as_i_do",
        source_robot=str(wrong_robot),
        provenance_status="ok",
    )

    report = audit_cells(matrix, base_manifest(), allow_legacy_unaligned=False)

    assert any("source.robot is not bound to the exact route root" in error for error in report["errors"])
    assert cell not in report["admitted_routes"]


def test_dai_unavailability_does_not_disable_exact_spider_route() -> None:
    dai = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    spider = "traj_egoinfinity__hand_aoe__retarget_spider"

    result = route_availability(spider, {dai: {"input_rc": 4}})

    assert result["status"] == "unknown"
    assert result["explicit_input_unavailable"] is False


def test_pure_dai_rejects_ego_fusion(tmp_path: Path) -> None:
    manifest = base_manifest()
    manifest["spider_fuse_ego_object_for_dai"] = True
    provenance = manifest["trajectory_asset_validation"]["object_provenance"]["do_as_i_do"]
    provenance["object_track_source"] = "fused_ego_for_spider"
    provenance["object_mesh_source"] = "egoinfinity"
    provenance["retarget_object_source"] = "fused_ego_for_spider"
    report = audit_cells(make_matrix(tmp_path), manifest, allow_legacy_unaligned=False)
    assert any("spider_fuse_ego_object_for_dai" in error for error in report["errors"])
    assert any("invalid DAI baseline provenance" in error for error in report["errors"])


def test_pure_dai_tracking_failure_is_error() -> None:
    path = Path("traj_do_as_i_do__hand_estimated__retarget_do_as_i_do/mjwp_alignment_manifest.json")
    errors, warnings = dai_tracking_quality_findings(
        path,
        {"object_tracking_quality": {"status": "invalid", "errors": ["object flyout"]}},
        allow_legacy_unaligned=False,
    )
    assert any("baseline numerical tracking error" in error for error in errors)
    assert warnings == []


def test_dai_alignment_rejects_incomplete_source_coverage(tmp_path: Path) -> None:
    video = tmp_path / "visualization_mjwp_act_aligned.mp4"
    video.touch()
    manifest = tmp_path / "mjwp_alignment_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "applied": True,
                "alignment_kind": "drop_warmup_and_sample_exact_source_timestamps",
                "start_index": 599,
                "warmup_steps": 600,
                "target_count": 68,
                "source_frame_count": 68,
                "complete_source_coverage": False,
                "exact_source_timestamps": True,
                "sampling_semantics": "exact_source_timestamp_indices",
                "duration_coverage_ratio": 0.67,
                "aligned_video": str(video),
            }
        ),
        encoding="utf-8",
    )
    errors = dai_alignment_errors(manifest, allow_legacy_unaligned=False)
    assert any("strict canonical artifact/source-video" in error for error in errors)


def test_dai_audit_delegates_to_shared_strict_alignment_validator() -> None:
    source = inspect.getsource(dai_alignment_errors)
    assert "valid_dai_alignment(manifest_path.parent, REPO_ROOT)" in source


def test_ego_to_dai_tracking_failure_is_numerical_error() -> None:
    path = Path("traj_egoinfinity__hand_estimated__retarget_do_as_i_do/mjwp_alignment_manifest.json")
    errors, warnings = dai_tracking_quality_findings(
        path,
        {"object_tracking_quality": {"status": "invalid", "errors": ["object flyout"]}},
        allow_legacy_unaligned=False,
    )
    assert any("object tracking status=invalid" in error for error in errors)
    assert warnings == []


def test_dai_audit_requires_exact_route_raw_to_processed_invariance(
    tmp_path: Path,
) -> None:
    alignment = (
        tmp_path
        / "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
        / "retargeting_outputs"
        / "sharpa"
        / "left"
        / "fixture_left"
        / "0"
        / "mjwp_alignment_manifest.json"
    )
    alignment.parent.mkdir(parents=True)

    errors, evidence = dai_raw_to_processed_findings(alignment)
    assert any("missing exact-route raw_to_processed" in error for error in errors)

    report = (
        alignment.parents[4]
        / "mano"
        / "left"
        / "fixture_left"
        / "0"
        / "raw_to_processed_hoi_invariance.json"
    )
    report.parent.mkdir(parents=True)
    cell_root = alignment.parents[5]
    adapter = cell_root / "raw_dir" / "adapter_manifest.json"
    raw_hand = cell_root / "raw_dir" / "raw_hand.npz"
    raw_layout = cell_root / "raw_dir" / "layout.json"
    processed = report.with_name("trajectory_keypoints.npz")
    adapter.parent.mkdir(parents=True, exist_ok=True)
    adapter.write_text("{}", encoding="utf-8")
    raw_hand.write_bytes(b"raw-hand")
    raw_layout.write_text("{}", encoding="utf-8")
    processed.write_bytes(b"processed")

    def binding(path: Path) -> dict:
        return {"path": str(path), "sha256": sha256(path)}

    report.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "full_relative_se3_checked": True,
                "status": "ok",
                "production_eligible": True,
                "cross_hand_source_comparison": False,
                "comparison_semantics": "exact_route_exact_hand_raw_adapter_to_dai_processed",
                "trajectory_6dof": "egoinfinity",
                "hand_source": "aoe",
                "hand_type": "left",
                "joint_fit_components": ["object", "left"],
                "route_binding": {
                    "trajectory_6dof": "egoinfinity",
                    "hand_source": "aoe",
                    "hand_type": "left",
                    "requested_sides": ["left"],
                    "adapter_manifest": binding(adapter),
                    "raw_inputs": {
                        "hand_npz": binding(raw_hand),
                        "object_layout": binding(raw_layout),
                    },
                    "processed_keypoints": binding(processed),
                },
                "adapter_manifest": str(adapter),
                "adapter_manifest_sha256": sha256(adapter),
                "raw_hand_npz": str(raw_hand),
                "raw_hand_sha256": sha256(raw_hand),
                "raw_object_layout": str(raw_layout),
                "raw_object_layout_sha256": sha256(raw_layout),
                "processed_keypoints": str(processed),
                "processed_keypoints_sha256": sha256(processed),
            }
        ),
        encoding="utf-8",
    )

    errors, evidence = dai_raw_to_processed_findings(alignment)
    assert errors == []
    assert evidence["path"] == str(report)


def test_spider_tracking_failure_is_error(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    robot = matrix / "visualization_mjwp_act_aligned.mp4"
    robot.parent.mkdir(parents=True)
    robot.touch()
    cell = "traj_egoinfinity__hand_estimated__retarget_spider"
    expected_dai_cell = "traj_egoinfinity__hand_estimated__retarget_do_as_i_do"
    cell_root = matrix / "intermediates" / "retargeting" / "spider" / cell
    cell_root.mkdir(parents=True)
    (cell_root / "spider_input_manifest.json").write_text(
        json.dumps(
            {
                "expected_dai_cell_key": expected_dai_cell,
                "source_keypoints": f"{expected_dai_cell}/trajectory_keypoints.npz",
            }
        ),
        encoding="utf-8",
    )
    (cell_root / "spider_output_manifest.json").write_text(
        json.dumps(
            {
                "object_tracking_quality": {
                    "status": "invalid",
                    "metrics": {
                        "pos_median_m": 0.30,
                        "pos_p95_m": 0.50,
                        "lost_frame_fraction": 0.79,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    report = audit_spider(
        matrix,
        {
            "spider": [
                {
                    "cell": cell,
                    "status": "reviewable_invalid_tracking",
                    "robot": str(robot),
                }
            ]
        },
        allow_legacy_unaligned=True,
    )

    assert any("status=reviewable_invalid_tracking" in error for error in report["errors"])
    assert any("object tracking quality is not valid" in error for error in report["errors"])
    assert any("raw_to_processed_hoi_invariance" in error for error in report["errors"])
    assert any("rest_support_provenance" in error for error in report["errors"])
    assert any("authoritative aligned" in error for error in report["errors"])
    assert not any("tracking" in warning for warning in report["warnings"])


def test_spider_invalid_input_without_output_is_route_local_unavailable(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    cell = "traj_do_as_i_do__hand_aoe__retarget_spider"
    preflight = make_preflight_evidence(
        matrix, cell, input_rc=4, status="invalid_input"
    )

    report = audit_spider(
        matrix,
        {
            "spider": [
                {
                    "cell": cell,
                    "status": "invalid_input",
                    "input_status": "invalid_input",
                    "input_rc": 4,
                    "returncode": 4,
                    "robot": None,
                    "preflight_evidence": preflight,
                }
            ]
        },
        allow_legacy_unaligned=False,
    )

    assert report["errors"] == []
    assert report["admitted_routes"] == []
    assert report["unavailable_routes"] == [cell]
    assert report["items"][0]["admission_status"] == "unavailable"


def test_spider_manifest_item_cannot_claim_a_dai_route(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix"
    matrix.mkdir()
    cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"

    report = audit_spider(
        matrix,
        {
            "spider": [
                {
                    "cell": cell,
                    "status": "invalid_input",
                    "input_rc": 4,
                    "robot": None,
                }
            ]
        },
        allow_legacy_unaligned=False,
    )

    assert any("not an exact __retarget_spider route" in error for error in report["errors"])
    assert report["admitted_routes"] == []


def test_route_admission_requires_spider_backend_and_matrix_cell_checks() -> None:
    dai = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    spider = "traj_egoinfinity__hand_aoe__retarget_spider"
    backend_only = "traj_do_as_i_do__hand_aoe__retarget_spider"

    result = summarize_route_admission(
        {
            "admitted_routes": [dai],
            "spider_candidate_routes": [spider],
            "unavailable_routes": [],
        },
        {
            "admitted_routes": [spider, backend_only],
            "unavailable_routes": [],
            "items": [{"cell": spider}, {"cell": backend_only}],
        },
    )

    assert result["admitted_routes"] == [dai, spider]
    assert result["admitted_route_count"] == 2
    assert result["demo_count"] == 2
    assert result["scene_admitted"] is True
    assert any(backend_only in error for error in result["errors"])


def test_invalid_route_is_top_level_and_does_not_poison_admitted_sibling() -> None:
    admitted = "traj_egoinfinity__hand_estimated__retarget_do_as_i_do"
    invalid = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    result = summarize_route_admission(
        {
            "admitted_routes": [admitted],
            "spider_candidate_routes": [],
            "unavailable_routes": [],
            "invalid_routes": [invalid],
            "invalid_route_details": [
                {"cell": invalid, "reason": "post_retarget_invalid"}
            ],
        },
        {"admitted_routes": [], "unavailable_routes": [], "items": []},
    )

    assert result["admitted_routes"] == [admitted]
    assert result["invalid_routes"] == [invalid]
    assert result["invalid_route_details"][0]["reason"] == "post_retarget_invalid"


def test_ego_to_dai_requires_passing_adapter_input_qc(tmp_path: Path) -> None:
    cell = tmp_path / "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
    alignment = cell / "retargeting_outputs" / "robot" / "mjwp_alignment_manifest.json"
    alignment.parent.mkdir(parents=True)
    adapter = cell / "raw_dir" / "adapter_manifest.json"
    adapter.parent.mkdir(parents=True)
    adapter.write_text(
        json.dumps({"retarget_input_qc": {"status": "invalid", "errors": ["unstable scale"]}}),
        encoding="utf-8",
    )
    errors, evidence = ego_retarget_input_qc_findings(alignment, allow_legacy_unaligned=False)
    assert any("invalid retarget_input_qc" in error for error in errors)
    assert evidence["adapter_manifest"] == str(adapter)

    adapter.write_text(
        json.dumps({"retarget_input_qc": {"status": "ok", "errors": []}}),
        encoding="utf-8",
    )
    errors, evidence = ego_retarget_input_qc_findings(alignment, allow_legacy_unaligned=False)
    assert errors == []
    assert evidence["retarget_input_qc"]["status"] == "ok"


def test_pure_dai_does_not_require_ego_input_qc(tmp_path: Path) -> None:
    alignment = (
        tmp_path
        / "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do"
        / "retargeting_outputs"
        / "robot"
        / "mjwp_alignment_manifest.json"
    )
    errors, evidence = ego_retarget_input_qc_findings(alignment, allow_legacy_unaligned=False)
    assert errors == []
    assert evidence == {}


def test_robot_rgb_overlay_artifact_is_forbidden(tmp_path: Path) -> None:
    artifact = tmp_path / "visualization_mjwp_rgb_overlay_aligned.mp4"
    artifact.touch()
    assert forbidden_robot_rgb_artifacts(tmp_path) == [artifact]


def test_matrix_bottom_rejects_robot_rgb_overlay(tmp_path: Path) -> None:
    matrix = make_matrix(tmp_path)
    source = matrix / "retarget" / "visualization_mjwp_rgb_overlay_aligned.mp4"
    source.parent.mkdir(parents=True)
    source.touch()
    manifest_path = matrix / "cells" / "cell_00" / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "settings": {"retargeting": "do_as_i_do"},
                "assets": {"source.robot": str(source)},
            }
        ),
        encoding="utf-8",
    )
    report = audit_cells(matrix, base_manifest(), allow_legacy_unaligned=False)
    assert any("bottom panel is not a plain aligned MuJoCo render" in error for error in report["errors"])
