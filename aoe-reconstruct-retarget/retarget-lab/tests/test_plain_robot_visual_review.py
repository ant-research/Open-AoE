from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review" / "record_plain_robot_visual_review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("plain_robot_review", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_invariance_report(
    root: Path,
    *,
    trajectory: str,
    hand_source: str,
    hand_type: str,
    adapter_manifest: Path,
    processed_keypoints: Path,
) -> dict:
    # Exact-route reports bind the raw adapter artifacts, never convenient
    # matrix-metadata copies.
    raw_hand = adapter_manifest.parent / "raw" / "all_hand_meshes.npz"
    raw_layout = (
        adapter_manifest.parent
        / "obj_tracking_out/object/combined_visualization/"
        "layout_camera_frame_optimized.json"
    )
    raw_hand.parent.mkdir(parents=True, exist_ok=True)
    raw_layout.parent.mkdir(parents=True, exist_ok=True)
    raw_hand.write_bytes(b"raw-hand")
    raw_layout.write_text("{}", encoding="utf-8")
    sides = ["left", "right"] if hand_type == "bimanual" else [hand_type]

    def binding(path: Path) -> dict:
        return {"path": str(path), "sha256": sha256(path)}

    adapter_binding = binding(adapter_manifest)
    raw_hand_binding = binding(raw_hand)
    raw_layout_binding = binding(raw_layout)
    processed_binding = binding(processed_keypoints)
    return {
        "schema_version": 3,
        "full_relative_se3_checked": True,
        "status": "ok",
        "production_eligible": True,
        "cross_hand_source_comparison": False,
        "comparison_semantics": "exact_route_exact_hand_raw_adapter_to_dai_processed",
        "trajectory_6dof": trajectory,
        "hand_source": hand_source,
        "hand_type": hand_type,
        "joint_fit_components": ["object", *sides],
        "route_binding": {
            "trajectory_6dof": trajectory,
            "hand_source": hand_source,
            "hand_type": hand_type,
            "requested_sides": sides,
            "adapter_manifest": adapter_binding,
            "raw_inputs": {
                "hand_npz": raw_hand_binding,
                "object_layout": raw_layout_binding,
            },
            "processed_keypoints": processed_binding,
        },
        "adapter_manifest": str(adapter_manifest),
        "adapter_manifest_sha256": sha256(adapter_manifest),
        "raw_hand_npz": str(raw_hand),
        "raw_hand_sha256": sha256(raw_hand),
        "raw_object_layout": str(raw_layout),
        "raw_object_layout_sha256": sha256(raw_layout),
        "processed_keypoints": str(processed_keypoints),
        "processed_keypoints_sha256": sha256(processed_keypoints),
        "errors": [],
    }


def cell_labels(cell: str) -> tuple[str, str]:
    route = cell[5:] if cell.startswith("traj_") else cell
    trajectory, remainder = route.split("__hand_", 1)
    hand_source = remainder.split("__retarget_", 1)[0]
    return trajectory, hand_source


def make_validated_cell(
    matrix: Path,
    cell: str,
    retargeter: str,
    *,
    route_status: str = "ok",
    tracking_status: str = "ok",
    interaction_status: str = "ok",
    provenance_status: str = "ok",
) -> dict[str, Path]:
    trajectory, hand_source = cell_labels(cell)
    expected_dai_cell = (
        cell
        if retargeter == "do_as_i_do"
        else cell.rsplit("__retarget_", 1)[0] + "__retarget_do_as_i_do"
    )
    route_input_qc = {
        "status": route_status,
        "errors": [] if route_status == "ok" else ["route mismatch"],
        "expected_cell_key": expected_dai_cell,
        "trajectory": trajectory,
        "hand_source": hand_source,
    }
    interaction_quality = {
        "status": interaction_status,
        "errors": [] if interaction_status == "ok" else ["contact retention"],
        "active_sides": ["left"],
        "thresholds": {
            "reference_contact_m": 0.05,
            "executed_contact_m": 0.08,
            "distance_error_median_m": 0.03,
            "distance_error_p95_m": 0.08,
            "min_reference_contact_frames": 3,
            "min_contact_retention": 0.5,
        },
        "sides": {
            "left": {
                "status": interaction_status,
                "errors": [] if interaction_status == "ok" else ["contact retention"],
                "distance_error_median_m": 0.01,
                "distance_error_p95_m": 0.02,
                "contact_retention_fraction": 1.0,
            }
        },
        "wrist_object_relative_translation": {
            "status": "ok",
            "available": True,
            "finite": True,
            "errors": [],
            "active_sides": ["left"],
            "thresholds": {
                "error_median_m": 0.03,
                "error_p95_m": 0.08,
            },
            "sides": {
                "left": {
                    "status": "ok",
                    "available": True,
                    "finite": True,
                    "errors": [],
                    "frames": 10,
                    "error_median_m": 0.01,
                    "error_p95_m": 0.02,
                    "binding": {
                        "side": "left",
                        "wrist_site_name": "left_palm",
                        "wrist_site_id": 1,
                        "wrist_body_name": "left_wrist",
                        "wrist_body_id": 2,
                        "object_body_name": "left_object",
                        "object_body_id": 3,
                        "resolution_errors": [],
                    },
                }
            },
        },
    }
    tracking_quality = {
        "status": tracking_status,
        "errors": [] if tracking_status == "ok" else ["object flyout"],
        "metrics": {
            "pos_median_m": 0.02,
            "pos_p95_m": 0.04,
            "rot_median_rad": 0.10,
            "rot_p95_rad": 0.20,
            "lost_frame_fraction": 0.0,
        },
        "thresholds": {
            "pos_median_m": 0.05,
            "pos_p95_m": 0.10,
            "rot_median_rad": 0.75,
            "rot_p95_rad": 2.0,
            "lost_pos_m": 0.10,
            "lost_fraction": 0.10,
        },
        "interaction_quality": interaction_quality,
    }
    cell_dir = matrix / "cells" / cell
    cell_dir.mkdir(parents=True)
    (cell_dir / "robot.mp4").touch()
    assets: dict[str, str] = {}
    paths: dict[str, Path] = {}
    if retargeter == "do_as_i_do":
        metadata = (
            matrix
            / "assets"
            / "cells"
            / cell
            / "retargeting"
            / "do_as_i_do"
            / "metadata"
        )
        metadata.mkdir(parents=True, exist_ok=True)
        route_path = metadata / "route_input_qc.json"
        tracking_path = metadata / "mjwp_object_tracking_quality.json"
        raw_to_processed_path = metadata / "raw_to_processed_hoi_invariance.json"
        robot = (
            matrix
            / "intermediates"
            / "retargeting"
            / "do_as_i_do"
            / cell
            / "retargeting_outputs"
            / "sharpa"
            / "left"
            / "task"
            / "0"
            / "visualization_mjwp_act_aligned.mp4"
        )
        robot.parent.mkdir(parents=True)
        robot.touch()
        route_root = next(
            parent for parent in robot.parents if parent.name == cell
        )
        adapter_path = route_root / "raw_dir" / "adapter_manifest.json"
        processed_keypoints = (
            route_root
            / "retargeting_outputs"
            / "mano"
            / "left"
            / "task"
            / "0"
            / "trajectory_keypoints.npz"
        )
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        processed_keypoints.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_text("{}", encoding="utf-8")
        processed_keypoints.write_bytes(b"processed-keypoints")
        raw_to_processed = raw_invariance_report(
            metadata,
            trajectory=trajectory,
            hand_source=hand_source,
            hand_type="left",
            adapter_manifest=adapter_path,
            processed_keypoints=processed_keypoints,
        )
        write_json(route_path, route_input_qc)
        write_json(tracking_path, tracking_quality)
        write_json(raw_to_processed_path, raw_to_processed)
        assets.update(
            {
                "source.robot": str(robot),
                "retargeting.quality.route_input_qc": str(route_path),
                "retargeting.quality.mjwp_object_tracking": str(tracking_path),
                "retargeting.quality.raw_to_processed_hoi_invariance": str(
                    raw_to_processed_path
                ),
            }
        )
        paths.update(
            {
                "route_input_qc": route_path,
                "tracking_quality": tracking_path,
                "raw_to_processed_hoi": raw_to_processed_path,
            }
        )
    else:
        root = matrix / "intermediates" / "retargeting" / "spider" / cell
        robot = root / "visualization_mjwp_act_aligned.mp4"
        robot.parent.mkdir(parents=True)
        robot.touch()
        input_path = root / "spider_input_manifest.json"
        output_path = root / "spider_output_manifest.json"
        exact_dai_root = root / expected_dai_cell
        adapter_path = exact_dai_root / "adapter_manifest.json"
        source_keypoints = exact_dai_root / "trajectory_keypoints.npz"
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        adapter_path.write_text("{}", encoding="utf-8")
        source_keypoints.write_bytes(b"route-specific-keypoints")
        raw_to_processed = raw_invariance_report(
            exact_dai_root,
            trajectory=trajectory,
            hand_source=hand_source,
            hand_type="left",
            adapter_manifest=adapter_path,
            processed_keypoints=source_keypoints,
        )
        aligned_trajectory = root / "trajectory_mjwp_act_aligned.npz"
        reference_trajectory = root / "trajectory_kinematic_act.npz"
        scene = root / "scene_act.xml"
        aligned_trajectory.touch()
        reference_trajectory.touch()
        scene.write_text("<mujoco/>", encoding="utf-8")
        tracking_quality.update(
            {
                "scene": str(scene),
                "mjwp_trajectory": str(aligned_trajectory),
                "reference_trajectory": str(reference_trajectory),
            }
        )
        tracking_path = root / "mesh_body_tracking_quality.json"
        write_json(tracking_path, tracking_quality)
        rest_support = {
            "schema_version": 1,
            "status": "ok",
            "policy": {
                "endpoints": ["start", "end"],
                "trajectory_write_allowed": False,
                "support_only_when_endpoint_not_in_hand": True,
                "support_application_mode": "audit_only",
                "diagnostic_only": True,
                "production_scene_geometry_modified": False,
            },
            "supports_added": 0,
            "trajectory_integrity": {"unchanged": True},
            "endpoint_accounting": {"complete": True},
            "route_binding": {
                "cell_key": cell,
                "trajectory_6dof": trajectory,
                "hand_source": hand_source,
                "task": "task",
                "hand_type": "left",
                "data_id": 0,
                "adapter_manifest": str(adapter_path),
                "adapter_manifest_sha256": sha256(adapter_path),
                "source_keypoints": str(source_keypoints),
                "source_keypoints_sha256": sha256(source_keypoints),
            },
            "outputs": {
                "scene_act_xml": str(scene),
                "scene_act_sha256_after": sha256(scene),
            },
        }
        write_json(
            input_path,
            {
                "status": "ok",
                "errors": [],
                "trajectory_6dof": trajectory,
                "hand_source": hand_source,
                "expected_dai_cell_key": expected_dai_cell,
                "route_input_qc": route_input_qc,
                "adapter_manifest": str(adapter_path),
                "source_keypoints": str(source_keypoints),
                "raw_to_processed_hoi_invariance": raw_to_processed,
            },
        )
        write_json(
            output_path,
            {
                "cell_key": cell,
                "trajectory_6dof": trajectory,
                "hand_source": hand_source,
                "task": "task",
                "resolved_hand_type": "left",
                "data_id": 0,
                "route_input_qc": route_input_qc,
                "raw_to_processed_hoi_invariance": raw_to_processed,
                "rest_support_provenance": rest_support,
                "alignment": {
                    "applied": True,
                    "aligned_trajectory": str(aligned_trajectory),
                },
                "object_tracking_quality": tracking_quality,
                "object_tracking_quality_file": str(tracking_path),
                "mjwp_fallback": {"applied": False},
            },
        )
        assets["source.robot"] = str(robot)
        paths.update({"spider_input": input_path, "spider_output": output_path})

    write_json(
        cell_dir / "manifest.json",
        {
            "settings": {
                "trajectory_6dof": trajectory,
                "hand_source": hand_source,
                "retargeting": retargeter,
            },
            "assets": assets,
            "provenance_validation": {
                "status": provenance_status,
                "errors": [] if provenance_status == "ok" else ["wrong object source"],
            },
        },
    )
    paths["manifest"] = cell_dir / "manifest.json"
    return paths


def run_review(module, matrix: Path, passed_cell: str) -> tuple[int, dict]:
    output = matrix / "plain_robot_visual_review.json"
    argv = [
        str(SCRIPT),
        "--matrix-run",
        str(matrix),
        "--reviewer",
        "unit-test",
        "--passed-cell",
        passed_cell,
        "--output-json",
        str(output),
    ]
    with mock.patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
        return_code = module.main()
    return return_code, json.loads(output.read_text(encoding="utf-8"))


class PlainRobotVisualReviewTest(unittest.TestCase):
    def test_collects_exactly_four_cells_per_retargeter(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            for retargeter in ("do_as_i_do", "spider"):
                for trajectory in ("do_as_i_do", "egoinfinity"):
                    for hand_source in ("aoe", "estimated"):
                        cell = (
                            f"traj_{trajectory}__hand_{hand_source}__retarget_{retargeter}"
                        )
                        make_validated_cell(matrix, cell, retargeter)
            reviewed, errors = module.collect_retarget_cells(matrix)
            self.assertEqual(len(reviewed), 8)
            self.assertEqual(errors, [])
            self.assertEqual(module.completeness_errors(reviewed), [])

    def test_rejects_unaligned_robot_source(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            cell = matrix / "cells" / "bad"
            cell.mkdir(parents=True)
            (cell / "robot.mp4").touch()
            (cell / "manifest.json").write_text(
                json.dumps(
                    {
                        "settings": {"retargeting": "do_as_i_do"},
                        "assets": {"source.robot": "robot/visualization_mjwp.mp4"},
                        "provenance_validation": {"status": "ok", "errors": []},
                    }
                ),
                encoding="utf-8",
            )
            _, errors = module.collect_retarget_cells(matrix)
            self.assertTrue(any("not a plain aligned" in error for error in errors))

    def test_rejects_non_actuator_aligned_robot_source(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            cell = matrix / "cells" / "non_act"
            cell.mkdir(parents=True)
            (cell / "robot.mp4").touch()
            (cell / "manifest.json").write_text(
                json.dumps(
                    {
                        "settings": {"retargeting": "do_as_i_do"},
                        "assets": {
                            "source.robot": "robot/visualization_mjwp_aligned.mp4"
                        },
                        "provenance_validation": {"status": "ok", "errors": []},
                    }
                ),
                encoding="utf-8",
            )
            _, errors = module.collect_retarget_cells(matrix)
            self.assertTrue(any("not a plain aligned" in error for error in errors))

    def test_incomplete_matrix_is_reported_separately_from_route_review(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            make_validated_cell(
                matrix,
                "traj_egoinfinity__hand_estimated__retarget_spider",
                "spider",
            )
            reviewed, errors = module.collect_retarget_cells(matrix)
            self.assertEqual(errors, [])
            self.assertEqual(len(reviewed), 1)
            self.assertTrue(module.completeness_errors(reviewed))

    def test_top_level_manifest_controls_structural_completeness(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            reviewed = [
                {"cell": f"dai_{index}", "retargeter": "do_as_i_do"}
                for index in range(4)
            ] + [
                {"cell": f"spider_{index}", "retargeter": "spider"}
                for index in range(4)
            ]
            (matrix / "reuse_12_demo_manifest.json").write_text(
                json.dumps(
                    {
                        "cells": 12,
                        "composed": 11,
                        "missing_cells": {"spider_3": ["robot"]},
                    }
                ),
                encoding="utf-8",
            )
            errors = module.matrix_completeness_errors(matrix, reviewed, [])
            self.assertIn("manifest composed=11, expected 12", errors)
            self.assertTrue(any("missing_cells is not empty" in error for error in errors))

    def test_success_candidate_requires_valid_provenance(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            make_validated_cell(
                matrix,
                "traj_egoinfinity__hand_estimated__retarget_do_as_i_do",
                "do_as_i_do",
                provenance_status="invalid",
            )
            _, errors = module.collect_retarget_cells(matrix)
            self.assertTrue(any("provenance validation is not ok" in error for error in errors))

    def test_v3_admits_visual_pass_only_when_exact_dai_numerical_artifacts_pass(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            cell = "traj_egoinfinity__hand_estimated__retarget_do_as_i_do"
            make_validated_cell(matrix, cell, "do_as_i_do")

            return_code, review = run_review(module, matrix, cell)

            self.assertEqual(return_code, 0)
            self.assertEqual(
                review["policy"],
                "retarget_only_plain_mujoco_numerical_and_visual_v3",
            )
            self.assertEqual(review["status"], "pass")
            self.assertEqual(review["successful_cells"], [cell])
            self.assertEqual(review["visually_passed_cells"], [cell])
            self.assertEqual(review["numerically_rejected_passed_cells"], [])
            self.assertEqual(
                review["reviewed_cells"][0]["numerical_validation"]["status"],
                "ok",
            )

    def test_dai_tracking_invalid_remains_visual_review_but_not_success(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
            make_validated_cell(
                matrix,
                cell,
                "do_as_i_do",
                tracking_status="invalid",
                interaction_status="invalid",
            )

            return_code, review = run_review(module, matrix, cell)

            self.assertEqual(return_code, 1)
            self.assertEqual(review["status"], "fail")
            self.assertEqual(review["visual_review"]["status"], "pass")
            self.assertEqual(review["visual_review"]["passed_cells"], [cell])
            self.assertEqual(review["successful_cells"], [])
            self.assertEqual(review["numerically_rejected_passed_cells"], [cell])
            self.assertEqual(review["retargeters"]["do_as_i_do"], "fail")
            self.assertTrue(
                any("object_tracking_quality status=invalid" in error for error in review["validation_errors"])
            )

    def test_status_ok_cannot_hide_hard_metric_or_threshold_failure(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
            paths = make_validated_cell(matrix, cell, "do_as_i_do")
            tracking = json.loads(paths["tracking_quality"].read_text(encoding="utf-8"))
            tracking["metrics"]["pos_median_m"] = 0.20
            tracking["thresholds"]["pos_median_m"] = 0.50
            write_json(paths["tracking_quality"], tracking)

            return_code, review = run_review(module, matrix, cell)

            self.assertEqual(return_code, 1)
            self.assertEqual(review["successful_cells"], [])
            errors = review["validation_errors"]
            self.assertTrue(any("metrics.pos_median_m" in error for error in errors))
            self.assertTrue(any("is looser than hard limit" in error for error in errors))

    def test_status_only_tracking_is_diagnostic_not_visual_success(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
            paths = make_validated_cell(matrix, cell, "do_as_i_do")
            write_json(paths["tracking_quality"], {"status": "ok", "errors": []})

            return_code, review = run_review(module, matrix, cell)

            self.assertEqual(return_code, 1)
            self.assertEqual(review["successful_cells"], [])
            self.assertTrue(
                any("status-only/legacy" in error for error in review["validation_errors"])
            )

    def test_missing_dai_route_input_qc_cannot_be_visual_success(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            cell = "traj_egoinfinity__hand_estimated__retarget_do_as_i_do"
            paths = make_validated_cell(matrix, cell, "do_as_i_do")
            paths["route_input_qc"].unlink()

            return_code, review = run_review(module, matrix, cell)

            self.assertEqual(return_code, 1)
            self.assertEqual(review["successful_cells"], [])
            self.assertEqual(review["numerically_rejected_passed_cells"], [cell])
            self.assertTrue(
                any("missing route_input_qc" in error for error in review["validation_errors"])
            )

    def test_missing_raw_to_processed_invariance_cannot_be_visual_success(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            cell = "traj_egoinfinity__hand_estimated__retarget_do_as_i_do"
            paths = make_validated_cell(matrix, cell, "do_as_i_do")
            paths["raw_to_processed_hoi"].unlink()

            return_code, review = run_review(module, matrix, cell)

            self.assertEqual(return_code, 1)
            self.assertEqual(review["successful_cells"], [])
            self.assertTrue(
                any(
                    "raw_to_processed_hoi_invariance" in error
                    for error in review["validation_errors"]
                )
            )

    def test_cross_task_dai_invariance_cannot_be_visual_success(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            cell = "traj_egoinfinity__hand_estimated__retarget_do_as_i_do"
            paths = make_validated_cell(matrix, cell, "do_as_i_do")
            report = json.loads(
                paths["raw_to_processed_hoi"].read_text(encoding="utf-8")
            )
            correct = Path(report["processed_keypoints"])
            wrong = (
                correct.parents[2]
                / "another_task"
                / correct.parent.name
                / correct.name
            )
            wrong.parent.mkdir(parents=True, exist_ok=True)
            wrong.write_bytes(correct.read_bytes())
            report["processed_keypoints"] = str(wrong)
            report["processed_keypoints_sha256"] = sha256(wrong)
            report["route_binding"]["processed_keypoints"] = {
                "path": str(wrong),
                "sha256": sha256(wrong),
            }
            write_json(paths["raw_to_processed_hoi"], report)

            return_code, review = run_review(module, matrix, cell)

            self.assertEqual(return_code, 1)
            self.assertEqual(review["successful_cells"], [])
            self.assertTrue(
                any(
                    "processed_keypoints is stale or cross-route" in error
                    for error in review["validation_errors"]
                )
            )

    def test_spider_applied_rest_support_cannot_be_visual_success(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            cell = "traj_egoinfinity__hand_estimated__retarget_spider"
            paths = make_validated_cell(matrix, cell, "spider")
            output = json.loads(paths["spider_output"].read_text(encoding="utf-8"))
            output["rest_support_provenance"]["supports_added"] = 1
            output["rest_support_provenance"]["policy"][
                "production_scene_geometry_modified"
            ] = True
            write_json(paths["spider_output"], output)

            return_code, review = run_review(module, matrix, cell)

            self.assertEqual(return_code, 1)
            self.assertEqual(review["successful_cells"], [])
            self.assertTrue(
                any(
                    "added artificial support geometry" in error
                    for error in review["validation_errors"]
                )
            )

    def test_spider_raw_qc_cannot_masquerade_as_authoritative_aligned_qc(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            cell = "traj_egoinfinity__hand_estimated__retarget_spider"
            paths = make_validated_cell(matrix, cell, "spider")
            output = json.loads(paths["spider_output"].read_text(encoding="utf-8"))
            tracking_path = Path(output["object_tracking_quality_file"])
            raw_trajectory = tracking_path.parent / "trajectory_mjwp_act.npz"
            raw_trajectory.touch()
            tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
            tracking["mjwp_trajectory"] = str(raw_trajectory)
            write_json(tracking_path, tracking)

            return_code, review = run_review(module, matrix, cell)

            self.assertEqual(return_code, 1)
            self.assertEqual(review["successful_cells"], [])
            self.assertTrue(
                any(
                    "did not evaluate alignment.aligned_trajectory" in error
                    for error in review["validation_errors"]
                )
            )

    def test_spider_interaction_invalid_cannot_be_overridden_by_visual_pass(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            cell = "traj_egoinfinity__hand_estimated__retarget_spider"
            make_validated_cell(
                matrix,
                cell,
                "spider",
                tracking_status="ok",
                interaction_status="invalid",
            )

            return_code, review = run_review(module, matrix, cell)

            self.assertEqual(return_code, 1)
            self.assertEqual(review["status"], "fail")
            self.assertEqual(review["visual_review"]["passed_cells"], [cell])
            self.assertEqual(review["successful_cells"], [])
            self.assertEqual(review["numerically_rejected_passed_cells"], [cell])
            self.assertTrue(
                any("interaction_quality status=invalid" in error for error in review["validation_errors"])
            )

    def test_non_active_not_applicable_interaction_side_does_not_block(self):
        module = load_module()
        interaction, errors = module.interaction_quality_errors(
            {
                "interaction_quality": {
                    "status": "ok",
                    "active_sides": ["left"],
                    "sides": {
                        "left": {"status": "ok", "errors": []},
                        "right": {"status": "not_applicable", "errors": []},
                    },
                }
            }
        )

        self.assertEqual(interaction["status"], "ok")
        self.assertEqual(errors, [])

    def test_dai_artifact_from_other_cell_is_rejected(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory)
            cell = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
            other = "traj_egoinfinity__hand_estimated__retarget_do_as_i_do"
            paths = make_validated_cell(matrix, cell, "do_as_i_do")
            other_paths = make_validated_cell(matrix, other, "do_as_i_do")
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["assets"]["retargeting.quality.route_input_qc"] = str(
                other_paths["route_input_qc"]
            )
            write_json(paths["manifest"], manifest)
            paths["route_input_qc"].unlink()

            return_code, review = run_review(module, matrix, cell)

            self.assertEqual(return_code, 1)
            self.assertEqual(review["successful_cells"], [])
            self.assertTrue(
                any("does not belong to exact cell" in error for error in review["validation_errors"])
            )


if __name__ == "__main__":
    unittest.main()
