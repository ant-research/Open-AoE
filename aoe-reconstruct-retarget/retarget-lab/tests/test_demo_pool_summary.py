from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review" / "summarize_retarget_demo_pool.py"


def load_summary_module():
    spec = importlib.util.spec_from_file_location("retarget_demo_pool_summary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_tracking_report() -> dict:
    return {
        "status": "ok",
        "errors": [],
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
        "interaction_quality": {
            "status": "ok",
            "errors": [],
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
                    "status": "ok",
                    "errors": [],
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
        },
    }


class DemoPoolGravityTest(unittest.TestCase):
    def make_run(self, root: Path, gravity: dict | None, *, dai_valid: bool = True):
        source_name = "sample_full_source_fresh01"
        matrix = root / "sample_matrix12_fresh01"
        (matrix / "videos").mkdir(parents=True)
        for index in range(12):
            (matrix / "videos" / f"cell_{index:02d}__triptych.mp4").touch()
        write_json(
            matrix / "reuse_12_demo_manifest.json",
            {
                "source_run": source_name,
                "cells": 12,
                "composed": 12,
                "missing_cells": {},
                "trajectory_asset_validation": {"do_as_i_do_quality_valid": dai_valid},
            },
        )
        alignment = {
            "applied": True,
            "alignment_kind": "drop_warmup_and_sample_exact_source_timestamps",
            "start_index": 89,
            "warmup_steps": 90,
            "target_count": 32,
            "source_frame_count": 32,
            "exact_source_timestamps": True,
            "sampling_semantics": "exact_source_timestamp_indices",
            "aligned_video": "visualization_mjwp_aligned.mp4",
            "aligned_videos": [
                {
                    "path": "visualization_mjwp_aligned.mp4",
                    "probe": {"exists": True},
                }
            ],
            "object_tracking_quality": valid_tracking_report(),
        }
        spider_item = {
            "status": "ok",
            "alignment": {
                "applied": True,
                "alignment_kind": "sample_exact_source_timestamps",
                "source_kind": "original_spider_mjwp",
                "exact_source_timestamps": True,
                "sampling_semantics": "exact_source_timestamp_indices",
                "start_index": 0,
                "warmup_steps": 0,
            },
            "aligned_video": {"exists": True},
            "mjwp_fallback": {"applied": False},
            "object_tracking_quality": valid_tracking_report(),
        }
        write_json(
            matrix / "audit_retarget_run.json",
            {
                "status": "ok",
                "source": {"errors": [], "dai_alignments": [alignment]},
                "spider": {"items": [dict(spider_item) for _ in range(4)]},
                "errors": [],
            },
        )
        write_json(
            matrix / "plain_robot_visual_review.json",
            {
                "policy": "retarget_only_plain_mujoco_numerical_and_visual_v3",
                "evidence_kind": "plain_aligned_mujoco",
                "status": "pass",
                "successful_cell_count": 1,
                "successful_cells": [
                    "traj_egoinfinity__hand_aoe__retarget_spider"
                ],
                "reviewed_cells": [
                    {
                        "cell": "traj_egoinfinity__hand_aoe__retarget_spider",
                        "provenance_validation": {"status": "ok", "errors": []},
                        "numerical_validation": {
                            "status": "ok",
                            "errors": [],
                            "evidence": {
                                "object_tracking_quality": {
                                    "artifact": valid_tracking_report()
                                }
                            },
                        },
                    }
                ],
                "validation_errors": [],
                "failed_cells": [],
            },
        )
        source = root / source_name
        write_json(source / "fresh_run_metadata.json", {"git": {"commit": "abc", "dirty": False}})
        if gravity is not None:
            write_json(
                source
                / "intermediates"
                / "trajectory_6dof"
                / "do_as_i_do"
                / "reconstruction"
                / "clip"
                / "gravity.json",
                gravity,
            )
        return matrix

    def test_plain_robot_review_and_measured_gravity_are_numerically_ready(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(
                root,
                {
                    "estimation_status": "estimated",
                    "estimation_source": "geocalib",
                    "is_fallback": False,
                    "geocalib_commit": "97b8968",
                    "weights_sha256": "sha",
                    "gravity_quality": {
                        "status": "dynamic_camera_reference_frame",
                        "selection_policy": "nearest_object_reconstruction_reference_frame",
                        "reference_frame": 82,
                        "selected_frame": 84,
                        "angular_error_p95_deg": 61.6,
                    },
                },
            )
            result = module.classify_run(matrix, root)
            self.assertEqual(result["classification"], "showcase")
            self.assertEqual(result["gravity_status"], "estimated")
            self.assertEqual(result["gravity_selection_status"], "dynamic_camera_reference_frame")
            self.assertEqual(
                result["gravity_selection_policy"],
                "nearest_object_reconstruction_reference_frame",
            )
            self.assertEqual(result["gravity_reference_frame"], 82)
            self.assertEqual(result["gravity_selected_frame"], 84)
            self.assertEqual(result["gravity_angular_error_p95_deg"], 61.6)
            self.assertTrue(result["recommended_route_ready"])
            self.assertTrue(result["numerical_quality_ready"])
            self.assertTrue(result["dai_numerical_ok"])

    def test_invalid_dai_baseline_does_not_hide_ready_recommended_route(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(
                root,
                {
                    "estimation_status": "estimated",
                    "estimation_source": "geocalib",
                    "is_fallback": False,
                },
                dai_valid=False,
            )
            result = module.classify_run(matrix, root)
            self.assertEqual(result["classification"], "showcase")
            self.assertTrue(result["dai_baseline_invalid"])
            self.assertTrue(result["recommended_route_ready"])
            self.assertTrue(result["plain_robot_visual_success"])

    def test_fallback_gravity_blocks_demo_readiness(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(
                root,
                {
                    "estimation_status": "fallback_assumption",
                    "estimation_source": "upright_camera_assumption",
                    "is_fallback": True,
                },
            )
            result = module.classify_run(matrix, root)
            self.assertEqual(result["classification"], "usable_with_caveat")
            self.assertIn("gravity_fallback", result["failure_tags"])
            self.assertFalse(result["numerical_quality_ready"])
            self.assertFalse(result["demo_ready"])

    def test_missing_gravity_blocks_demo_readiness(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(root, None)
            result = module.classify_run(matrix, root)
            self.assertEqual(result["classification"], "usable_with_caveat")
            self.assertIn("gravity_missing_or_unknown", result["failure_tags"])
            self.assertFalse(result["numerical_quality_ready"])
            self.assertFalse(result["demo_ready"])

    def test_missing_plain_robot_review_cannot_be_showcase(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(root, None)
            (matrix / "plain_robot_visual_review.json").unlink()
            result = module.classify_run(matrix, root)
            self.assertEqual(result["classification"], "usable_with_caveat")
            self.assertFalse(result["plain_robot_visual_success"])
            self.assertFalse(result["demo_ready"])

    def test_legacy_v1_visual_review_is_diagnostic_only(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(
                root,
                {
                    "estimation_status": "estimated",
                    "estimation_source": "geocalib",
                    "is_fallback": False,
                },
            )
            write_json(
                matrix / "plain_robot_visual_review.json",
                {
                    "policy": "retarget_only_plain_mujoco_v1",
                    "evidence_kind": "plain_aligned_mujoco",
                    "status": "pass",
                    "retargeters": {"do_as_i_do": "pass", "spider": "pass"},
                    "reviewed_cell_count": 8,
                    "failed_cells": [],
                },
            )

            result = module.classify_run(matrix, root)

            self.assertFalse(result["plain_robot_visual_success"])
            self.assertFalse(result["demo_ready"])
            self.assertIn(
                "diagnostic-only",
                result["plain_robot_visual_review"]["admission_note"],
            )

    def test_summary_recomputes_status_ok_metrics(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(
                root,
                {
                    "estimation_status": "estimated",
                    "estimation_source": "geocalib",
                    "is_fallback": False,
                },
            )
            audit_path = matrix / "audit_retarget_run.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            tracking = audit["source"]["dai_alignments"][0]["object_tracking_quality"]
            tracking["metrics"]["pos_p95_m"] = 0.20
            tracking["status"] = "ok"
            write_json(audit_path, audit)

            review_path = matrix / "plain_robot_visual_review.json"
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review_tracking = review["reviewed_cells"][0]["numerical_validation"]["evidence"][
                "object_tracking_quality"
            ]["artifact"]
            review_tracking["interaction_quality"]["sides"]["left"][
                "contact_retention_fraction"
            ] = 0.1
            review_tracking["status"] = "ok"
            write_json(review_path, review)

            result = module.classify_run(matrix, root)

            self.assertFalse(result["dai_numerical_ok"])
            self.assertFalse(result["plain_robot_visual_success"])
            self.assertFalse(result["demo_ready"])

    def test_invalid_tracking_cannot_be_overridden_by_visual_pass(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(root, None)
            audit_path = matrix / "audit_retarget_run.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["status"] = "failed"
            audit["errors"] = ["SPIDER cell: object tracking quality is not valid"]
            audit["source"]["errors"] = ["DAI object tracking threshold exceeded"]
            audit["source"]["dai_alignments"][0]["object_tracking_quality"] = {
                "status": "invalid"
            }
            for item in audit["spider"]["items"]:
                item["status"] = "reviewable_invalid_tracking"
            write_json(audit_path, audit)
            result = module.classify_run(matrix, root)
            self.assertEqual(result["classification"], "usable_with_caveat")
            self.assertTrue(result["spider_aligned"])
            self.assertFalse(result["numerical_quality_ready"])
            self.assertFalse(result["dai_numerical_ok"])
            self.assertTrue(result["plain_robot_visual_success"])
            self.assertTrue(result["successful_route_provenance_ok"])
            self.assertFalse(result["demo_ready"])
            self.assertFalse(result["recommended_route_ready"])

    def test_one_successful_plain_route_is_not_demo_ready_without_numerical_gate(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(root, None)
            for video in list((matrix / "videos").glob("*.mp4"))[1:]:
                video.unlink()
            manifest_path = matrix / "reuse_12_demo_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({"composed": 1, "missing_cells": {"other": ["robot"]}})
            write_json(manifest_path, manifest)
            write_json(
                matrix / "plain_robot_visual_review.json",
                {
                    "policy": "retarget_only_plain_mujoco_numerical_and_visual_v3",
                    "evidence_kind": "plain_aligned_mujoco",
                    "status": "pass",
                    "successful_cell_count": 1,
                    "successful_cells": [
                        "traj_egoinfinity__hand_estimated__retarget_spider"
                    ],
                    "reviewed_cells": [
                        {
                            "cell": "traj_egoinfinity__hand_estimated__retarget_spider",
                            "provenance_validation": {"status": "ok", "errors": []},
                            "numerical_validation": {"status": "ok", "errors": []},
                        }
                    ],
                    "validation_errors": [],
                    "failed_cells": ["traj_do_as_i_do__hand_aoe__retarget_spider"],
                },
            )
            result = module.classify_run(matrix, root)
            self.assertFalse(result["cells"] == 12 and result["composed"] == 12)
            self.assertFalse(result["plain_robot_visual_success"])
            self.assertFalse(result["numerical_quality_ready"])
            self.assertFalse(result["demo_ready"])
            self.assertFalse(result["recommended_route_ready"])
            self.assertEqual(result["classification"], "reject_or_stress_only")

    def test_v3_visual_pass_rejects_successful_cell_with_invalid_route_numerics(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(
                root,
                {
                    "estimation_status": "estimated",
                    "estimation_source": "geocalib",
                    "is_fallback": False,
                },
            )
            write_json(
                matrix / "plain_robot_visual_review.json",
                {
                    "policy": "retarget_only_plain_mujoco_numerical_and_visual_v3",
                    "evidence_kind": "plain_aligned_mujoco",
                    "status": "pass",
                    "successful_cell_count": 1,
                    "successful_cells": ["recommended_bad_numerics"],
                    "reviewed_cells": [
                        {
                            "cell": "recommended_bad_numerics",
                            "provenance_validation": {"status": "ok", "errors": []},
                            "numerical_validation": {
                                "status": "invalid",
                                "errors": ["contact retention below threshold"],
                            },
                        }
                    ],
                    "validation_errors": [],
                },
            )

            result = module.classify_run(matrix, root)

            self.assertFalse(result["plain_robot_visual_success"])
            self.assertFalse(result["demo_ready"])

    def test_failed_cell_provenance_does_not_block_successful_route(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(root, None)
            write_json(
                matrix / "plain_robot_visual_review.json",
                {
                    "policy": "retarget_only_plain_mujoco_numerical_and_visual_v3",
                    "evidence_kind": "plain_aligned_mujoco",
                    "status": "pass",
                    "successful_cell_count": 1,
                    "successful_cells": ["recommended_ok"],
                    "reviewed_cells": [
                        {
                            "cell": "recommended_ok",
                            "provenance_validation": {"status": "ok", "errors": []},
                            "numerical_validation": {"status": "ok", "errors": []},
                        },
                        {
                            "cell": "baseline_missing",
                            "provenance_validation": {
                                "status": "invalid",
                                "errors": ["missing baseline asset"],
                            },
                            "numerical_validation": {"status": "ok", "errors": []},
                        },
                    ],
                    "validation_errors": [],
                },
            )
            audit_path = matrix / "audit_retarget_run.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["status"] = "failed"
            audit["errors"] = ["baseline provenance validation invalid"]
            write_json(audit_path, audit)
            result = module.classify_run(matrix, root)
            self.assertFalse(result["matrix_provenance_ok"])
            self.assertTrue(result["successful_route_provenance_ok"])
            self.assertFalse(result["numerical_quality_ready"])
            self.assertFalse(result["demo_ready"])
            self.assertEqual(result["classification"], "usable_with_caveat")

    def test_successful_cell_requires_valid_provenance(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(
                root,
                {
                    "estimation_status": "estimated",
                    "estimation_source": "geocalib",
                    "is_fallback": False,
                },
            )
            write_json(
                matrix / "plain_robot_visual_review.json",
                {
                    "policy": "retarget_only_plain_mujoco_numerical_and_visual_v3",
                    "evidence_kind": "plain_aligned_mujoco",
                    "status": "pass",
                    "successful_cell_count": 1,
                    "successful_cells": ["recommended_bad"],
                    "reviewed_cells": [
                        {
                            "cell": "recommended_bad",
                            "provenance_validation": {
                                "status": "invalid",
                                "errors": ["wrong object source"],
                            },
                            "numerical_validation": {"status": "ok", "errors": []},
                        }
                    ],
                    "validation_errors": [],
                },
            )
            result = module.classify_run(matrix, root)
            self.assertFalse(result["successful_route_provenance_ok"])
            self.assertFalse(result["demo_ready"])

    def test_retarget_audit_filename_is_supported(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(
                root,
                {
                    "estimation_status": "estimated",
                    "estimation_source": "geocalib",
                    "is_fallback": False,
                },
            )
            (matrix / "audit_retarget_run.json").rename(matrix / "retarget_audit.json")
            result = module.classify_run(matrix, root)
            self.assertEqual(result["classification"], "showcase")
            self.assertEqual(result["audit_status"], "ok")

    def test_missing_manifest_is_not_classified_as_rejected(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "active_matrix12_fresh01"
            matrix.mkdir()
            result = module.classify_run(matrix, root)
            self.assertEqual(result["classification"], "incomplete_or_in_progress")
            self.assertEqual(result["failure_bucket"], "missing_manifest_or_in_progress")

    def test_ego_preflight_failure_without_manifest_remains_route_isolated(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "sample_matrix12_fresh01"
            matrix.mkdir()
            source = root / "sample_full_source_fresh01"
            failures = source / "logs" / "stage_failures.tsv"
            failures.parent.mkdir(parents=True)
            failures.write_text(
                "2026-07-12T04:00:00+08:00\tegoinfinity_retarget_input_preflight\t4\n",
                encoding="utf-8",
            )
            result = module.classify_run(matrix, root)
            self.assertEqual(result["classification"], "incomplete_or_in_progress")
            self.assertEqual(result["run_state"], "incomplete_or_in_progress")
            self.assertEqual(result["failure_bucket"], "retarget_input_preflight_failed")

    def test_aborted_source_outcome_is_not_classified_as_in_progress(self):
        module = load_summary_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = self.make_run(root, None)
            manifest = json.loads(
                (matrix / "reuse_12_demo_manifest.json").read_text(encoding="utf-8")
            )
            source = root / manifest["source_run"]
            (matrix / "reuse_12_demo_manifest.json").unlink()
            write_json(
                source / "fresh_run_outcome.json",
                {
                    "status": "aborted",
                    "return_code": 143,
                    "termination_signal": "SIGTERM",
                    "stage": "do_as_i_do_fresh_reconstruction",
                },
            )
            result = module.classify_run(matrix, root)
            self.assertEqual(result["classification"], "reject_or_stress_only")
            self.assertEqual(result["run_state"], "failed")
            self.assertEqual(result["source_return_code"], 143)
            self.assertIn("source_run_failed", result["failure_tags"])
            self.assertEqual(result["failure_bucket"], "source_run_failed")

    def test_scene_summary_deduplicates_reruns_without_merging_delivery_gates(self):
        module = load_summary_module()
        rows = [
            {
                "matrix_run": "sample_matrix12_retry01",
                "demo_ready": True,
                "recommended_route_ready": True,
                "matrix_complete": False,
                "plain_robot_visual_review": {
                    "policy": "retarget_only_plain_mujoco_any_route_v2",
                    "successful_cells": [
                        "traj_egoinfinity__hand_aoe__retarget_spider"
                    ],
                },
            },
            {
                "matrix_run": "sample_matrix12_retry02",
                "demo_ready": True,
                "recommended_route_ready": False,
                "matrix_complete": True,
                "plain_robot_visual_review": {
                    "policy": "retarget_only_plain_mujoco_any_route_v2",
                    "successful_cells": [
                        "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do"
                    ],
                },
            },
            {
                "matrix_run": "other_matrix12_full01",
                "demo_ready": False,
                "recommended_route_ready": False,
                "matrix_complete": True,
                "plain_robot_visual_review": {},
            },
        ]
        counts, inventory = module.summarize_scenes(rows)
        self.assertEqual(counts["tested_matrix_scenes"], 2)
        self.assertEqual(counts["demo_ready_any_route"], 1)
        self.assertEqual(counts["dai_success_any_route"], 1)
        self.assertEqual(counts["spider_success_any_route"], 1)
        self.assertEqual(counts["matrix_complete_any_quality"], 2)
        self.assertEqual(counts["deliverable_12_of_12"], 1)
        sample = next(item for item in inventory if item["scene_id"] == "sample")
        self.assertEqual(sample["run_count"], 2)
        self.assertTrue(sample["deliverable_12_of_12"])
        self.assertEqual(sample["best_run"], "sample_matrix12_retry02")


if __name__ == "__main__":
    unittest.main()
