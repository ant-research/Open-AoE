import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))

from filter_egoinfinity_objects import choose_objects  # noqa: E402
from prepare_fresh_aoe_clip import (  # noqa: E402
    DEFAULT_BBOX_FORMAT,
    automatic_ref_frame_fraction,
    clip_window,
    select_reference_frame,
    transform_bbox,
)
from screen_aoe_retarget_candidates import (  # noqa: E402
    BboxSpace,
    GOOD_VERBS,
    RIGID_TOKENS,
    candidate_history_summary,
    canonical_scene_id,
    classify_suitability,
    existing_scene_ids,
    existing_run_records,
    is_likely_deformable_object,
    is_supported_rigid_object,
    scale_bbox,
)
from validate_sam3_prompt_gate import validate_prompt_qc  # noqa: E402


def packed_mask(center_x: int, center_y: int, size: int = 12) -> dict:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[max(center_y - 1, 0) : center_y + 2, max(center_x - 1, 0) : center_x + 2] = 1
    return {"mask_packed": np.packbits(mask.reshape(-1)), "mask_shape": mask.shape}


class BboxCoordinateTest(unittest.TestCase):
    def test_reference_frame_uses_actual_encoded_clip_length(self) -> None:
        frame, fraction, strategy = select_reference_frame(
            actual_num_frames=92,
            start_frame=1000,
            ref_source_frame=-1,
            ref_frame_fraction=-1.0,
            verb="hold",
        )
        self.assertEqual(frame, 77)
        self.assertEqual(fraction, 0.85)
        self.assertEqual(strategy, "verb_hold_late_visibility")

    def test_explicit_reference_frame_fails_closed_past_actual_clip_end(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside actual encoded clip"):
            select_reference_frame(
                actual_num_frames=92,
                start_frame=1000,
                ref_source_frame=1152,
                ref_frame_fraction=-1.0,
                verb="hold",
            )

    def test_fresh_clip_start_offset_is_explicit_and_bounded(self) -> None:
        self.assertEqual(clip_window(41.0, 60.0, 2.0, 0.0, 0.0), (43.0, 17.0))
        self.assertEqual(clip_window(41.0, 60.0, 2.0, 4.0, 0.25), (43.0, 4.25))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            clip_window(41.0, 60.0, -0.1, 4.0, 0.0)
        with self.assertRaisesRegex(ValueError, "invalid clip window"):
            clip_window(41.0, 42.0, 2.0, 0.0, 0.0)

    def test_dataset_action_and_rigid_aliases_cover_reviewed_candidates(self) -> None:
        self.assertIn("put_down", GOOD_VERBS)
        self.assertIn("rotate", GOOD_VERBS)
        self.assertNotIn("toss", GOOD_VERBS)
        self.assertIn("vape", RIGID_TOKENS)

    def test_rigid_object_names_override_ambiguous_material_tokens(self) -> None:
        self.assertTrue(is_supported_rigid_object("water gun"))
        self.assertTrue(is_supported_rigid_object("water pump"))
        self.assertTrue(is_supported_rigid_object("water_bottle"))
        self.assertFalse(is_likely_deformable_object("water gun"))
        self.assertFalse(is_likely_deformable_object("water pump"))
        self.assertFalse(is_likely_deformable_object("water bottle"))
        self.assertTrue(is_likely_deformable_object("water"))

    def test_detachable_devices_require_stress_review(self) -> None:
        tier, flags = classify_suitability(
            "accept", "right", {"water", "pump"}, [], "grasp"
        )
        self.assertEqual(tier, "stress")
        self.assertIn("detachable_or_articulated_device", flags)

    def test_scene_aliases_deduplicate_name_order(self) -> None:
        self.assertEqual(canonical_scene_id("bottle of cooking wine"), "cooking_wine_bottle")
        self.assertEqual(canonical_scene_id("cooking wine bottle"), "cooking_wine_bottle")
        self.assertEqual(canonical_scene_id("water_bottle"), "water_bottle")
        self.assertNotEqual(canonical_scene_id("bottle of cooking wine"), "kitchen")

    def test_existing_scene_ids_use_source_run_prefix(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gold_jar_full_source_20260713_fresh01").mkdir()
            (root / "gold_jar_matrix12_20260713_fresh01").mkdir()
            (root / "knife_full_source_20260713_fresh01").mkdir()
            self.assertEqual(existing_scene_ids(root), {"gold_jar", "knife"})

    def test_candidate_history_promotes_passed_preflight_only(self) -> None:
        summary = candidate_history_summary(
            [{"stage": "preflight_passed", "source_run": "phone_preflight"}]
        )
        self.assertTrue(summary["already_run"])
        self.assertTrue(summary["promotion_eligible"])
        self.assertEqual(summary["recommended_next_action"], "launch_fresh_full_run")

        summary = candidate_history_summary(
            [
                {"stage": "preflight_passed"},
                {"stage": "full_run_failed"},
            ]
        )
        self.assertFalse(summary["promotion_eligible"])
        self.assertEqual(summary["recommended_next_action"], "inspect_full_run_failure")

    def test_existing_run_records_distinguish_preflight_outcomes(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, qa_pass, stale_gate in (
                ("passed", True, False),
                ("failed", False, False),
                ("stale", True, True),
            ):
                run = root / f"box_full_source_20260713_{name}"
                qc = run / (
                    "intermediates/trajectory_6dof/do_as_i_do/reconstruction/clip/"
                    "video_segmentation/mask_qc_summary.json"
                )
                qc.parent.mkdir(parents=True)
                frame_stats = [
                    {
                        "frame_idx": index,
                        "valid": qa_pass,
                        "mask_pixels": (
                            25000 if stale_gate and index == 5 else (2500 if qa_pass else 0)
                        ),
                    }
                    for index in range(10)
                ]
                qc.write_text(
                    json.dumps(
                        {
                            "qa_pass": qa_pass,
                            "valid_ratio": 1.0 if qa_pass else 0.0,
                            "text": "white box",
                            "target_bbox": [0.0, 0.0, 100.0, 100.0],
                            "selected_sam3_obj_id": 0 if qa_pass else None,
                            "prompt_frame_idx": 9,
                            "prompt_candidates": [
                                {
                                    "sam3_obj_id": 0,
                                    "mask_pixels": 2500,
                                    "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
                                }
                            ] if qa_pass else [],
                            "frame_stats": frame_stats,
                            "max_centroid_jump_px": 0.0,
                        }
                    )
                )
                (run / "fresh_run_metadata.json").write_text(
                    json.dumps(
                        {
                            "run_command": "runner --sam3-preflight-only",
                            "dataset": {"segment": "segment_a", "annotation_id": "7"},
                            "task": {"object_name": "  White   Box "},
                            "runs": {"matrix_run": f"box_matrix12_{name}"},
                        }
                    )
                )
            records = existing_run_records(root)
            history = records[("segment_a", "7", "white box")]
            self.assertEqual(
                {item["stage"] for item in history},
                {"preflight_passed", "preflight_failed"},
            )
            stale = next(item for item in history if item["source_run"].endswith("_stale"))
            self.assertTrue(stale["mask_qa_pass"])
            self.assertEqual(stale["current_prompt_gate_status"], "failed")
            self.assertTrue(
                any("area changes too abruptly" in error for error in stale["current_prompt_gate_errors"])
            )

    def test_scene_suitability_tiers_separate_primary_and_stress_inputs(self) -> None:
        self.assertEqual(
            classify_suitability("accept", "left", {"phone"}, []),
            ("primary", []),
        )
        tier, flags = classify_suitability("accept", "bimanual", {"jar"}, [])
        self.assertEqual(tier, "conditional")
        self.assertIn("bimanual_interaction", flags)
        tier, flags = classify_suitability("accept", "right", {"frying", "pan"}, [])
        self.assertEqual(tier, "stress")
        self.assertIn("thin_open_or_articulated_geometry", flags)
        tier, flags = classify_suitability("accept", "left", {"pen"}, [])
        self.assertEqual(tier, "stress")
        self.assertIn("thin_open_or_articulated_geometry", flags)
        tier, flags = classify_suitability("reject", "right", {"bowl"}, [])
        self.assertEqual(tier, "reject")
        self.assertIn("thin_open_or_articulated_geometry", flags)
        tier, flags = classify_suitability("accept", "bimanual", {"box"}, [], "open")
        self.assertEqual(tier, "stress")
        self.assertIn("articulated_action", flags)

    def test_normalized_1000_to_1280x720(self) -> None:
        self.assertEqual(DEFAULT_BBOX_FORMAT, "xyxy")
        expected = [640.0, 288.0, 896.0, 432.0]
        self.assertEqual(
            transform_bbox(
                [500, 400, 700, 600],
                (1000, 1000),
                (1280, 720),
                bbox_format="xyxy",
            ),
            expected,
        )
        self.assertEqual(
            transform_bbox(
                [500, 400, 200, 200],
                (1000, 1000),
                (1280, 720),
                bbox_format="xywh",
            ),
            expected,
        )
        self.assertEqual(
            scale_bbox(
                [500.0, 400.0, 700.0, 600.0],
                BboxSpace(width=1000, height=1000, source="normalized_0_1000"),
                1280,
                720,
            ),
            expected,
        )

    def test_target_point_selects_matching_duplicate_prompt(self) -> None:
        frames = []
        for index in range(5):
            frames.append(
                {
                    "sam3_obj_data": {
                        0: packed_mask(2 + index // 3, 2),
                        1: packed_mask(9, 9),
                    }
                }
            )
        data = {
            "sam3_mesh_info": {0: {"n_points": 100}, 1: {"n_points": 200}},
            "sam3_prompt_mapping": [
                {"prompt": "bottle", "score": 0.9},
                {"prompt": "bottle", "score": 0.99},
            ],
            "frame_data": frames,
        }
        keep, rows = choose_objects(
            data,
            target_prompt="bottle",
            target_point=(2.0, 2.0),
            max_jump_px=3.0,
            selection_mode="best",
        )
        self.assertEqual(keep, {0})
        selected = min(rows, key=lambda row: row["selection_rank"])
        self.assertEqual(selected["oid"], 0)

    def test_articulated_actions_choose_less_occluded_reference_side(self) -> None:
        self.assertEqual(automatic_ref_frame_fraction("uncap"), (0.10, "verb_uncap_early_visibility"))
        self.assertEqual(
            automatic_ref_frame_fraction("put_down"),
            (0.25, "verb_put_down_pre_placement_visibility"),
        )
        self.assertEqual(
            automatic_ref_frame_fraction("set_down"),
            (0.25, "verb_set_down_pre_placement_visibility"),
        )
        self.assertEqual(
            automatic_ref_frame_fraction("place"),
            (0.25, "verb_place_pre_placement_visibility"),
        )
        self.assertEqual(automatic_ref_frame_fraction("close"), (0.85, "verb_close_late_visibility"))
        self.assertEqual(automatic_ref_frame_fraction("grasp"), (0.85, "verb_grasp_late_visibility"))
        self.assertEqual(automatic_ref_frame_fraction("lift"), (0.85, "verb_lift_late_visibility"))
        self.assertEqual(automatic_ref_frame_fraction("shake"), (0.50, "verb_default_midpoint"))

    def test_production_prompt_gate_accepts_text_bbox_selection(self) -> None:
        qc = {
            "qa_pass": True,
            "text": "gold jar",
            "target_bbox": [512.0, 180.0, 768.0, 324.0],
            "target_point": None,
            "prompt_frame_idx": 10,
            "frame_stats": [{"frame_idx": 10, "valid": True, "mask_pixels": 16852}],
            "selected_sam3_obj_id": 2,
            "prompt_candidates": [
                {
                    "sam3_obj_id": 2,
                    "mask_pixels": 16852,
                    "bbox_xyxy": [512.0, 180.0, 768.0, 324.0],
                }
            ],
        }
        payload = validate_prompt_qc(qc, "gold jar", [512.0, 180.0, 768.0, 324.0])
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["errors"], [])
        self.assertEqual(payload["warnings"], [])

    def test_production_prompt_gate_flags_large_mask_for_manual_review(self) -> None:
        qc = {
            "qa_pass": True,
            "text": "phone",
            "target_bbox": [618.0, 411.0, 730.0, 493.0],
            "target_point": None,
            "prompt_frame_idx": 10,
            "frame_stats": [{"frame_idx": 10, "valid": True, "mask_pixels": 21248}],
            "selected_sam3_obj_id": 0,
            "prompt_candidates": [
                {
                    "sam3_obj_id": 0,
                    "mask_pixels": 21248,
                    "bbox_xyxy": [642.0, 368.0, 762.0, 603.0],
                }
            ],
        }
        payload = validate_prompt_qc(qc, "phone", [618.0, 411.0, 730.0, 493.0])
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["errors"], [])
        self.assertTrue(any("manual_visual_review" in item for item in payload["warnings"]))

    def test_production_prompt_gate_rejects_point_prompt(self) -> None:
        qc = {
            "qa_pass": True,
            "text": None,
            "target_bbox": [512.0, 180.0, 768.0, 324.0],
            "target_point": [640.0, 252.0],
            "prompt_frame_idx": 10,
            "frame_stats": [{"frame_idx": 10, "valid": True, "mask_pixels": 5000}],
            "selected_sam3_obj_id": 1,
            "prompt_candidates": [{"sam3_obj_id": 1, "mask_pixels": 5000}],
        }
        payload = validate_prompt_qc(qc, "gold jar", [512.0, 180.0, 768.0, 324.0])
        self.assertEqual(payload["status"], "failed")
        self.assertTrue(any("text prompt" in error for error in payload["errors"]))
        self.assertTrue(any("target_point" in error for error in payload["errors"]))

    def test_production_prompt_gate_accepts_small_object_inside_interaction_region(self) -> None:
        qc = {
            "qa_pass": True,
            "text": "large white bowl",
            "target_bbox": [448.0, 288.0, 832.0, 432.0],
            "target_point": None,
            "prompt_frame_idx": 78,
            "frame_stats": [{"frame_idx": 78, "valid": True, "mask_pixels": 8978}],
            "selected_sam3_obj_id": 4,
            "prompt_candidates": [
                {
                    "sam3_obj_id": 4,
                    "mask_pixels": 8978,
                    "bbox_xyxy": [520.0, 310.0, 700.0, 410.0],
                }
            ],
        }
        payload = validate_prompt_qc(qc, "large white bowl", [448.0, 288.0, 832.0, 432.0])
        self.assertEqual(payload["status"], "ok")
        self.assertAlmostEqual(payload["mask_to_bbox_area_ratio"], 8978 / (384 * 144))
        self.assertTrue(any("small relative" in warning for warning in payload["warnings"]))

    def test_production_prompt_gate_rejects_ambiguous_nonoverlapping_instance(self) -> None:
        qc = {
            "qa_pass": True,
            "text": "bottle",
            "target_bbox": [100.0, 100.0, 300.0, 300.0],
            "target_point": None,
            "prompt_frame_idx": 50,
            "frame_stats": [{"frame_idx": 50, "valid": True, "mask_pixels": 20000}],
            "selected_sam3_obj_id": 1,
            "prompt_candidates": [
                {"sam3_obj_id": 1, "mask_pixels": 20000, "bbox_xyxy": [700, 100, 900, 300]},
                {"sam3_obj_id": 2, "mask_pixels": 18000, "bbox_xyxy": [500, 100, 680, 300]},
            ],
        }
        payload = validate_prompt_qc(qc, "bottle", [100.0, 100.0, 300.0, 300.0])
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["prompt_candidate_count"], 2)
        self.assertEqual(payload["selected_bbox_iou"], 0.0)
        self.assertTrue(any("multiple SAM3 prompt candidates" in error for error in payload["errors"]))

    def test_production_prompt_gate_rejects_missing_written_prompt_mask(self) -> None:
        qc = {
            "qa_pass": True,
            "text": "lamp",
            "target_bbox": [768.0, 72.6, 1024.0, 290.4],
            "target_point": None,
            "prompt_frame_idx": 108,
            "frame_stats": [{"frame_idx": 108, "valid": False, "mask_pixels": 0}],
            "selected_sam3_obj_id": 0,
            "prompt_candidates": [
                {"sam3_obj_id": 0, "mask_pixels": 38318, "bbox_xyxy": [588, 86, 867, 396]}
            ],
        }
        payload = validate_prompt_qc(qc, "lamp", [768.0, 72.6, 1024.0, 290.4])
        self.assertEqual(payload["status"], "failed")
        self.assertTrue(any("no final written mask" in error for error in payload["errors"]))

    def test_production_prompt_gate_rejects_temporal_hand_switch(self) -> None:
        frame_stats = [
            {"frame_idx": index, "valid": True, "mask_pixels": 18000}
            for index in range(10)
        ]
        frame_stats[5]["mask_pixels"] = 8000
        qc = {
            "qa_pass": True,
            "text": "large white bowl",
            "target_bbox": [448.0, 288.0, 832.0, 432.0],
            "target_point": None,
            "prompt_frame_idx": 9,
            "frame_stats": frame_stats,
            "max_centroid_jump_px": 131.7,
            "selected_sam3_obj_id": 0,
            "prompt_candidates": [
                {
                    "sam3_obj_id": 0,
                    "mask_pixels": 18000,
                    "bbox_xyxy": [500, 300, 750, 430],
                }
            ],
        }
        payload = validate_prompt_qc(
            qc, "large white bowl", [448.0, 288.0, 832.0, 432.0]
        )
        self.assertEqual(payload["status"], "failed")
        self.assertTrue(any("centroid jump" in error for error in payload["errors"]))
        self.assertTrue(any("area changes too abruptly" in error for error in payload["errors"]))
        self.assertGreater(
            payload["worst_consecutive_mask_area_change"]["ratio"], 2.0
        )

    def test_temporal_area_gate_does_not_compare_across_missing_frames(self) -> None:
        frame_stats = [
            {"frame_idx": 0, "valid": True, "mask_pixels": 1000},
            {"frame_idx": 2, "valid": True, "mask_pixels": 10000},
        ] + [
            {"frame_idx": index, "valid": True, "mask_pixels": 10000}
            for index in range(3, 10)
        ]
        qc = {
            "qa_pass": True,
            "text": "box",
            "target_bbox": [0.0, 0.0, 100.0, 100.0],
            "target_point": None,
            "prompt_frame_idx": 9,
            "frame_stats": frame_stats,
            "max_centroid_jump_px": 10.0,
            "selected_sam3_obj_id": 0,
            "prompt_candidates": [
                {"sam3_obj_id": 0, "mask_pixels": 10000, "bbox_xyxy": [0, 0, 100, 100]}
            ],
        }
        payload = validate_prompt_qc(qc, "box", [0.0, 0.0, 100.0, 100.0])
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["worst_consecutive_mask_area_change"]["ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
