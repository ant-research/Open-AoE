import unittest

from aoe_retarget_lab.retarget_input_qc import assess_ego_retarget_input


def surface_geometry(
    selected: str = "left",
    *,
    left_contacts: int = 6,
    right_contacts: int = 0,
    distance_thresh: float = 0.05,
) -> dict:
    return {
        "available": True,
        "distance_thresh": distance_thresh,
        "sampled_all_frames": True,
        "source_frame_count": 20,
        "selected_hand": selected,
        "coverage_by_hand": {
            side: {
                "source_frame_count": 20,
                "planned_frame_count": 20,
                "evaluated_frame_count": 20,
                "failed_frame_count": 0,
                "complete_source_coverage": True,
                "failure_reasons": {},
            }
            for side in ("left", "right")
        },
        "scores": {
            "left": {"frames": 20, "in_hand_frames": left_contacts},
            "right": {"frames": 20, "in_hand_frames": right_contacts},
        },
    }


def projection(*sides: str, inside_fraction: float = 1.0) -> dict:
    return {
        "sides": {
            side: {
                "available": True,
                "anchor_hand": side,
                "sampling_policy": "start_middle_end",
                "expected_frame_count": 3,
                "available_frame_count": 3,
                "complete_sample_coverage": True,
                "frames": [
                    {
                        "available": True,
                        "inside_fraction": inside_fraction,
                        "positive_depth_fraction": 1.0,
                    },
                    {
                        "available": True,
                        "inside_fraction": inside_fraction,
                        "positive_depth_fraction": 1.0,
                    },
                    {
                        "available": True,
                        "inside_fraction": inside_fraction,
                        "positive_depth_fraction": 1.0,
                    },
                ],
            }
            for side in sides
        }
    }


class EgoRetargetInputQCTest(unittest.TestCase):
    def test_stable_scale_and_close_geometry_pass(self):
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband", "p90_p10_ratio": 1.1},
            visual_interaction={"scores": {"left": {"overlap_frames": 12}}},
            geometry_interaction=surface_geometry(),
            hand_projection=projection("left"),
            selected_hand="left",
            hoi_contact_alignment={"applied": False, "reason": "geometry_already_close"},
        )
        self.assertEqual(result["status"], "ok")

    def test_temporally_unstable_scale_is_rejected(self):
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "unstable_projected_scale_ratio"},
            visual_interaction={"scores": {"left": {"overlap_frames": 12}}},
            geometry_interaction=surface_geometry(),
            hand_projection=projection("left"),
            selected_hand="left",
            hoi_contact_alignment={"applied": False, "reason": "geometry_already_close"},
        )
        self.assertIn("object_scale_ratio_temporally_unstable", result["errors"])

    def test_large_coordinate_offset_is_rejected(self):
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "stable_projected_scale_mismatch"},
            visual_interaction={"scores": {"right": {"overlap_frames": 20}}},
            geometry_interaction=surface_geometry(
                "right", left_contacts=0, right_contacts=8
            ),
            hand_projection=projection("right"),
            selected_hand="right",
            hoi_contact_alignment={
                "applied": False,
                "reason": "offset_exceeds_max",
                "offset_norm": 0.78,
            },
        )
        self.assertIn("hoi_coordinate_offset_exceeds_max", result["errors"])

    def test_legacy_object_only_translation_is_rejected(self):
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={"scores": {"right": {"overlap_frames": 20}}},
            geometry_interaction=surface_geometry(
                "right", left_contacts=0, right_contacts=8
            ),
            hand_projection=projection("right"),
            selected_hand="right",
            hoi_contact_alignment={
                "applied": True,
                "method": "median_nearest_surface_translation",
                "offset_norm": 0.10,
            },
        )
        self.assertEqual(result["status"], "invalid")
        self.assertIn("object_only_hoi_translation_applied", result["errors"])

    def test_diagnostic_offset_does_not_modify_or_invalidate_input(self):
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={"scores": {"right": {"overlap_frames": 20}}},
            geometry_interaction=surface_geometry(
                "right", left_contacts=0, right_contacts=8
            ),
            hand_projection=projection("right"),
            selected_hand="right",
            hoi_contact_alignment={
                "applied": False,
                "would_apply": True,
                "diagnostic_only": True,
                "reason": "object_only_translation_forbidden",
                "offset_norm": 0.10,
            },
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn(
            "object_only_hoi_translation_rejected_diagnostic",
            result["warnings"],
        )

    def test_no_selected_hand_overlap_is_rejected(self):
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={"scores": {"right": {"overlap_frames": 0}}},
            geometry_interaction=surface_geometry(
                "right", left_contacts=0, right_contacts=8
            ),
            hand_projection=projection("right"),
            selected_hand="right",
            hoi_contact_alignment={"applied": False, "reason": "weak_visual_overlap"},
        )
        self.assertIn("weak_visual_hand_object_overlap:right:0", result["errors"])

    def test_bimanual_input_requires_both_hands_to_overlap(self):
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={
                "scores": {
                    "left": {"overlap_frames": 9},
                    "right": {"overlap_frames": 2},
                }
            },
            geometry_interaction=surface_geometry(
                "bimanual", left_contacts=8, right_contacts=8
            ),
            hand_projection=projection("left", "right"),
            selected_hand="bimanual",
            hoi_contact_alignment={"applied": False, "reason": "geometry_already_close"},
        )
        self.assertIn("weak_visual_bimanual_overlap:right:2", result["errors"])

    def test_bimanual_input_passes_with_two_hand_evidence(self):
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={
                "scores": {
                    "left": {"overlap_frames": 9},
                    "right": {"overlap_frames": 7},
                }
            },
            geometry_interaction=surface_geometry(
                "bimanual", left_contacts=8, right_contacts=8
            ),
            hand_projection=projection("left", "right"),
            selected_hand="bimanual",
            hoi_contact_alignment={"applied": False, "reason": "geometry_already_close"},
        )
        self.assertEqual(result["status"], "ok")

    def test_selected_side_requires_three_true_surface_contact_frames(self):
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={"scores": {"left": {"overlap_frames": 12}}},
            geometry_interaction=surface_geometry(
                "right", left_contacts=2, right_contacts=14
            ),
            hand_projection=projection("left"),
            selected_hand="left",
            hoi_contact_alignment={"applied": False, "reason": "disabled"},
        )
        self.assertEqual(result["status"], "invalid")
        self.assertIn(
            "insufficient_surface_contact_frames:left:2<3@0.050m",
            result["errors"],
        )

    def test_point_one_meter_in_hand_threshold_is_not_contact_qc(self):
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={"scores": {"left": {"overlap_frames": 12}}},
            geometry_interaction=surface_geometry(
                "left", left_contacts=12, distance_thresh=0.10
            ),
            hand_projection=projection("left"),
            selected_hand="left",
            hoi_contact_alignment={"applied": False, "reason": "disabled"},
        )
        self.assertIn(
            "surface_contact_qc_threshold_too_loose:0.100000>0.050000",
            result["errors"],
        )

    def test_sampled_geometry_cannot_claim_production_contact(self):
        geometry = surface_geometry("left", left_contacts=12)
        geometry["sampled_all_frames"] = False
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={"scores": {"left": {"overlap_frames": 12}}},
            geometry_interaction=geometry,
            hand_projection=projection("left"),
            selected_hand="left",
            hoi_contact_alignment={"applied": False, "reason": "disabled"},
        )
        self.assertIn("surface_contact_qc_not_full_frame_coverage", result["errors"])

    def test_surface_contact_requires_actual_selected_side_full_coverage(self):
        geometry = surface_geometry("left", left_contacts=3)
        geometry["source_frame_count"] = 100
        geometry["coverage_by_hand"]["left"].update(
            {
                "source_frame_count": 100,
                "planned_frame_count": 100,
                "evaluated_frame_count": 3,
                "failed_frame_count": 97,
                "complete_source_coverage": False,
                "failure_reasons": {"missing_object_mesh": 97},
            }
        )
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={"scores": {"left": {"overlap_frames": 12}}},
            geometry_interaction=geometry,
            hand_projection=projection("left"),
            selected_hand="left",
            hoi_contact_alignment={"applied": False, "reason": "disabled"},
        )
        self.assertEqual(result["status"], "invalid")
        self.assertIn(
            "surface_contact_qc_incomplete_evaluation:"
            "left:evaluated=3/100:failed=97:planned=100",
            result["errors"],
        )

    def test_obvious_visual_geometry_side_conflict_is_rejected(self):
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={
                "selected_hand": "left",
                "scores": {"left": {"overlap_frames": 19}},
            },
            geometry_interaction=surface_geometry(
                "right", left_contacts=2, right_contacts=14
            ),
            hand_projection=projection("left"),
            selected_hand="left",
            hoi_contact_alignment={"applied": False, "reason": "disabled"},
        )
        self.assertTrue(
            any(
                error.startswith("selected_hand_visual_geometry_conflict:")
                for error in result["errors"]
            )
        )

    def test_mostly_offscreen_selected_hand_is_rejected(self):
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={"scores": {"right": {"overlap_frames": 12}}},
            geometry_interaction=surface_geometry(
                "right", left_contacts=0, right_contacts=14
            ),
            hand_projection=projection("right", inside_fraction=0.0),
            selected_hand="right",
            hoi_contact_alignment={"applied": False, "reason": "disabled"},
        )
        self.assertIn(
            "hand_projection_mostly_outside_image:right:0.000000<0.250000",
            result["errors"],
        )

    def test_projection_requires_all_start_middle_end_frames(self):
        projection_report = projection("left")
        side = projection_report["sides"]["left"]
        side["frames"][0] = {
            "available": False,
            "reason": "missing_intrinsics",
        }
        side["available_frame_count"] = 2
        side["complete_sample_coverage"] = False
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={"scores": {"left": {"overlap_frames": 12}}},
            geometry_interaction=surface_geometry("left"),
            hand_projection=projection_report,
            selected_hand="left",
            hoi_contact_alignment={"applied": False, "reason": "disabled"},
        )
        self.assertIn(
            "hand_projection_qc_incomplete_start_middle_end:left:available=2/3",
            result["errors"],
        )

    def test_projection_uses_minimum_not_best_frame(self):
        projection_report = projection("left")
        projection_report["sides"]["left"]["frames"][0]["inside_fraction"] = 0.0
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={"scores": {"left": {"overlap_frames": 12}}},
            geometry_interaction=surface_geometry("left"),
            hand_projection=projection_report,
            selected_hand="left",
            hoi_contact_alignment={"applied": False, "reason": "disabled"},
        )
        self.assertIn(
            "hand_projection_mostly_outside_image:left:0.000000<0.250000",
            result["errors"],
        )

    def test_projection_requires_positive_depth_vertex_coverage(self):
        projection_report = projection("left")
        projection_report["sides"]["left"]["frames"][1][
            "positive_depth_fraction"
        ] = 0.10
        result = assess_ego_retarget_input(
            scale_fit={"available": True, "reason": "within_scale_deadband"},
            visual_interaction={"scores": {"left": {"overlap_frames": 12}}},
            geometry_interaction=surface_geometry("left"),
            hand_projection=projection_report,
            selected_hand="left",
            hoi_contact_alignment={"applied": False, "reason": "disabled"},
        )
        self.assertIn(
            "hand_projection_insufficient_positive_depth:left:0.100000<0.950000",
            result["errors"],
        )


if __name__ == "__main__":
    unittest.main()
