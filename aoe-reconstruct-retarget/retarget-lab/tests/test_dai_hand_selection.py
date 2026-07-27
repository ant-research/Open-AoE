import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reuse_v4_for_12_demos import (  # noqa: E402
    dai_interaction_hand_types,
    parse_args,
    spider_cell_order,
    spider_single_object_hand_selection,
)
from select_do_as_i_do_hand_type import resolve_hand_type, strong_single_hand_mask_choice  # noqa: E402


class DAIHandSelectionTest(unittest.TestCase):
    def test_matrix_auto_selection_applies_to_native_and_ego_trajectories(self):
        source = (ROOT / "scripts" / "reuse_v4_for_12_demos.py").read_text(encoding="utf-8")
        self.assertNotIn('args.dai_auto_select_interaction_hand and trajectory == "egoinfinity"', source)

    def test_bimanual_searches_resolved_single_hand_outputs(self):
        self.assertEqual(
            dai_interaction_hand_types("bimanual", auto_select=True),
            ["bimanual", "left", "right"],
        )

    def test_single_hand_keeps_requested_side_first(self):
        self.assertEqual(
            dai_interaction_hand_types("left", auto_select=True),
            ["left", "right", "bimanual"],
        )

    def test_auto_selection_can_be_disabled(self):
        self.assertEqual(
            dai_interaction_hand_types("bimanual", auto_select=False),
            ["bimanual"],
        )

    def test_spider_defaults_to_per_trajectory_auto_selection(self):
        with patch.object(sys, "argv", ["reuse_v4_for_12_demos.py", "--source-run", "sample"]):
            args = parse_args()
        self.assertEqual(args.spider_hand_type, "auto")

    def test_spider_auto_collapses_bimanual_single_object_to_contacting_hand(self):
        quality = {
            "available": True,
            "sides": {
                "left": {
                    "contact_active": 24,
                    "contact_mean": 0.7,
                    "finger_object_distance_median": 0.05,
                    "finger_object_distance_min": 0.01,
                },
                "right": {
                    "contact_active": 5,
                    "contact_mean": 0.2,
                    "finger_object_distance_median": 0.03,
                    "finger_object_distance_min": 0.005,
                },
            },
        }
        selection = spider_single_object_hand_selection("auto", "bimanual", quality)
        self.assertEqual(selection["resolved_hand_type"], "left")
        self.assertEqual(selection["reason"], "single_object_best_interaction_hand")

    def test_spider_auto_keeps_explicit_single_hand_task_side(self):
        selection = spider_single_object_hand_selection(
            "auto", "right", {"available": False}
        )
        self.assertEqual(selection["resolved_hand_type"], "right")
        self.assertEqual(selection["reason"], "explicit_source_task_side")

    def test_explicit_spider_bimanual_request_is_preserved(self):
        selection = spider_single_object_hand_selection(
            "bimanual", "bimanual", {"available": True}
        )
        self.assertEqual(selection["resolved_hand_type"], "bimanual")
        self.assertEqual(selection["reason"], "explicit_spider_hand_type")

    def test_spider_runs_recommended_ego_routes_before_dai_baselines(self):
        self.assertEqual(
            spider_cell_order("all"),
            [
                ("egoinfinity", "aoe"),
                ("egoinfinity", "estimated"),
                ("do_as_i_do", "aoe"),
                ("do_as_i_do", "estimated"),
            ],
        )

    def test_full_pipeline_requests_auto_dai_slot_and_disables_hand_only_fusion(self):
        script = (ROOT / "scripts" / "run_v4_two_full_pipelines.sh").read_text(encoding="utf-8")
        self.assertIn('dai_retarget_hand_type="${V4_DAI_RETARGET_HAND_TYPE:-auto}"', script)
        self.assertNotIn('--hand-type "$dai_retarget_hand_type"', script)
        self.assertIn('--hand-type "$dai_resolved_hand_type"', script)
        self.assertIn('${trajectory_6dof}_${hand_source}_adapter_hand=', script)
        self.assertIn('${trajectory_6dof}_${hand_source}_resolved_hand=', script)
        self.assertIn("resolve_native_dai_hand", script)
        self.assertIn('test "$resolved_hand_type" = "$adapter_hand_type"', script)
        self.assertNotIn(
            'run_adapted_do_as_i_do_retarget do_as_i_do estimated "$dai_estimated_raw_dir" source ego',
            script,
        )
        self.assertIn(
            "DAI-object/Ego-hand estimated route disabled: production hand-only canonical fusion is forbidden",
            script,
        )
        self.assertIn('"disabled_reason": "hand_only_canonical_fusion_forbidden"', script)
        self.assertIn(
            'run_adapted_do_as_i_do_retarget egoinfinity aoe "$ego_dai_aoe_raw_dir" ego source',
            script,
        )
        self.assertIn("scale_anchor_hand_selection.json", script)
        self.assertIn('get("selected_anchor_hand")', script)

    def test_egoinfinity_workers_stop_before_dai_retarget(self):
        script = (ROOT / "scripts" / "run_v4_two_full_pipelines.sh").read_text(encoding="utf-8")
        ego_selection = script.index(
            "\nrun_stage egoinfinity_object_selection select_egoinfinity_object\n",
            script.index("\nstart_workers\n"),
        )
        stop_workers = script.index("\nstop_workers\n", ego_selection)
        dai_retarget = script.index(
            "run_stage do_as_i_do_aoe_retarget",
            stop_workers,
        )
        self.assertLess(ego_selection, stop_workers)
        self.assertLess(stop_workers, dai_retarget)
        self.assertIn(
            "stop_after_egoinfinity_object_selection_before_dai_spider",
            script,
        )

    def test_strong_mask_evidence_can_override_bimanual_adapter(self):
        result = {
            "mask_normalized": {"left": 4.18, "right": 1.22},
            "mask_scores": {
                "left": {"frames": 32, "touch_frames": 25, "overlap_frames": 22},
                "right": {"frames": 32, "touch_frames": 4, "overlap_frames": 2},
            },
        }
        choice = strong_single_hand_mask_choice(result)
        self.assertTrue(choice["strong"])
        self.assertEqual(choice["selected_hand"], "left")

    def test_ambiguous_two_hand_masks_do_not_force_single_hand(self):
        result = {
            "mask_normalized": {"left": 3.4, "right": 3.0},
            "mask_scores": {
                "left": {"frames": 32, "touch_frames": 25, "overlap_frames": 20},
                "right": {"frames": 32, "touch_frames": 23, "overlap_frames": 18},
            },
        }
        choice = strong_single_hand_mask_choice(result)
        self.assertFalse(choice["strong"])
        self.assertIsNone(choice["selected_hand"])

    @patch("select_do_as_i_do_hand_type.adapter_geometry_hand_choice")
    @patch("select_do_as_i_do_hand_type.score_hand_sides")
    def test_explicit_left_task_wins_ambiguous_bimanual_evidence(self, score, adapter):
        score.return_value = self._selection_scores(3.43, 3.50, 23, 23)
        adapter.return_value = {"available": True, "selected_hand": "bimanual", "reason": "geometry-bimanual"}
        result = resolve_hand_type(Path("/tmp/not-read"), "phone_left", 32)
        self.assertEqual(result["selected_hand_type"], "left")
        self.assertEqual(result["selection_source"], "task")

    @patch("select_do_as_i_do_hand_type.adapter_geometry_hand_choice")
    @patch("select_do_as_i_do_hand_type.score_hand_sides")
    def test_strong_opposite_mask_can_override_explicit_task(self, score, adapter):
        score.return_value = self._selection_scores(0.7, 4.2, 2, 26)
        adapter.return_value = {"available": False}
        result = resolve_hand_type(Path("/tmp/not-read"), "sample_left", 32)
        self.assertEqual(result["selected_hand_type"], "right")
        self.assertEqual(result["selection_source"], "hand-object-mask")

    @patch("select_do_as_i_do_hand_type.adapter_geometry_hand_choice")
    @patch("select_do_as_i_do_hand_type.score_hand_sides")
    def test_adapter_visual_agreement_blocks_conflicting_derived_mask(self, score, adapter):
        score.return_value = self._selection_scores(0.7, 4.2, 2, 26)
        adapter.return_value = {
            "available": True,
            "selected_hand": "left",
            "reason": "adapter-manifest-visual-agreement",
            "manifest_visual_agree": True,
        }
        result = resolve_hand_type(Path("/tmp/not-read"), "sample_left", 32)
        self.assertEqual(result["selected_hand_type"], "left")
        self.assertEqual(result["selection_source"], "adapter-geometry")

    @patch("select_do_as_i_do_hand_type.adapter_geometry_hand_choice")
    @patch("select_do_as_i_do_hand_type.score_hand_sides")
    def test_bimanual_task_keeps_ambiguous_bimanual_choice(self, score, adapter):
        score.return_value = self._selection_scores(3.4, 3.0, 25, 23)
        adapter.return_value = {"available": False}
        result = resolve_hand_type(Path("/tmp/not-read"), "sample_bimanual", 32)
        self.assertEqual(result["selected_hand_type"], "bimanual")

    @staticmethod
    def _selection_scores(left_score, right_score, left_touch, right_touch):
        return {
            "mask_normalized": {"left": left_score, "right": right_score},
            "normalized": {"left": left_score, "right": right_score},
            "mask_scores": {
                "left": {"score": left_score * 32, "frames": 32, "overlap_frames": left_touch, "touch_frames": left_touch},
                "right": {"score": right_score * 32, "frames": 32, "overlap_frames": right_touch, "touch_frames": right_touch},
            },
            "scores": {
                "left": {"score": left_score * 32, "frames": 32, "overlap_frames": left_touch},
                "right": {"score": right_score * 32, "frames": 32, "overlap_frames": right_touch},
            },
            "config_anchor_hand": "left",
        }


if __name__ == "__main__":
    unittest.main()
