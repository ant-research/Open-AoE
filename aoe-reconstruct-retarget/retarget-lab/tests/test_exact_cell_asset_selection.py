import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reuse_v4_for_12_demos.py"


def load_reuse_module():
    spec = importlib.util.spec_from_file_location("reuse_v4_for_12_demos", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExactCellAssetSelectionTest(unittest.TestCase):
    def test_invalid_exact_hand_source_is_not_admitted_or_replaced_by_fallback(self):
        module = load_reuse_module()
        original = module.find_dai_robot_candidates
        module.find_dai_robot_candidates = lambda *args, **kwargs: [
        {
            "robot": Path("/run/traj_do_as_i_do__hand_estimated__retarget_do_as_i_do/robot.mp4"),
            "task": "task",
            "trajectory": "do_as_i_do",
            "hand_source": "estimated",
            "hand_type": "left",
            "quality_valid": True,
            "score": (1, 1, 10, 0, 1, 1),
        },
        {
            "robot": Path("/run/traj_do_as_i_do__hand_aoe__retarget_do_as_i_do/robot.mp4"),
            "task": "task",
            "trajectory": "do_as_i_do",
            "hand_source": "aoe",
            "hand_type": "left",
            "quality_valid": False,
            "score": (0, 0, 0, -1, 1, 1),
        },
        ]
        try:
            robot, note = module.find_quality_dai_robot_in_source(
                Path("/run"), "do_as_i_do", "aoe", "task", "left", SimpleNamespace()
            )
        finally:
            module.find_dai_robot_candidates = original
        self.assertIsNone(robot)
        self.assertIsNone(note)


    def test_invalid_exact_override_is_not_admitted_or_replaced_by_fallback(self):
        module = load_reuse_module()
        original = module.find_dai_robot_candidates
        module.find_dai_robot_candidates = lambda *args, **kwargs: [
        {
            "robot": Path("/override/traj_do_as_i_do__hand_estimated__retarget_do_as_i_do/robot.mp4"),
            "task": "task",
            "trajectory": "do_as_i_do",
            "hand_source": "estimated",
            "hand_type": "left",
            "quality_valid": True,
            "score": (1, 1, 10, 0, 1, 1),
        },
        {
            "robot": Path("/override/traj_do_as_i_do__hand_aoe__retarget_do_as_i_do/robot.mp4"),
            "task": "task",
            "trajectory": "do_as_i_do",
            "hand_source": "aoe",
            "hand_type": "left",
            "quality_valid": False,
            "score": (0, 0, 0, -1, 1, 1),
        },
        ]
        args = SimpleNamespace(
            allow_cross_trajectory_dai_robot_fallback=False,
            dai_task="task",
            dai_source_task="task",
            dai_hand_type="left",
            dai_source_hand_type="left",
        )
        try:
            robot, note = module.find_dai_robot_in_overrides(
                [Path("/override")], "do_as_i_do", "aoe", "task", "left", args
            )
        finally:
            module.find_dai_robot_candidates = original
        self.assertIsNone(robot)
        self.assertIsNone(note)

    def test_invalid_exact_route_is_available_only_with_diagnostic_opt_in(self):
        module = load_reuse_module()
        original = module.find_dai_robot_candidates
        module.find_dai_robot_candidates = lambda *args, **kwargs: [{
            "robot": Path("/run/traj_do_as_i_do__hand_aoe__retarget_do_as_i_do/robot.mp4"),
            "task": "task",
            "trajectory": "do_as_i_do",
            "hand_source": "aoe",
            "hand_type": "left",
            "quality_valid": False,
            "score": (0, 0, 0, -1, 1, 1),
        }]
        try:
            robot, note = module.find_quality_dai_robot_in_source(
                Path("/run"),
                "do_as_i_do",
                "aoe",
                "task",
                "left",
                SimpleNamespace(include_numerically_invalid_aligned_robot=True),
            )
        finally:
            module.find_dai_robot_candidates = original
        self.assertIn("hand_aoe", str(robot))
        self.assertIn("quality-unverified", note)


    def test_spider_receives_exact_source_cell_bundle(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("find_exact_spider_input_bundle", text)
        self.assertIn('cell_env["SPIDER_SOURCE_KEYPOINTS"] = str(input_bundle["keypoints"])', text)
        self.assertIn('cell_env["SPIDER_OBJECT_VISUAL"] = str(input_bundle["object_visual"])', text)
        self.assertIn('subprocess.run(cmd, cwd=REPO_ROOT, env=cell_env', text)

    def test_spider_rejects_invalid_ego_adapter_bundle(self):
        module = load_reuse_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = root / "adapter_manifest.json"
            adapter.write_text(
                json.dumps({"retarget_input_qc": {"status": "invalid"}}),
                encoding="utf-8",
            )
            original = module.find_dai_asset_bundles
            module.find_dai_asset_bundles = lambda *args, **kwargs: [{
                "adapter_manifest": adapter,
                "keypoints": root / "trajectory_keypoints.npz",
                "object_visual": root / "visual.obj",
                "task": "task",
                "trajectory": "egoinfinity",
                "hand_source": "aoe",
                "hand_type": "left",
            }]
            args = SimpleNamespace(
                dai_auto_select_interaction_hand=True,
                dai_asset_min_distance_threshold=0.35,
                dai_asset_median_distance_threshold=0.45,
                dai_asset_no_contact_min_distance_threshold=0.08,
                dai_asset_no_contact_median_distance_threshold=0.18,
                dai_asset_require_contact=True,
            )
            try:
                bundle = module.find_exact_spider_input_bundle(
                    root, "egoinfinity", "aoe", "task", "left", args
                )
            finally:
                module.find_dai_asset_bundles = original
        self.assertIsNone(bundle)


    def test_dai_hand_source_fallback_is_opt_in(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            '"--allow-dai-hand-fallback", action=argparse.BooleanOptionalAction, default=False',
            text,
        )


if __name__ == "__main__":
    unittest.main()
