import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_fresh_aoe_auto_window.py"
SPEC = importlib.util.spec_from_file_location("fresh_auto_window", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FreshAutoWindowTest(unittest.TestCase):
    def test_candidates_are_quantized_deduplicated_and_longest_first(self):
        self.assertEqual(
            MODULE.quantize_candidates("2.2,2.4,2.21,3.0", 15.0),
            [3.0, 2.4, 2.2],
        )

    def test_only_temporal_failures_are_retryable(self):
        temporal = {
            "errors": [
                "SAM3 mask area changes too abruptly between consecutive frames: frames 69->70"
            ]
        }
        self.assertEqual(
            MODULE.classify_preflight_failure(temporal, {}),
            "retryable_temporal_window_failure",
        )
        wrong_identity = {"errors": ["semantic prompt selected wrong object"]}
        self.assertEqual(
            MODULE.classify_preflight_failure(wrong_identity, {}),
            "non_retryable_prompt_or_identity_failure",
        )

    def test_controlled_runner_options_are_rejected(self):
        for values in (["--duration-sec", "2.2"], ["--source-run=x"]):
            with self.assertRaises(ValueError):
                MODULE.validate_base_args(list(values))

    def test_direct_duration_uses_same_ego_clock_quantization(self):
        self.assertEqual(MODULE.quantize_candidates("2.5", 15.0), [38 / 15])


if __name__ == "__main__":
    unittest.main()
