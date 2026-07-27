import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_dai_scale_anchor_hand.py"
SPEC = importlib.util.spec_from_file_location("select_dai_scale_anchor_hand", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ScaleAnchorSelectionTest(unittest.TestCase):
    def test_explicit_single_hand_is_preserved(self):
        evidence = {"normalized": {"left": 0.1, "right": 10.0}}
        selected, decision = MODULE.select_anchor("left", evidence)
        self.assertEqual(selected, "left")
        self.assertEqual(decision["reason"], "explicit_single_hand")

    def test_bimanual_resolves_to_stronger_visual_contact(self):
        evidence = {"normalized": {"left": 2.0, "right": 0.5}}
        selected, decision = MODULE.select_anchor("bimanual", evidence)
        self.assertEqual(selected, "left")
        self.assertEqual(decision["score_ratio"], 4.0)
        self.assertFalse(decision["ambiguous"])

    def test_close_bimanual_scores_are_recorded_as_ambiguous(self):
        evidence = {"normalized": {"left": 1.0, "right": 1.1}}
        selected, decision = MODULE.select_anchor("bimanual", evidence)
        self.assertEqual(selected, "right")
        self.assertTrue(decision["ambiguous"])


if __name__ == "__main__":
    unittest.main()
