import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_fresh_aoe_scene_12_demos.sh"


class FreshRunOutcomeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_exit_trap_records_machine_readable_outcome(self):
        self.assertIn('run_outcome="$exp/fresh_run_outcome.json"', self.text)
        self.assertIn("trap record_run_outcome EXIT", self.text)
        self.assertIn("trap 'handle_signal 129' HUP", self.text)
        self.assertIn("trap 'handle_signal 130' INT", self.text)
        self.assertIn("trap 'handle_signal 143' TERM", self.text)
        self.assertIn('129: "SIGHUP"', self.text)
        self.assertIn('status="aborted"', self.text)
        self.assertIn('payload["termination_signal"]', self.text)
        self.assertIn('"return_code": int(return_code)', self.text)
        self.assertIn('"stage": stage', self.text)

    def test_major_fresh_stages_are_named(self):
        for stage in (
            "prepare_fresh_aoe_clip",
            "do_as_i_do_fresh_reconstruction",
            "full_12_demos",
            "strict_postflight_audit",
            "complete",
        ):
            self.assertIn(f'current_stage="{stage}"', self.text)


if __name__ == "__main__":
    unittest.main()
