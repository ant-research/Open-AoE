import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "maintenance" / "patch_fast_sam3d_cluster_guard.py"


def load_patcher():
    spec = importlib.util.spec_from_file_location("fast_sam3d_guard_patcher", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FastSam3DGuardPatcherTest(unittest.TestCase):
    def test_installs_candidate_and_postopt_guards_idempotently(self):
        patcher = load_patcher()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "track_object.py"
            target.write_text(patcher.OLD + "\n" + patcher.OLD_POSTOPT, encoding="utf-8")

            subprocess.run(
                [sys.executable, str(SCRIPT), "--track-object", str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            first = target.read_text(encoding="utf-8")
            self.assertIn(patcher.CLUSTER_V4_SENTINEL, first)
            self.assertIn(patcher.POSTOPT_V2_SENTINEL, first)
            self.assertIn("all pose candidates are non-finite", first)
            self.assertIn('cluster_info["selected_cluster_size"] = len(all_samples)', first)
            self.assertIn('best_result["post_opt_rejected"] = "invalid_pose"', first)

            subprocess.run(
                [sys.executable, str(SCRIPT), "--track-object", str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(first, target.read_text(encoding="utf-8"))

    def test_upgrades_v2_cluster_and_v1_postopt_guards(self):
        patcher = load_patcher()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "track_object.py"
            target.write_text(patcher.NEW + "\n" + patcher.NEW_POSTOPT, encoding="utf-8")

            subprocess.run(
                [sys.executable, str(SCRIPT), "--track-object", str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            upgraded = target.read_text(encoding="utf-8")
            self.assertIn(patcher.CLUSTER_V4_SENTINEL, upgraded)
            self.assertIn(patcher.POSTOPT_V2_SENTINEL, upgraded)

    def test_upgrades_v3_cluster_metadata(self):
        patcher = load_patcher()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "track_object.py"
            target.write_text(
                patcher.NEW_V3 + "\n" + patcher.NEW_POSTOPT_V2,
                encoding="utf-8",
            )

            subprocess.run(
                [sys.executable, str(SCRIPT), "--track-object", str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
            upgraded = target.read_text(encoding="utf-8")
            self.assertIn(patcher.CLUSTER_V4_SENTINEL, upgraded)
            self.assertIn('cluster_info["best_candidate_score"]', upgraded)


if __name__ == "__main__":
    unittest.main()
