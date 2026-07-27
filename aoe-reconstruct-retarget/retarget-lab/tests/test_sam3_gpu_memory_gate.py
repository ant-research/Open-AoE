import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_do_as_i_do_reconstruction_fresh.sh"


class Sam3GpuMemoryGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = SCRIPT.read_text(encoding="utf-8")

    def test_sam3_and_sam3d_share_the_memory_selector(self) -> None:
        self.assertIn("select_gpu_by_free_memory()", self.script)
        self.assertIn('"SAM3" "$sam3_cuda" "$sam3_min_free_mb"', self.script)
        self.assertIn('"SAM3D" "$sam3d_cuda" "$sam3d_min_free_mb"', self.script)

    def test_sam3_gate_runs_before_segmentation(self) -> None:
        gate = self.script.index("select_sam3_gpu_if_needed\n")
        segmentation = self.script.index(
            'run_sam3_py "$repo_root/scripts/compatibility/run_sam3_video.py"'
        )
        self.assertLess(gate, segmentation)
        self.assertIn('sam3_min_free_mb="${SAM3_MIN_FREE_MB:-12000}"', self.script)
        self.assertIn("exit 86", self.script)


if __name__ == "__main__":
    unittest.main()
