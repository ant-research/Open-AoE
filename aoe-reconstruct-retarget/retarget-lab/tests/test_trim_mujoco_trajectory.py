from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TrimMujocoTrajectoryTest(unittest.TestCase):
    def run_trim(self, include_time: bool) -> tuple[dict, dict[str, np.ndarray]]:
        script = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics" / "trim_mujoco_trajectory.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            output = root / "aligned.npz"
            manifest = root / "alignment.json"
            payload = {"qpos": np.arange(144, dtype=np.float64).reshape(36, 4)}
            if include_time:
                payload["time"] = np.arange(36, dtype=np.float64) / 180.0
            np.savez(source, **payload)
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(source),
                    "--output",
                    str(output),
                    "--start-index",
                    "5",
                    "--target-count",
                    "6",
                    "--target-fps",
                    "30",
                    "--sim-dt",
                    str(1.0 / 180.0),
                    "--metadata-json",
                    str(manifest),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with np.load(output) as data:
                arrays = {key: np.asarray(data[key]).copy() for key in data.files}
            return json.loads(manifest.read_text(encoding="utf-8")), arrays

    def test_rebases_existing_time_to_source_clip_timeline(self) -> None:
        manifest, arrays = self.run_trim(include_time=True)
        np.testing.assert_allclose(arrays["time"], np.arange(6) / 30.0)
        self.assertEqual(manifest["aligned_time_semantics"], "source_clip_frame_time")
        self.assertAlmostEqual(manifest["aligned_duration_sec"], 5.0 / 30.0)
        self.assertTrue(manifest["complete_source_coverage"])
        self.assertAlmostEqual(manifest["duration_coverage_ratio"], 1.0)
        self.assertEqual(manifest["sample_stride"], 6)
        self.assertEqual(
            manifest["sample_indices_relative_to_post_warmup"],
            [0, 6, 12, 18, 24, 30],
        )
        self.assertEqual(manifest["ignored_tail_samples"], 0)
        np.testing.assert_array_equal(
            arrays["qpos"],
            np.arange(144, dtype=np.float64).reshape(36, 4)[[5, 11, 17, 23, 29, 35]],
        )

    def test_rejects_missing_recorded_time_for_exact_alignment(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_trim(include_time=False)

    def test_rejects_non_integer_timestamp_stride(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "diagnostics" / "trim_mujoco_trajectory.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            np.savez(source, qpos=np.zeros((100, 4), dtype=np.float64))
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--input",
                    str(source),
                    "--output",
                    str(root / "aligned.npz"),
                    "--start-index",
                    "0",
                    "--target-count",
                    "6",
                    "--target-fps",
                    "30",
                    "--sim-dt",
                    "0.005",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("integer source-frame stride", result.stderr)


if __name__ == "__main__":
    unittest.main()
