import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "diagnostics"))

import numpy as np
import pytest

pytest.importorskip("mujoco")

from render_mujoco_trajectory import load_qpos


class RenderMujocoTrajectoryTest(unittest.TestCase):
    def test_qpos_width_mismatch_is_strict_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            trajectory = Path(directory) / "trajectory.npz"
            np.savez_compressed(trajectory, qpos=np.zeros((2, 3)))
            with self.assertRaisesRegex(ValueError, "qpos width=3 does not match model.nq=4"):
                load_qpos(trajectory, 4, np.zeros(4))

    def test_legacy_shape_coercion_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            trajectory = Path(directory) / "trajectory.npz"
            np.savez_compressed(trajectory, qpos=np.ones((2, 3)))
            loaded = load_qpos(
                trajectory,
                4,
                np.zeros(4),
                allow_shape_coercion=True,
            )
            self.assertEqual(loaded.shape, (2, 4))
            np.testing.assert_allclose(loaded[:, :3], 1.0)
            np.testing.assert_allclose(loaded[:, 3], 0.0)


if __name__ == "__main__":
    unittest.main()
