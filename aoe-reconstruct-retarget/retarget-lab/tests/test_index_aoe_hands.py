import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from index_cell_assets import aoe_hand_candidates  # noqa: E402


class IndexAoEHandsTest(unittest.TestCase):
    def test_exact_route_hand_mesh_is_preferred_without_diagnostic_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            (
                exp
                / "intermediates"
                / "retargeting"
                / "do_as_i_do"
                / "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do"
                / "retargeting_outputs"
            ).mkdir(parents=True)
            candidates = aoe_hand_candidates(
                exp,
                "phone_left",
                "left",
                "do_as_i_do",
                "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do",
            )
            self.assertTrue(str(candidates[0]).endswith("raw_dir/raw/all_hand_meshes.npz"))
            self.assertEqual(
                candidates[0],
                exp
                / "intermediates"
                / "retargeting"
                / "do_as_i_do"
                / "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do"
                / "raw_dir"
                / "raw"
                / "all_hand_meshes.npz",
            )
            self.assertFalse(
                any("dai_native_contact_aligned_raw_dir" in str(path) for path in candidates)
            )
            self.assertFalse(
                any("dai_native_diagnostic_raw_dir" in str(path) for path in candidates)
            )

    def test_ego_aoe_adapter_hand_mesh_is_available_for_spider_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            candidates = aoe_hand_candidates(
                exp,
                "mug_right",
                "right",
                "spider",
                "traj_egoinfinity__hand_aoe__retarget_spider",
            )
            self.assertIn(
                exp
                / "intermediates"
                / "trajectory_6dof"
                / "egoinfinity"
                / "do_as_i_do_raw_dir_hand_aoe"
                / "raw"
                / "all_hand_meshes.npz",
                candidates,
            )


if __name__ == "__main__":
    unittest.main()
