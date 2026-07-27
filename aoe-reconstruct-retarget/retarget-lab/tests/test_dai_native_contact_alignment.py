import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from align_dai_native_contact import (  # noqa: E402
    align_layout,
    assess_dai_native_retarget_input,
    assess_object_pose_continuity,
)


class DAINativeContactAlignmentTest(unittest.TestCase):
    def test_object_pose_continuity_accepts_smooth_track(self):
        layout = {
            "objects": [
                {
                    "frame_idx": idx,
                    "local_to_scene": {
                        "translation_camera_frame": [0.01 * idx, 0.0, 1.0],
                        "quat_wxyz_camera_frame": [1.0, 0.0, 0.0, 0.0],
                    },
                }
                for idx in range(20)
            ]
        }
        result = assess_object_pose_continuity(layout)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["lost_frame_fraction"], 0.0)

    def test_object_pose_continuity_rejects_catastrophic_flip_and_flyout(self):
        objects = []
        for idx in range(20):
            translation = [0.01 * idx, 0.0, 1.0]
            quaternion = [1.0, 0.0, 0.0, 0.0]
            if idx >= 10:
                translation = [0.09 + 0.30 * (idx - 9), 0.0, 1.0]
                quaternion = [0.0, 1.0, 0.0, 0.0]
            objects.append(
                {
                    "frame_idx": idx,
                    "local_to_scene": {
                        "translation_camera_frame": translation,
                        "quat_wxyz_camera_frame": quaternion,
                    },
                }
            )
        result = assess_object_pose_continuity({"objects": objects})
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("translation_step_p95" in error for error in result["errors"]))
        self.assertTrue(any("rotation_step_max" in error for error in result["errors"]))

    def test_object_pose_continuity_counts_invalid_frame_once(self):
        objects = [
            {
                "frame_idx": idx,
                "local_to_scene": {
                    "translation_camera_frame": [0.01 * idx, 0.0, 1.0],
                    "quat_wxyz_camera_frame": [1.0, 0.0, 0.0, 0.0],
                },
            }
            for idx in range(10)
        ]
        objects[5]["local_to_scene"]["translation_camera_frame"] = [float("nan"), 0.0, 1.0]
        result = assess_object_pose_continuity(
            {"objects": objects},
            max_lost_frame_fraction=0.2,
        )
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["lost_frame_fraction"], 0.1)
        self.assertEqual(result["invalid_frames"], [5])

    def test_retarget_input_rejects_invalid_source_object_track(self):
        result = assess_dai_native_retarget_input(
            {"applied": False, "reason": "geometry_already_close"},
            overlap_frames=18,
            object_pose_quality={
                "status": "invalid",
                "errors": ["object_pose_rotation_step_max_deg:179"],
            },
        )
        self.assertEqual(result["status"], "invalid")
        self.assertIn("source_object_pose_rotation_step_max_deg:179", result["errors"])

    def test_retarget_input_requires_visual_overlap(self):
        result = assess_dai_native_retarget_input(
            {"applied": False, "reason": "weak_visual_overlap"},
            overlap_frames=0,
        )
        self.assertEqual(result["status"], "invalid")
        self.assertIn("weak_visual_hand_object_overlap:0", result["errors"])

    def test_retarget_input_accepts_close_geometry_with_overlap(self):
        result = assess_dai_native_retarget_input(
            {"applied": False, "reason": "geometry_already_close"},
            overlap_frames=18,
        )
        self.assertEqual(result["status"], "ok")

    def test_far_native_object_translation_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh = root / "object.obj"
            mesh.write_text(
                "v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\nv 0 0 0.01\n"
                "f 1 2 3\nf 1 2 4\n",
                encoding="utf-8",
            )
            hand = np.zeros((8, 4, 3), dtype=np.float32)
            hand[:, :, 0] = np.asarray([0.0, 0.01, 0.0, 0.0])
            hand_npz = root / "hands.npz"
            np.savez(hand_npz, left_vertices=hand, left_valid=np.ones(8))
            layout = {
                "translation_scale_optimization": {"mesh_scale": 1.0},
                "objects": [
                    {
                        "frame_idx": idx,
                        "local_to_scene": {
                            "translation_camera_frame": [0.20, 0.0, 0.0],
                            "quat_wxyz_camera_frame": [1.0, 0.0, 0.0, 0.0],
                        },
                    }
                    for idx in range(8)
                ],
            }
            aligned, report = align_layout(
                layout,
                hand_npz,
                mesh,
                "left",
                8,
                min_bad_mean_distance=0.10,
                max_offset=0.30,
            )
            self.assertFalse(report["applied"])
            self.assertTrue(report["diagnostic_only"])
            self.assertTrue(report["would_apply"])
            self.assertLess(
                report["hypothetical_after"]["mean"],
                report["before"]["mean"] * 0.5,
            )
            self.assertEqual(report["retarget_object_source"], "dai_native")
            self.assertEqual(aligned, layout)

    def test_auto_rejects_large_required_translation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh = root / "object.obj"
            mesh.write_text("v 0 0 0\nv .01 0 0\nv 0 .01 0\nv 0 0 .01\n", encoding="utf-8")
            hand_npz = root / "hands.npz"
            np.savez(
                hand_npz,
                left_vertices=np.zeros((8, 4, 3), dtype=np.float32),
                left_valid=np.ones(8),
            )
            layout = {
                "translation_scale_optimization": {"mesh_scale": 1.0},
                "objects": [
                    {
                        "frame_idx": idx,
                        "local_to_scene": {
                            "translation_camera_frame": [0.16, 0.0, 0.0],
                            "quat_wxyz_camera_frame": [1.0, 0.0, 0.0, 0.0],
                        },
                    }
                    for idx in range(8)
                ],
            }

            aligned, report = align_layout(layout, hand_npz, mesh, "left", 8)

            self.assertFalse(report["applied"])
            self.assertEqual(report["reason"], "offset_exceeds_max")
            self.assertGreater(report["required_offset_norm"], report["max_offset"])
            self.assertEqual(aligned, layout)

    def test_close_native_object_is_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mesh = root / "object.obj"
            mesh.write_text("v 0 0 0\nv .01 0 0\nv 0 .01 0\nv 0 0 .01\n", encoding="utf-8")
            hand_npz = root / "hands.npz"
            np.savez(hand_npz, right_vertices=np.zeros((4, 4, 3)), right_valid=np.ones(4))
            layout = {
                "objects": [
                    {"frame_idx": idx, "local_to_scene": {"translation_camera_frame": [0.02, 0, 0], "quat_wxyz_camera_frame": [1, 0, 0, 0]}}
                    for idx in range(4)
                ]
            }
            aligned, report = align_layout(layout, hand_npz, mesh, "right", 4)
            self.assertFalse(report["applied"])
            self.assertEqual(report["reason"], "geometry_already_close")
            self.assertEqual(aligned, layout)

    def test_auto_requires_visual_overlap(self):
        layout = {"objects": []}
        aligned, report = align_layout(layout, Path("missing"), Path("missing"), "left", 0)
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "weak_visual_overlap")
        self.assertEqual(aligned, layout)


if __name__ == "__main__":
    unittest.main()
