from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from aoe_retarget_lab.hoi_geometry import (
    DEFAULT_FINGERTIP_INDICES,
    apply_common_se3,
    summarize_adapter_rigid_invariance,
    validate_rigid_transform,
)


def arbitrary_se3() -> np.ndarray:
    axis = np.asarray([0.31, -0.77, 0.55], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = 1.173
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray([0.43, -0.28, 1.17])
    return transform


def object_pose_sequence(frames: int) -> np.ndarray:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], frames, axis=0)
    for frame in range(frames):
        angle = 0.07 * frame
        poses[frame, :3, :3] = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        poses[frame, :3, 3] = np.asarray(
            [0.22 + 0.01 * frame, -0.11 + 0.004 * frame, 0.72]
        )
    return poses


def hand_joints(frames: int) -> np.ndarray:
    joints = np.zeros((frames, 21, 3), dtype=np.float64)
    for frame in range(frames):
        joints[frame, :, 0] = np.linspace(0.02, 0.18, 21) + 0.005 * frame
        joints[frame, :, 1] = np.linspace(-0.08, 0.12, 21)
        joints[frame, :, 2] = 0.64 + np.linspace(0.0, 0.09, 21)
    return joints


class HOIRigidTransformTest(unittest.TestCase):
    def test_validates_true_se3_and_rejects_non_rigid_matrices(self):
        transform = arbitrary_se3()
        np.testing.assert_allclose(validate_rigid_transform(transform), transform)

        scaled = transform.copy()
        scaled[:3, 0] *= 1.02
        with self.assertRaisesRegex(ValueError, "orthonormal"):
            validate_rigid_transform(scaled)

        reflected = transform.copy()
        reflected[:3, 0] *= -1.0
        with self.assertRaisesRegex(ValueError, "determinant"):
            validate_rigid_transform(reflected)

        projective = transform.copy()
        projective[3, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "bottom row"):
            validate_rigid_transform(projective)

        nonfinite = transform.copy()
        nonfinite[0, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_rigid_transform(nonfinite)

    def test_common_se3_preserves_full_hoi_relative_geometry(self):
        rng = np.random.default_rng(20260713)
        frames = 7
        transform = arbitrary_se3()
        joints = hand_joints(frames)
        hand_vertices = rng.normal(size=(frames, 32, 3)) * 0.08
        poses = object_pose_sequence(frames)
        contacts = rng.normal(size=(frames, 5, 3)) * 0.04
        mesh = rng.normal(size=(128, 3)) * 0.12

        transformed = apply_common_se3(
            transform,
            hand_joints=joints,
            hand_vertices=hand_vertices,
            object_poses=poses,
            contact_points=contacts,
            mesh_vertices=mesh,
        )

        for name, before in (
            ("hand_joints", joints),
            ("hand_vertices", hand_vertices),
            ("contact_points", contacts),
            ("mesh_vertices", mesh),
        ):
            expected = (
                np.einsum("ij,...j->...i", transform[:3, :3], before)
                + transform[:3, 3]
            )
            np.testing.assert_allclose(transformed[name], expected, atol=1e-14)
        np.testing.assert_allclose(
            transformed["object_poses"], transform @ poses, atol=1e-14
        )

        before_distance = np.linalg.norm(
            joints - poses[:, None, :3, 3], axis=-1
        )
        after_distance = np.linalg.norm(
            transformed["hand_joints"]
            - transformed["object_poses"][:, None, :3, 3],
            axis=-1,
        )
        np.testing.assert_allclose(after_distance, before_distance, atol=1e-14)

        wrist_pose = np.repeat(np.eye(4)[None], frames, axis=0)
        wrist_pose[:, :3, 3] = joints[:, 0]
        transformed_wrist_pose = transform @ wrist_pose
        before_relative_pose = np.linalg.inv(wrist_pose) @ poses
        after_relative_pose = (
            np.linalg.inv(transformed_wrist_pose) @ transformed["object_poses"]
        )
        np.testing.assert_allclose(
            after_relative_pose, before_relative_pose, atol=1e-14
        )

    def test_summary_is_json_safe_and_near_zero_for_common_se3(self):
        frames = 9
        transform = arbitrary_se3()
        before_joints = hand_joints(frames)
        before_poses = object_pose_sequence(frames)
        after = apply_common_se3(
            transform,
            hand_joints=before_joints,
            hand_vertices=np.zeros((frames, 1, 3)),
            object_poses=before_poses,
            contact_points=np.zeros((frames, 1, 3)),
            mesh_vertices=np.zeros((1, 3)),
        )

        with tempfile.TemporaryDirectory() as directory:
            before_hand_path = Path(directory) / "pre_canonical_hands.npz"
            hand_path = Path(directory) / "post_canonical_hands.npz"
            np.savez(
                before_hand_path,
                right_joints=before_joints,
                right_valid=np.ones(frames, dtype=np.float32),
            )
            np.savez(
                hand_path,
                right_joints=after["hand_joints"],
                right_valid=np.ones(frames, dtype=np.float32),
            )
            summary = summarize_adapter_rigid_invariance(
                before_hand_path,
                hand_path,
                before_object_poses=before_poses,
                after_object_poses=after["object_poses"],
                canonical_transform=transform,
            )

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["errors"], [])
        self.assertLess(summary["median"], 1e-14)
        self.assertLess(summary["p95"], 1e-14)
        self.assertLess(summary["max"], 1e-14)
        self.assertEqual(
            summary["thresholds_m"],
            {"median": 1e-8, "p95": 1e-7, "max": 1e-6},
        )
        self.assertEqual(summary["fingertips_object"]["count"], frames * 5)
        self.assertEqual(summary["wrist_object"]["count"], frames)
        self.assertEqual(
            summary["sides"]["right"]["fingertips_object"]["indices"],
            list(DEFAULT_FINGERTIP_INDICES),
        )
        json.dumps(summary, allow_nan=False)

    def test_summary_detects_object_only_translation(self):
        frames = 8
        joints = hand_joints(frames)
        before_poses = object_pose_sequence(frames)
        after_poses = before_poses.copy()
        after_poses[:, :3, 3] += np.asarray([0.25, -0.13, 0.06])

        hand_data = {
            "left_joints": joints,
            "left_valid": np.ones(frames, dtype=np.float32),
        }
        summary = summarize_adapter_rigid_invariance(
            hand_data,
            hand_data,
            before_object_poses=before_poses,
            after_object_poses=after_poses,
        )

        self.assertEqual(summary["status"], "invalid")
        self.assertGreater(summary["median"], 1e-3)
        self.assertGreater(summary["p95"], 1e-3)
        self.assertGreater(summary["max"], 1e-3)
        self.assertIn(
            "relative_distance_max_error_exceeds_tolerance", summary["errors"]
        )

    def test_summary_detects_object_only_rotation_about_unchanged_center(self):
        frames = 6
        joints = hand_joints(frames)
        before_poses = object_pose_sequence(frames)
        after_poses = before_poses.copy()
        angle = 0.4
        object_only_rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        after_poses[:, :3, :3] = (
            object_only_rotation @ after_poses[:, :3, :3]
        )

        summary = summarize_adapter_rigid_invariance(
            {
                "right_joints": joints,
                "right_valid": np.ones(frames, dtype=np.float32),
            },
            {
                "right_joints": joints,
                "right_valid": np.ones(frames, dtype=np.float32),
            },
            before_object_poses=before_poses,
            after_object_poses=after_poses,
        )

        self.assertEqual(summary["status"], "invalid")
        self.assertIn(
            "object_pose_rotation_error_exceeds_tolerance", summary["errors"]
        )
        self.assertGreater(summary["object_pose"]["rotation_error_rad"]["max"], 0.1)

    def test_identity_pose_with_source_orthonormality_drift_is_zero(self):
        frames = 4
        poses = object_pose_sequence(frames)
        poses[:, 0, 0] += 1.0e-7
        hands = {
            "right_joints": hand_joints(frames),
            "right_valid": np.ones(frames, dtype=np.float32),
        }

        summary = summarize_adapter_rigid_invariance(
            hands,
            hands,
            before_object_poses=poses,
            after_object_poses=poses.copy(),
        )

        self.assertEqual(summary["status"], "ok", summary["errors"])
        self.assertLess(
            summary["object_pose"]["rotation_error_rad"]["max"], 1e-12
        )

    def test_summary_detects_hand_only_translation(self):
        frames = 6
        before_joints = hand_joints(frames)
        after_joints = before_joints + np.asarray([0.25, -0.04, 0.02])
        poses = object_pose_sequence(frames)
        before = {
            "left_joints": before_joints,
            "left_valid": np.ones(frames, dtype=np.float32),
        }
        after = {
            "left_joints": after_joints,
            "left_valid": np.ones(frames, dtype=np.float32),
        }

        summary = summarize_adapter_rigid_invariance(
            before,
            after,
            before_object_poses=poses,
            after_object_poses=poses.copy(),
        )

        self.assertEqual(summary["status"], "invalid")
        self.assertIn(
            "hand_canonical_position_error_exceeds_tolerance",
            summary["errors"],
        )
        self.assertGreater(summary["hand_canonical_position_error_m"]["max"], 0.2)

    def test_summary_detects_hand_slot_swap(self):
        frames = 5
        left = hand_joints(frames)
        right = left + np.asarray([0.4, 0.0, 0.0])
        poses = object_pose_sequence(frames)
        before = {
            "left_joints": left,
            "right_joints": right,
            "left_valid": np.ones(frames, dtype=np.float32),
            "right_valid": np.ones(frames, dtype=np.float32),
        }
        after = {
            "left_joints": right,
            "right_joints": left,
            "left_valid": np.ones(frames, dtype=np.float32),
            "right_valid": np.ones(frames, dtype=np.float32),
        }

        summary = summarize_adapter_rigid_invariance(
            before,
            after,
            before_object_poses=poses,
            after_object_poses=poses.copy(),
        )

        self.assertEqual(summary["status"], "invalid")
        self.assertIn(
            "hand_canonical_position_error_exceeds_tolerance",
            summary["errors"],
        )

    def test_common_helper_does_not_offer_object_only_mode(self):
        with self.assertRaisesRegex(ValueError, "hand_joints are required"):
            apply_common_se3(
                arbitrary_se3(),
                hand_joints=None,
                hand_vertices=None,
                object_poses=object_pose_sequence(1),
                contact_points=None,
                mesh_vertices=None,
            )


if __name__ == "__main__":
    unittest.main()
