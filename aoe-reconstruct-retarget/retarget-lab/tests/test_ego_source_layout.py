from __future__ import annotations

import json
import gzip
import pickle
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import trimesh  # noqa: F401
except ModuleNotFoundError:
    sys.modules["trimesh"] = types.ModuleType("trimesh")

from prepare_egoinfinity_do_as_i_do_raw_dir import (  # noqa: E402
    adapter_object_provenance,
    align_object_to_interacting_hand,
    build_ego_aoe_camera_intrinsics_binding,
    infer_geometry_interaction_hand,
    initial_ego_mask_scale_fit_diagnostic,
    maybe_swap_hand_slots_for_visual_geometry,
    normalize_aoe_hand_camera_to_ego,
    replace_hand_geometry_with_ego,
    resolve_visual_bimanual_tie,
    run_ego_scale_optimization_diagnostic,
    source_layout_path,
)
import prepare_egoinfinity_do_as_i_do_raw_dir as ego_adapter_module  # noqa: E402
from prepare_do_as_i_do_scene_adapter import hand_projection_qc  # noqa: E402


class EgoSourceLayoutTest(unittest.TestCase):
    def test_camera_normalization_rewrites_hand_and_frame_intrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hand_npz = root / "all_hand_meshes.npz"
            points = np.array([[[0.1, -0.03, 0.6]]], dtype=np.float32)
            payload = {}
            for side in ("left", "right"):
                payload[f"{side}_vertices"] = points.copy()
                payload[f"{side}_joints"] = points.copy()
                payload[f"{side}_trans"] = points[:, 0, :].copy()
            np.savez(hand_npz, **payload)
            source_frames = root / "source_frames"
            source_frames.mkdir()
            ego_adapter_module.cv2.imwrite(
                str(source_frames / "000000.png"),
                np.zeros((1080, 1920, 3), dtype=np.uint8),
            )
            np.save(source_frames / "000000_intrinsics.npy", np.eye(3))
            frames = root / "all_frames"
            frames.symlink_to(source_frames, target_is_directory=True)
            object_camera = {
                "intrinsics": {
                    "width": 1920, "height": 1080,
                    "fx": 962.0, "fy": 962.0, "cx": 960.0, "cy": 540.0,
                }
            }
            hand_camera = {
                "intrinsics": {
                    "width": 1920, "height": 1080,
                    "fx": 788.0, "fy": 791.0, "cx": 961.8, "cy": 539.9,
                }
            }
            binding = ego_adapter_module.build_camera_intrinsics_binding(
                object_camera, hand_camera
            )
            self.assertEqual(binding["status"], "invalid")
            resolved = normalize_aoe_hand_camera_to_ego(hand_npz, binding, frames)
            self.assertEqual(resolved["status"], "ok")
            self.assertTrue(resolved["geometry_modified"])
            self.assertLess(
                resolved["normalization"]["reprojection_error_px_max"], 1e-3
            )
            self.assertFalse(frames.is_symlink())
            target_k = np.load(frames / "000000_intrinsics.npy")
            self.assertEqual(target_k[0, 0], 962.0)
            with np.load(hand_npz, allow_pickle=False) as normalized:
                self.assertIn("camera_intrinsics_normalization_matrix", normalized)
                self.assertEqual(normalized["left_vertices"][0, 0, 2], points[0, 0, 2])

    def test_scale_optimizer_candidate_is_diagnostic_and_preserves_production_scale_source(self):
        optimizer_outcome = {
            "requested": True,
            "status": "diagnostic_candidate_generated",
            "succeeded": True,
            "candidate_generated": True,
            "applied": False,
            "method": "unit_test_optimizer",
            "fallback": False,
            "reason": None,
            "returncode": 0,
            "output": "/tmp/production_layout.json",
            "mesh_scale": 1.0,
            "candidate_output": "/tmp/candidate_layout.json",
            "candidate_mesh_scale": 1.75,
        }
        with mock.patch.object(
            ego_adapter_module,
            "run_scale_optimization",
            return_value=optimizer_outcome,
        ):
            diagnostic = run_ego_scale_optimization_diagnostic(
                requested=True,
                python_bin=sys.executable,
                prepared_dir=Path("/tmp/prepared"),
                object_id="object",
                anchor_hand="left",
                ref_frame=3,
                viz_dir=None,
                production_mesh_scale=0.42,
                production_mesh_scale_source="pose_track_info.scale_correction",
            )

        self.assertFalse(diagnostic["applied"])
        self.assertTrue(diagnostic["diagnostic_only"])
        self.assertEqual(diagnostic["production_mesh_scale"], 0.42)
        self.assertEqual(
            diagnostic["production_mesh_scale_source"],
            "pose_track_info.scale_correction",
        )
        self.assertEqual(diagnostic["candidate_mesh_scale"], 1.75)
        self.assertEqual(
            diagnostic["optimizer_outcome"]["candidate_output"],
            "/tmp/candidate_layout.json",
        )
        skipped_fit = initial_ego_mask_scale_fit_diagnostic(
            object_geometry_source="ego",
            optimize_scale=True,
            no_fit_object_scale_to_ego_mask=False,
        )
        self.assertEqual(
            skipped_fit["reason"],
            "skipped_for_scale_optimization_diagnostic",
        )
        self.assertNotEqual(skipped_fit["reason"], "non_ego_object_geometry")

    def test_shared_se3_updates_complete_mano_global_pose_and_geometry(self):
        frames = 3
        angle = 0.63
        common_rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        common_translation = np.array([0.31, -0.27, 0.14], dtype=np.float64)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = common_rotation
        transform[:3, 3] = common_translation

        left_rot = np.array(
            [[0.13, -0.21, 0.08], [0.0, 0.3, -0.1], [-0.2, 0.04, 0.17]],
            dtype=np.float32,
        )
        right_rot = np.array(
            [[-0.07, 0.12, 0.19], [0.2, -0.11, 0.03], [0.09, 0.06, -0.15]],
            dtype=np.float32,
        )
        left_trans = np.array(
            [[0.1, 0.2, 0.8], [0.12, 0.18, 0.82], [0.14, 0.16, 0.84]],
            dtype=np.float32,
        )
        right_trans = left_trans + np.array([0.22, -0.08, 0.03], dtype=np.float32)
        left_vertices = np.stack(
            [np.full((5, 3), 0.02 * frame, dtype=np.float32) for frame in range(frames)]
        )
        right_vertices = left_vertices + np.array([0.18, 0.02, 0.01], dtype=np.float32)
        left_joints = np.stack(
            [np.full((21, 3), 0.03 * frame, dtype=np.float32) for frame in range(frames)]
        )
        right_joints = left_joints + np.array([0.16, -0.01, 0.02], dtype=np.float32)
        result = {
            "frame_data": [
                {
                    "vertices_3d": [left_vertices[frame], right_vertices[frame]],
                    "joints_3d_pred": [left_joints[frame], right_joints[frame]],
                    "joints_2d_pred": [
                        np.zeros((21, 2), dtype=np.float32),
                        np.zeros((21, 2), dtype=np.float32),
                    ],
                    "hand_is_right": [False, True],
                }
                for frame in range(frames)
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            hand_npz = Path(tmp) / "hands.npz"
            hand_pose = np.arange(frames * 45, dtype=np.float32).reshape(frames, 45)
            np.savez(
                hand_npz,
                left_vertices=np.zeros_like(left_vertices),
                right_vertices=np.zeros_like(right_vertices),
                left_joints=np.zeros_like(left_joints),
                right_joints=np.zeros_like(right_joints),
                left_rot=left_rot,
                right_rot=right_rot,
                left_trans=left_trans,
                right_trans=right_trans,
                left_hand_pose=hand_pose,
                right_hand_pose=hand_pose + 1.0,
            )
            report = replace_hand_geometry_with_ego(
                result, hand_npz, alignment_transform=transform
            )
            with np.load(hand_npz, allow_pickle=False) as transformed:
                left_rot_after = ego_adapter_module._mano_rotvecs_to_matrices(
                    transformed["left_rot"], field="left_rot"
                )
                right_rot_after = ego_adapter_module._mano_rotvecs_to_matrices(
                    transformed["right_rot"], field="right_rot"
                )
                left_rot_before = ego_adapter_module._mano_rotvecs_to_matrices(
                    left_rot, field="left_rot"
                )
                right_rot_before = ego_adapter_module._mano_rotvecs_to_matrices(
                    right_rot, field="right_rot"
                )
                np.testing.assert_allclose(
                    left_rot_after,
                    common_rotation @ left_rot_before,
                    atol=2e-6,
                )
                np.testing.assert_allclose(
                    right_rot_after,
                    common_rotation @ right_rot_before,
                    atol=2e-6,
                )
                np.testing.assert_allclose(
                    transformed["left_trans"],
                    left_trans @ common_rotation.T + common_translation,
                    atol=2e-7,
                )
                np.testing.assert_allclose(
                    transformed["right_trans"],
                    right_trans @ common_rotation.T + common_translation,
                    atol=2e-7,
                )
                np.testing.assert_allclose(
                    transformed["left_joints"],
                    left_joints @ common_rotation.T + common_translation,
                    atol=2e-7,
                )
                np.testing.assert_allclose(
                    transformed["right_joints"],
                    right_joints @ common_rotation.T + common_translation,
                    atol=2e-7,
                )
                np.testing.assert_allclose(
                    transformed["left_vertices"],
                    left_vertices @ common_rotation.T + common_translation,
                    atol=2e-7,
                )
                np.testing.assert_allclose(transformed["left_hand_pose"], hand_pose)

        pose_report = report["mano_global_pose_transform"]
        self.assertEqual(pose_report["status"], "ok")
        self.assertTrue(pose_report["applied"])
        self.assertEqual(
            pose_report["rotation_expected_relation"],
            "R_after = R_common @ R_before",
        )
        self.assertEqual(
            set(pose_report["transformed_fields"]),
            {"left_rot", "right_rot", "left_trans", "right_trans"},
        )
        self.assertLess(
            pose_report["sides"]["left"]["rotation_matrix_max_abs_error"],
            2e-6,
        )

    def test_shared_se3_rejects_malformed_or_nonfinite_mano_global_pose(self):
        transform = np.eye(4, dtype=np.float64)
        transform[0, 3] = 0.1
        base = {
            "left_joints": np.zeros((2, 21, 3), dtype=np.float32),
            "left_vertices": np.zeros((2, 5, 3), dtype=np.float32),
            "left_rot": np.zeros((2, 3), dtype=np.float32),
            "left_trans": np.zeros((2, 3), dtype=np.float32),
        }
        malformed = dict(base)
        malformed["left_rot"] = np.zeros((2, 1, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "left_rot must have shape"):
            ego_adapter_module.transform_mano_global_pose_fields(
                malformed, transform, frame_count=2
            )
        nonfinite = dict(base)
        nonfinite["left_trans"] = np.array(
            [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]], dtype=np.float32
        )
        with self.assertRaisesRegex(ValueError, "left_trans contains non-finite"):
            ego_adapter_module.transform_mano_global_pose_fields(
                nonfinite, transform, frame_count=2
            )

    def test_surface_geometry_records_actual_per_side_coverage_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout_path = root / "layout.json"
            layout_path.write_text(
                json.dumps(
                    {
                        "objects": [
                            {
                                "frame_idx": frame_id,
                                "local_to_scene": {
                                    "translation_camera_frame": [0.0, 0.0, 1.0],
                                    "quat_wxyz_camera_frame": [1.0, 0.0, 0.0, 0.0],
                                    "scale": [1.0, 1.0, 1.0],
                                },
                            }
                            for frame_id in range(3)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            mesh_dir = (
                root
                / "video_segmentation/masks/frame_000000_masks/object"
            )
            mesh_dir.mkdir(parents=True)
            (mesh_dir / "object.obj").write_text(
                "v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\nv 0 0 0.01\n"
                "f 1 2 3\nf 1 2 4\n",
                encoding="utf-8",
            )
            hand_vertices = np.repeat(
                np.array(
                    [[[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.0, 0.01, 1.0], [0.0, 0.0, 1.01]]]
                ),
                3,
                axis=0,
            )
            hand_npz = root / "hands.npz"
            np.savez(
                hand_npz,
                left_vertices=hand_vertices,
                right_vertices=hand_vertices,
                left_valid=np.array([1.0, 0.0, 1.0]),
                right_valid=np.ones(3),
            )

            report = infer_geometry_interaction_hand(
                hand_npz,
                layout_path,
                root,
                "object",
                distance_thresh=0.05,
                max_sample_frames=None,
            )

        self.assertEqual(
            report["distance_method"],
            "subsampled_mesh_vertex_to_vertex_minimum_conservative",
        )
        left = report["coverage_by_hand"]["left"]
        self.assertEqual(left["planned_frame_count"], 3)
        self.assertEqual(left["evaluated_frame_count"], 2)
        self.assertEqual(left["failed_frame_count"], 1)
        self.assertEqual(left["failure_reasons"], {"invalid_hand_frame": 1})
        self.assertFalse(left["complete_source_coverage"])
        self.assertTrue(report["coverage_by_hand"]["right"]["complete_source_coverage"])

    def test_projection_records_positive_depth_against_all_vertices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames_dir = root / "all_frames"
            frames_dir.mkdir()
            vertices = np.zeros((3, 10, 3), dtype=np.float64)
            vertices[..., 2] = -1.0
            vertices[:, 0] = [0.0, 0.0, 1.0]
            hand_npz = root / "hands.npz"
            np.savez(
                hand_npz,
                left_vertices=vertices,
                left_valid=np.ones(3, dtype=np.float64),
            )
            intrinsics = np.array(
                [[100.0, 0.0, 5.0], [0.0, 100.0, 5.0], [0.0, 0.0, 1.0]]
            )
            for frame_id in range(3):
                np.save(
                    frames_dir / f"{frame_id:06d}_intrinsics.npy",
                    intrinsics,
                )
                np.save(
                    frames_dir / f"{frame_id:06d}_pointmap.npy",
                    np.zeros((10, 10, 3), dtype=np.float32),
                )

            report = hand_projection_qc(hand_npz, frames_dir, "left")

        self.assertTrue(report["complete_sample_coverage"])
        self.assertEqual(report["available_frame_count"], 3)
        for frame in report["frames"]:
            self.assertEqual(frame["positive_depth_vertex_count"], 1)
            self.assertAlmostEqual(frame["positive_depth_fraction"], 0.1)
            self.assertAlmostEqual(frame["inside_fraction"], 0.1)
            self.assertAlmostEqual(frame["inside_fraction_of_positive_depth"], 1.0)

    def test_ego_aoe_camera_binding_traces_calibration_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ego_process = root / "segment" / "ego_process"
            hawor_dir = ego_process / "ego_hands_reconstruction"
            calibration_dir = ego_process / "ego_undistorted_video"
            hawor_dir.mkdir(parents=True)
            calibration_dir.mkdir(parents=True)
            source_hands = hawor_dir / "hands.npz"
            np.savez(source_hands, focal=np.array(788.0))
            adapted_hands = root / "all_hand_meshes.npz"
            np.savez(adapted_hands, source_hands_npz=np.array(str(source_hands)))
            (calibration_dir / "undistorted_video_info.json").write_text(
                json.dumps(
                    {
                        "cameraParams": {
                            "resolution": "1920x1080",
                            "fx_pixels": 788.0,
                            "fy_pixels": 791.0,
                            "cx_pixels": 960.0,
                            "cy_pixels": 540.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            result = {
                "dp_focal": 1077.0,
                "cx": 960.0,
                "cy": 540.0,
                "bg_rgb": np.zeros((1080, 1920, 3), dtype=np.uint8),
            }
            pipeline_result = root / "pipeline_result.pkl.gz"
            with gzip.open(pipeline_result, "wb") as handle:
                pickle.dump(result, handle)

            binding = build_ego_aoe_camera_intrinsics_binding(
                result, pipeline_result, adapted_hands
            )

        self.assertEqual(binding["status"], "invalid")
        self.assertIn("camera_intrinsics_mismatch", binding["errors"])
        self.assertEqual(
            Path(binding["hand_camera"]["source"]["source_hands_npz"]),
            source_hands.resolve(),
        )
        self.assertEqual(
            Path(binding["hand_camera"]["source"]["calibration_path"]),
            (calibration_dir / "undistorted_video_info.json").resolve(),
        )
        self.assertFalse(binding["geometry_modified"])
        self.assertEqual(
            len(
                binding["object_camera"]["source"]["file_bindings"]
                ["pipeline_result"]["sha256"]
            ),
            64,
        )
        for key in ("adapted_hand_npz", "source_hands_npz", "calibration"):
            self.assertEqual(
                len(binding["hand_camera"]["source"]["file_bindings"][key]["sha256"]),
                64,
            )

    def test_hand_slot_disagreement_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            hand_npz = Path(tmp) / "hands.npz"
            np.savez(
                hand_npz,
                left_joints=np.zeros((2, 21, 3), dtype=np.float32),
                right_joints=np.ones((2, 21, 3), dtype=np.float32),
            )
            before = hand_npz.read_bytes()
            with mock.patch.object(
                ego_adapter_module, "strong_visual_single_hand", return_value="left"
            ), mock.patch.object(
                ego_adapter_module, "strong_geometry_single_hand", return_value="right"
            ):
                report = maybe_swap_hand_slots_for_visual_geometry(
                    hand_npz, {}, {}
                )

            self.assertFalse(report["applied"])
            self.assertTrue(report["would_apply"])
            self.assertTrue(report["diagnostic_only"])
            self.assertFalse(report["hand_npz_modified"])
            self.assertEqual(hand_npz.read_bytes(), before)

    def test_production_wrapper_keeps_ego_object_in_camera_frame(self):
        wrapper = (ROOT / "scripts" / "run_v4_two_full_pipelines.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--source-layout-alignment none", wrapper)
        self.assertIn('source_layout_mode="none"', wrapper)
        self.assertIn('--source-layout-alignment "$source_layout_mode"', wrapper)
        self.assertNotIn("--hoi-contact-alignment auto", wrapper)
        self.assertIn("--hoi-contact-alignment diagnostic", wrapper)
        self.assertIn('--hand-source "$hand_source"', wrapper)
        self.assertGreaterEqual(
            wrapper.count('--source-dir "$dai_native_source_raw_dir"'),
            1,
        )
        self.assertIn('adapter_source_dir="$dai_native_source_raw_dir"', wrapper)
        self.assertNotIn("dai_native_contact_aligned_raw_dir", wrapper)
        self.assertGreaterEqual(wrapper.count('retarget_hand_request="bimanual"'), 2)
        self.assertGreaterEqual(
            wrapper.count('--retarget-hand "$retarget_hand_request"'),
            2,
        )

    def test_bimanual_contact_alignment_uses_both_hand_meshes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hand_npz = root / "hands.npz"
            left = np.array(
                [[[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]] * 2,
                dtype=np.float64,
            )
            right = left + np.array([0.0, 0.1, 0.0])
            np.savez(
                hand_npz,
                left_vertices=left,
                right_vertices=right,
                left_valid=np.ones(2),
                right_valid=np.ones(2),
            )
            mesh = root / "object.obj"
            mesh.write_text(
                "v 0.2 0 0\nv 0.2 0.01 0\nv 0.2 0 0.01\nf 1 2 3\n",
                encoding="utf-8",
            )
            transforms = np.repeat(np.eye(4)[None, ...], 2, axis=0)
            aligned, report = align_object_to_interacting_hand(
                transforms,
                hand_npz,
                mesh,
                1.0,
                "bimanual",
                {
                    "available": True,
                    "scores": {
                        "left": {"overlap_frames": 5},
                        "right": {"overlap_frames": 5},
                    },
                },
                "force",
                0.25,
                0.0,
                0.3,
            )
            self.assertFalse(report["applied"])
            self.assertTrue(report["diagnostic_only"])
            self.assertTrue(report["would_apply"])
            self.assertEqual(report["selected_hand"], "bimanual")
            self.assertEqual(report["frames_used"], 2)
            np.testing.assert_array_equal(aligned, transforms)

    def test_auto_alignment_closes_gap_above_contact_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hand_npz = root / "hands.npz"
            hand = np.array(
                [[[0.0, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]] * 4,
                dtype=np.float64,
            )
            np.savez(hand_npz, left_vertices=hand, left_valid=np.ones(4))
            mesh = root / "object.obj"
            mesh.write_text(
                "v 0.12 0 0\nv 0.13 0 0\nv 0.12 0.01 0\nf 1 2 3\n",
                encoding="utf-8",
            )
            transforms = np.repeat(np.eye(4)[None, ...], 4, axis=0)
            aligned, report = align_object_to_interacting_hand(
                transforms,
                hand_npz,
                mesh,
                1.0,
                "left",
                {"available": True, "scores": {"left": {"overlap_frames": 4}}},
                "auto",
                0.075,
                0.025,
                0.30,
            )
            self.assertFalse(report["applied"])
            self.assertTrue(report["would_apply"])
            self.assertTrue(report["diagnostic_only"])
            self.assertGreater(report["before"]["mean"], 0.075)
            self.assertLess(report["hypothetical_after"]["mean"], 0.05)
            np.testing.assert_array_equal(aligned, transforms)

    def test_single_hand_task_breaks_visual_bimanual_tie_with_evidence(self):
        visual = {
            "scores": {
                "left": {"overlap_frames": 13.0},
                "right": {"overlap_frames": 5.0},
            },
            "normalized": {"left": 0.491, "right": 0.410},
        }
        self.assertEqual(resolve_visual_bimanual_tie("bimanual", visual, "left"), "left")

    def test_single_hand_task_does_not_override_weak_visual_evidence(self):
        visual = {
            "scores": {
                "left": {"overlap_frames": 2.0},
                "right": {"overlap_frames": 12.0},
            },
            "normalized": {"left": 0.1, "right": 0.6},
        }
        self.assertEqual(
            resolve_visual_bimanual_tie("bimanual", visual, "left"),
            "bimanual",
        )

    def test_bimanual_task_remains_bimanual(self):
        visual = {
            "scores": {"left": {"overlap_frames": 10.0}},
            "normalized": {"left": 0.5, "right": 0.4},
        }
        self.assertEqual(resolve_visual_bimanual_tie("bimanual", visual, None), "bimanual")

    def make_layouts(self, root: Path) -> tuple[Path, Path]:
        layout_dir = root / "obj_tracking_out" / "case" / "combined_visualization"
        layout_dir.mkdir(parents=True)
        clean = layout_dir / "layout_camera_frame.json"
        optimized = layout_dir / "layout_camera_frame_optimized.json"
        clean.write_text("{}", encoding="utf-8")
        optimized.write_text("{}", encoding="utf-8")
        return clean, optimized

    def test_legacy_contact_aligned_native_adapter_keeps_clean_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            clean, _ = self.make_layouts(source)
            (source / "adapter_manifest.json").write_text(
                json.dumps({"retarget_object_source": "dai_native_contact_aligned"}),
                encoding="utf-8",
            )
            self.assertEqual(source_layout_path(source, "case"), clean)

    def test_untrusted_adapter_keeps_clean_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            clean, _ = self.make_layouts(source)
            (source / "adapter_manifest.json").write_text(
                json.dumps({"retarget_object_source": "dai_native"}),
                encoding="utf-8",
            )
            self.assertEqual(source_layout_path(source, "case"), clean)

    def test_legacy_source_without_manifest_keeps_clean_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            clean, _ = self.make_layouts(source)
            self.assertEqual(source_layout_path(source, "case"), clean)

    def test_ego_adapter_provenance_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                adapter_object_provenance(Path(tmp), "ego"),
                {
                    "object_track_source": "egoinfinity",
                    "object_mesh_source": "egoinfinity",
                    "retarget_object_source": "egoinfinity",
                },
            )

    def test_source_geometry_rejects_legacy_contact_alignment_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / "adapter_manifest.json").write_text(
                json.dumps(
                    {
                        "object_track_source": "dai_native",
                        "object_mesh_source": "dai_native",
                        "retarget_object_source": "dai_native_contact_aligned",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                adapter_object_provenance(source, "source"),
                {
                    "object_track_source": "dai_native",
                    "object_mesh_source": "dai_native",
                    "retarget_object_source": "dai_native",
                },
            )


if __name__ == "__main__":
    unittest.main()
