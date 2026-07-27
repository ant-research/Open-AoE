from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# The layout/invariance unit tests do not exercise mesh I/O. Keep them runnable
# in the lightweight local test environment where trimesh is optional.
try:
    import trimesh  # noqa: F401
except ModuleNotFoundError:
    sys.modules["trimesh"] = types.ModuleType("trimesh")

from prepare_do_as_i_do_scene_adapter import (  # noqa: E402
    layout_object_pose_sequence as dai_layout_object_pose_sequence,
    link_tracking_assets as link_dai_tracking_assets,
    run_scale_optimization,
    scale_refinement_provenance,
)
import prepare_do_as_i_do_scene_adapter as dai_adapter_module  # noqa: E402
from prepare_egoinfinity_do_as_i_do_raw_dir import (  # noqa: E402
    copy_tracking_assets_for_adapter,
    layout_object_pose_sequence as ego_layout_object_pose_sequence,
    summarize_source_object_layout_invariance,
)
from aoe_retarget_lab.hoi_geometry import summarize_adapter_rigid_invariance  # noqa: E402


def make_layout(frames: int, offset: np.ndarray | None = None) -> dict:
    delta = np.zeros(3, dtype=np.float64) if offset is None else np.asarray(offset, dtype=np.float64)
    objects = []
    for frame in range(frames):
        angle = 0.04 * frame
        objects.append(
            {
                "index": frame,
                "frame_idx": frame,
                "local_to_scene": {
                    "translation_camera_frame": (
                        np.asarray([0.2 + 0.01 * frame, -0.1, 0.7]) + delta
                    ).tolist(),
                    "quat_wxyz_camera_frame": [
                        float(np.cos(angle / 2.0)),
                        0.0,
                        0.0,
                        float(np.sin(angle / 2.0)),
                    ],
                },
            }
        )
    return {"objects": objects}


def make_hands(frames: int) -> dict[str, np.ndarray]:
    joints = np.zeros((frames, 21, 3), dtype=np.float64)
    joints[..., 0] = np.linspace(0.03, 0.19, 21)
    joints[..., 1] = np.linspace(-0.08, 0.11, 21)
    joints[..., 2] = 0.66
    joints[:, :, 0] += np.arange(frames, dtype=np.float64)[:, None] * 0.005
    return {
        "right_joints": joints,
        "right_valid": np.ones(frames, dtype=np.float32),
    }


class AdapterRigidInvarianceContractTest(unittest.TestCase):
    def test_failed_scale_optimizer_restores_clean_layout_and_never_claims_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout_dir = root / "obj_tracking_out/object/combined_visualization"
            layout_dir.mkdir(parents=True)
            clean = make_layout(3)
            clean_path = layout_dir / "layout_camera_frame.json"
            clean_path.write_text(json.dumps(clean), encoding="utf-8")
            mesh = root / "object.obj"
            mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
            with mock.patch.object(
                dai_adapter_module, "require", return_value=None
            ), mock.patch.object(
                dai_adapter_module, "choose_object_mesh", return_value=mesh
            ), mock.patch.object(
                dai_adapter_module.subprocess,
                "run",
                side_effect=dai_adapter_module.subprocess.CalledProcessError(
                    17, ["optimizer"]
                ),
            ):
                outcome = run_scale_optimization(
                    sys.executable, root, "object", "right", 0, None
                )
            production = json.loads(
                (layout_dir / "layout_camera_frame_optimized.json").read_text(
                    encoding="utf-8"
                )
            )
            provenance = scale_refinement_provenance(
                requested=True,
                optimization=outcome,
                contact_diagnostic={"applied": False},
            )

        self.assertEqual(production, clean)
        self.assertEqual(outcome["status"], "fallback_unoptimized")
        self.assertFalse(outcome["succeeded"])
        self.assertFalse(outcome["applied"])
        self.assertTrue(outcome["fallback"])
        self.assertFalse(provenance["applied"])
        self.assertEqual(provenance["method"], "none")
        self.assertTrue(provenance["fallback"])

    def test_successful_scale_optimizer_is_saved_only_as_diagnostic_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout_dir = root / "obj_tracking_out/object/combined_visualization"
            layout_dir.mkdir(parents=True)
            clean = make_layout(3)
            candidate = make_layout(3, np.array([0.4, -0.2, 0.1]))
            candidate["translation_scale_optimization"] = {
                "method": "unit_test_optimizer",
                "mesh_scale": 1.7,
            }
            clean_path = layout_dir / "layout_camera_frame.json"
            clean_path.write_text(json.dumps(clean), encoding="utf-8")
            mesh = root / "object.obj"
            mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")

            def write_candidate(*_args, **_kwargs):
                (layout_dir / "layout_camera_frame_optimized.json").write_text(
                    json.dumps(candidate), encoding="utf-8"
                )
                return mock.Mock(returncode=0)

            with mock.patch.object(
                dai_adapter_module, "require", return_value=None
            ), mock.patch.object(
                dai_adapter_module, "choose_object_mesh", return_value=mesh
            ), mock.patch.object(
                dai_adapter_module.subprocess,
                "run",
                side_effect=write_candidate,
            ):
                outcome = run_scale_optimization(
                    sys.executable, root, "object", "right", 0, None
                )
            production = json.loads(
                (layout_dir / "layout_camera_frame_optimized.json").read_text(
                    encoding="utf-8"
                )
            )
            candidate_saved = json.loads(
                Path(outcome["candidate_output"]).read_text(encoding="utf-8")
            )
            provenance = scale_refinement_provenance(
                requested=True,
                optimization=outcome,
                contact_diagnostic={"applied": False},
            )

        self.assertEqual(production, clean)
        self.assertEqual(candidate_saved, candidate)
        self.assertTrue(outcome["succeeded"])
        self.assertTrue(outcome["candidate_generated"])
        self.assertFalse(outcome["applied"])
        self.assertEqual(outcome["candidate_mesh_scale"], 1.7)
        self.assertFalse(provenance["applied"])
        self.assertEqual(provenance["method"], "none")
        self.assertEqual(
            provenance["diagnostic_candidate_method"], "unit_test_optimizer"
        )

    def test_both_adapters_parse_identical_layout_poses(self):
        layout = make_layout(6)
        np.testing.assert_allclose(
            dai_layout_object_pose_sequence(layout),
            ego_layout_object_pose_sequence(layout),
            atol=1e-14,
        )

    def test_measured_identity_adapter_is_ok(self):
        frames = 7
        poses = dai_layout_object_pose_sequence(make_layout(frames))
        result = summarize_adapter_rigid_invariance(
            make_hands(frames),
            make_hands(frames),
            before_object_poses=poses,
            after_object_poses=poses.copy(),
            canonical_transform=None,
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["errors"], [])
        self.assertLess(result["max"], 1e-14)

    def test_object_only_layout_translation_fails_measured_invariance(self):
        frames = 7
        before = ego_layout_object_pose_sequence(make_layout(frames))
        after = ego_layout_object_pose_sequence(
            make_layout(frames, np.asarray([0.16, -0.04, 0.02]))
        )
        result = summarize_adapter_rigid_invariance(
            make_hands(frames),
            make_hands(frames),
            before_object_poses=before,
            after_object_poses=after,
            canonical_transform=None,
        )

        self.assertEqual(result["status"], "invalid")
        self.assertGreater(result["max"], 1e-3)
        self.assertIn(
            "relative_distance_max_error_exceeds_tolerance",
            result["errors"],
        )

    def test_identity_adapters_replace_stale_optimized_layout_with_clean_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            layout_dir = source / "obj_tracking_out/object/combined_visualization"
            layout_dir.mkdir(parents=True)
            clean = make_layout(3)
            stale = make_layout(3, np.asarray([0.2, 0.0, 0.0]))
            (layout_dir / "layout_camera_frame.json").write_text(
                json.dumps(clean), encoding="utf-8"
            )
            (layout_dir / "layout_camera_frame_optimized.json").write_text(
                json.dumps(stale), encoding="utf-8"
            )

            dai_output = root / "dai_output"
            link_dai_tracking_assets(source, dai_output, "object", optimize_scale=False)
            dai_final = json.loads(
                (
                    dai_output
                    / "obj_tracking_out/object/combined_visualization/layout_camera_frame_optimized.json"
                ).read_text(encoding="utf-8")
            )

            ego_output = root / "ego_output"
            copy_tracking_assets_for_adapter(
                source, ego_output, "object", optimize_scale=False
            )
            ego_final = json.loads(
                (
                    ego_output
                    / "obj_tracking_out/object/combined_visualization/layout_camera_frame_optimized.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(dai_final, clean)
        self.assertEqual(ego_final, clean)

    def test_source_object_measurement_stays_in_one_source_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before_path = root / "before.json"
            after_path = root / "after.json"
            hand_path = root / "post_adapter_hands.npz"
            before_path.write_text(json.dumps(make_layout(5)), encoding="utf-8")
            after_path.write_text(json.dumps(make_layout(5)), encoding="utf-8")
            np.savez(hand_path, **make_hands(5))

            result = summarize_source_object_layout_invariance(
                hand_path, hand_path, before_path, after_path
            )

        self.assertEqual(result["status"], "ok")
        self.assertFalse(result["canonical_transform_applied"])
        self.assertEqual(
            result["comparison_semantics"],
            "true_pre_adapter_vs_post_adapter_hand_and_object",
        )
        self.assertEqual(
            result["hand_coordinate_frame"],
            "explicit_adapter_boundary_frames",
        )

    def test_scene_adapter_writes_measured_exact_route_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "all_frames").mkdir(parents=True)
            (source / "raw").mkdir()
            (source / "raw.mp4").write_bytes(b"")
            (source / "config.json").write_text(
                json.dumps(
                    {
                        "object_names": ["object"],
                        "anchor_hand": "right",
                        "frame_number": 0,
                    }
                ),
                encoding="utf-8",
            )
            (source / "gravity.json").write_text("{}", encoding="utf-8")
            layout_dir = source / "obj_tracking_out/object/combined_visualization"
            layout_dir.mkdir(parents=True)
            (layout_dir / "layout_camera_frame.json").write_text(
                json.dumps(make_layout(4)), encoding="utf-8"
            )
            mesh = source / "video_segmentation/masks/frame_000000_masks/object/object.obj"
            mesh.parent.mkdir(parents=True)
            mesh.write_text(
                "v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\nf 1 2 3\n",
                encoding="utf-8",
            )
            hands = make_hands(4)
            vertices = np.repeat(
                hands["right_joints"][:, :1], 746, axis=1
            )
            np.savez(
                source / "raw/all_hand_meshes.npz",
                **hands,
                right_vertices=vertices,
            )

            manifests = {}
            for hand_source in ("aoe", "estimated"):
                output = root / f"output_{hand_source}"
                argv = [
                    "prepare_do_as_i_do_scene_adapter.py",
                    "--source-dir",
                    str(source),
                    "--output-dir",
                    str(output),
                    "--anchor-hand",
                    "right",
                    "--hand-source",
                    hand_source,
                    "--no-optimize-scale",
                    "--hoi-contact-alignment",
                    "none",
                ]
                with mock.patch.object(
                    dai_adapter_module,
                    "score_hand_sides",
                    return_value={
                        "mask_scores": {"right": {"overlap_frames": 8}}
                    },
                ), mock.patch.object(
                    dai_adapter_module,
                    "hand_projection_qc",
                    return_value={"available": False, "reason": "unit_test"},
                ), mock.patch.object(
                    dai_adapter_module,
                    "generate_hand_masks",
                    return_value={"sides": {}},
                ), mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    return_code = dai_adapter_module.main()
                self.assertEqual(return_code, 0, hand_source)
                manifests[hand_source] = json.loads(
                    (output / "adapter_manifest.json").read_text(encoding="utf-8")
                )

        for hand_source, manifest in manifests.items():
            self.assertEqual(manifest["hand_source"], hand_source)
            self.assertEqual(manifest["adapter_rigid_invariance"]["status"], "ok")
            self.assertTrue(
                manifest["adapter_rigid_invariance"][
                    "true_pre_post_hand_comparison"
                ]
            )
            self.assertGreater(manifest["adapter_rigid_invariance"]["samples"], 0)
            self.assertFalse(manifest["hoi_refinement"]["applied"])
            self.assertEqual(manifest["object_track_source"], "dai_native")

    def test_production_wrappers_pass_exact_pure_dai_hand_sources(self):
        v4 = (ROOT / "scripts" / "run_v4_two_full_pipelines.sh").read_text(
            encoding="utf-8"
        )
        full = (ROOT / "scripts" / "run_full_12_demos.sh").read_text(
            encoding="utf-8"
        )
        adapter = (ROOT / "scripts" / "prepare_do_as_i_do_scene_adapter.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('"--hand-source",', adapter)
        self.assertIn('choices=["aoe", "estimated", "unknown"]', adapter)
        self.assertIn("--hand-source aoe", v4)
        self.assertIn('--hand-source "$hand_source"', v4)
        self.assertIn("--no-optimize-scale", v4)
        self.assertIn("hand_only_canonical_fusion_forbidden", v4)
        self.assertIn("--hand-source aoe", full)
        self.assertIn("adapter_cmd+=(--no-optimize-scale)", full)
        self.assertIn("scale_refinement_provenance", adapter)
        self.assertIn("object-only scale optimization cannot modify production HOI", adapter)


if __name__ == "__main__":
    unittest.main()
