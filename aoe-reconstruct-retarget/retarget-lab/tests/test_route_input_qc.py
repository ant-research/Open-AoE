from __future__ import annotations

import gzip
import json
import pickle
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from aoe_retarget_lab.route_input_qc import (
    SPIDER_CORE_HOI_ARRAYS,
    build_camera_intrinsics_binding,
    camera_intrinsics_fingerprint,
    camera_ray_depth_transform,
    expected_cell_key,
    file_sha256,
    validate_route_input,
    validate_spider_postprocess_semantics,
)


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


class CameraRayDepthTransformTest(unittest.TestCase):
    def test_preserves_pixels_and_depth(self):
        source = {
            "width": 1920, "height": 1080,
            "fx": 787.8279111480762, "fy": 791.2080732719322,
            "cx": 961.8658327877191, "cy": 539.9728967260908,
        }
        target = {
            "width": 1920, "height": 1080,
            "fx": 962.0146214962006, "fy": 962.0146214962006,
            "cx": 960.0, "cy": 540.0,
        }
        points = np.array([[0.1, -0.03, 0.6], [-0.2, 0.15, 1.2]])
        matrix = camera_ray_depth_transform(source, target)
        converted = np.einsum("ij,nj->ni", matrix, points)
        source_uv = np.column_stack(
            [source["fx"] * points[:, 0] / points[:, 2] + source["cx"],
             source["fy"] * points[:, 1] / points[:, 2] + source["cy"]]
        )
        target_uv = np.column_stack(
            [target["fx"] * converted[:, 0] / converted[:, 2] + target["cx"],
             target["fy"] * converted[:, 1] / converted[:, 2] + target["cy"]]
        )
        np.testing.assert_allclose(target_uv, source_uv, rtol=0.0, atol=1e-10)
        np.testing.assert_array_equal(converted[:, 2], points[:, 2])


_CAMERA_FIXTURE_DIR = tempfile.TemporaryDirectory()
_CAMERA_BINDING: dict | None = None


def production_camera_binding() -> dict:
    global _CAMERA_BINDING
    if _CAMERA_BINDING is not None:
        return deepcopy(_CAMERA_BINDING)
    root = Path(_CAMERA_FIXTURE_DIR.name)
    pipeline_result = root / "pipeline_result.pkl.gz"
    result = {
        "dp_focal": 2.0,
        "cx": 2.0,
        "cy": 1.0,
        "bg_rgb": np.zeros((2, 4, 3), dtype=np.uint8),
    }
    with gzip.open(pipeline_result, "wb") as handle:
        pickle.dump(result, handle)
    source_hands = root / "ego_process/ego_hands_reconstruction/hands.npz"
    source_hands.parent.mkdir(parents=True)
    np.savez(source_hands, focal=np.asarray(2.0))
    adapted_hands = root / "adapted/all_hand_meshes.npz"
    adapted_hands.parent.mkdir(parents=True)
    np.savez(adapted_hands, source_hands_npz=np.asarray(str(source_hands)))
    calibration = (
        root / "ego_process/ego_undistorted_video/undistorted_video_info.json"
    )
    calibration.parent.mkdir(parents=True)
    calibration.write_text(
        json.dumps(
            {
                "cameraParams": {
                    "resolution": "4x2",
                    "fx_pixels": 2.0,
                    "fy_pixels": 2.0,
                    "cx_pixels": 2.0,
                    "cy_pixels": 1.0,
                }
            }
        ),
        encoding="utf-8",
    )

    def binding(path: Path) -> dict[str, str]:
        return {"path": str(path.resolve()), "sha256": file_sha256(path)}

    pipeline_binding = binding(pipeline_result)
    adapted_binding = binding(adapted_hands)
    source_binding = binding(source_hands)
    calibration_binding = binding(calibration)
    intrinsics = {
        "model": "pinhole",
        "width": 4,
        "height": 2,
        "fx": 2.0,
        "fy": 2.0,
        "cx": 2.0,
        "cy": 1.0,
    }
    object_camera = {
        "source": {
            "kind": "egoinfinity_pipeline_result",
            "path": pipeline_binding["path"],
            "sha256": pipeline_binding["sha256"],
            "file_bindings": {"pipeline_result": pipeline_binding},
        },
        "intrinsics": intrinsics,
    }
    hand_camera = {
        "source": {
            "kind": "aoe_hawor_calibrated_camera",
            "adapted_hand_npz": adapted_binding["path"],
            "adapted_hand_npz_sha256": adapted_binding["sha256"],
            "source_hands_npz": source_binding["path"],
            "source_hands_npz_sha256": source_binding["sha256"],
            "calibration_path": calibration_binding["path"],
            "calibration_sha256": calibration_binding["sha256"],
            "file_bindings": {
                "adapted_hand_npz": adapted_binding,
                "source_hands_npz": source_binding,
                "calibration": calibration_binding,
            },
        },
        "intrinsics": intrinsics,
    }
    _CAMERA_BINDING = build_camera_intrinsics_binding(object_camera, hand_camera)
    return deepcopy(_CAMERA_BINDING)


def valid_manifest(hand_source: str, trajectory: str = "egoinfinity") -> dict:
    object_source = "egoinfinity" if trajectory == "egoinfinity" else "dai_native"
    return {
        "adapter_schema_version": 3,
        "production_hoi_policy": "preserve_source_relative_transform",
        "hand_source": hand_source,
        "object_track_source": object_source,
        "object_mesh_source": object_source,
        "retarget_object_source": object_source,
        "canonical_transform": {
            "applied": False,
            "kind": "identity",
            "matrix": IDENTITY,
            "from_frame": "egoinfinity_camera",
            "to_frame": "egoinfinity_camera",
            "applies_to": [
                "hand_joints",
                "hand_vertices",
                "mano_global_orient",
                "mano_global_translation",
                "object_pose",
                "contact_points_generated_post_transform",
                "object_mesh_world_via_object_pose",
            ],
        },
        "hoi_refinement": {
            "applied": False,
            "method": "none",
        },
        "hoi_contact_alignment": {
            "applied": False,
            "diagnostic_only": True,
            "transforms_modified": False,
        },
        "adapter_rigid_invariance": {
            "status": "ok",
            "true_pre_post_hand_comparison": True,
            "wrist_object_max_abs_error": 2.0e-16,
            "fingertip_object_max_abs_error": 4.0e-16,
        },
        "retarget_input_qc": {"status": "ok", "errors": []},
        "camera_intrinsics_binding": production_camera_binding(),
    }


def with_valid_shared_se3_mano_evidence(manifest: dict, matrix: list[list[float]]) -> dict:
    result = deepcopy(manifest)
    result["canonical_transform"].update(
        {
            "applied": True,
            "kind": "shared_se3",
            "matrix": deepcopy(matrix),
            "to_frame": "do_as_i_do_source_camera",
        }
    )
    applies_to = result["canonical_transform"].setdefault("applies_to", [])
    for field in ("mano_global_orient", "mano_global_translation"):
        if field not in applies_to:
            applies_to.append(field)
    transformed_fields = [
        "left_rot",
        "left_trans",
        "right_rot",
        "right_trans",
    ]
    result["ego_hand_geometry"] = {
        "available": True,
        "mano_global_pose_transform": {
            "status": "ok",
            "applied": True,
            "frame_count": 4,
            "rotation_expected_relation": "R_after = R_common @ R_before",
            "translation_expected_relation": (
                "t_after = R_common @ t_before + t_common"
            ),
            "common_rotation": [row[:3] for row in matrix[:3]],
            "common_translation": [row[3] for row in matrix[:3]],
            "transformed_fields": transformed_fields,
            "sides": {
                side: {
                    "geometry_present": True,
                    "rotation_field": f"{side}_rot",
                    "translation_fields": [f"{side}_trans"],
                    "rotation_frame_count": 4,
                    "rotation_matrix_max_abs_error": 1.0e-8,
                    "translation_max_abs_error_m": 1.0e-9,
                }
                for side in ("left", "right")
            },
        },
    }
    return result


class RouteFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sources: dict[str, Path] = {}
        self.processed: dict[str, Path] = {}
        self.source_meshes: dict[str, Path] = {}
        self.processed_meshes: dict[str, Path] = {}
        self.manifests: dict[str, Path] = {}
        for hand_source, payload in (("aoe", b"aoe-keypoints"), ("estimated", b"estimated-keypoints")):
            dai_cell = expected_cell_key("egoinfinity", hand_source)
            source = (
                root
                / "intermediates"
                / "retargeting"
                / "do_as_i_do"
                / dai_cell
                / "retargeting_outputs"
                / "mano"
                / "right"
                / "fixture_task"
                / "0"
                / "trajectory_keypoints.npz"
            )
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(payload)

            spider_cell = f"traj_egoinfinity__hand_{hand_source}__retarget_spider"
            processed = (
                root
                / "intermediates"
                / "retargeting"
                / "spider"
                / spider_cell
                / "dataset"
                / "processed"
                / "fixture"
                / "mano"
                / "right"
                / "fixture_task"
                / "0"
                / "trajectory_keypoints.npz"
            )
            processed.parent.mkdir(parents=True, exist_ok=True)
            processed.write_bytes(payload)

            source_mesh = (
                root
                / "intermediates"
                / "retargeting"
                / "do_as_i_do"
                / dai_cell
                / "retargeting_outputs"
                / "assets"
                / "objects"
                / "fixture_task"
                / "visual.obj"
            )
            source_mesh.parent.mkdir(parents=True, exist_ok=True)
            source_mesh.write_bytes(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
            processed_mesh = (
                root
                / "intermediates"
                / "retargeting"
                / "spider"
                / spider_cell
                / "dataset"
                / "processed"
                / "fixture"
                / "assets"
                / "objects"
                / "fixture_task"
                / "visual.obj"
            )
            processed_mesh.parent.mkdir(parents=True, exist_ok=True)
            processed_mesh.write_bytes(source_mesh.read_bytes())

            manifest = (
                root
                / "intermediates"
                / "retargeting"
                / "do_as_i_do"
                / dai_cell
                / "raw_dir"
                / "adapter_manifest.json"
            )
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(valid_manifest(hand_source)), encoding="utf-8")

            self.sources[hand_source] = source
            self.processed[hand_source] = processed
            self.source_meshes[hand_source] = source_mesh
            self.processed_meshes[hand_source] = processed_mesh
            self.manifests[hand_source] = manifest


def semantic_fixture(root: Path) -> tuple[Path, Path, dict[str, np.ndarray]]:
    dai_cell = expected_cell_key("egoinfinity", "aoe")
    spider_cell = "traj_egoinfinity__hand_aoe__retarget_spider"
    source = (
        root
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / dai_cell
        / "retargeting_outputs"
        / "mano"
        / "left"
        / "fixture_task"
        / "0"
        / "trajectory_keypoints.npz"
    )
    processed = (
        root
        / "intermediates"
        / "retargeting"
        / "spider"
        / spider_cell
        / "dataset"
        / "processed"
        / "fixture"
        / "mano"
        / "left"
        / "fixture_task"
        / "0"
        / "trajectory_keypoints.npz"
    )
    source.parent.mkdir(parents=True)
    processed.parent.mkdir(parents=True)
    frames = 4
    arrays = {
        "qpos_wrist_left": np.arange(frames * 7, dtype=np.float64).reshape(frames, 7),
        "qpos_wrist_right": np.arange(frames * 7, dtype=np.float64).reshape(frames, 7) + 10.0,
        "qpos_finger_left": np.arange(frames * 5 * 7, dtype=np.float64).reshape(frames, 5, 7),
        "qpos_finger_right": np.arange(frames * 5 * 7, dtype=np.float64).reshape(frames, 5, 7) + 20.0,
        "qpos_obj_left": np.arange(frames * 7, dtype=np.float64).reshape(frames, 7) + 30.0,
        "qpos_obj_right": np.arange(frames * 7, dtype=np.float64).reshape(frames, 7) + 40.0,
    }
    return source, processed, arrays


class RouteInputQCTest(unittest.TestCase):
    def test_camera_binding_accepts_exact_resize_and_has_stable_fingerprints(self):
        object_camera = {
            "intrinsics": {
                "width": 1920,
                "height": 1080,
                "fx": 800.0,
                "fy": 810.0,
                "cx": 960.0,
                "cy": 540.0,
            }
        }
        hand_camera = {
            "intrinsics": {
                "width": 1280,
                "height": 720,
                "fx": 800.0 * 2.0 / 3.0,
                "fy": 810.0 * 2.0 / 3.0,
                "cx": 640.0,
                "cy": 360.0,
            }
        }
        binding = build_camera_intrinsics_binding(object_camera, hand_camera)

        self.assertEqual(binding["status"], "ok")
        self.assertTrue(binding["comparison"]["matches"])
        self.assertEqual(
            binding["object_camera"]["fingerprint"],
            camera_intrinsics_fingerprint(object_camera),
        )

    def test_ego_aoe_camera_binding_missing_or_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            common = {
                "source_keypoints": fixture.sources["aoe"],
                "processed_keypoints": fixture.processed["aoe"],
                "processed_backend": "spider",
                "source_object_mesh": fixture.source_meshes["aoe"],
                "processed_object_mesh": fixture.processed_meshes["aoe"],
                "trajectory": "egoinfinity",
                "hand_source": "aoe",
                "task": "fixture_task",
                "hand_type": "right",
                "data_id": 0,
            }
            missing = valid_manifest("aoe")
            missing.pop("camera_intrinsics_binding")
            missing_result = validate_route_input(adapter_manifest=missing, **common)

            mismatch = valid_manifest("aoe")
            object_camera = mismatch["camera_intrinsics_binding"]["object_camera"]
            hand_camera = deepcopy(mismatch["camera_intrinsics_binding"]["hand_camera"])
            hand_camera["intrinsics"]["fx"] = 500.0
            hand_camera["intrinsics"]["fy"] = 500.0
            mismatch["camera_intrinsics_binding"] = build_camera_intrinsics_binding(
                object_camera, hand_camera
            )
            mismatch_result = validate_route_input(adapter_manifest=mismatch, **common)

        self.assertIn(
            "adapter_camera_intrinsics_binding_missing_or_invalid",
            missing_result["errors"],
        )
        self.assertTrue(
            any(
                item.startswith("adapter_camera_intrinsics_binding_not_ok")
                or item.startswith("adapter_camera_intrinsics_binding_recomputed_invalid")
                for item in mismatch_result["errors"]
            )
        )

    def test_camera_source_files_are_rehashed_and_intrinsics_are_reread(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            common = {
                "source_keypoints": fixture.sources["aoe"],
                "processed_keypoints": fixture.processed["aoe"],
                "processed_backend": "spider",
                "source_object_mesh": fixture.source_meshes["aoe"],
                "processed_object_mesh": fixture.processed_meshes["aoe"],
                "trajectory": "egoinfinity",
                "hand_source": "aoe",
                "task": "fixture_task",
                "hand_type": "right",
                "data_id": 0,
            }
            stale_hash = valid_manifest("aoe")
            stale_hash["camera_intrinsics_binding"]["object_camera"]["source"][
                "file_bindings"
            ]["pipeline_result"]["sha256"] = "0" * 64
            stale_result = validate_route_input(
                adapter_manifest=stale_hash,
                **common,
            )

            fabricated = valid_manifest("aoe")
            object_camera = deepcopy(
                fabricated["camera_intrinsics_binding"]["object_camera"]
            )
            hand_camera = deepcopy(
                fabricated["camera_intrinsics_binding"]["hand_camera"]
            )
            for camera in (object_camera, hand_camera):
                camera["intrinsics"]["fx"] = 1.5
                camera["intrinsics"]["fy"] = 1.5
            fabricated["camera_intrinsics_binding"] = build_camera_intrinsics_binding(
                object_camera, hand_camera
            )
            fabricated_result = validate_route_input(
                adapter_manifest=fabricated,
                **common,
            )

        self.assertIn(
            "adapter_camera_pipeline_result_sha256_mismatch",
            stale_result["errors"],
        )
        self.assertIn(
            "adapter_camera_pipeline_result_intrinsics_mismatch",
            fabricated_result["errors"],
        )
        self.assertIn(
            "adapter_camera_calibration_intrinsics_mismatch",
            fabricated_result["errors"],
        )

    def test_exact_route_and_hand_source_copy_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            result = validate_route_input(
                adapter_manifest=fixture.manifests["aoe"],
                source_keypoints=fixture.sources["aoe"],
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                source_object_mesh=fixture.source_meshes["aoe"],
                processed_object_mesh=fixture.processed_meshes["aoe"],
                trajectory="egoinfinity",
                hand_source="aoe",
                task="fixture_task",
                hand_type="right",
                data_id=0,
            )
            expected_hash = file_sha256(fixture.sources["aoe"])
            expected_mesh_hash = file_sha256(fixture.source_meshes["aoe"])

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["hash"]["matches"])
        self.assertEqual(result["hash"]["source_keypoints"], expected_hash)
        self.assertTrue(result["hash"]["object_mesh"]["validated"])
        self.assertTrue(result["hash"]["object_mesh"]["matches"])
        self.assertEqual(result["hash"]["object_mesh"]["source"], expected_mesh_hash)
        self.assertEqual(
            result["paths"]["exact_dai_cell_root"],
            str(fixture.sources["aoe"].resolve().parents[5]),
        )
        self.assertEqual(result["provenance"]["hand_source"], "aoe")
        self.assertEqual(result["provenance"]["retarget_object_source"], "egoinfinity")
        self.assertEqual(result["provenance"]["adapter_rigid_invariance"]["status"], "ok")
        self.assertEqual(result["route_artifact_identity"]["task"], "fixture_task")
        self.assertEqual(result["processed_backend"], "spider")
        json.dumps(result)

    def test_backend_and_both_visual_mesh_paths_are_mandatory(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            common = {
                "adapter_manifest": fixture.manifests["aoe"],
                "source_keypoints": fixture.sources["aoe"],
                "processed_keypoints": fixture.processed["aoe"],
                "trajectory": "egoinfinity",
                "hand_source": "aoe",
                "task": "fixture_task",
                "hand_type": "right",
                "data_id": 0,
            }
            invalid_backend = validate_route_input(
                **common,
                processed_backend="unknown",
                source_object_mesh=fixture.source_meshes["aoe"],
                processed_object_mesh=fixture.processed_meshes["aoe"],
            )
            missing_source = validate_route_input(
                **common,
                processed_backend="spider",
                processed_object_mesh=fixture.processed_meshes["aoe"],
            )
            missing_processed = validate_route_input(
                **common,
                processed_backend="spider",
                source_object_mesh=fixture.source_meshes["aoe"],
            )

        self.assertIn("processed_backend_invalid:unknown", invalid_backend["errors"])
        self.assertIn("source_object_mesh_path_missing", missing_source["errors"])
        self.assertIn("processed_object_mesh_path_missing", missing_processed["errors"])

    def test_route_artifact_identity_is_required_even_when_all_fields_are_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            result = validate_route_input(
                adapter_manifest=fixture.manifests["aoe"],
                source_keypoints=fixture.sources["aoe"],
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                source_object_mesh=fixture.source_meshes["aoe"],
                processed_object_mesh=fixture.processed_meshes["aoe"],
                trajectory="egoinfinity",
                hand_source="aoe",
            )

        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["errors"], ["route_artifact_identity_incomplete"])
        self.assertEqual(
            result["route_artifact_identity"],
            {"task": None, "hand_type": None, "data_id": None},
        )

    def test_exact_cell_raw_dir_symlink_to_route_adapter_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            manifest = fixture.manifests["aoe"]
            raw_dir = manifest.parent
            manifest.unlink()
            raw_dir.rmdir()
            adapter_dir = fixture.root / "intermediates" / "trajectory_6dof" / "egoinfinity" / "do_as_i_do_raw_dir_hand_aoe"
            adapter_dir.mkdir(parents=True)
            external_manifest = adapter_dir / "adapter_manifest.json"
            external_manifest.write_text(
                json.dumps(valid_manifest("aoe")), encoding="utf-8"
            )
            raw_dir.symlink_to(adapter_dir, target_is_directory=True)

            result = validate_route_input(
                adapter_manifest=manifest,
                source_keypoints=fixture.sources["aoe"],
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                source_object_mesh=fixture.source_meshes["aoe"],
                processed_object_mesh=fixture.processed_meshes["aoe"],
                trajectory="egoinfinity",
                hand_source="aoe",
                task="fixture_task",
                hand_type="right",
                data_id=0,
            )

        self.assertEqual(result["status"], "ok", result["errors"])
        self.assertEqual(
            result["paths"]["adapter_manifest_resolved"],
            str(external_manifest.resolve()),
        )

    def test_unified_matrix_source_is_forbidden_even_when_bytes_match(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            unified = (
                fixture.root
                / "assets"
                / "trajectory_6dof"
                / "egoinfinity"
                / "fixture_task"
                / "source_trajectory_keypoints.npz"
            )
            unified.parent.mkdir(parents=True, exist_ok=True)
            unified.write_bytes(fixture.sources["aoe"].read_bytes())
            result = validate_route_input(
                adapter_manifest=valid_manifest("aoe"),
                source_keypoints=unified,
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                trajectory="egoinfinity",
                hand_source="aoe",
            )

        self.assertEqual(result["status"], "invalid")
        self.assertIn("source_keypoints_unified_asset_forbidden", result["errors"])
        self.assertTrue(result["hash"]["matches"])

    def test_mesh_symlink_to_other_hand_cell_is_rejected_after_resolve(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            aoe_cell = expected_cell_key("egoinfinity", "aoe")
            indexed_mesh = (
                fixture.root
                / "assets"
                / "cells"
                / aoe_cell
                / "retargeting"
                / "do_as_i_do"
                / "assets"
                / "objects"
                / "fixture_task"
                / "visual.obj"
            )
            indexed_mesh.parent.mkdir(parents=True, exist_ok=True)
            indexed_mesh.symlink_to(fixture.source_meshes["estimated"])
            result = validate_route_input(
                adapter_manifest=fixture.manifests["aoe"],
                source_keypoints=fixture.sources["aoe"],
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                source_object_mesh=indexed_mesh,
                processed_object_mesh=fixture.processed_meshes["aoe"],
                trajectory="egoinfinity",
                hand_source="aoe",
            )

        self.assertEqual(result["status"], "invalid")
        self.assertTrue(
            any(
                error.startswith("source_object_mesh_resolved_not_exact_dai_cell")
                for error in result["errors"]
            )
        )
        self.assertTrue(
            any(
                error.startswith("source_object_mesh_resolved_cross_cell")
                for error in result["errors"]
            )
        )
        self.assertTrue(
            any(
                error.startswith("exact_dai_cell_component_root_mismatch")
                for error in result["errors"]
            )
        )

    def test_exact_cell_index_symlinks_resolve_to_one_dai_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            aoe_cell = expected_cell_key("egoinfinity", "aoe")
            indexed_root = (
                fixture.root
                / "assets"
                / "cells"
                / aoe_cell
                / "retargeting"
                / "do_as_i_do"
            )
            indexed_keypoints = indexed_root / "trajectories" / "trajectory_keypoints.npz"
            indexed_mesh = (
                indexed_root / "assets" / "objects" / "fixture_task" / "visual.obj"
            )
            indexed_keypoints.parent.mkdir(parents=True, exist_ok=True)
            indexed_mesh.parent.mkdir(parents=True, exist_ok=True)
            indexed_keypoints.symlink_to(fixture.sources["aoe"])
            indexed_mesh.symlink_to(fixture.source_meshes["aoe"])
            result = validate_route_input(
                adapter_manifest=fixture.manifests["aoe"],
                source_keypoints=indexed_keypoints,
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                source_object_mesh=indexed_mesh,
                processed_object_mesh=fixture.processed_meshes["aoe"],
                trajectory="egoinfinity",
                hand_source="aoe",
                task="fixture_task",
                hand_type="right",
                data_id=0,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["hash"]["keypoints"]["matches"])
        self.assertTrue(result["hash"]["object_mesh"]["matches"])

    def test_mesh_from_same_route_but_other_experiment_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = RouteFixture(root / "experiment_a")
            other = RouteFixture(root / "experiment_b")
            result = validate_route_input(
                adapter_manifest=fixture.manifests["aoe"],
                source_keypoints=fixture.sources["aoe"],
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                source_object_mesh=other.source_meshes["aoe"],
                processed_object_mesh=fixture.processed_meshes["aoe"],
                trajectory="egoinfinity",
                hand_source="aoe",
            )

        self.assertEqual(result["status"], "invalid")
        self.assertTrue(
            any(
                error.startswith("exact_dai_cell_component_root_mismatch")
                for error in result["errors"]
            )
        )

    def test_mesh_from_other_task_in_same_cell_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            other_task_mesh = (
                fixture.source_meshes["aoe"].parents[1]
                / "other_task"
                / "visual.obj"
            )
            other_task_mesh.parent.mkdir(parents=True, exist_ok=True)
            other_task_mesh.write_bytes(fixture.source_meshes["aoe"].read_bytes())
            result = validate_route_input(
                adapter_manifest=fixture.manifests["aoe"],
                source_keypoints=fixture.sources["aoe"],
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                source_object_mesh=other_task_mesh,
                processed_object_mesh=fixture.processed_meshes["aoe"],
                trajectory="egoinfinity",
                hand_source="aoe",
                task="fixture_task",
                hand_type="right",
                data_id=0,
            )

        self.assertEqual(result["status"], "invalid")
        self.assertIn(
            "source_object_mesh_route_artifact_identity_mismatch:expected_task=fixture_task",
            result["errors"],
        )

    def test_processed_mesh_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            fixture.processed_meshes["aoe"].write_bytes(b"stale-mesh")
            result = validate_route_input(
                adapter_manifest=fixture.manifests["aoe"],
                source_keypoints=fixture.sources["aoe"],
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                source_object_mesh=fixture.source_meshes["aoe"],
                processed_object_mesh=fixture.processed_meshes["aoe"],
                trajectory="egoinfinity",
                hand_source="aoe",
            )

        self.assertEqual(result["status"], "invalid")
        self.assertIn("source_processed_object_mesh_sha256_mismatch", result["errors"])
        self.assertFalse(result["hash"]["object_mesh"]["matches"])

    def test_spider_wrapper_has_no_trajectory_or_legacy_mesh_fallback(self):
        script = (
            Path(__file__).resolve().parents[1] / "scripts" / "run_spider_retarget.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn('traj_assets / "object_meshes" / "visual.obj"', script)
        self.assertNotIn('legacy_root / "assets" / "objects"', script)
        self.assertIn("missing object asset", script)
        self.assertIn('payload[key] = f"processed/do_as_i_do/{value}"', script)
        self.assertIn('"object_assets": [', script)
        self.assertIn('"byte_identical": any(', script)
        self.assertNotIn("detect_contact.py", script)
        self.assertNotIn("validate_spider_postprocess_semantics", script)

    def test_postprocess_semantic_qc_allows_contact_rewrite_and_mano_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            source, processed, core = semantic_fixture(Path(directory))
            source_payload = {
                **{name: value.copy() for name, value in core.items()},
                "contact_left": np.zeros((4, 5), dtype=np.float32),
                "contact_pos_left": np.zeros((5, 3), dtype=np.float32),
                "qpos_pip_left": np.zeros((4, 5, 3), dtype=np.float64),
                "mano_verts_left": np.zeros((4, 778, 3), dtype=np.float32),
            }
            processed_payload = {
                **{name: value.copy() for name, value in core.items()},
                "contact_left": np.ones((4, 5), dtype=np.float32),
                "contact_pos_left": np.ones((5, 3), dtype=np.float32),
                "contact_world_left": np.ones((4, 5, 3), dtype=np.float32),
            }
            np.savez(source, **source_payload)
            np.savez(processed, **processed_payload)

            result = validate_spider_postprocess_semantics(
                source_keypoints=source,
                processed_keypoints=processed,
                trajectory="egoinfinity",
                hand_source="aoe",
                task="fixture_task",
                hand_type="left",
                data_id=0,
            )

        self.assertEqual(result["status"], "ok", result["errors"])
        self.assertFalse(result["archive_sha256"]["matches"])
        self.assertFalse(result["archive_sha256"]["match_required"])
        self.assertEqual(set(result["core_arrays"]), set(SPIDER_CORE_HOI_ARRAYS))
        self.assertTrue(
            all(field["status"] == "ok" for field in result["core_arrays"].values())
        )
        allowed = {
            (item["field"], item["change"], item["category"])
            for item in result["allowed_noncore_changes"]
        }
        self.assertIn(("contact_left", "modified", "contact"), allowed)
        self.assertIn(("contact_world_left", "added", "contact"), allowed)
        self.assertIn(("mano_verts_left", "removed", "derived_or_auxiliary"), allowed)
        self.assertIn(("qpos_pip_left", "removed", "derived_or_auxiliary"), allowed)

    def test_postprocess_semantic_qc_preserves_usable_route_source_contact(self):
        with tempfile.TemporaryDirectory() as directory:
            source, processed, core = semantic_fixture(Path(directory))
            source_contact = np.zeros((6, 5), dtype=np.float32)
            source_contact[[0, 2, 4], 1] = 1.0
            source_contact_pos = np.arange(15, dtype=np.float32).reshape(5, 3)
            source_payload = {
                **{name: value.copy() for name, value in core.items()},
                "contact_left": source_contact,
                "contact_pos_left": source_contact_pos,
            }
            processed_payload = {
                **{name: value.copy() for name, value in core.items()},
                "contact_left": np.ones((6, 5), dtype=np.float32),
                "contact_pos_left": source_contact_pos.copy(),
            }
            np.savez(source, **source_payload)
            np.savez(processed, **processed_payload)

            result = validate_spider_postprocess_semantics(
                source_keypoints=source,
                processed_keypoints=processed,
                trajectory="egoinfinity",
                hand_source="aoe",
                task="fixture_task",
                hand_type="left",
                data_id=0,
            )

        self.assertEqual(result["status"], "invalid")
        self.assertIn(
            "postprocess_source_contact_modified:contact_left:modified",
            result["errors"],
        )
        self.assertEqual(
            result["contact_provenance"]["left"]["policy"],
            "preserve_exact_route_source",
        )
        self.assertEqual(
            result["contact_provenance"]["left"]["source_active_frames"], 3
        )

    def test_postprocess_semantic_qc_rejects_any_material_core_change(self):
        with tempfile.TemporaryDirectory() as directory:
            source, processed, core = semantic_fixture(Path(directory))
            source_payload = {name: value.copy() for name, value in core.items()}
            processed_payload = {name: value.copy() for name, value in core.items()}
            processed_payload["qpos_obj_left"][0, 0] += 1.0e-6
            processed_payload["qpos_finger_right"] = processed_payload[
                "qpos_finger_right"
            ].astype(np.float32)
            processed_payload["qpos_wrist_right"][0, 0] = np.nan
            processed_payload["qpos_obj_right"] = processed_payload["qpos_obj_right"][:-1]
            np.savez(source, **source_payload)
            np.savez(processed, **processed_payload)

            result = validate_spider_postprocess_semantics(
                source_keypoints=source,
                processed_keypoints=processed,
                trajectory="egoinfinity",
                hand_source="aoe",
                task="fixture_task",
                hand_type="left",
                data_id=0,
            )

        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["production_allowed"])
        self.assertTrue(
            any(
                error.startswith("postprocess_core_value_mismatch:qpos_obj_left")
                for error in result["errors"]
            )
        )
        self.assertTrue(
            any(
                error.startswith("postprocess_core_dtype_mismatch:qpos_finger_right")
                for error in result["errors"]
            )
        )
        self.assertIn(
            "postprocess_core_processed_nonfinite:qpos_wrist_right", result["errors"]
        )
        self.assertTrue(
            any(
                error.startswith("postprocess_core_shape_mismatch:qpos_obj_right")
                for error in result["errors"]
            )
        )

    def test_postprocess_semantic_qc_records_byte_drift_within_strict_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            source, processed, core = semantic_fixture(Path(directory))
            source_payload = {name: value.copy() for name, value in core.items()}
            processed_payload = {name: value.copy() for name, value in core.items()}
            processed_payload["qpos_wrist_left"][0, 1] = np.nextafter(
                processed_payload["qpos_wrist_left"][0, 1], np.inf
            )
            np.savez(source, **source_payload)
            np.savez(processed, **processed_payload)

            result = validate_spider_postprocess_semantics(
                source_keypoints=source,
                processed_keypoints=processed,
                trajectory="egoinfinity",
                hand_source="aoe",
                task="fixture_task",
                hand_type="left",
                data_id=0,
            )

        self.assertEqual(result["status"], "ok", result["errors"])
        wrist = result["core_arrays"]["qpos_wrist_left"]
        self.assertTrue(wrist["strict_numeric_match"])
        self.assertFalse(wrist["payload_sha256"]["bytes_match"])
        self.assertGreater(wrist["max_abs_error"], 0.0)

    def test_postprocess_semantic_qc_rejects_unclassified_field_rewrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source, processed, core = semantic_fixture(Path(directory))
            source_payload = {
                **{name: value.copy() for name, value in core.items()},
                "unclassified_semantic": np.array([1], dtype=np.int64),
            }
            processed_payload = {
                **{name: value.copy() for name, value in core.items()},
                "unclassified_semantic": np.array([2], dtype=np.int64),
            }
            np.savez(source, **source_payload)
            np.savez(processed, **processed_payload)

            result = validate_spider_postprocess_semantics(
                source_keypoints=source,
                processed_keypoints=processed,
                trajectory="egoinfinity",
                hand_source="aoe",
                task="fixture_task",
                hand_type="left",
                data_id=0,
            )

        self.assertEqual(result["status"], "invalid")
        self.assertIn(
            "postprocess_unexpected_noncore_change:unclassified_semantic:modified",
            result["errors"],
        )

    def test_postprocess_semantic_qc_requires_core_fields_and_hand_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            source, processed, core = semantic_fixture(Path(directory))
            wrong_hand_processed = Path(str(processed).replace("/left/", "/right/"))
            wrong_hand_processed.parent.mkdir(parents=True)
            incomplete = {name: value.copy() for name, value in core.items()}
            incomplete.pop("qpos_obj_left")
            np.savez(source, **core)
            np.savez(wrong_hand_processed, **incomplete)

            result = validate_spider_postprocess_semantics(
                source_keypoints=source,
                processed_keypoints=wrong_hand_processed,
                trajectory="egoinfinity",
                hand_source="aoe",
                task="fixture_task",
                hand_type="left",
                data_id=0,
            )

        self.assertEqual(result["status"], "invalid")
        self.assertIn(
            "postprocess_core_processed_field_missing:qpos_obj_left", result["errors"]
        )
        self.assertIn(
            "postprocess_processed_route_artifact_identity_mismatch:"
            "expected=mano/left/fixture_task/0",
            result["errors"],
        )

    def test_pure_dai_aoe_and_estimated_routes_accept_exact_native_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = {}
            for hand_source in ("aoe", "estimated"):
                dai_cell = expected_cell_key("do_as_i_do", hand_source)
                source = (
                    root
                    / "intermediates/retargeting/do_as_i_do"
                    / dai_cell
                    / "retargeting_outputs/mano/right/task/0/trajectory_keypoints.npz"
                )
                processed = (
                    root
                    / "assets/cells"
                    / dai_cell
                    / "retargeting/do_as_i_do/trajectories/mano/right/task/0/trajectory_keypoints.npz"
                )
                source_mesh = (
                    source.parents[4] / "assets/objects/task/visual.obj"
                )
                processed_mesh = (
                    processed.parents[5] / "assets/objects/task/visual.obj"
                )
                manifest = source.parents[5] / "raw_dir/adapter_manifest.json"
                source.parent.mkdir(parents=True)
                processed.parent.mkdir(parents=True)
                source_mesh.parent.mkdir(parents=True)
                processed_mesh.parent.mkdir(parents=True)
                manifest.parent.mkdir(parents=True)
                source.write_bytes(f"pure-dai-{hand_source}-keypoints".encode())
                processed.write_bytes(source.read_bytes())
                source_mesh.write_text("v 0 0 0\n", encoding="utf-8")
                processed_mesh.write_bytes(source_mesh.read_bytes())
                manifest.write_text(
                    json.dumps(valid_manifest(hand_source, "do_as_i_do")),
                    encoding="utf-8",
                )
                results[hand_source] = validate_route_input(
                    adapter_manifest=manifest,
                    source_keypoints=source,
                    processed_keypoints=processed,
                    processed_backend="do_as_i_do",
                    source_object_mesh=source_mesh,
                    processed_object_mesh=processed_mesh,
                    trajectory="do_as_i_do",
                    hand_source=hand_source,
                    task="task",
                    hand_type="right",
                    data_id=0,
                )

        for hand_source, result in results.items():
            self.assertEqual(result["status"], "ok", hand_source)
            self.assertEqual(result["provenance"]["hand_source"], hand_source)
            self.assertEqual(result["provenance"]["object_track_source"], "dai_native")
            self.assertEqual(result["provenance"]["object_mesh_source"], "dai_native")
            self.assertEqual(result["provenance"]["retarget_object_source"], "dai_native")

    def test_cross_hand_source_is_rejected_not_compared_as_same_route(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            result = validate_route_input(
                adapter_manifest=fixture.manifests["aoe"],
                source_keypoints=fixture.sources["estimated"],
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                trajectory="egoinfinity",
                hand_source="aoe",
            )

        self.assertEqual(result["status"], "invalid")
        self.assertTrue(
            any(error.startswith("source_keypoints_not_exact_dai_cell") for error in result["errors"])
        )
        self.assertIn("source_processed_sha256_mismatch", result["errors"])
        self.assertFalse(result["hash"]["matches"])

    def test_processed_copy_from_other_hand_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            result = validate_route_input(
                adapter_manifest=fixture.manifests["aoe"],
                source_keypoints=fixture.sources["aoe"],
                processed_keypoints=fixture.processed["estimated"],
                processed_backend="spider",
                trajectory="egoinfinity",
                hand_source="aoe",
            )

        self.assertEqual(result["status"], "invalid")
        self.assertTrue(
            any(
                error.startswith("processed_keypoints_not_exact_spider_cell")
                for error in result["errors"]
            )
        )
        self.assertIn("source_processed_sha256_mismatch", result["errors"])

    def test_legacy_applied_contact_translation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            manifest = valid_manifest("aoe")
            manifest["hoi_contact_alignment"] = {
                "applied": True,
                "method": "median_nearest_surface_translation",
                "offset": [0.12, -0.03, 0.01],
            }
            result = validate_route_input(
                adapter_manifest=manifest,
                source_keypoints=fixture.sources["aoe"],
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                trajectory="egoinfinity",
                hand_source="aoe",
            )

        self.assertEqual(result["status"], "invalid")
        self.assertIn("adapter_hoi_contact_alignment_applied_or_unrecorded", result["errors"])

    def test_legacy_or_fused_object_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            manifest = valid_manifest("aoe")
            manifest["object_track_source"] = "dai_native_contact_aligned"
            manifest["object_mesh_source"] = "fused_ego_for_spider"
            result = validate_route_input(
                adapter_manifest=manifest,
                source_keypoints=fixture.sources["aoe"],
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                trajectory="egoinfinity",
                hand_source="aoe",
            )

        self.assertEqual(result["status"], "invalid")
        self.assertTrue(
            any(error.startswith("adapter_object_track_source_mismatch") for error in result["errors"])
        )
        self.assertTrue(
            any(error.startswith("adapter_object_mesh_source_mismatch") for error in result["errors"])
        )

    def test_canonical_transform_rejects_non_se3_and_false_identity(self):
        invalid_matrices = {
            "scale": [
                [1.1, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "shear": [
                [1.0, 0.2, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "reflection": [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            for name, matrix in invalid_matrices.items():
                with self.subTest(name=name):
                    manifest = deepcopy(valid_manifest("aoe"))
                    manifest["canonical_transform"]["matrix"] = matrix
                    result = validate_route_input(
                        adapter_manifest=manifest,
                        source_keypoints=fixture.sources["aoe"],
                        processed_keypoints=fixture.processed["aoe"],
                        processed_backend="spider",
                        trajectory="egoinfinity",
                        hand_source="aoe",
                    )
                    self.assertEqual(result["status"], "invalid")
                    self.assertTrue(
                        any(
                            error.startswith("adapter_canonical_transform_not_se3")
                            for error in result["errors"]
                        )
                    )

            translated_identity = deepcopy(valid_manifest("aoe"))
            translated_identity["canonical_transform"]["matrix"][0][3] = 0.3
            result = validate_route_input(
                adapter_manifest=translated_identity,
                source_keypoints=fixture.sources["aoe"],
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                source_object_mesh=fixture.source_meshes["aoe"],
                processed_object_mesh=fixture.processed_meshes["aoe"],
                trajectory="egoinfinity",
                hand_source="aoe",
                task="fixture_task",
                hand_type="right",
                data_id=0,
            )

            shared_se3 = with_valid_shared_se3_mano_evidence(
                translated_identity,
                translated_identity["canonical_transform"]["matrix"],
            )
            fixture.manifests["aoe"].write_text(
                json.dumps(shared_se3), encoding="utf-8"
            )
            shared_result = validate_route_input(
                adapter_manifest=fixture.manifests["aoe"],
                source_keypoints=fixture.sources["aoe"],
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                source_object_mesh=fixture.source_meshes["aoe"],
                processed_object_mesh=fixture.processed_meshes["aoe"],
                trajectory="egoinfinity",
                hand_source="aoe",
                task="fixture_task",
                hand_type="right",
                data_id=0,
            )

        self.assertEqual(result["status"], "invalid")
        self.assertIn(
            "adapter_canonical_transform_identity_matrix_not_identity",
            result["errors"],
        )
        self.assertEqual(shared_result["status"], "ok")

    def test_shared_se3_requires_complete_mano_global_pose_evidence(self):
        matrix = deepcopy(IDENTITY)
        matrix[0][3] = 0.25
        matrix[1][3] = -0.12
        base = with_valid_shared_se3_mano_evidence(
            valid_manifest("aoe"), matrix
        )
        cases: dict[str, tuple[dict, str]] = {}

        missing_target = deepcopy(base)
        missing_target["canonical_transform"]["applies_to"].remove(
            "mano_global_translation"
        )
        cases["missing_target"] = (
            missing_target,
            "adapter_canonical_transform_incomplete_targets:mano_global_translation",
        )

        missing_report = deepcopy(base)
        missing_report.pop("ego_hand_geometry")
        cases["missing_report"] = (
            missing_report,
            "adapter_ego_hand_geometry_missing_for_shared_se3",
        )

        wrong_translation = deepcopy(base)
        wrong_translation["ego_hand_geometry"]["mano_global_pose_transform"][
            "common_translation"
        ][0] += 0.1
        cases["wrong_translation"] = (
            wrong_translation,
            "adapter_mano_global_pose_common_translation_mismatch",
        )

        missing_transformed_field = deepcopy(base)
        missing_transformed_field["ego_hand_geometry"][
            "mano_global_pose_transform"
        ]["transformed_fields"].remove("left_trans")
        cases["missing_transformed_field"] = (
            missing_transformed_field,
            "adapter_mano_global_pose_left_translation_not_transformed",
        )

        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            for name, (manifest, expected_error) in cases.items():
                with self.subTest(name=name):
                    fixture.manifests["aoe"].write_text(
                        json.dumps(manifest), encoding="utf-8"
                    )
                    result = validate_route_input(
                        adapter_manifest=fixture.manifests["aoe"],
                        source_keypoints=fixture.sources["aoe"],
                        processed_keypoints=fixture.processed["aoe"],
                        processed_backend="spider",
                        source_object_mesh=fixture.source_meshes["aoe"],
                        processed_object_mesh=fixture.processed_meshes["aoe"],
                        trajectory="egoinfinity",
                        hand_source="aoe",
                        task="fixture_task",
                        hand_type="right",
                        data_id=0,
                    )
                    self.assertEqual(result["status"], "invalid")
                    self.assertIn(expected_error, result["errors"])

    def test_manifest_fields_and_rigid_invariance_are_hard_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RouteFixture(Path(directory))
            manifest = deepcopy(valid_manifest("aoe"))
            manifest["hand_source"] = "estimated"
            manifest["object_track_source"] = ""
            manifest.pop("object_mesh_source")
            manifest.pop("retarget_object_source")
            manifest["hoi_refinement"]["applied"] = True
            manifest["hoi_refinement"]["hand_contact_translation_applied"] = True
            manifest["hoi_contact_alignment"]["transforms_modified"] = True
            manifest["hoi_contact_alignment"]["layout_modified"] = True
            manifest["adapter_rigid_invariance"]["status"] = "invalid"
            manifest["adapter_rigid_invariance"]["true_pre_post_hand_comparison"] = False
            result = validate_route_input(
                adapter_manifest=manifest,
                source_keypoints=fixture.sources["aoe"],
                processed_keypoints=fixture.processed["aoe"],
                processed_backend="spider",
                trajectory="egoinfinity",
                hand_source="aoe",
            )

        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any(error.startswith("adapter_hand_source_mismatch") for error in result["errors"]))
        self.assertIn("adapter_object_track_source_missing", result["errors"])
        self.assertIn("adapter_object_mesh_source_missing", result["errors"])
        self.assertIn("adapter_retarget_object_source_missing", result["errors"])
        self.assertIn("adapter_hoi_refinement_applied_or_unrecorded", result["errors"])
        self.assertIn(
            "adapter_hoi_refinement_component_applied:hand_contact_translation_applied",
            result["errors"],
        )
        self.assertIn(
            "adapter_hoi_contact_alignment_modified_production_input:layout_modified,transforms_modified",
            result["errors"],
        )
        self.assertIn("adapter_rigid_invariance_not_ok:invalid", result["errors"])
        self.assertIn(
            "adapter_rigid_invariance_not_true_pre_post_hand_comparison",
            result["errors"],
        )


if __name__ == "__main__":
    unittest.main()
