from __future__ import annotations

import importlib.util
import gzip
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aoe_retarget_lab.route_input_qc import (
    build_camera_intrinsics_binding,
    file_sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


_CAMERA_FIXTURE_DIR = tempfile.TemporaryDirectory()
_CAMERA_BINDING: dict | None = None


def production_camera_binding() -> dict:
    global _CAMERA_BINDING
    if _CAMERA_BINDING is not None:
        return json.loads(json.dumps(_CAMERA_BINDING))
    root = Path(_CAMERA_FIXTURE_DIR.name)
    pipeline = root / "pipeline_result.pkl.gz"
    with gzip.open(pipeline, "wb") as handle:
        pickle.dump(
            {
                "dp_focal": 2.0,
                "cx": 2.0,
                "cy": 1.0,
                "bg_rgb": np.zeros((2, 4, 3), dtype=np.uint8),
            },
            handle,
        )
    source_hands = root / "ego_process/ego_hands_reconstruction/hands.npz"
    source_hands.parent.mkdir(parents=True)
    np.savez(source_hands, focal=np.asarray(2.0))
    adapted_hands = root / "adapted/all_hand_meshes.npz"
    adapted_hands.parent.mkdir(parents=True)
    np.savez(adapted_hands, source_hands_npz=np.asarray(str(source_hands)))
    calibration = root / "ego_process/ego_undistorted_video/undistorted_video_info.json"
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

    def bound(path: Path) -> dict[str, str]:
        return {"path": str(path.resolve()), "sha256": file_sha256(path)}

    pipeline_binding = bound(pipeline)
    adapted_binding = bound(adapted_hands)
    source_binding = bound(source_hands)
    calibration_binding = bound(calibration)
    intrinsics = {
        "model": "pinhole",
        "width": 4,
        "height": 2,
        "fx": 2.0,
        "fy": 2.0,
        "cx": 2.0,
        "cy": 1.0,
    }
    _CAMERA_BINDING = build_camera_intrinsics_binding(
        {
            "source": {
                "kind": "egoinfinity_pipeline_result",
                "path": pipeline_binding["path"],
                "sha256": pipeline_binding["sha256"],
                "file_bindings": {"pipeline_result": pipeline_binding},
            },
            "intrinsics": intrinsics,
        },
        {
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
        },
    )
    return json.loads(json.dumps(_CAMERA_BINDING))


def route_payload(trajectory: str, hand_source: str) -> dict:
    source = "egoinfinity" if trajectory == "egoinfinity" else "dai_native"
    payload = {
        "adapter_schema_version": 3,
        "production_hoi_policy": "preserve_source_relative_transform",
        "hand_source": hand_source,
        "object_track_source": source,
        "object_mesh_source": source,
        "retarget_object_source": source,
        "canonical_transform": {
            "applied": False,
            "kind": "identity",
            "matrix": IDENTITY,
        },
        "hoi_refinement": {"applied": False, "method": "none"},
        "hoi_contact_alignment": {
            "applied": False,
            "diagnostic_only": True,
            "transforms_modified": False,
        },
        "adapter_rigid_invariance": {
            "status": "ok",
            "max_abs_error": 1.0e-15,
            "true_pre_post_hand_comparison": True,
        },
        "retarget_input_qc": {"status": "ok", "errors": []},
    }
    if trajectory == "egoinfinity" and hand_source == "aoe":
        payload["camera_intrinsics_binding"] = production_camera_binding()
    return payload


def route_qc(
    trajectory: str,
    hand_source: str,
    processed_backend: str = "do_as_i_do",
) -> dict:
    return {
        "status": "ok",
        "errors": [],
        "trajectory": trajectory,
        "hand_source": hand_source,
        "processed_backend": processed_backend,
        "route_artifact_identity": {
            "task": "fixture_left",
            "hand_type": "left",
            "data_id": 0,
        },
        "expected_cell_key": f"traj_{trajectory}__hand_{hand_source}__retarget_do_as_i_do",
    }


def postprocess_qc(trajectory: str, hand_source: str) -> dict:
    return {
        "schema_version": 1,
        "status": "ok",
        "errors": [],
        "trajectory": trajectory,
        "hand_source": hand_source,
        "route_artifact_identity": {
            "task": "fixture_left",
            "hand_type": "left",
            "data_id": 0,
        },
        "stage": "after_detect_contact_before_simulation",
    }


def hard_tracking_report() -> dict:
    side = {
        "status": "ok",
        "errors": [],
        "distance_error_median_m": 0.0,
        "distance_error_p95_m": 0.0,
        "contact_retention_fraction": 1.0,
    }
    return {
        "status": "ok",
        "errors": [],
        "metrics": {
            "pos_median_m": 0.0,
            "pos_p95_m": 0.0,
            "rot_median_rad": 0.0,
            "rot_p95_rad": 0.0,
            "lost_frame_fraction": 0.0,
        },
        "thresholds": {
            "pos_median_m": 0.05,
            "pos_p95_m": 0.10,
            "rot_median_rad": 0.75,
            "rot_p95_rad": 2.0,
            "lost_pos_m": 0.10,
            "lost_fraction": 0.10,
        },
        "interaction_quality": {
            "status": "ok",
            "errors": [],
            "active_sides": ["left"],
            "sides": {"left": side},
            "thresholds": {
                "distance_error_median_m": 0.03,
                "distance_error_p95_m": 0.08,
                "executed_contact_m": 0.08,
                "min_contact_retention": 0.50,
                "reference_contact_m": 0.05,
                "min_reference_contact_frames": 3,
            },
            "wrist_object_relative_translation": {
                "status": "ok",
                "available": True,
                "finite": True,
                "errors": [],
                "active_sides": ["left"],
                "thresholds": {
                    "error_median_m": 0.03,
                    "error_p95_m": 0.08,
                },
                "sides": {
                    "left": {
                        "status": "ok",
                        "available": True,
                        "finite": True,
                        "errors": [],
                        "frames": 10,
                        "error_median_m": 0.0,
                        "error_p95_m": 0.0,
                        "binding": {
                            "side": "left",
                            "wrist_site_name": "left_palm",
                            "wrist_site_id": 1,
                            "wrist_body_name": "left_wrist",
                            "wrist_body_id": 2,
                            "object_body_name": "left_object",
                            "object_body_id": 3,
                            "resolution_errors": [],
                        },
                    }
                },
            },
        },
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class IndexProvenancePropagationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = load_script("index_cell_assets")

    def args(self, trajectory: str, hand_source: str, retargeting: str):
        return SimpleNamespace(
            trajectory_6dof=trajectory,
            hand_source=hand_source,
            retargeting=retargeting,
            task="fixture_left",
            hand_type="left",
        )

    def test_dai_canonical_keypoint_paths_preserve_route_identity_in_copy_and_symlink_modes(self):
        for mode in ("copy", "symlink"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                exp = Path(directory)
                canonical_by_hand = {}
                canonical_mesh_by_hand = {}
                alias_by_hand = {}
                source_by_hand = {}
                source_mesh_by_hand = {}
                manifest_by_hand = {}
                for hand_source in ("aoe", "estimated"):
                    cell = self.index.cell_key(
                        "egoinfinity", hand_source, "do_as_i_do"
                    )
                    source = (
                        exp
                        / "intermediates"
                        / "retargeting"
                        / "do_as_i_do"
                        / cell
                        / "retargeting_outputs"
                        / "mano"
                        / "left"
                        / "fixture_left"
                        / "0"
                        / "trajectory_keypoints.npz"
                    )
                    source.parent.mkdir(parents=True)
                    source.write_bytes(f"{hand_source}-route".encode())
                    source_mesh = (
                        source.parents[4]
                        / "assets"
                        / "objects"
                        / "fixture_left"
                        / "visual.obj"
                    )
                    source_mesh.parent.mkdir(parents=True)
                    source_mesh.write_text(
                        f"v {len(hand_source)} 0 0\n", encoding="utf-8"
                    )
                    manifest = source.parents[5] / "raw_dir" / "adapter_manifest.json"
                    write_json(manifest, route_payload("egoinfinity", hand_source))
                    retargeting_dir = (
                        exp
                        / "assets"
                        / "cells"
                        / cell
                        / "retargeting"
                        / "do_as_i_do"
                    )
                    canonical, alias = self.index.dai_indexed_keypoint_paths(
                        retargeting_dir,
                        hand_type="left",
                        task="fixture_left",
                        data_id=0,
                    )
                    self.index.link_or_copy(source, canonical, mode, exp)
                    self.index.link_or_copy(canonical, alias, mode, exp)
                    canonical_mesh = self.index.dai_indexed_visual_mesh_path(
                        retargeting_dir, task="fixture_left"
                    )
                    self.index.link_or_copy(source_mesh, canonical_mesh, mode, exp)

                    result = self.index.validate_route_input(
                        adapter_manifest=manifest,
                        source_keypoints=source,
                        processed_keypoints=canonical,
                        processed_backend="do_as_i_do",
                        source_object_mesh=source_mesh,
                        processed_object_mesh=canonical_mesh,
                        trajectory="egoinfinity",
                        hand_source=hand_source,
                        task="fixture_left",
                        hand_type="left",
                        data_id=0,
                    )
                    self.assertEqual(result["status"], "ok", result["errors"])
                    self.assertEqual(
                        canonical.parts[-6:],
                        (
                            "trajectories",
                            "mano",
                            "left",
                            "fixture_left",
                            "0",
                            "trajectory_keypoints.npz",
                        ),
                    )
                    self.assertEqual(
                        alias,
                        retargeting_dir / "trajectories" / "trajectory_keypoints.npz",
                    )
                    self.assertEqual(alias.read_bytes(), source.read_bytes())
                    self.assertEqual(alias.is_symlink(), mode == "symlink")
                    self.assertEqual(canonical_mesh.is_symlink(), mode == "symlink")
                    self.assertEqual(canonical_mesh.read_bytes(), source_mesh.read_bytes())
                    canonical_by_hand[hand_source] = canonical
                    canonical_mesh_by_hand[hand_source] = canonical_mesh
                    alias_by_hand[hand_source] = alias
                    source_by_hand[hand_source] = source
                    source_mesh_by_hand[hand_source] = source_mesh
                    manifest_by_hand[hand_source] = manifest

                self.assertNotEqual(
                    alias_by_hand["aoe"].read_bytes(),
                    alias_by_hand["estimated"].read_bytes(),
                )
                cross_route = self.index.validate_route_input(
                    adapter_manifest=manifest_by_hand["aoe"],
                    source_keypoints=source_by_hand["aoe"],
                    processed_keypoints=canonical_by_hand["estimated"],
                    processed_backend="do_as_i_do",
                    source_object_mesh=source_mesh_by_hand["aoe"],
                    processed_object_mesh=canonical_mesh_by_hand["estimated"],
                    trajectory="egoinfinity",
                    hand_source="aoe",
                    task="fixture_left",
                    hand_type="left",
                    data_id=0,
                )
                self.assertEqual(cross_route["status"], "invalid")
                self.assertTrue(
                    any(
                        error.startswith(
                            "processed_keypoints_not_exact_dai_processed_cell"
                        )
                        for error in cross_route["errors"]
                    )
                )
                self.assertIn(
                    "source_processed_sha256_mismatch", cross_route["errors"]
                )

    def test_dai_cell_uses_exact_adapter_and_route_qc_not_generic_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory)
            args = self.args("egoinfinity", "aoe", "do_as_i_do")
            key = self.index.cell_key("egoinfinity", "aoe", "do_as_i_do")
            adapter_path = (
                exp
                / "intermediates/retargeting/do_as_i_do"
                / key
                / "raw_dir/adapter_manifest.json"
            )
            adapter = route_payload("egoinfinity", "aoe")
            write_json(adapter_path, adapter)
            qc = route_qc("egoinfinity", "aoe")
            write_json(
                exp
                / "assets/cells"
                / key
                / "retargeting/do_as_i_do/metadata/route_input_qc.json",
                qc,
            )
            write_json(
                exp / "assets/trajectory_6dof/egoinfinity/fixture_left/egoinfinity_adapter_manifest.json",
                route_payload("egoinfinity", "estimated"),
            )

            provenance, evidence = self.index.exact_route_provenance(exp, args, key)

        self.assertEqual(provenance["hand_source"], "aoe")
        self.assertEqual(provenance["route_input_qc"], qc)
        self.assertEqual(provenance["canonical_transform"], adapter["canonical_transform"])
        self.assertIn(key, evidence["adapter_manifest"])
        self.assertEqual(
            self.index.validate_exact_route_provenance(
                provenance, trajectory="egoinfinity", hand_source="aoe"
            ),
            [],
        )

    def test_missing_pure_dai_adapter_is_not_fabricated(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory)
            args = self.args("do_as_i_do", "estimated", "do_as_i_do")
            key = self.index.cell_key("do_as_i_do", "estimated", "do_as_i_do")
            provenance, _ = self.index.exact_route_provenance(exp, args, key)

        self.assertIsNone(provenance["hand_source"])
        self.assertIsNone(provenance["object_track_source"])
        errors = self.index.validate_exact_route_provenance(
            provenance, trajectory="do_as_i_do", hand_source="estimated"
        )
        self.assertTrue(any("object_track_source mismatch" in error for error in errors))
        self.assertIn("missing route_input_qc provenance", errors)

    def test_spider_propagates_exact_manifests_and_invalid_qc_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory)
            trajectory = "egoinfinity"
            hand_source = "aoe"
            args = self.args(trajectory, hand_source, "spider")
            spider_key = self.index.cell_key(trajectory, hand_source, "spider")
            dai_key = self.index.cell_key(trajectory, hand_source, "do_as_i_do")
            adapter_path = (
                exp
                / "intermediates/retargeting/do_as_i_do"
                / dai_key
                / "raw_dir/adapter_manifest.json"
            )
            adapter = route_payload(trajectory, hand_source)
            raw_hand = adapter_path.parent / "raw/all_hand_meshes.npz"
            raw_hand.parent.mkdir(parents=True, exist_ok=True)
            raw_hand.write_bytes(b"exact-route-aoe-hand")
            raw_layout = (
                adapter_path.parent
                / "obj_tracking_out/object/combined_visualization/"
                "layout_camera_frame_optimized.json"
            )
            raw_layout.parent.mkdir(parents=True, exist_ok=True)
            raw_layout.write_text('{"objects": []}', encoding="utf-8")
            adapter["layout"] = str(raw_layout)
            write_json(adapter_path, adapter)
            root = exp / "intermediates/retargeting/spider" / spider_key
            source_keypoints = (
                exp
                / "intermediates/retargeting/do_as_i_do"
                / dai_key
                / "retargeting_outputs/mano/left/fixture_left/0/"
                "trajectory_keypoints.npz"
            )
            source_keypoints.parent.mkdir(parents=True, exist_ok=True)
            source_keypoints.write_bytes(b"exact-route-keypoints")
            def file_binding(path: Path) -> dict:
                return {
                    "path": str(path.resolve()),
                    "sha256": self.index.file_sha256(path),
                }

            raw_to_processed_invariance = {
                "schema_version": 3,
                "full_relative_se3_checked": True,
                "status": "ok",
                "production_eligible": True,
                "cross_hand_source_comparison": False,
                "trajectory_6dof": trajectory,
                "hand_source": hand_source,
                "hand_type": "left",
                "joint_fit_components": ["object", "left"],
                "route_binding": {
                    "trajectory_6dof": trajectory,
                    "hand_source": hand_source,
                    "hand_type": "left",
                    "requested_sides": ["left"],
                    "adapter_manifest": file_binding(adapter_path),
                    "raw_inputs": {
                        "hand_npz": file_binding(raw_hand),
                        "object_layout": file_binding(raw_layout),
                    },
                    "processed_keypoints": file_binding(source_keypoints),
                },
                "adapter_manifest": str(adapter_path.resolve()),
                "adapter_manifest_sha256": self.index.file_sha256(adapter_path),
                "raw_hand_npz": str(raw_hand.resolve()),
                "raw_hand_sha256": self.index.file_sha256(raw_hand),
                "raw_object_layout": str(raw_layout.resolve()),
                "raw_object_layout_sha256": self.index.file_sha256(raw_layout),
                "processed_keypoints": str(source_keypoints.resolve()),
                "processed_keypoints_sha256": self.index.file_sha256(source_keypoints),
            }
            write_json(
                source_keypoints.with_name("raw_to_processed_hoi_invariance.json"),
                raw_to_processed_invariance,
            )
            root.mkdir(parents=True, exist_ok=True)
            consumed_scene = root / "scene_act.xml"
            consumed_scene.write_text("<mujoco/>", encoding="utf-8")
            payload = {
                **adapter,
                "trajectory_6dof": trajectory,
                "hand_source": hand_source,
                "expected_dai_cell_key": dai_key,
                "task": "fixture_left",
                "validated_hand_type": "left",
                "data_id": 0,
                "adapter_manifest": str(adapter_path),
                "source_keypoints": str(source_keypoints),
                "route_input_qc": route_qc(
                    trajectory, hand_source, processed_backend="spider"
                ),
                "post_preprocess_semantic_qc": postprocess_qc(
                    trajectory, hand_source
                ),
                "object_tracking_quality": hard_tracking_report(),
                "dai_preprocess_invariance": {
                    "status": "pass",
                    "production_eligible": True,
                    "object_only_freeze": {
                        "requested": False,
                        "applied": False,
                        "diagnostic_only": True,
                    },
                },
                "raw_to_processed_hoi_invariance": raw_to_processed_invariance,
                "rest_support_provenance": {
                    "schema_version": 1,
                    "status": "ok",
                    "policy": {
                        "endpoints": ["start", "end"],
                        "trajectory_write_allowed": False,
                        "support_only_when_endpoint_not_in_hand": True,
                        "support_application_mode": "audit_only",
                        "diagnostic_only": True,
                        "production_scene_geometry_modified": False,
                    },
                    "supports_added": 0,
                    "trajectory_integrity": {"unchanged": True},
                    "endpoint_accounting": {
                        "records": 2,
                        "expected_records": 2,
                        "complete": True,
                    },
                    "route_binding": {
                        "cell_key": spider_key,
                        "trajectory_6dof": trajectory,
                        "hand_source": hand_source,
                        "task": "fixture_left",
                        "hand_type": "left",
                        "data_id": 0,
                        "adapter_manifest": str(adapter_path),
                        "adapter_manifest_sha256": self.index.file_sha256(adapter_path),
                        "source_keypoints": str(source_keypoints),
                        "source_keypoints_sha256": self.index.file_sha256(source_keypoints),
                    },
                    "outputs": {
                        "scene_act_xml": str(consumed_scene),
                        "scene_act_sha256_after": self.index.file_sha256(consumed_scene),
                    },
                },
            }
            write_json(root / "spider_input_manifest.json", payload)
            write_json(root / "spider_output_manifest.json", payload)
            write_json(
                root / "spider_postprocess_semantic_qc.json",
                payload["post_preprocess_semantic_qc"],
            )

            provenance, _ = self.index.exact_route_provenance(exp, args, spider_key)
            original = self.index.valid_spider_alignment
            self.index.valid_spider_alignment = lambda _root: True
            try:
                valid = self.index.validate_index_provenance(exp, args, spider_key)
                legacy_tracking_payload = {
                    **payload,
                    "object_tracking_quality": {"status": "ok", "errors": []},
                }
                write_json(root / "spider_output_manifest.json", legacy_tracking_payload)
                legacy_tracking = self.index.validate_index_provenance(
                    exp, args, spider_key
                )
                invalid_post_payload = {
                    **payload,
                    "post_preprocess_semantic_qc": {
                        **payload["post_preprocess_semantic_qc"],
                        "status": "invalid",
                        "errors": ["qpos_obj_left changed"],
                    },
                }
                write_json(root / "spider_output_manifest.json", invalid_post_payload)
                write_json(root / "spider_input_manifest.json", invalid_post_payload)
                invalid_post = self.index.validate_index_provenance(
                    exp, args, spider_key
                )
                # The indexed standalone report is authoritative: stale
                # embedded ok payloads must not mask an invalid report.
                write_json(root / "spider_output_manifest.json", payload)
                write_json(root / "spider_input_manifest.json", payload)
                write_json(
                    root / "spider_postprocess_semantic_qc.json",
                    invalid_post_payload["post_preprocess_semantic_qc"],
                )
                invalid_standalone = self.index.validate_index_provenance(
                    exp, args, spider_key
                )
                wrong_task_payload = {**payload, "task": "other_left"}
                write_json(root / "spider_output_manifest.json", wrong_task_payload)
                write_json(root / "spider_input_manifest.json", wrong_task_payload)
                write_json(
                    root / "spider_postprocess_semantic_qc.json",
                    payload["post_preprocess_semantic_qc"],
                )
                wrong_task = self.index.validate_index_provenance(
                    exp, args, spider_key
                )
                payload["route_input_qc"] = {
                    **payload["route_input_qc"],
                    "status": "invalid",
                    "errors": ["cross-hand source"],
                }
                write_json(root / "spider_output_manifest.json", payload)
                write_json(root / "spider_input_manifest.json", payload)
                invalid = self.index.validate_index_provenance(exp, args, spider_key)
            finally:
                self.index.valid_spider_alignment = original

        self.assertEqual(provenance["hand_source"], hand_source)
        self.assertEqual(provenance["adapter_rigid_invariance"]["status"], "ok")
        self.assertEqual(
            provenance["post_preprocess_semantic_qc"],
            postprocess_qc(trajectory, hand_source),
        )
        self.assertEqual(valid["status"], "ok", valid["errors"])
        self.assertEqual(legacy_tracking["status"], "invalid")
        self.assertTrue(
            any("SPIDER hard tracking admission" in error for error in legacy_tracking["errors"])
        )
        self.assertEqual(invalid_post["status"], "invalid")
        self.assertTrue(
            any(
                "post_preprocess_semantic_qc is not ok" in error
                for error in invalid_post["errors"]
            )
        )
        self.assertEqual(invalid_standalone["status"], "invalid")
        self.assertTrue(
            any(
                "post_preprocess_semantic_qc" in error
                for error in invalid_standalone["errors"]
            )
        )
        self.assertEqual(wrong_task["status"], "invalid")
        self.assertTrue(
            any("task does not match cell" in error for error in wrong_task["errors"])
        )
        self.assertEqual(invalid["status"], "invalid")
        self.assertIn("route_input_qc is not ok", invalid["errors"])


class ReuseProvenancePropagationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reuse = load_script("reuse_v4_for_12_demos")

    def test_spider_cell_manifest_payload_comes_from_exact_route(self):
        with tempfile.TemporaryDirectory() as directory:
            dst = Path(directory)
            trajectory = "do_as_i_do"
            hand_source = "estimated"
            spider_key = self.reuse.cell_key(trajectory, hand_source, "spider")
            dai_key = self.reuse.cell_key(trajectory, hand_source, "do_as_i_do")
            adapter_path = (
                dst
                / "intermediates/retargeting/do_as_i_do"
                / dai_key
                / "raw_dir/adapter_manifest.json"
            )
            adapter = route_payload(trajectory, hand_source)
            write_json(adapter_path, adapter)
            payload = {
                **adapter,
                "trajectory_6dof": trajectory,
                "hand_source": hand_source,
                "expected_dai_cell_key": dai_key,
                "adapter_manifest": str(adapter_path),
                "route_input_qc": route_qc(trajectory, hand_source),
            }
            root = dst / "intermediates/retargeting/spider" / spider_key
            write_json(root / "spider_input_manifest.json", payload)
            write_json(root / "spider_output_manifest.json", payload)

            provenance = self.reuse.cell_object_provenance(
                dst,
                trajectory,
                "spider",
                hand_source,
                None,
                SimpleNamespace(),
                {"dai_native_adapter_manifest": dst / "generic_adapter_manifest.json"},
            )
            conflicting_adapter = {**adapter, "hand_source": "aoe"}
            write_json(adapter_path, conflicting_adapter)
            validation = self.reuse.validate_cell_provenance(
                dst=dst,
                source=dst,
                trajectory=trajectory,
                retargeting="spider",
                hand_source=hand_source,
                robot_src=None,
                srcs={
                    "dai_keypoints": None,
                    "dai_object_visual": None,
                    "dai_object_convex": None,
                },
                args=SimpleNamespace(spider_fuse_ego_object_for_dai=False),
            )

        for field in self.reuse.ROUTE_PROVENANCE_FIELDS:
            expected = payload[field]
            self.assertEqual(provenance[field], expected, field)
        self.assertTrue(provenance["baseline_only"])
        self.assertEqual(validation["status"], "invalid")
        self.assertIn(
            "SPIDER hand_source does not match exact DAI source adapter",
            validation["errors"],
        )

    def test_reuse_pure_dai_missing_adapter_remains_explicitly_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            provenance = self.reuse.cell_object_provenance(
                Path(directory),
                "do_as_i_do",
                "do_as_i_do",
                "aoe",
                None,
                SimpleNamespace(),
                {},
            )

        self.assertIsNone(provenance["object_track_source"])
        self.assertIsNone(provenance["hand_source"])
        self.assertTrue(provenance["baseline_only"])
        self.assertEqual(provenance["route"], "unverified_object_source__do_as_i_do_retargeting")
        self.assertIn(
            "missing route_input_qc provenance",
            self.reuse.validate_route_provenance_payload(
                provenance, trajectory="do_as_i_do", hand_source="aoe"
            ),
        )

    def test_dai_spider_preflight_reads_exact_bundle_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exact = root / "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do/raw_dir/adapter_manifest.json"
            write_json(
                exact,
                {
                    **route_payload("do_as_i_do", "aoe"),
                    "object_pose_quality": {"status": "invalid", "errors": ["pose_jump"]},
                    "retarget_input_qc": {"status": "ok", "errors": []},
                },
            )
            result = self.reuse.dai_native_spider_input_preflight(
                {"adapter_manifest": exact}, "aoe"
            )
            missing = self.reuse.dai_native_spider_input_preflight(None, "estimated")

        self.assertEqual(result["status"], "invalid")
        self.assertEqual(result["provenance_scope"], "exact_dai_cell")
        self.assertIn("source_pose_jump", result["errors"])
        self.assertEqual(missing["status"], "invalid")
        self.assertIn("missing exact DAI hand-source input bundle", missing["errors"])


if __name__ == "__main__":
    unittest.main()
