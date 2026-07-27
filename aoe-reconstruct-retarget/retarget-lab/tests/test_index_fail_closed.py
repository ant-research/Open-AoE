from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_index_module():
    path = ROOT / "scripts" / "index_cell_assets.py"
    spec = importlib.util.spec_from_file_location("index_cell_assets_fail_closed", path)
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


def valid_adapter(trajectory: str = "egoinfinity", hand_source: str = "aoe") -> dict:
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
        },
        "hoi_refinement": {"applied": False, "method": "none"},
        "hoi_contact_alignment": {
            "applied": False,
            "diagnostic_only": True,
            "transforms_modified": False,
        },
        "adapter_rigid_invariance": {
            "status": "ok",
            "max_abs_error": 0.0,
            "true_pre_post_hand_comparison": True,
        },
        "retarget_input_qc": {"status": "ok", "errors": []},
    }


class IndexFailClosedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_index_module()

    def test_exact_raw_dir_preflight_accepts_clean_route_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            (raw_dir / "adapter_manifest.json").write_text(
                json.dumps(valid_adapter()), encoding="utf-8"
            )
            result = self.module.validate_dai_raw_dir_preflight(
                raw_dir,
                trajectory="egoinfinity",
                hand_source="aoe",
            )

        self.assertEqual(result["status"], "ok", result["errors"])
        self.assertEqual(
            result["expected_cell_key"],
            "traj_egoinfinity__hand_aoe__retarget_do_as_i_do",
        )

    def test_exact_raw_dir_preflight_rejects_wrong_hand_and_invalid_adapter_qc(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            adapter = valid_adapter(hand_source="estimated")
            adapter["retarget_input_qc"] = {
                "status": "invalid",
                "errors": ["object_only_hoi_translation_applied"],
            }
            (raw_dir / "adapter_manifest.json").write_text(
                json.dumps(adapter), encoding="utf-8"
            )
            result = self.module.validate_dai_raw_dir_preflight(
                raw_dir,
                trajectory="egoinfinity",
                hand_source="aoe",
            )

        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("hand_source mismatch" in error for error in result["errors"]))
        self.assertTrue(any("retarget_input_qc is not ok" in error for error in result["errors"]))

    def test_exact_raw_dir_preflight_rejects_schema2_or_self_comparison_invariance(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_dir = Path(directory)
            adapter = valid_adapter()
            adapter["adapter_schema_version"] = 2
            adapter["adapter_rigid_invariance"].pop("true_pre_post_hand_comparison")
            (raw_dir / "adapter_manifest.json").write_text(
                json.dumps(adapter), encoding="utf-8"
            )
            result = self.module.validate_dai_raw_dir_preflight(
                raw_dir,
                trajectory="egoinfinity",
                hand_source="aoe",
            )

        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("adapter_schema_version must be >=3" in error for error in result["errors"]))
        self.assertTrue(any("true pre/post hand comparison" in error for error in result["errors"]))

    def test_preflight_cli_returns_nonzero_and_persists_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            output = root / "reports" / "preflight.json"
            argv = [
                "index_cell_assets.py",
                "--run-name",
                "fixture",
                "--trajectory-6dof",
                "egoinfinity",
                "--hand-source",
                "aoe",
                "--retargeting",
                "do_as_i_do",
                "--preflight-raw-dir",
                str(raw_dir),
                "--preflight-output",
                str(output),
            ]
            with mock.patch.object(sys, "argv", argv):
                return_code = self.module.main()
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(return_code, 6)
        self.assertEqual(report["status"], "invalid")
        self.assertTrue(any("adapter manifest" in error for error in report["errors"]))

    def test_raw_to_processed_binding_rejects_stale_hash_and_wrong_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw_dir"
            (raw_dir / "raw").mkdir(parents=True)
            raw_hand = raw_dir / "raw" / "all_hand_meshes.npz"
            raw_hand.write_bytes(b"route-specific-hand")
            layout = (
                raw_dir
                / "obj_tracking_out/object/combined_visualization/"
                "layout_camera_frame_optimized.json"
            )
            layout.parent.mkdir(parents=True)
            layout.write_text('{"objects": []}', encoding="utf-8")
            adapter_path = raw_dir / "adapter_manifest.json"
            adapter = valid_adapter()
            adapter["layout"] = str(layout)
            adapter_path.write_text(json.dumps(adapter), encoding="utf-8")
            processed = root / "trajectory_keypoints.npz"
            processed.write_bytes(b"processed-keypoints")

            def binding(path: Path):
                return {
                    "path": str(path.resolve()),
                    "sha256": self.module.file_sha256(path),
                }

            report = {
                "schema_version": 3,
                "full_relative_se3_checked": True,
                "status": "ok",
                "production_eligible": True,
                "trajectory_6dof": "egoinfinity",
                "hand_source": "aoe",
                "hand_type": "left",
                "joint_fit_components": ["object", "left"],
                "route_binding": {
                    "trajectory_6dof": "egoinfinity",
                    "hand_source": "aoe",
                    "hand_type": "left",
                    "requested_sides": ["left"],
                    "adapter_manifest": binding(adapter_path),
                    "raw_inputs": {
                        "hand_npz": binding(raw_hand),
                        "object_layout": binding(layout),
                    },
                    "processed_keypoints": binding(processed),
                },
                "adapter_manifest": str(adapter_path.resolve()),
                "adapter_manifest_sha256": self.module.file_sha256(adapter_path),
                "raw_hand_npz": str(raw_hand.resolve()),
                "raw_hand_sha256": self.module.file_sha256(raw_hand),
                "raw_object_layout": str(layout.resolve()),
                "raw_object_layout_sha256": self.module.file_sha256(layout),
                "processed_keypoints": str(processed.resolve()),
                "processed_keypoints_sha256": self.module.file_sha256(processed),
            }

            clean = self.module.validate_raw_to_processed_hoi_binding(
                report,
                adapter_manifest=adapter_path,
                processed_keypoints=processed,
                trajectory="egoinfinity",
                hand_source="aoe",
                hand_type="left",
            )
            self.assertEqual(clean, [])

            external_layout = root / "other_route" / layout.name
            external_layout.parent.mkdir()
            external_layout.write_bytes(layout.read_bytes())
            external_adapter = dict(adapter)
            external_adapter["layout"] = str(external_layout)
            adapter_path.write_text(json.dumps(external_adapter), encoding="utf-8")
            cross_layout_report = json.loads(json.dumps(report))
            cross_layout_report["route_binding"]["adapter_manifest"] = binding(
                adapter_path
            )
            cross_layout_report["adapter_manifest_sha256"] = self.module.file_sha256(
                adapter_path
            )
            cross_layout_report["route_binding"]["raw_inputs"][
                "object_layout"
            ] = binding(external_layout)
            cross_layout_report["raw_object_layout"] = str(
                external_layout.resolve()
            )
            cross_layout_report["raw_object_layout_sha256"] = (
                self.module.file_sha256(external_layout)
            )
            cross_layout = self.module.validate_raw_to_processed_hoi_binding(
                cross_layout_report,
                adapter_manifest=adapter_path,
                processed_keypoints=processed,
                trajectory="egoinfinity",
                hand_source="aoe",
                hand_type="left",
            )
            self.assertTrue(
                any("object layout is stale or cross-route" in error for error in cross_layout),
                cross_layout,
            )
            adapter_path.write_text(json.dumps(adapter), encoding="utf-8")

            processed.write_bytes(b"mutated-after-validation")
            stale = self.module.validate_raw_to_processed_hoi_binding(
                report,
                adapter_manifest=adapter_path,
                processed_keypoints=processed,
                trajectory="egoinfinity",
                hand_source="aoe",
                hand_type="left",
            )
            self.assertTrue(any("SHA256 mismatch" in error for error in stale), stale)

            wrong_route = self.module.validate_raw_to_processed_hoi_binding(
                report,
                adapter_manifest=adapter_path,
                processed_keypoints=processed,
                trajectory="egoinfinity",
                hand_source="estimated",
                hand_type="left",
            )
            self.assertTrue(any("hand_source mismatch" in error for error in wrong_route))

    def test_spider_runs_exact_binding_gate_before_dataset_preprocessing(self):
        wrapper = (ROOT / "scripts" / "run_spider_retarget.sh").read_text(
            encoding="utf-8"
        )
        indexer = (ROOT / "scripts" / "index_cell_assets.py").read_text(
            encoding="utf-8"
        )
        matrix_runner = (ROOT / "scripts" / "reuse_v4_for_12_demos.py").read_text(
            encoding="utf-8"
        )
        gate_marker = 'expected one exact task_info, found'
        preprocessing_marker = 'dataset_root="$staging/dataset"'
        self.assertIn(gate_marker, wrapper)
        self.assertIn('"byte_identical": digest(staged_keypoints)', wrapper)
        self.assertLess(wrapper.index(gate_marker), wrapper.index(preprocessing_marker))
        self.assertIn(
            'bound_spider_artifact(roots["root"], "aligned_trajectory")',
            indexer,
        )
        self.assertIn(
            'bound_spider_artifact(roots["root"], "aligned_video")',
            indexer,
        )
        self.assertIn(
            'bound_spider_artifact(roots["root"], "render_scene")',
            indexer,
        )
        self.assertIn("require_tracking=False", matrix_runner)
        self.assertIn(
            'alignment.get("aligned_video")',
            matrix_runner,
        )

    def test_manifest_admission_rejects_missing_route_qc_and_provenance(self):
        manifest = {
            "missing": ["cell.robot"],
            "provenance": {
                "status": "invalid",
                "invalid_reasons": ["wrong hand source"],
                "route_input_qc": {
                    "status": "invalid",
                    "errors": ["source_processed_sha256_mismatch"],
                },
            },
            "provenance_validation": {
                "status": "invalid",
                "errors": ["route_input_qc is not ok"],
            },
        }

        errors = self.module.manifest_admission_errors(
            manifest, retargeting="do_as_i_do"
        )

        self.assertTrue(any("missing required indexed assets" in error for error in errors))
        self.assertTrue(any("provenance_validation is not ok" in error for error in errors))
        self.assertTrue(any("manifest provenance is invalid" in error for error in errors))
        self.assertTrue(any("route_input_qc is not ok" in error for error in errors))

    def test_postprocess_semantic_qc_is_required_only_for_spider_admission(self):
        manifest = {
            "missing": [],
            "provenance": {
                "route_input_qc": {"status": "ok", "errors": []},
            },
            "provenance_validation": {"status": "ok", "errors": []},
        }

        dai_errors = self.module.manifest_admission_errors(
            manifest, retargeting="do_as_i_do"
        )
        missing_spider_errors = self.module.manifest_admission_errors(
            manifest, retargeting="spider"
        )
        manifest["provenance"]["post_preprocess_semantic_qc"] = {
            "status": "invalid",
            "errors": ["core qpos_obj_left changed"],
        }
        invalid_spider_errors = self.module.manifest_admission_errors(
            manifest, retargeting="spider"
        )

        self.assertEqual(dai_errors, [])
        self.assertIn(
            "missing post_preprocess_semantic_qc in exact SPIDER provenance",
            missing_spider_errors,
        )
        self.assertTrue(
            any(
                "post_preprocess_semantic_qc is not ok" in error
                for error in invalid_spider_errors
            )
        )
        source = (ROOT / "scripts" / "index_cell_assets.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '"retargeting.quality.post_preprocess_semantic_qc"', source
        )

    def test_pristine_wrapper_validates_before_launch_and_review_gates_before_index(self):
        wrapper = (
            ROOT / "scripts" / "run_do_as_i_do_official_retarget.sh"
        ).read_text(encoding="utf-8")
        pipeline = (ROOT / "scripts" / "run_v4_two_full_pipelines.sh").read_text(
            encoding="utf-8"
        )

        self.assertLess(
            wrapper.index('validate_input_schema "$raw_dir"'),
            wrapper.index('env -i "${backend_env[@]}" "$python_bin" "${backend_argv[@]}"'),
        )
        self.assertIn("input_schema_path=", wrapper)
        self.assertIn("backend_rc_authoritative", wrapper)
        self.assertLess(
            pipeline.index('"${cell}_materialize_and_qc"'),
            pipeline.index('"${cell}_index"'),
        )
        self.assertIn("native DAI completed successfully; review/index", pipeline)

    def test_dai_index_qc_uses_canonical_identity_path_and_complete_identity(self):
        source = (ROOT / "scripts" / "index_cell_assets.py").read_text(
            encoding="utf-8"
        )
        call_start = source.index("route_input_qc = validate_route_input(")
        call_end = source.index("route_input_qc_path =", call_start)
        call = source[call_start:call_end]

        self.assertIn("dai_indexed_keypoint_paths(", source)
        self.assertIn('/ "mano"', source)
        self.assertIn("processed_keypoints=processed_keypoints", call)
        self.assertIn('processed_backend="do_as_i_do"', call)
        self.assertIn('source_object_mesh=roots["object_visual"]', call)
        self.assertIn("processed_object_mesh=processed_object_mesh", call)
        self.assertIn("task=args.task", call)
        self.assertIn("hand_type=args.hand_type", call)
        self.assertIn("data_id=0", call)
        self.assertIn("dai_indexed_visual_mesh_path(", source)
        self.assertIn('"retargeting.object_mesh.visual.route_canonical"', source)


if __name__ == "__main__":
    unittest.main()
