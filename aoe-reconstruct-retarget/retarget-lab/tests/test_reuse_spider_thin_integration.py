from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reuse_v4_for_12_demos.py"
PRISTINE_SPIDER_COMMIT = "2e54f19ee6ab8e0690c0f585beb1e9f8f53a6898"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_native_runner_evidence(
    *,
    native_run: Path,
    input_root: Path,
    task: str,
    hand_type: str,
    returncode: int,
) -> None:
    keypoints = (
        input_root
        / "mano"
        / hand_type
        / task
        / "0"
        / "trajectory_keypoints.npz"
    )
    keypoints.parent.mkdir(parents=True, exist_ok=True)
    keypoints.write_bytes(b"route-keypoints")
    robot_assets = input_root / "assets" / "robots" / "sharpa"
    robot_assets.mkdir(parents=True, exist_ok=True)
    (robot_assets / "robot.xml").write_text("<mujoco/>", encoding="utf-8")
    native_run.mkdir(parents=True, exist_ok=True)
    (native_run / "vanilla_spider_run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "pristine_official_spider_input_staging_only",
                "backend": {
                    "commit": PRISTINE_SPIDER_COMMIT,
                    "clean_before_launch": True,
                    "patches_applied": [],
                    "behavior_overrides": [],
                },
                "route": {
                    "dataset_name": "do_as_i_do",
                    "task": task,
                    "robot_type": "sharpa",
                    "embodiment_type": hand_type,
                    "data_id": 0,
                    "ref_dt": 1.0 / 30.0,
                },
                "input_timebase_binding": "exact_dai_keypoint_frame_grid",
                "input_root": str(input_root.resolve()),
                "trajectory_keypoints": {
                    "source": str(keypoints.resolve()),
                    "source_sha256": sha256(keypoints),
                    "byte_identical": True,
                    "modified": False,
                },
                "robot_assets": {
                    "source": str(robot_assets.resolve()),
                    "byte_identical": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (native_run / "vanilla_spider_outcome.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "pristine_official_spider_native_returncode",
                "stage": "run_mjwp",
                "returncode": returncode,
                "backend_commit_before": PRISTINE_SPIDER_COMMIT,
                "backend_commit_after": PRISTINE_SPIDER_COMMIT,
                "backend_clean_after": True,
                "backend_patches_applied": [],
                "backend_behavior_overrides": [],
            }
        ),
        encoding="utf-8",
    )


def load_module():
    spec = importlib.util.spec_from_file_location(
        "reuse_spider_thin_integration", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReuseSpiderThinIntegrationTests(unittest.TestCase):
    def test_ref_dt_must_come_from_agreeing_route_bound_evidence(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "retargeting_outputs"
            task_info = (
                input_root / "mano" / "left" / "fixture_left" / "task_info.json"
            )
            task_info.parent.mkdir(parents=True)
            task_info.write_text(
                json.dumps(
                    {
                        "task": "fixture_left",
                        "dataset_name": "do_as_i_do",
                        "robot_type": "mano",
                        "embodiment_type": "left",
                        "data_id": 0,
                        "ref_dt": 1.0 / 30.0,
                    }
                ),
                encoding="utf-8",
            )
            adapter = root / "adapter_manifest.json"
            adapter.write_text(
                json.dumps(
                    {
                        "temporal_alignment": {
                            "status": "ok",
                            "hard_gate_passed": True,
                            "target_clock": {"fps": 30.0, "frame_count": 90},
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = module.audited_spider_input_timebase(
                input_root,
                task="fixture_left",
                hand_type="left",
                data_id=0,
                adapter_manifest=adapter,
            )

            self.assertEqual(result["status"], "ok")
            self.assertAlmostEqual(result["ref_dt"], 1.0 / 30.0)
            self.assertEqual(len(result["evidence"]), 2)
            self.assertFalse(result["backend_tuning"])

            adapter.write_text(
                json.dumps(
                    {
                        "temporal_alignment": {
                            "status": "ok",
                            "hard_gate_passed": True,
                            "target_clock": {"fps": 25.0, "frame_count": 75},
                        }
                    }
                ),
                encoding="utf-8",
            )
            conflict = module.audited_spider_input_timebase(
                input_root,
                task="fixture_left",
                hand_type="left",
                data_id=0,
                adapter_manifest=adapter,
            )
            self.assertEqual(conflict["status"], "invalid")
            self.assertIn(
                "spider_input_timebase_evidence_conflict", conflict["errors"]
            )

    def test_ref_dt_missing_route_evidence_fails_closed(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = module.audited_spider_input_timebase(
                root / "retargeting_outputs",
                task="fixture_left",
                hand_type="left",
                data_id=0,
                adapter_manifest=root / "adapter_manifest.json",
            )

        self.assertEqual(result["status"], "invalid")
        self.assertIsNone(result["ref_dt"])
        self.assertIn("spider_input_timebase_no_audited_ref_dt", result["errors"])

    def test_native_rc_and_plain_video_drive_report_without_legacy_qc(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            dst = Path(temp_dir) / "matrix"
            route = "traj_egoinfinity__hand_estimated__retarget_spider"
            route_root = (
                dst
                / "intermediates"
                / "retargeting"
                / "spider"
                / route
            )
            route_root.mkdir(parents=True)
            preflight = route_root / "spider_source_input_preflight.json"
            preflight.write_text(
                json.dumps(
                    {
                        "route": route,
                        "trajectory_6dof": "egoinfinity",
                        "hand_source": "estimated",
                        "status": "ok",
                        "input_rc": 0,
                        "input_assessment": {"status": "ok"},
                        "resolved_hand_type": "left",
                        "resolved_task": "fixture_left",
                    }
                ),
                encoding="utf-8",
            )
            input_root = (
                Path(temp_dir)
                / "traj_egoinfinity__hand_estimated__retarget_do_as_i_do"
                / "retargeting_outputs"
            )
            native_run = route_root / "native_runs" / "attempt01"
            write_native_runner_evidence(
                native_run=native_run,
                input_root=input_root,
                task="fixture_left",
                hand_type="left",
                returncode=0,
            )
            output = (
                native_run
                / "dataset"
                / "processed"
                / "do_as_i_do"
                / "sharpa"
                / "left"
                / "fixture_left"
                / "0"
            )
            output.mkdir(parents=True)
            video = output / "visualization_mjwp.mp4"
            video.write_bytes(b"plain-native-video")
            log = native_run / "native.log"
            log.write_text(
                "Final object tracking error: pos=0.12, quat=2.3\n",
                encoding="utf-8",
            )
            binding = module.write_native_spider_run_binding(
                route_root=route_root,
                route=route,
                trajectory_6dof="egoinfinity",
                hand_source="estimated",
                hand_type="left",
                task="fixture_left",
                robot_type="sharpa",
                data_id=0,
                ref_dt_evidence={"status": "ok", "ref_dt": 1.0 / 30.0},
                input_root=input_root,
                native_run_dir=native_run,
                preflight_path=preflight,
                log_path=log,
                returncode=0,
            )
            args = SimpleNamespace(
                prefer_spider_ik_video=False,
                allow_spider_mjwp_fallback_video=False,
                include_numerically_invalid_aligned_robot=False,
            )

            found = module.find_spider_robot(dst, route)
            report = module.spider_report_item(args, dst, route)
            binding_errors = module.existing_spider_input_binding_errors(
                route_root,
                route=route,
                trajectory_6dof="egoinfinity",
                hand_source="estimated",
                resolved_hand_type="left",
                resolved_task="fixture_left",
                preflight_payload=json.loads(preflight.read_text(encoding="utf-8")),
            )

            self.assertEqual(found, video)
            self.assertIsNotNone(report)
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["returncode"], 0)
            self.assertTrue(report["native_returncode_authoritative"])
            self.assertEqual(report["external_qc"]["role"], "diagnostic_only")
            self.assertEqual(binding_errors, [])
            self.assertEqual(
                binding["native_metrics"]["final_object_position_error_m"], 0.12
            )
            self.assertFalse((route_root / "spider_output_manifest.json").exists())

            (route_root / "spider_output_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            self.assertIsNone(module.find_spider_robot(dst, route))

    def test_native_failure_remains_failed_even_when_video_exists(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            dst = Path(temp_dir) / "matrix"
            route = "traj_egoinfinity__hand_estimated__retarget_spider"
            route_root = (
                dst
                / "intermediates"
                / "retargeting"
                / "spider"
                / route
            )
            route_root.mkdir(parents=True)
            preflight = route_root / "spider_source_input_preflight.json"
            preflight.write_text(
                json.dumps(
                    {
                        "route": route,
                        "trajectory_6dof": "egoinfinity",
                        "hand_source": "estimated",
                        "status": "ok",
                        "input_rc": 0,
                        "resolved_hand_type": "left",
                        "resolved_task": "fixture_left",
                    }
                ),
                encoding="utf-8",
            )
            native_run = route_root / "native_runs" / "attempt01"
            input_root = (
                Path(temp_dir)
                / "traj_egoinfinity__hand_estimated__retarget_do_as_i_do"
                / "retargeting_outputs"
            )
            write_native_runner_evidence(
                native_run=native_run,
                input_root=input_root,
                task="fixture_left",
                hand_type="left",
                returncode=7,
            )
            output = (
                native_run
                / "dataset"
                / "processed"
                / "do_as_i_do"
                / "sharpa"
                / "left"
                / "fixture_left"
                / "0"
            )
            output.mkdir(parents=True)
            (output / "visualization_mjwp.mp4").write_bytes(b"partial-video")
            module.write_native_spider_run_binding(
                route_root=route_root,
                route=route,
                trajectory_6dof="egoinfinity",
                hand_source="estimated",
                hand_type="left",
                task="fixture_left",
                robot_type="sharpa",
                data_id=0,
                ref_dt_evidence={"status": "ok", "ref_dt": 1.0 / 30.0},
                input_root=input_root,
                native_run_dir=native_run,
                preflight_path=preflight,
                log_path=native_run / "native.log",
                returncode=7,
            )
            args = SimpleNamespace(
                prefer_spider_ik_video=False,
                allow_spider_mjwp_fallback_video=False,
                include_numerically_invalid_aligned_robot=False,
            )

            self.assertIsNone(module.find_spider_robot(dst, route))
            report = module.spider_report_item(args, dst, route)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["returncode"], 7)


if __name__ == "__main__":
    unittest.main()
