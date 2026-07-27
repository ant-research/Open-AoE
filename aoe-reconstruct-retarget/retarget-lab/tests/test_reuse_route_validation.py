from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reuse_v4_for_12_demos.py"


def load_module():
    spec = importlib.util.spec_from_file_location("reuse_route_validation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReuseRouteValidationTest(unittest.TestCase):
    def test_spider_invalid_input_preflight_is_route_and_sha_bound(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            matrix = root / "matrix"
            adapter = source / "adapter_manifest.json"
            adapter.parent.mkdir(parents=True)
            adapter.write_text(
                json.dumps(
                    {
                        "adapter_schema_version": 3,
                        "hand_source": "aoe",
                        "object_pose_quality": {"status": "invalid", "errors": ["lost"]},
                        "retarget_input_qc": {"status": "invalid"},
                        "adapter_rigid_invariance": {"status": "ok"},
                    }
                ),
                encoding="utf-8",
            )
            bundle = {"adapter_manifest": adapter}
            assessment = module.dai_native_spider_input_preflight(bundle, "aoe")
            route = "traj_do_as_i_do__hand_aoe__retarget_spider"
            payload = module.spider_source_input_preflight_payload(
                route=route,
                trajectory_6dof="do_as_i_do",
                hand_source="aoe",
                input_bundle=bundle,
                assessment=assessment,
            )
            adapter_sha = module.file_sha256(adapter)
            preflight = (
                matrix
                / "intermediates"
                / "retargeting"
                / "spider"
                / route
                / "spider_source_input_preflight.json"
            )
            module.write_json(preflight, payload)
            reference = {
                "path": module.rel(preflight, matrix),
                "path_base": "matrix_root",
                "sha256": module.file_sha256(preflight),
                "trajectory_6dof": "do_as_i_do",
                "hand_source": "aoe",
                "status": "invalid_input",
                "input_rc": 4,
            }
            collected = module.collect_route_availability(
                source,
                [
                    {
                        "cell": route,
                        "status": "invalid_input",
                        "input_rc": 4,
                        "preflight_evidence": reference,
                    }
                ],
            )

        self.assertEqual(payload["route"], route)
        self.assertEqual(payload["retargeting"], "spider")
        self.assertEqual(payload["underlying_adapter_manifest"]["sha256"], adapter_sha)
        self.assertEqual(collected[route]["preflight_evidence"]["sha256"], reference["sha256"])

    def test_exact_route_failure_does_not_poison_valid_sibling_route(self) -> None:
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            dst = tmp_path / "matrix"
            (dst / "reuse").mkdir(parents=True)

            bundles: dict[tuple[str, str], dict[str, object] | None] = {}
            for trajectory, hand_source, valid in (
                ("egoinfinity", "estimated", True),
                ("do_as_i_do", "aoe", True),
                ("do_as_i_do", "estimated", False),
            ):
                route = f"{trajectory}__{hand_source}"
                keypoints = tmp_path / route / "trajectory_keypoints.npz"
                retarget_quality = tmp_path / route / "retarget_quality.json"
                keypoints.parent.mkdir(parents=True)
                keypoints.write_bytes(route.encode())
                retarget_quality.write_text(
                    json.dumps({"status": "ok" if valid else "invalid"}),
                    encoding="utf-8",
                )
                bundles[(trajectory, hand_source)] = {
                    "keypoints": keypoints,
                    "retarget_quality": retarget_quality,
                    "adapter_manifest": tmp_path / route / "adapter_manifest.json",
                    "hand_type": "left",
                    "task": "fixture_left",
                }
            bundles[("egoinfinity", "aoe")] = None

            requested_routes: list[tuple[str, str]] = []

            def exact_bundle(_source, trajectory, hand_source, *_args, **_kwargs):
                requested_routes.append((trajectory, hand_source))
                return bundles[(trajectory, hand_source)]

            def keypoint_quality(path: Path) -> dict[str, object]:
                return {
                    "available": True,
                    "valid": "do_as_i_do__estimated" not in str(path),
                }

            with (
                patch.object(module, "find_dai_asset_bundle", side_effect=exact_bundle),
                patch.object(
                    module,
                    "summarize_keypoint_quality",
                    side_effect=keypoint_quality,
                ),
                patch.object(
                    module,
                    "is_keypoint_quality_valid",
                    side_effect=lambda quality, *_args, **_kwargs: bool(quality["valid"]),
                ),
            ):
                report = module.validate_trajectory_assets(
                    dst,
                    "fixture_left",
                    dai_task="fixture_left",
                    dai_hand_type="left",
                    dai_source_task="fixture_left",
                    dai_source_hand_type="left",
                    spider_fuse_ego_object_for_dai=False,
                    dai_override_asset_report={},
                    dai_asset_min_distance_threshold=0.2,
                    dai_asset_median_distance_threshold=0.45,
                    dai_asset_no_contact_min_distance_threshold=0.08,
                    dai_asset_no_contact_median_distance_threshold=0.18,
                    dai_asset_require_contact=True,
                )

        self.assertEqual(
            requested_routes,
            [
                ("egoinfinity", "aoe"),
                ("egoinfinity", "estimated"),
                ("do_as_i_do", "aoe"),
                ("do_as_i_do", "estimated"),
            ],
        )
        self.assertEqual(report["routes"]["egoinfinity__aoe"]["status"], "missing")
        self.assertEqual(report["routes"]["egoinfinity__estimated"]["status"], "ok")
        self.assertEqual(report["routes"]["do_as_i_do__aoe"]["status"], "ok")
        self.assertEqual(report["routes"]["do_as_i_do__estimated"]["status"], "invalid")

        # The trajectory remains usable through its valid exact route. Completeness
        # remains false and the failed sibling route remains explicit.
        self.assertIs(report["do_as_i_do_quality_valid"], True)
        self.assertIs(report["do_as_i_do_all_routes_quality_valid"], False)
        self.assertEqual(
            report["do_as_i_do_quality_valid_by_hand_source"],
            {"aoe": True, "estimated": False},
        )
        self.assertEqual(report["object_provenance"]["do_as_i_do"]["status"], "ok")
        self.assertIn("do_as_i_do__estimated", report["invalid_routes"])
        self.assertNotIn("errors", report)

    def test_numerical_invalid_video_is_opt_in_and_full_wrapper_disables_it(self) -> None:
        module = load_module()
        with patch.object(
            sys,
            "argv",
            ["reuse_v4_for_12_demos.py", "--source-run", "fixture"],
        ):
            self.assertIs(
                module.parse_args().include_numerically_invalid_aligned_robot,
                False,
            )

        with patch.object(
            sys,
            "argv",
            [
                "reuse_v4_for_12_demos.py",
                "--source-run",
                "fixture",
                "--include-numerically-invalid-aligned-robot",
            ],
        ):
            self.assertIs(
                module.parse_args().include_numerically_invalid_aligned_robot,
                True,
            )

        wrapper = (ROOT / "scripts" / "run_full_12_demos.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--no-include-numerically-invalid-aligned-robot", wrapper)
        self.assertIn("--no-allow-dai-hand-fallback", wrapper)
        self.assertIn("--no-allow-cross-trajectory-dai-robot-fallback", wrapper)
        self.assertIn("--no-allow-spider-mjwp-fallback-video", wrapper)
