import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v4_two_full_pipelines.sh"


class StageFailureIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_stage_runner_uses_errexit_subshell_and_records_failures(self):
        self.assertIn('if (set -euo pipefail; "$@"); then', self.text)
        self.assertIn('stage failed but matrix execution will continue', self.text)
        self.assertIn('stage_failures_file="$exp/logs/stage_failures.tsv"', self.text)
        self.assertIn("last_stage_rc=0", self.text)
        self.assertIn("last_stage_rc=$status", self.text)
        self.assertIn("return 0", self.text)

    def test_enabled_retarget_variants_are_isolated_and_forbidden_fusion_fails_closed(self):
        for stage in (
            "do_as_i_do_aoe_retarget",
            "egoinfinity_aoe_retarget",
            "egoinfinity_estimated_retarget",
        ):
            self.assertIn(f"run_stage {stage}", self.text)
        self.assertNotIn("run_stage do_as_i_do_estimated_retarget", self.text)
        self.assertIn(
            '"do_as_i_do_estimated_retarget" "$dai_native_object_pose_rc"',
            self.text,
        )
        self.assertIn("hand_only_canonical_fusion_forbidden", self.text)

    def test_official_dai_launch_failures_are_explicitly_propagated(self):
        calls = self.text.split("scripts/run_do_as_i_do_official_retarget.sh")[1:]
        self.assertEqual(len(calls), 2)
        for call in calls:
            command = call.split("|| launcher_rc=$?", 1)[0]
            # The pristine launcher accepts only the native upstream surface.
            # Retarget Lab resolves hand identity and packages review artifacts
            # after native success instead of injecting legacy tuning flags.
            self.assertNotIn("--max-sim-steps", command)
            self.assertNotIn("--force", command)
            self.assertNotIn("--hand-type", command)
            self.assertIn("--output-root-dir", command)
            self.assertIn("--no-wait", command)
            self.assertIn('if (( launcher_rc != 0 )); then', call)
            self.assertIn('return "$launcher_rc"', call)
            self.assertLess(
                call.index('return "$launcher_rc"'),
                call.index('index_cell_assets.py'),
            )

    def test_failed_launcher_clears_partial_index_and_triptych(self):
        self.assertIn("clear_route_demo_artifacts()", self.text)
        self.assertIn('rm -rf "$exp/cells/$cell" "$exp/assets/cells/$cell"', self.text)
        self.assertIn('rm -f "$exp/videos/${cell}__triptych.mp4"', self.text)
        self.assertIn('rm -f "$triptych"', self.text)
        self.assertGreaterEqual(self.text.count('clear_route_demo_artifacts "$cell"'), 4)

    def test_ego_reconstruction_and_selection_are_isolated(self):
        self.assertIn("run_stage egoinfinity_full run_egoinfinity_full", self.text)
        self.assertIn("run_stage egoinfinity_object_selection select_egoinfinity_object", self.text)

    def test_invalid_ego_inputs_only_disable_dependent_routes(self):
        preflight = self.text.index("if preflight_ego_retarget_inputs; then")
        dai = self.text.index("run_stage do_as_i_do_aoe_retarget", preflight)
        self.assertLess(preflight, dai)
        self.assertIn("--require-retarget-input-qc", self.text)
        self.assertIn("--force || return $?", self.text)
        self.assertNotIn("scene rejected before DAI/SPIDER simulation", self.text)
        self.assertIn("Ego-dependent routes selectively disabled", self.text)
        self.assertIn("if (( ego_aoe_preflight_rc == 0 )); then", self.text)
        self.assertIn("if (( ego_estimated_preflight_rc == 0 )); then", self.text)

    def test_invalid_dai_native_input_does_not_block_ego_routes(self):
        self.assertIn("if prepare_dai_native_retarget_input; then", self.text)
        self.assertIn("pure DAI route disabled by DAI-native input QC", self.text)
        self.assertIn('"route_availability"', self.text)
        self.assertIn('"dai_native_object_pose_preflight"', self.text)
        self.assertIn("dai_native_object_pose_rc=4", self.text)
        self.assertNotIn("if (( dai_native_object_pose_rc == 0 )); then", self.text)
        self.assertIn(
            "DAI-object/Ego-hand estimated route disabled: production hand-only canonical fusion is forbidden",
            self.text,
        )
        self.assertIn(
            '"input_rc": $dai_native_object_pose_rc,',
            self.text,
        )

    def test_each_exact_dai_route_records_preflight_and_post_retarget_status(self):
        for route in (
            "traj_do_as_i_do__hand_aoe__retarget_do_as_i_do",
            "traj_do_as_i_do__hand_estimated__retarget_do_as_i_do",
            "traj_egoinfinity__hand_aoe__retarget_do_as_i_do",
            "traj_egoinfinity__hand_estimated__retarget_do_as_i_do",
        ):
            self.assertIn(f'route_preflight__{route}.json', self.text)
        self.assertEqual(self.text.count('"post_retarget_rc": $'), 4)
        self.assertEqual(self.text.count('"preflight_evidence": {"path":'), 4)
        self.assertEqual(self.text.count('"sha256": "$'), 4)
        self.assertIn('"underlying_adapter_manifest": {', self.text)
        self.assertIn('"retarget_input_qc_status":', self.text)
        self.assertIn('"adapter_rigid_invariance_status":', self.text)
        self.assertIn('"available": $ego_aoe_route_available', self.text)
        self.assertIn('ego_aoe_post_retarget_status="failed"', self.text)

    def test_embedded_preflight_evidence_binds_adapter_hash_and_identity(self):
        marker = '"$input_rc" "$status" "$adapter_manifest" "$exp" <<\'PY\'\n'
        start = self.text.index(marker) + len(marker)
        body = self.text[start : self.text.index("\nPY\n}", start)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "raw" / "adapter_manifest.json"
            adapter.parent.mkdir()
            adapter.write_text(
                json.dumps(
                    {
                        "adapter_schema_version": 3,
                        "hand_source": "aoe",
                        "retarget_input_qc": {"status": "ok"},
                        "adapter_rigid_invariance": {"status": "ok"},
                    }
                ),
                encoding="utf-8",
            )
            evidence = root / "logs" / "route.json"
            args = [
                sys.executable,
                "-",
                str(evidence),
                "traj_egoinfinity__hand_aoe__retarget_do_as_i_do",
                "egoinfinity",
                "aoe",
                "0",
                "ok",
                str(adapter),
                str(root),
            ]
            result = subprocess.run(args, input=body, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertEqual(payload["trajectory_6dof"], "egoinfinity")
            self.assertEqual(payload["hand_source"], "aoe")
            self.assertEqual(payload["input_rc"], 0)
            self.assertEqual(payload["evidence_validation"]["status"], "ok")
            self.assertEqual(
                payload["underlying_adapter_manifest"]["sha256"],
                hashlib.sha256(adapter.read_bytes()).hexdigest(),
            )

            args[5] = "estimated"
            invalid = subprocess.run(args, input=body, text=True, capture_output=True)
            self.assertEqual(invalid.returncode, 5)
            payload = json.loads(evidence.read_text(encoding="utf-8"))
            self.assertEqual(payload["evidence_validation"]["status"], "invalid")

    def test_recommended_ego_routes_run_before_dai_native_baselines(self):
        ego_aoe = self.text.index("run_stage egoinfinity_aoe_retarget")
        ego_estimated = self.text.index("run_stage egoinfinity_estimated_retarget")
        dai_aoe = self.text.index("run_stage do_as_i_do_aoe_retarget")
        dai_estimated = self.text.index(
            '"do_as_i_do_estimated_retarget" "$dai_native_object_pose_rc"'
        )
        self.assertLess(ego_aoe, dai_aoe)
        self.assertLess(ego_estimated, dai_aoe)
        self.assertLess(ego_aoe, dai_estimated)
        self.assertLess(ego_estimated, dai_estimated)

    def test_explicit_hand_locks_both_ego_adapter_selection_paths(self):
        self.assertEqual(self.text.count('retarget_hand_request="$hand_type"'), 2)
        self.assertGreaterEqual(
            self.text.count('anchor_args=(--anchor-hand "$hand_type")'),
            2,
        )
        self.assertGreaterEqual(
            self.text.count('local retarget_hand_request="auto"'),
            2,
        )
        self.assertIn('--retarget-hand "$retarget_hand_request"', self.text)

        marker = '  if [[ "$hand_type" == "left" || "$hand_type" == "right" ]]; then\n'
        starts = []
        offset = 0
        while True:
            try:
                start = self.text.index(marker, offset)
            except ValueError:
                break
            starts.append(start)
            offset = start + len(marker)
        self.assertEqual(len(starts), 2)
        for start in starts:
            end = self.text.index("\n  fi\n", start) + len("\n  fi\n")
            block = self.text[start:end]
            probe = (
                "probe() {\n"
                "  local hand_type=\"$1\"\n"
                "  local anchor_args=()\n"
                "  local retarget_hand_request=\"auto\"\n"
                + block
                + '\n  printf \'%s|%s\\n\' "$retarget_hand_request" "${anchor_args[*]}"\n'
                + "}\nprobe \"$1\"\n"
            )
            for hand, expected in (
                ("left", "left|--anchor-hand left"),
                ("right", "right|--anchor-hand right"),
                ("bimanual", "bimanual|"),
            ):
                result = subprocess.run(
                    ["bash", "-c", probe, "probe", hand],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_adapted_dai_backend_uses_exact_adapter_hand(self):
        adapted = self.text[
            self.text.index("run_adapted_do_as_i_do_retarget()") :
        ]
        launch = adapted.split(
            "scripts/run_do_as_i_do_official_retarget.sh", 1
        )[1].split("index_cell_assets.py", 1)[0]
        self.assertNotIn("--hand-type", launch)
        self.assertIn("resolve_native_dai_hand", launch)
        self.assertIn('test "$resolved_hand_type" = "$adapter_hand_type"', launch)
        self.assertIn(
            'materialize_and_validate_pristine_dai \\\n'
            '            "$retarget_output_root" "$output_dir" "$trajectory_6dof"',
            launch,
        )


if __name__ == "__main__":
    unittest.main()
