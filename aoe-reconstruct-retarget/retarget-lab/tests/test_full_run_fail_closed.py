import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_full_12_demos.sh"
FRESH_SCRIPT = ROOT / "scripts" / "run_fresh_aoe_scene_12_demos.sh"


class FullRunFailClosedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_production_wrapper_rejects_positive_spider_fallback(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "--allow-spider-fallback"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("diagnostic-only and forbidden", result.stderr)

    def test_all_route_and_video_fallbacks_are_explicitly_disabled(self):
        for flag in (
            "--no-allow-spider-fallback",
            "--no-allow-spider-mjwp-fallback-video",
            "--no-allow-dai-hand-fallback",
            "--no-allow-cross-trajectory-dai-robot-fallback",
            "--no-include-numerically-invalid-aligned-robot",
        ):
            self.assertIn(flag, self.text)

    def test_matrix_audit_is_strict_without_requiring_twelve_cells(self):
        self.assertIn("audit.audit_cells", self.text)
        self.assertIn("audit.audit_spider", self.text)
        self.assertIn("audit.summarize_route_admission", self.text)
        self.assertIn('if not candidate_routes:', self.text)
        self.assertIn(
            "strict matrix audit admitted no exact numerical-valid retarget route",
            self.text,
        )
        self.assertNotIn('if not cell_report.get("matrix_complete")', self.text)
        self.assertIn('"scope": "matrix_only"', self.text)
        self.assertIn('"source_audit_status": "deferred_to_fresh_full_run_audit"', self.text)
        self.assertIn(
            "candidate_routes_blocked_by_audit_errors = candidate_routes if errors else []",
            self.text,
        )
        self.assertIn('admitted_routes = [] if errors else candidate_routes', self.text)
        self.assertIn('"candidate_routes_blocked_by_audit_errors":', self.text)
        self.assertIn('require_triptych=compose_enabled', self.text)
        self.assertIn('"demo_count": admitted_route_count if compose_enabled else 0', self.text)

    def test_matrix_audit_never_reports_demo_when_any_audit_error_remains(self):
        marker = '"$repo_root" "$matrix_run_name" "$source_run" "$matrix_audit_json" "$compose" <<\'PY\' \\\n'
        marker_start = self.text.index(marker) + len(marker)
        start = self.text.index("\nimport json\n", marker_start) + 1
        body = self.text[start : self.text.index("\nPY\nif (( matrix_audit_rc", start)]
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            scripts = repo / "scripts"
            scripts.mkdir()
            matrix_run = "matrix"
            matrix_root = repo / "experiments" / matrix_run
            matrix_root.mkdir(parents=True)
            (matrix_root / "reuse_12_demo_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            route = "traj_egoinfinity__hand_aoe__retarget_do_as_i_do"
            (scripts / "audit_retarget_run.py").write_text(
                "\n".join(
                    (
                        "import json",
                        "def load_json(path): return {}",
                        "def audit_cells(*args, **kwargs): return {'errors': ['route error'], 'warnings': []}",
                        "def audit_spider(*args, **kwargs): return {'errors': [], 'warnings': []}",
                        f"def summarize_route_admission(*args, **kwargs): return {{'errors': [], 'admitted_routes': ['{route}'], 'unavailable_routes': [], 'unavailable_route_details': []}}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            output = matrix_root / "audit_retarget_run.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-",
                    str(repo),
                    matrix_run,
                    "source",
                    str(output),
                    "1",
                ],
                input=body,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["admitted_routes"], [])
            self.assertEqual(report["admitted_route_count"], 0)
            self.assertEqual(report["demo_count"], 0)
            self.assertIs(report["scene_admitted"], False)
            self.assertEqual(report["candidate_routes_blocked_by_audit_errors"], [route])

    def test_fresh_wrapper_locks_dai_hand_and_can_disable_composition(self):
        text = FRESH_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('V4_DAI_RETARGET_HAND_TYPE="$hand_type"', text)
        self.assertIn('--no-compose) compose=0', text)
        self.assertIn('full12_compose_args+=(--no-compose)', text)
        self.assertIn('"${full12_compose_args[@]}"', text)
        self.assertIn('postflight_audit_args+=(--no-require-triptych)', text)
        result = subprocess.run(
            ["bash", str(FRESH_SCRIPT), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--no-compose", result.stdout)


if __name__ == "__main__":
    unittest.main()
