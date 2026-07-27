from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSE_SCRIPT = ROOT / "scripts/diagnostics/diagnose_egoinfinity_scale.py"
REPAIR_SCRIPT = ROOT / "scripts/diagnostics/repair_do_as_i_do_object_scale.py"
ADAPTER_SCRIPT = ROOT / "scripts" / "prepare_egoinfinity_do_as_i_do_raw_dir.py"
FULL_RUNNER = ROOT / "scripts" / "run_v4_two_full_pipelines.sh"


def load_script(name: str, path: Path):
    optional_stubs = {
        "cv2": types.ModuleType("cv2"),
        "numpy": types.ModuleType("numpy"),
        "trimesh": types.ModuleType("trimesh"),
    }
    installed = []
    for module_name, stub in optional_stubs.items():
        try:
            __import__(module_name)
        except ModuleNotFoundError:
            sys.modules[module_name] = stub
            installed.append(module_name)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for module_name in installed:
            sys.modules.pop(module_name, None)


class ObjectScaleToolPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repair = load_script("repair_object_scale_policy", REPAIR_SCRIPT)
        cls.diagnose = load_script("diagnose_object_scale_policy", DIAGNOSE_SCRIPT)

    def test_repair_identifiers_reject_path_traversal(self):
        for value in ("../other", "run/other", "run\\other", ".", "..", ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.repair.validate_path_component(value, "run")
        self.assertEqual(
            self.repair.validate_path_component("vinegar_probe-01", "run"),
            "vinegar_probe-01",
        )

    def test_repair_requires_explicit_diagnostic_acknowledgement(self):
        argv = [
            str(REPAIR_SCRIPT),
            "--source-run",
            "source",
            "--override-run",
            "override",
            "--trajectory-6dof",
            "egoinfinity",
            "--hand-type",
            "left",
            "--task",
            "fixture",
            "--target-object-visual",
            "fixture.obj",
            "--raw-dir",
            "raw",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as caught:
                self.repair.main()
        self.assertEqual(caught.exception.code, 2)

    def test_hybrid_object_reference_is_opt_in_and_default_off(self):
        captured = {}

        def stop_after_parse(args):
            captured["args"] = args
            raise RuntimeError("parsed")

        argv = [
            str(REPAIR_SCRIPT),
            "--source-run",
            "source",
            "--override-run",
            "override",
            "--diagnostic-override-only",
            "--trajectory-6dof",
            "egoinfinity",
            "--hand-type",
            "left",
            "--task",
            "fixture",
            "--target-object-visual",
            "fixture.obj",
            "--raw-dir",
            "raw",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            self.repair, "build_paths", side_effect=stop_after_parse
        ):
            with self.assertRaisesRegex(RuntimeError, "parsed"):
                self.repair.main()
        self.assertFalse(captured["args"].object_ref_fallback)

    def test_diagnose_requires_trusted_pickle_acknowledgement(self):
        argv = [
            str(DIAGNOSE_SCRIPT),
            "--pipeline-result",
            "pipeline.pkl.gz",
            "--output-dir",
            "diagnostic",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as caught:
                self.diagnose.main()
        self.assertEqual(caught.exception.code, 2)

    def test_posthoc_outputs_and_manifest_are_unambiguously_diagnostic(self):
        source = REPAIR_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("diagnostic_only_posthoc_object_scale_override", source)
        self.assertIn('"eligible_for_pristine_backend_success": False', source)
        self.assertIn('"eligible_for_demo_admission": False', source)
        self.assertIn("visualization_diagnostic_scale_override_", source)
        self.assertNotIn('default=True,\n        help=(\n            "Explicit diagnostic opt-in', source)

    def test_read_only_report_is_labeled_and_hash_bound(self):
        source = DIAGNOSE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("diagnostic_only_object_scale_evidence", source)
        self.assertIn('"automatic_scale_application_allowed": False', source)
        self.assertIn('"pipeline_result_sha256"', source)
        self.assertIn('"ply_sha256"', source)

    def test_adapter_scale_fit_remains_candidate_only(self):
        source = ADAPTER_SCRIPT.read_text(encoding="utf-8")
        required_fragments = (
            '"applied": False',
            '"would_apply": would_apply_scale',
            '"diagnostic_only": True',
            '"mesh_modified": False',
            '"policy": "object_only_mesh_scale_refinement_forbidden"',
            '"object_mesh_scale_refinement_applied": False',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

    def test_full_runner_disables_mask_scale_fit_without_removing_visuals(self):
        source = FULL_RUNNER.read_text(encoding="utf-8")
        self.assertGreaterEqual(
            source.count("--no-fit-object-scale-to-ego-mask"), 2
        )
        self.assertIn("materialize_do_as_i_do_visuals.py", source)
        self.assertIn("compose_triptych.py", source)

    def test_standalone_scale_tools_have_no_production_runner_callers(self):
        forbidden = (
            "diagnose_egoinfinity_scale.py",
            "repair_do_as_i_do_object_scale.py",
        )
        for path in (ROOT / "scripts").glob("run_*"):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for tool in forbidden:
                with self.subTest(runner=path.name, tool=tool):
                    self.assertNotIn(tool, text)


if __name__ == "__main__":
    unittest.main()
