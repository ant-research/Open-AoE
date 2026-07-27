from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnostics" / "validate_adapter_geometry_contact.py"


def load_module():
    spec = importlib.util.spec_from_file_location("adapter_geometry_contact", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class AdapterGeometryContactTest(unittest.TestCase):
    def test_v4_runner_gates_only_dai_object_with_ego_hand(self):
        runner = (ROOT / "scripts" / "run_v4_two_full_pipelines.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '[[ "$object_geometry_source" == "source" && "$hand_geometry_source" == "ego" ]]',
            runner,
        )
        self.assertIn("scripts/diagnostics/validate_adapter_geometry_contact.py", runner)

    def test_rejects_meter_scale_cross_source_separation(self):
        module = load_module()
        qc = module.validate_geometry_contact(
            {
                "selected_hand": "bimanual",
                "geometry_interaction_hand": {
                    "available": True,
                    "contact_fraction": {"left": 0.0, "right": 0.0},
                    "mean_min_distance": {"left": 1.31, "right": 1.22},
                },
            },
            min_contact_fraction=0.01,
            max_mean_distance_m=0.25,
        )
        self.assertEqual(qc["status"], "invalid")
        self.assertIn("no_geometric_hand_object_contact", qc["errors"])
        self.assertIn("hand_object_geometry_separated", qc["errors"])

    def test_accepts_contacting_geometry(self):
        module = load_module()
        qc = module.validate_geometry_contact(
            {
                "selected_hand": "left",
                "geometry_interaction_hand": {
                    "available": True,
                    "contact_fraction": {"left": 0.4},
                    "mean_min_distance": {"left": 0.03},
                },
            },
            min_contact_fraction=0.01,
            max_mean_distance_m=0.25,
        )
        self.assertEqual(qc["status"], "ok")
        self.assertEqual(qc["errors"], [])

    def test_cli_persists_invalid_qc_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "selected_hand": "right",
                        "geometry_interaction_hand": {
                            "available": True,
                            "contact_fraction": {"right": 0.0},
                            "mean_min_distance": {"right": 0.8},
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--adapter-manifest", str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 4)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["cross_source_geometry_qc"]["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
