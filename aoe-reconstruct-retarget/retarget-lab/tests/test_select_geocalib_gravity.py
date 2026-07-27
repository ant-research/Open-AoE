import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "select_geocalib_gravity.py"
SPEC = importlib.util.spec_from_file_location("select_geocalib_gravity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def sample(frame: int, roll: float, pitch: float) -> dict:
    return {
        "path": f"/frames/{frame:06d}.png",
        "roll_deg": roll,
        "pitch_deg": pitch,
        "confidence": 0.5,
        "outlier": False,
    }


class SelectGeoCalibGravityTest(unittest.TestCase):
    def test_static_camera_keeps_aggregate(self):
        payload = {
            "vec3d": [0.0, -1.0, 0.0],
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "per_frame": [sample(0, 0.0, 0.0), sample(10, 2.0, 1.0)],
        }
        result = MODULE.select_gravity(payload, reference_frame=10, max_p95_deg=25.0)
        self.assertEqual(result["gravity_quality"]["status"], "static_camera_aggregate")
        self.assertEqual(result["vec3d"], [0.0, -1.0, 0.0])

    def test_dynamic_camera_uses_reconstruction_reference_frame(self):
        payload = {
            "vec3d": [0.0, -1.0, 0.0],
            "roll_deg": 0.0,
            "pitch_deg": 0.0,
            "per_frame": [
                sample(0, 0.0, 0.0),
                sample(40, 35.0, -20.0),
                sample(82, -30.0, -40.0),
            ],
        }
        result = MODULE.select_gravity(payload, reference_frame=80, max_p95_deg=25.0)
        quality = result["gravity_quality"]
        self.assertEqual(quality["status"], "dynamic_camera_reference_frame")
        self.assertEqual(quality["selected_frame"], 82)
        self.assertEqual(result["roll_deg"], -30.0)
        self.assertEqual(result["pitch_deg"], -40.0)
        self.assertEqual(quality["original_aggregate"]["vec3d"], [0.0, -1.0, 0.0])


if __name__ == "__main__":
    unittest.main()
