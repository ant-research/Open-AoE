import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review" / "render_aoe_candidate_contact_sheets.py"
SPEC = importlib.util.spec_from_file_location("render_aoe_candidate_contact_sheets", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FailedCapture:
    def __init__(self, _path: str):
        pass

    def set(self, _key: int, _value: int) -> None:
        pass

    def read(self):
        return False, None

    def release(self) -> None:
        pass


class CandidateContactSheetTest(unittest.TestCase):
    def test_ffmpeg_fallback_decodes_frame_after_opencv_seek_failure(self):
        expected = np.full((8, 12, 3), 127, dtype=np.uint8)
        encoded_ok, encoded = cv2.imencode(".png", expected)
        self.assertTrue(encoded_ok)
        completed = SimpleNamespace(returncode=0, stdout=encoded.tobytes(), stderr=b"")
        with patch.object(MODULE.cv2, "VideoCapture", FailedCapture), patch.object(
            MODULE.subprocess, "run", return_value=completed
        ) as run:
            frame = MODULE.read_frame(Path("long_gop.mp4"), frame_id=60, fps=30.0)
        self.assertEqual(frame.shape, expected.shape)
        self.assertIn("2.000000", run.call_args.args[0])

    def test_sam3_preflight_stops_before_sam3d_and_full_matrix(self):
        root = Path(__file__).resolve().parents[1]
        reconstruction = (root / "scripts/run_do_as_i_do_reconstruction_fresh.sh").read_text()
        fresh = (root / "scripts/run_fresh_aoe_scene_12_demos.sh").read_text()
        self.assertIn('stop_after_sam3="${DAI_RECON_STOP_AFTER_SAM3:-0}"', reconstruction)
        self.assertLess(
            reconstruction.index('if [[ "$stop_after_sam3" == "1" ]]'),
            reconstruction.index('echo "=== [2a] SAM3D masks -> object mesh ==="'),
        )
        self.assertIn("--sam3-preflight-only) sam3_preflight_only=1", fresh)
        self.assertIn('DAI_RECON_STOP_AFTER_SAM3="$sam3_preflight_only"', fresh)
        self.assertIn('reconstruction_rc="${PIPESTATUS[0]}"', fresh)
        self.assertIn('if [[ "$reconstruction_rc" -ne 0 ]]', fresh)
        self.assertIn('prompt_gate.get("status") != "ok"', fresh)
        self.assertIn('mask_qc.get("qa_pass") is not True', fresh)
        self.assertLess(
            fresh.index('if [[ "$sam3_preflight_only" == "1" ]]'),
            fresh.index('"$repo_root/scripts/run_full_12_demos.sh"'),
        )


if __name__ == "__main__":
    unittest.main()
