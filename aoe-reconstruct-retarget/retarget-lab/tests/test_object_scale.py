import unittest

from aoe_retarget_lab.object_scale import robust_projected_scale_fit, scale_bbox_xyxy


class ObjectScaleFitTest(unittest.TestCase):
    def test_ego_mask_bbox_is_scaled_into_source_frame_space(self):
        result = scale_bbox_xyxy(
            (960.0, 540.0, 1920.0, 1080.0),
            source_size=(1920, 1080),
            target_size=(1280, 720),
        )
        self.assertEqual(result, (640.0, 360.0, 1280.0, 720.0))

    def test_stable_oversized_mesh_is_shrunk(self):
        result = robust_projected_scale_fit([0.39, 0.40, 0.38, 0.41, 0.40, 0.39])
        self.assertTrue(result["available"])
        self.assertTrue(result["applied"])
        self.assertAlmostEqual(result["multiplier"], 0.395, places=3)

    def test_small_stable_difference_stays_in_deadband(self):
        result = robust_projected_scale_fit([0.98, 1.01, 1.02, 0.99, 1.00])
        self.assertTrue(result["available"])
        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "within_scale_deadband")

    def test_unstable_scale_evidence_is_not_applied(self):
        result = robust_projected_scale_fit([0.30, 0.35, 0.40, 1.4, 1.8, 2.2])
        self.assertTrue(result["available"])
        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "unstable_projected_scale_ratio")

    def test_too_few_frames_is_unavailable(self):
        result = robust_projected_scale_fit([0.4, 0.4, 0.4, 0.4])
        self.assertFalse(result["available"])
        self.assertFalse(result["applied"])


if __name__ == "__main__":
    unittest.main()
