import unittest

import numpy as np

from aoe_retarget_lab.tracking_quality import (
    InteractionThresholds,
    TrackingThresholds,
    _hand_object_distance_sequences,
    rotation_geodesic_rad,
    summarize_pose_errors,
    summarize_interaction_distances,
    summarize_wrist_object_relative_translation,
    validate_hard_tracking_admission,
)


class _FakeContact:
    def __init__(self, geom1, geom2):
        self.geom1 = geom1
        self.geom2 = geom2


class _FakeData:
    def __init__(self, contacts):
        self.qpos = np.zeros(1, dtype=np.float64)
        self.contact = contacts
        self.ncon = len(contacts)


class _FalseZeroMujoco:
    def __init__(self, contacts=()):
        self.contacts = list(contacts)

    def MjData(self, _model):
        return _FakeData(self.contacts)

    @staticmethod
    def mj_forward(_model, _data):
        return None

    @staticmethod
    def mj_geomDistance(_model, _data, _geom1, _geom2, _distmax, fromto):
        fromto[:] = [0.0, 0.0, 0.0, 0.2, 0.0, 0.0]
        return 0.0


class TrackingQualityTest(unittest.TestCase):
    @staticmethod
    def exact_binding(side="left"):
        return {
            "side": side,
            "wrist_site_name": f"{side}_palm",
            "wrist_site_id": 10,
            "wrist_body_name": f"{side}_hand_link",
            "wrist_body_id": 9,
            "object_body_name": f"{side}_object",
            "object_body_id": 20,
            "resolution_errors": [],
        }

    @staticmethod
    def valid_tracking_report():
        report = summarize_pose_errors(
            np.full(40, 0.02), np.full(40, 0.10), TrackingThresholds()
        )
        report["interaction_quality"] = summarize_interaction_distances(
            {"left": np.full(40, 0.03)},
            {"left": np.full(40, 0.02)},
            InteractionThresholds(),
        )
        wrist = np.zeros((40, 3), dtype=np.float64)
        wrist_rotation = np.repeat(np.eye(3)[None], 40, axis=0)
        reference_object = np.tile([0.02, 0.0, 0.0], (40, 1))
        report["interaction_quality"][
            "wrist_object_relative_translation"
        ] = summarize_wrist_object_relative_translation(
            {"left": wrist.copy()},
            {"left": wrist.copy()},
            {"left": wrist_rotation.copy()},
            {"left": wrist_rotation.copy()},
            {"left": reference_object.copy()},
            {"left": reference_object.copy()},
            ["left"],
            {"left": TrackingQualityTest.exact_binding("left")},
            InteractionThresholds(),
        )
        return report

    def test_rotation_geodesic_handles_pi_rotation(self):
        identity = np.eye(3)[None]
        rotate_x_pi = np.diag([1.0, -1.0, -1.0])[None]
        self.assertAlmostEqual(float(rotation_geodesic_rad(identity, rotate_x_pi)[0]), np.pi)

    def test_late_object_flyout_is_invalid(self):
        position = np.concatenate([np.full(80, 0.03), np.linspace(0.3, 2.0, 20)])
        rotation = np.zeros(100)
        result = summarize_pose_errors(position, rotation, TrackingThresholds())
        self.assertEqual(result["status"], "invalid")
        self.assertGreater(result["metrics"]["lost_frame_fraction"], 0.10)
        self.assertGreater(result["metrics"]["pos_p95_m"], 0.25)

    def test_small_bounded_pose_error_is_ok(self):
        position = np.linspace(0.01, 0.04, 100)
        rotation = np.linspace(0.01, 0.20, 100)
        result = summarize_pose_errors(position, rotation, TrackingThresholds())
        self.assertEqual(result["status"], "ok")

    def test_post_retarget_position_hard_thresholds(self):
        position = np.concatenate([np.full(94, 0.05), np.full(6, 0.10)])
        result = summarize_pose_errors(position, np.zeros(100), TrackingThresholds())
        self.assertEqual(result["status"], "ok")
        position[-6:] = 0.101
        result = summarize_pose_errors(position, np.zeros(100), TrackingThresholds())
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("pos_p95_m" in error for error in result["errors"]))

    def test_hand_object_contact_tracking_is_ok(self):
        reference = np.concatenate([np.full(20, 0.03), np.full(20, 0.15)])
        executed = reference + np.linspace(0.0, 0.02, 40)
        result = summarize_interaction_distances(
            {"right": executed}, {"right": reference}, InteractionThresholds()
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["active_sides"], ["right"])
        self.assertEqual(result["sides"]["right"]["contact_retention_fraction"], 1.0)

    def test_hand_object_detachment_is_invalid(self):
        reference = np.full(40, 0.02)
        executed = np.full(40, 0.25)
        result = summarize_interaction_distances(
            {"right": executed}, {"right": reference}, InteractionThresholds()
        )
        self.assertEqual(result["status"], "invalid")
        self.assertIn("right:contact_retention_fraction", result["errors"])

    def test_hand_object_distance_hard_thresholds(self):
        reference = np.full(100, 0.02)
        executed = reference + np.concatenate([np.full(94, 0.03), np.full(6, 0.08)])
        result = summarize_interaction_distances(
            {"right": executed}, {"right": reference}, InteractionThresholds()
        )
        self.assertEqual(result["status"], "ok")
        executed[-6:] = reference[-6:] + 0.081
        result = summarize_interaction_distances(
            {"right": executed}, {"right": reference}, InteractionThresholds()
        )
        self.assertEqual(result["status"], "invalid")

    def test_missing_reference_contact_is_invalid(self):
        reference = np.full(40, 0.25)
        result = summarize_interaction_distances(
            {"right": reference}, {"right": reference}, InteractionThresholds()
        )
        self.assertEqual(result["status"], "invalid")
        self.assertIn("no_reference_hand_object_contact", result["errors"])

    def test_relative_translation_uses_vector_not_scalar_distance(self):
        wrist = np.zeros((40, 3), dtype=np.float64)
        wrist_rotation = np.repeat(np.eye(3)[None], 40, axis=0)
        reference_object = np.tile([0.05, 0.0, 0.0], (40, 1))
        executed_object = np.tile([-0.05, 0.0, 0.0], (40, 1))
        self.assertTrue(
            np.allclose(
                np.linalg.norm(reference_object - wrist, axis=1),
                np.linalg.norm(executed_object - wrist, axis=1),
            )
        )
        result = summarize_wrist_object_relative_translation(
            {"left": wrist},
            {"left": wrist},
            {"left": wrist_rotation},
            {"left": wrist_rotation},
            {"left": executed_object},
            {"left": reference_object},
            ["left"],
            {"left": self.exact_binding("left")},
            InteractionThresholds(),
        )
        self.assertEqual(result["status"], "invalid")
        self.assertAlmostEqual(result["sides"]["left"]["error_median_m"], 0.10)
        self.assertAlmostEqual(result["sides"]["left"]["error_p95_m"], 0.10)

    def test_relative_translation_missing_exact_geometry_fails_closed(self):
        points = np.zeros((5, 3), dtype=np.float64)
        rotations = np.repeat(np.eye(3)[None], 5, axis=0)
        binding = self.exact_binding("left")
        binding["wrist_site_name"] = "right_palm"
        result = summarize_wrist_object_relative_translation(
            {"left": points},
            {"left": points},
            {"left": rotations},
            {"left": rotations},
            {"left": points},
            {"left": points},
            ["left"],
            {"left": binding},
            InteractionThresholds(),
        )
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["sides"]["left"]["available"])
        self.assertTrue(
            any("wrist_site_name_not_exact" in error for error in result["errors"])
        )

    def test_relative_translation_is_invariant_to_common_se3(self):
        frames = 12
        reference_wrist = np.tile([0.2, -0.1, 0.4], (frames, 1))
        reference_object = reference_wrist + np.tile([0.04, 0.02, -0.01], (frames, 1))
        reference_rotation = np.repeat(np.eye(3)[None], frames, axis=0)
        angle = np.deg2rad(73.0)
        common_rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        common_translation = np.array([1.5, -0.7, 0.25])
        executed_wrist = reference_wrist @ common_rotation.T + common_translation
        executed_object = reference_object @ common_rotation.T + common_translation
        executed_rotation = np.repeat(common_rotation[None], frames, axis=0)
        result = summarize_wrist_object_relative_translation(
            {"left": executed_wrist},
            {"left": reference_wrist},
            {"left": executed_rotation},
            {"left": reference_rotation},
            {"left": executed_object},
            {"left": reference_object},
            ["left"],
            {"left": self.exact_binding("left")},
            InteractionThresholds(),
        )
        self.assertEqual(result["status"], "ok")
        self.assertLess(result["sides"]["left"]["error_p95_m"], 1.0e-12)

    def test_false_zero_geom_distance_uses_closest_point_segment_without_contact(self):
        evidence = {}
        values = _hand_object_distance_sequences(
            object(),
            _FalseZeroMujoco(),
            np.zeros((1, 1), dtype=np.float64),
            [20],
            {"right": [10]},
            evidence,
        )
        self.assertAlmostEqual(values["right"][0], 0.2)
        self.assertEqual(evidence["false_zero_fallback_count"], 1)
        self.assertEqual(evidence["false_zero_fallback_frame_sample_by_side"]["right"], [0])

    def test_zero_geom_distance_is_preserved_for_actual_contact_pair(self):
        evidence = {}
        values = _hand_object_distance_sequences(
            object(),
            _FalseZeroMujoco([_FakeContact(20, 10)]),
            np.zeros((1, 1), dtype=np.float64),
            [20],
            {"right": [10]},
            evidence,
        )
        self.assertEqual(values["right"][0], 0.0)
        self.assertEqual(evidence["false_zero_fallback_count"], 0)

    def test_distance_sequences_bind_same_side_object_geoms(self):
        values = _hand_object_distance_sequences(
            object(),
            _FalseZeroMujoco(),
            np.zeros((1, 1), dtype=np.float64),
            {"left": [20], "right": []},
            {"left": [10], "right": [11]},
        )
        self.assertEqual(set(values), {"left"})

    def test_hard_admission_recomputes_raw_metrics(self):
        report = self.valid_tracking_report()
        self.assertEqual(validate_hard_tracking_admission(report)["status"], "ok")

        report["status"] = "ok"
        report["metrics"]["pos_median_m"] = 0.051
        result = validate_hard_tracking_admission(report)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("pos_median_m" in error for error in result["errors"]))

    def test_hard_admission_rejects_status_only_and_loose_thresholds(self):
        result = validate_hard_tracking_admission({"status": "ok", "errors": []})
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("status-only/legacy" in error for error in result["errors"]))

        report = self.valid_tracking_report()
        report["thresholds"]["pos_median_m"] = 0.50
        report["interaction_quality"]["thresholds"]["min_contact_retention"] = 0.1
        result = validate_hard_tracking_admission(report)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("looser" in error for error in result["errors"]))

    def test_hard_admission_requires_relative_translation_metrics_and_binding(self):
        report = self.valid_tracking_report()
        del report["interaction_quality"]["wrist_object_relative_translation"]
        result = validate_hard_tracking_admission(report)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(
            any("wrist_object_relative_translation" in error for error in result["errors"])
        )

        report = self.valid_tracking_report()
        relative = report["interaction_quality"][
            "wrist_object_relative_translation"
        ]
        relative["sides"]["left"]["binding"]["object_body_name"] = "right_object"
        result = validate_hard_tracking_admission(report)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("object_body_name_not_exact" in error for error in result["errors"]))

    def test_hard_admission_recomputes_relative_translation_limits(self):
        report = self.valid_tracking_report()
        relative = report["interaction_quality"][
            "wrist_object_relative_translation"
        ]
        relative["sides"]["left"]["error_median_m"] = 0.031
        relative["status"] = "ok"
        relative["sides"]["left"]["status"] = "ok"
        relative["errors"] = []
        relative["sides"]["left"]["errors"] = []
        result = validate_hard_tracking_admission(report)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("error_median_m" in error for error in result["errors"]))

        report = self.valid_tracking_report()
        relative = report["interaction_quality"][
            "wrist_object_relative_translation"
        ]
        relative["sides"]["left"]["error_p95_m"] = 0.081
        result = validate_hard_tracking_admission(report)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("error_p95_m" in error for error in result["errors"]))

        report = self.valid_tracking_report()
        relative = report["interaction_quality"][
            "wrist_object_relative_translation"
        ]
        relative["thresholds"]["error_median_m"] = 0.30
        result = validate_hard_tracking_admission(report)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("looser" in error for error in result["errors"]))

    def test_hard_admission_rechecks_rotation_and_rejects_relaxed_rotation_gate(self):
        report = self.valid_tracking_report()
        report["metrics"]["rot_median_rad"] = 0.751
        report["status"] = "ok"
        report["errors"] = []
        result = validate_hard_tracking_admission(report)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("rot_median_rad" in error for error in result["errors"]))

        report = self.valid_tracking_report()
        report["thresholds"]["rot_median_rad"] = 7.5
        result = validate_hard_tracking_admission(report)
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("looser" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
