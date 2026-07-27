from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnostics"
    / "validate_dai_raw_to_processed_hoi.py"
)
SPEC = importlib.util.spec_from_file_location("validate_dai_raw_to_processed_hoi", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _fixture(
    tmp_path: Path,
    *,
    object_only_offset: np.ndarray | None = None,
    bimanual: bool = False,
    constant_object: bool = False,
    right_local_rotation_angle: float | None = None,
    right_object_offset: np.ndarray | None = None,
    right_object_quaternion_sign_flip: bool = False,
    object_only_rotation_angle: float | None = None,
    invalid_object_quaternion: bool = False,
):
    raw_dir = tmp_path / "raw_dir"
    (raw_dir / "raw").mkdir(parents=True)
    layout_path = (
        raw_dir
        / "obj_tracking_out/object/combined_visualization/layout_camera_frame_optimized.json"
    )
    layout_path.parent.mkdir(parents=True)
    frames = 6
    if constant_object:
        object_position = np.repeat(
            np.array([[-0.2, 0.4, 0.8]], dtype=np.float64), frames, axis=0
        )
    else:
        object_position = np.stack(
            [
                np.linspace(-0.2, 0.3, frames),
                np.linspace(0.4, -0.1, frames),
                np.linspace(0.8, 1.0, frames),
            ],
            axis=1,
        )
    local_joints = np.stack(
        [
            np.array(
                [
                    0.012 * ((joint % 5) - 2),
                    0.018 * (joint // 5),
                    0.07 + 0.004 * ((joint * 3) % 7),
                ]
            )
            for joint in range(21)
        ]
    )
    # Match DAI's MCP wrist-frame construction with a non-degenerate palm.
    local_joints[0] = [0.0, 0.0, 0.0]
    local_joints[5] = [0.045, 0.075, 0.012]
    local_joints[9] = [0.0, 0.105, 0.018]
    local_joints[13] = [-0.045, 0.075, 0.012]
    joints = object_position[:, None] + local_joints[None]
    right_joints = joints + np.array([-0.16, 0.07, 0.03])
    raw_payload = {
        "left_joints": joints,
        "left_rot": np.zeros((frames, 3), dtype=np.float64),
    }
    if bimanual:
        raw_payload["right_joints"] = right_joints
        raw_payload["right_rot"] = np.zeros((frames, 3), dtype=np.float64)
    np.savez(raw_dir / "raw/all_hand_meshes.npz", **raw_payload)
    raw_object_angle = 0.37
    raw_object_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(raw_object_angle), -np.sin(raw_object_angle)],
            [0.0, np.sin(raw_object_angle), np.cos(raw_object_angle)],
        ]
    )
    raw_object_quaternion = MODULE._matrix_to_quat_wxyz(raw_object_rotation)
    layout_path.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "local_to_scene": {
                            "translation_camera_frame": position.tolist(),
                            "quat_wxyz_camera_frame": raw_object_quaternion.tolist(),
                        }
                    }
                    for position in object_position
                ]
            }
        ),
        encoding="utf-8",
    )
    angle = 0.71
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([0.7, -0.3, 0.2])

    def transform(points):
        return np.asarray(points) @ rotation.T + translation

    object_after = transform(object_position)
    if object_only_offset is not None:
        object_after = object_after + np.asarray(object_only_offset)
    wrist_after = transform(joints[:, 0])
    finger_after = transform(joints[:, MODULE.FINGERTIP_INDICES])
    processed_object_rotation = rotation @ raw_object_rotation
    if object_only_rotation_angle is not None:
        extra = np.array(
            [
                [np.cos(object_only_rotation_angle), -np.sin(object_only_rotation_angle), 0.0],
                [np.sin(object_only_rotation_angle), np.cos(object_only_rotation_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        processed_object_rotation = extra @ processed_object_rotation
    qpos_object = np.zeros((frames, 7), dtype=np.float64)
    qpos_object[:, :3] = object_after
    qpos_object[:, 3:7] = MODULE._matrix_to_quat_wxyz(processed_object_rotation)
    if invalid_object_quaternion:
        qpos_object[:, 3:7] = 0.0
    qpos_wrist = np.zeros((frames, 7), dtype=np.float64)
    qpos_wrist[:, :3] = wrist_after
    raw_left_wrist = MODULE._wrist_rotations(
        joints, raw_payload["left_rot"], "left"
    )
    qpos_wrist[:, 3:7] = MODULE._matrix_to_quat_wxyz(
        rotation @ raw_left_wrist
    )
    qpos_finger = np.zeros((frames, 5, 7), dtype=np.float64)
    qpos_finger[:, :, :3] = finger_after
    qpos_finger[:, :, 3] = 1.0
    processed = tmp_path / "trajectory_keypoints.npz"
    processed_payload = {
        "qpos_obj_left": qpos_object,
        "qpos_wrist_left": qpos_wrist,
        "qpos_finger_left": qpos_finger,
    }
    if bimanual:
        right_after = transform(right_joints)
        if right_local_rotation_angle is not None:
            right_rotation = np.array(
                [
                    [np.cos(right_local_rotation_angle), -np.sin(right_local_rotation_angle), 0.0],
                    [np.sin(right_local_rotation_angle), np.cos(right_local_rotation_angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            right_after = (
                (right_after - object_after[:, None]) @ right_rotation.T
                + object_after[:, None]
            )
        right_object = qpos_object.copy()
        if right_object_offset is not None:
            right_object[:, :3] += np.asarray(right_object_offset)
        if right_object_quaternion_sign_flip:
            right_object[:, 3:7] *= -1.0
        right_wrist = np.zeros((frames, 7), dtype=np.float64)
        right_wrist[:, :3] = right_after[:, 0]
        raw_right_wrist = MODULE._wrist_rotations(
            right_joints, raw_payload["right_rot"], "right"
        )
        right_wrist[:, 3:7] = MODULE._matrix_to_quat_wxyz(
            rotation @ raw_right_wrist
        )
        right_fingers = np.zeros((frames, 5, 7), dtype=np.float64)
        right_fingers[:, :, :3] = right_after[:, MODULE.FINGERTIP_INDICES]
        right_fingers[:, :, 3] = 1.0
        processed_payload.update(
            {
                "qpos_obj_right": right_object,
                "qpos_wrist_right": right_wrist,
                "qpos_finger_right": right_fingers,
            }
        )
    np.savez(processed, **processed_payload)
    manifest = raw_dir / "adapter_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "trajectory_6dof": "egoinfinity",
                "hand_source": "aoe",
                "object_track_source": "egoinfinity",
                "object_mesh_source": "egoinfinity",
                "retarget_object_source": "egoinfinity",
                "canonical_transform": {"applied": False, "kind": "identity"},
                "hoi_refinement": {"applied": False},
                "hoi_contact_alignment": {"applied": False},
                "layout": str(layout_path),
            }
        ),
        encoding="utf-8",
    )
    return manifest, processed


def test_arbitrary_common_se3_preserves_raw_to_processed_hoi(tmp_path: Path) -> None:
    manifest, processed = _fixture(tmp_path)
    report = MODULE.evaluate(
        adapter_manifest=manifest,
        processed_keypoints=processed,
        trajectory_6dof="egoinfinity",
        hand_type="left",
    )
    assert report["status"] == "ok"
    assert report["cross_hand_source_comparison"] is False
    assert report["relative_distance_error_m"]["max"] < 1.0e-12
    assert report["common_se3_position_residual_m"]["max"] < 1.0e-12
    assert report["common_se3_rotation_residual_rad"]["max"] < 1.0e-7
    assert report["wrist_object_relative_translation_error_m"]["max"] < 1.0e-12
    assert report["wrist_object_relative_rotation_error_rad"]["max"] < 1.0e-7
    assert report["schema_version"] == 3
    assert report["full_relative_se3_checked"] is True
    assert report["route_binding"]["hand_source"] == "aoe"
    assert report["route_binding"]["processed_keypoints"]["sha256"]


def test_object_only_translation_is_rejected(tmp_path: Path) -> None:
    manifest, processed = _fixture(
        tmp_path, object_only_offset=np.array([0.04, 0.0, 0.0])
    )
    report = MODULE.evaluate(
        adapter_manifest=manifest,
        processed_keypoints=processed,
        trajectory_6dof="egoinfinity",
        hand_type="left",
    )
    assert report["status"] == "invalid"
    assert "raw_to_processed_relative_distance_max_exceeds_threshold" in report["errors"]


def test_object_only_rotation_is_rejected_by_full_relative_se3(tmp_path: Path) -> None:
    manifest, processed = _fixture(tmp_path, object_only_rotation_angle=0.25)
    report = MODULE.evaluate(
        adapter_manifest=manifest,
        processed_keypoints=processed,
        trajectory_6dof="egoinfinity",
        hand_type="left",
    )
    assert report["status"] == "invalid"
    assert "raw_to_processed_relative_pose_rotation_exceeds_threshold" in report["errors"]
    assert "raw_to_processed_rotations_are_not_one_common_se3" in report["errors"]


def test_zero_processed_object_quaternion_is_rejected(tmp_path: Path) -> None:
    manifest, processed = _fixture(tmp_path, invalid_object_quaternion=True)
    try:
        MODULE.evaluate(
            adapter_manifest=manifest,
            processed_keypoints=processed,
            trajectory_6dof="egoinfinity",
            hand_type="left",
        )
    except ValueError as exc:
        assert "non-unit WXYZ quaternion" in str(exc)
    else:
        raise AssertionError("zero object quaternion must fail closed")


def test_bimanual_uses_one_joint_common_se3(tmp_path: Path) -> None:
    manifest, processed = _fixture(tmp_path, bimanual=True)
    report = MODULE.evaluate(
        adapter_manifest=manifest,
        processed_keypoints=processed,
        trajectory_6dof="egoinfinity",
        hand_type="bimanual",
    )
    assert report["status"] == "ok", report["errors"]
    assert report["joint_fit_components"] == ["object", "left", "right"]
    assert report["processed_object_track_consistency"]["required"] is True
    assert report["processed_object_track_consistency"]["absolute_component_error"]["max"] == 0.0


def test_independent_bimanual_hand_transform_fails_joint_fit(tmp_path: Path) -> None:
    manifest, processed = _fixture(
        tmp_path,
        bimanual=True,
        constant_object=True,
        right_local_rotation_angle=0.45,
    )
    report = MODULE.evaluate(
        adapter_manifest=manifest,
        processed_keypoints=processed,
        trajectory_6dof="egoinfinity",
        hand_type="bimanual",
    )
    assert report["relative_distance_error_m"]["max"] < 1.0e-12
    assert report["processed_object_track_consistency"]["absolute_component_error"]["max"] == 0.0
    assert report["status"] == "invalid"
    assert "raw_to_processed_is_not_one_common_se3" in report["errors"]


def test_bimanual_divergent_processed_object_tracks_are_rejected(tmp_path: Path) -> None:
    manifest, processed = _fixture(
        tmp_path,
        bimanual=True,
        right_object_offset=np.array([0.002, 0.0, 0.0]),
    )
    report = MODULE.evaluate(
        adapter_manifest=manifest,
        processed_keypoints=processed,
        trajectory_6dof="egoinfinity",
        hand_type="bimanual",
    )
    assert report["status"] == "invalid"
    assert "processed_bimanual_object_tracks_are_inconsistent" in report["errors"]


def test_bimanual_quaternion_sign_equivalence_is_accepted(tmp_path: Path) -> None:
    manifest, processed = _fixture(
        tmp_path,
        bimanual=True,
        right_object_quaternion_sign_flip=True,
    )
    report = MODULE.evaluate(
        adapter_manifest=manifest,
        processed_keypoints=processed,
        trajectory_6dof="egoinfinity",
        hand_type="bimanual",
    )
    assert report["status"] == "ok", report["errors"]
    assert (
        report["processed_object_track_consistency"]["absolute_component_error"][
            "max"
        ]
        == 0.0
    )
    assert (
        report["processed_object_track_consistency"]["rotation_error_rad"]["max"]
        == 0.0
    )


def test_layout_outside_exact_raw_dir_is_rejected(tmp_path: Path) -> None:
    manifest, processed = _fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    exact_layout = Path(payload["layout"])
    external_layout = tmp_path / "other_route" / exact_layout.name
    external_layout.parent.mkdir()
    external_layout.write_bytes(exact_layout.read_bytes())
    payload["layout"] = str(external_layout)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    try:
        MODULE.evaluate(
            adapter_manifest=manifest,
            processed_keypoints=processed,
            trajectory_6dof="egoinfinity",
            hand_type="left",
        )
    except ValueError as exc:
        assert "outside the exact-route raw directory" in str(exc)
    else:
        raise AssertionError("cross-route object layout must fail closed")


class RawToProcessedFullSE3Unittest(unittest.TestCase):
    """Run the regression cases under the repository's unittest entrypoint."""

    def _run(self, test) -> None:
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))

    def test_common_se3(self) -> None:
        self._run(test_arbitrary_common_se3_preserves_raw_to_processed_hoi)

    def test_object_only_translation(self) -> None:
        self._run(test_object_only_translation_is_rejected)

    def test_object_only_rotation(self) -> None:
        self._run(test_object_only_rotation_is_rejected_by_full_relative_se3)

    def test_invalid_quaternion(self) -> None:
        self._run(test_zero_processed_object_quaternion_is_rejected)

    def test_bimanual_common_se3(self) -> None:
        self._run(test_bimanual_uses_one_joint_common_se3)

    def test_independent_bimanual_transform(self) -> None:
        self._run(test_independent_bimanual_hand_transform_fails_joint_fit)

    def test_bimanual_object_track_consistency(self) -> None:
        self._run(test_bimanual_divergent_processed_object_tracks_are_rejected)

    def test_bimanual_quaternion_sign_equivalence(self) -> None:
        self._run(test_bimanual_quaternion_sign_equivalence_is_accepted)

    def test_cross_route_layout(self) -> None:
        self._run(test_layout_outside_exact_raw_dir_is_rejected)
