from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import mujoco
    import numpy as np

    from scripts.resolve_spider_rest_support import resolve_spider_rest_support

    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only in lean test envs.
    mujoco = None
    np = None
    resolve_spider_rest_support = None
    _IMPORT_ERROR = exc


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scene_xml(hand_x: float) -> str:
    return f"""\
<mujoco model="spider_rest_support_test">
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="left_hand_fixture" pos="{hand_x} 0 0.3">
      <geom name="collision_hand_left_palm_0" type="sphere" size="0.02" group="3"/>
    </body>
    <body name="left_object">
      <joint name="left_object_pos_x" type="slide" axis="1 0 0"/>
      <joint name="left_object_pos_y" type="slide" axis="0 1 0"/>
      <joint name="left_object_pos_z" type="slide" axis="0 0 1"/>
      <joint name="left_object_rot_x" type="hinge" axis="1 0 0"/>
      <joint name="left_object_rot_y" type="hinge" axis="0 1 0"/>
      <joint name="left_object_rot_z" type="hinge" axis="0 0 1"/>
      <geom name="left_object_collision_box" type="box" size="0.04 0.03 0.05"
            group="3" density="1000"/>
    </body>
  </worldbody>
</mujoco>
"""


@unittest.skipIf(_IMPORT_ERROR is not None, f"MuJoCo stack unavailable: {_IMPORT_ERROR}")
class SpiderRestSupportTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        hand_x: float,
        object_x_start: float,
        object_x_end: float | None = None,
    ) -> tuple[Path, Path, Path]:
        scene = root / "scene_act.xml"
        trajectory = root / "trajectory_kinematic_act.npz"
        provenance = root / "rest_support.json"
        scene.write_text(_scene_xml(hand_x), encoding="utf-8")
        if object_x_end is None:
            object_x_end = object_x_start
        qpos = np.asarray(
            [
                [object_x_start, 0.0, 0.3, 0.0, 0.0, 0.0],
                [object_x_end, 0.0, 0.3, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        np.savez(trajectory, qpos=qpos)
        return scene, trajectory, provenance

    def test_adds_static_pedestal_and_object_attached_support_without_touching_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, trajectory, provenance = self._fixture(
                root, hand_x=0.0, object_x_start=0.5
            )
            trajectory_sha = _sha256(trajectory)

            result = resolve_spider_rest_support(
                scene,
                trajectory,
                provenance_path=provenance,
            )

            self.assertEqual(result["supports_added"], 1)
            self.assertEqual(result["not_in_hand_endpoints"], 2)
            self.assertEqual(result["overlapping_endpoint_fixtures_reused"], 1)
            self.assertTrue(
                all(not record["endpoint_in_hand"] for record in result["objects"])
            )
            self.assertEqual(result["objects"][0]["action"], "added_rest_support")
            self.assertEqual(
                result["objects"][1]["action"],
                "reused_overlapping_rest_support",
            )
            self.assertTrue(result["trajectory_integrity"]["unchanged"])
            self.assertEqual(_sha256(trajectory), trajectory_sha)

            xml_root = ET.parse(scene).getroot()
            pedestal = xml_root.find(
                "./worldbody/geom[@name='spider_rest_pedestal_left_start']"
            )
            support = xml_root.find(
                ".//body[@name='left_object']/geom[@name='spider_rest_support_left_start']"
            )
            self.assertIsNotNone(pedestal)
            self.assertIsNotNone(support)
            self.assertEqual(support.get("density"), "0")
            self.assertEqual(support.get("contype"), "0")
            self.assertEqual(support.get("conaffinity"), "0")
            self.assertIsNotNone(
                xml_root.find(
                    "./contact/pair[@name='spider_rest_pair_spider_rest_support_left_start_spider_rest_pedestal_left_start']"
                )
            )
            mujoco.MjModel.from_xml_path(str(scene))

            saved = json.loads(provenance.read_text(encoding="utf-8"))
            self.assertFalse(saved["policy"]["trajectory_write_allowed"])
            self.assertTrue(saved["policy"]["support_only_when_endpoint_not_in_hand"])

            # Re-resolution strips its own generated elements before adding the
            # current endpoint decision, so geometry never accumulates.
            rerun = resolve_spider_rest_support(
                scene,
                trajectory,
                provenance_path=provenance,
            )
            self.assertGreater(rerun["previous_generated_elements_removed"], 0)
            rerun_root = ET.parse(scene).getroot()
            self.assertEqual(
                len(
                    rerun_root.findall(
                        "./worldbody/geom[@name='spider_rest_pedestal_left_start']"
                    )
                ),
                1,
            )
            self.assertEqual(
                len(
                    rerun_root.findall(
                        ".//body[@name='left_object']/geom[@name='spider_rest_support_left_start']"
                    )
                ),
                1,
            )
            self.assertEqual(_sha256(trajectory), trajectory_sha)

    def test_audit_only_records_need_without_modifying_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, trajectory, provenance = self._fixture(
                root, hand_x=0.0, object_x_start=0.5
            )
            scene_sha = _sha256(scene)

            result = resolve_spider_rest_support(
                scene,
                trajectory,
                provenance_path=provenance,
                apply_supports=False,
            )

            self.assertEqual(_sha256(scene), scene_sha)
            self.assertEqual(result["supports_added"], 0)
            self.assertEqual(result["supports_would_be_added"], 1)
            self.assertEqual(
                result["policy"]["support_application_mode"], "audit_only"
            )
            self.assertTrue(result["policy"]["diagnostic_only"])
            self.assertFalse(
                result["policy"]["production_scene_geometry_modified"]
            )
            self.assertEqual(
                result["objects"][0]["action"],
                "diagnostic_support_not_applied",
            )
            self.assertEqual(
                result["objects"][1]["action"],
                "diagnostic_overlapping_support_not_applied",
            )

    def test_in_hand_endpoint_is_a_strict_no_support_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, trajectory, provenance = self._fixture(
                root, hand_x=0.0, object_x_start=0.0
            )
            scene_sha = _sha256(scene)
            trajectory_sha = _sha256(trajectory)

            result = resolve_spider_rest_support(
                scene,
                trajectory,
                provenance_path=provenance,
            )

            self.assertEqual(result["supports_added"], 0)
            self.assertEqual(len(result["objects"]), 2)
            self.assertTrue(
                all(record["endpoint_in_hand"] for record in result["objects"])
            )
            self.assertTrue(
                all(
                    record["action"] == "skipped_endpoint_in_hand"
                    for record in result["objects"]
                )
            )
            self.assertEqual(_sha256(scene), scene_sha)
            self.assertEqual(_sha256(trajectory), trajectory_sha)
            xml_root = ET.parse(scene).getroot()
            self.assertFalse(
                any(
                    (element.get("name") or "").startswith("spider_rest_")
                    for element in xml_root.iter()
                )
            )

    def test_start_rest_is_supported_even_when_end_is_in_hand(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, trajectory, provenance = self._fixture(
                root,
                hand_x=0.0,
                object_x_start=0.5,
                object_x_end=0.0,
            )

            result = resolve_spider_rest_support(
                scene,
                trajectory,
                provenance_path=provenance,
            )

            records = {record["endpoint"]: record for record in result["objects"]}
            self.assertEqual(result["supports_added"], 1)
            self.assertFalse(records["start"]["endpoint_in_hand"])
            self.assertEqual(records["start"]["action"], "added_rest_support")
            self.assertTrue(records["end"]["endpoint_in_hand"])
            self.assertEqual(records["end"]["action"], "skipped_endpoint_in_hand")
            xml_root = ET.parse(scene).getroot()
            self.assertIsNotNone(
                xml_root.find(
                    "./worldbody/geom[@name='spider_rest_pedestal_left_start']"
                )
            )
            self.assertIsNone(
                xml_root.find(
                    "./worldbody/geom[@name='spider_rest_pedestal_left_end']"
                )
            )

    def test_distinct_not_in_hand_endpoints_get_distinct_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene, trajectory, provenance = self._fixture(
                root,
                hand_x=0.0,
                object_x_start=0.5,
                object_x_end=-0.5,
            )

            result = resolve_spider_rest_support(
                scene,
                trajectory,
                provenance_path=provenance,
            )

            self.assertEqual(result["not_in_hand_endpoints"], 2)
            self.assertEqual(result["supports_added"], 2)
            self.assertEqual(result["overlapping_endpoint_fixtures_reused"], 0)
            self.assertTrue(
                all(
                    record["action"] == "added_rest_support"
                    for record in result["objects"]
                )
            )
            xml_root = ET.parse(scene).getroot()
            for endpoint in ("start", "end"):
                self.assertIsNotNone(
                    xml_root.find(
                        f"./worldbody/geom[@name='spider_rest_pedestal_left_{endpoint}']"
                    )
                )


if __name__ == "__main__":
    unittest.main()
