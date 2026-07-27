#!/usr/bin/env python3
"""Audit a SPIDER post-IK scene for start/end rest support.

The resolver reads ``scene_act.xml`` and ``trajectory_kinematic_act.npz`` after
SPIDER IK.  It never rewrites the trajectory.  For each object side present in
the scene, it evaluates both IK endpoints against the same-side robot hand
collision geometry.  Production CLI use is audit-only: it records where a
rest support would be needed but does not add geometry or otherwise change the
simulation.  ``--apply-supports`` is retained only for explicit diagnostics.

The generated geometry is intentionally prefixed with ``spider_rest_`` so a
rerun can strip and recompute it deterministically instead of accumulating
duplicate support.  The scene is compiled before an atomic replacement, and a
JSON provenance record captures the decision and verifies that the trajectory
hash did not change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np


GENERATED_PREFIX = "spider_rest_"
DEFAULT_HAND_OBJECT_DISTANCE_M = 0.1
DEFAULT_DISTANCE_QUERY_MAX_M = 10.0
DEFAULT_BOTTOM_SLICE_M = 0.01
DEFAULT_MIN_SUPPORT_RADIUS_M = 0.005
DEFAULT_MAX_SUPPORT_RADIUS_M = 0.5
PEDESTAL_HALF_HEIGHT_M = 0.005
SUPPORT_HALF_HEIGHT_M = 0.002
PEDESTAL_OBJECT_GAP_M = 0.0001
DEFAULT_SOLREF = "0.01 1"
DEFAULT_SOLIMP = "0.9 0.95 0.001 0.5 2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt(value: float) -> str:
    return f"{float(value):.10g}"


def _vec(values: Iterable[float]) -> str:
    return " ".join(_fmt(v) for v in values)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _indent(root: ET.Element) -> None:
    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")
        return

    def visit(node: ET.Element, level: int = 0) -> None:
        prefix = "\n" + "  " * level
        child_prefix = "\n" + "  " * (level + 1)
        if len(node):
            if not node.text or not node.text.strip():
                node.text = child_prefix
            for child in node:
                visit(child, level + 1)
                if not child.tail or not child.tail.strip():
                    child.tail = child_prefix
            node[-1].tail = prefix
        elif level and (not node.tail or not node.tail.strip()):
            node.tail = prefix

    visit(root)


def _serialize_tree(root: ET.Element, path: Path) -> None:
    _indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _temporary_xml(root: ET.Element, directory: Path) -> Path:
    fd, name = tempfile.mkstemp(
        dir=str(directory), prefix=".spider_rest_support.", suffix=".xml"
    )
    os.close(fd)
    path = Path(name)
    try:
        _serialize_tree(root, path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _strip_generated(root: ET.Element) -> int:
    removed = 0
    for parent in root.iter():
        for child in list(parent):
            if child.tag not in {"geom", "pair"}:
                continue
            if (child.get("name") or "").startswith(GENERATED_PREFIX):
                parent.remove(child)
                removed += 1
    return removed


def _name(model: mujoco.MjModel, object_type: Any, object_id: int) -> str:
    value = mujoco.mj_id2name(model, object_type, int(object_id))
    return value or ""


def _contacting(data: mujoco.MjData, geom1: int, geom2: int) -> bool:
    pair = {int(geom1), int(geom2)}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        if {int(contact.geom1), int(contact.geom2)} == pair:
            return True
    return False


def _geom_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom1: int,
    geom2: int,
    query_max_m: float,
) -> dict[str, Any]:
    """Return a robust distance record for one hand/object geom pair.

    MuJoCo 3.4 can return zero for a separated mesh pair while ``fromto``
    contains two distinct closest points and no contact exists.  In precisely
    that case the segment norm is the usable separation.  Genuine contacts and
    penetrations keep MuJoCo's signed result.
    """

    fromto = np.zeros(6, dtype=np.float64)
    if hasattr(mujoco, "mj_geomDistance"):
        raw = float(
            mujoco.mj_geomDistance(
                model, data, int(geom1), int(geom2), float(query_max_m), fromto
            )
        )
        method = "mj_geomDistance"
    else:  # MuJoCo < 3.0 compatibility for local/unit-test environments.
        points1 = _geom_world_points(model, data, int(geom1))
        points2 = _geom_world_points(model, data, int(geom2))
        # This branch is not used by the production MuJoCo 3.4 environment.
        # Bound it for older runtimes that expose a dense mesh vertex array.
        if len(points1) > 2048:
            points1 = points1[np.linspace(0, len(points1) - 1, 2048).astype(int)]
        if len(points2) > 2048:
            points2 = points2[np.linspace(0, len(points2) - 1, 2048).astype(int)]
        best_distance = math.inf
        best1 = points1[0]
        best2 = points2[0]
        for start in range(0, len(points1), 128):
            delta = points1[start : start + 128, None, :] - points2[None, :, :]
            distance = np.linalg.norm(delta, axis=-1)
            flat = int(distance.argmin())
            row, column = divmod(flat, distance.shape[1])
            candidate = float(distance[row, column])
            if candidate < best_distance:
                best_distance = candidate
                best1 = points1[start + row]
                best2 = points2[column]
        fromto[:3] = best1
        fromto[3:] = best2
        raw = min(float(query_max_m), best_distance)
        method = "sampled_surface_fallback"
    segment = float(np.linalg.norm(fromto[:3] - fromto[3:]))
    has_contact = _contacting(data, geom1, geom2)
    false_zero = bool(abs(raw) <= 1e-12 and segment > 1e-9 and not has_contact)
    effective = segment if false_zero else raw
    return {
        "hand_geom": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
        "object_geom": _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
        "raw_distance_m": raw,
        "fromto_segment_m": segment,
        "effective_distance_m": effective,
        "false_zero_corrected": false_zero,
        "contact_present": has_contact,
        "distance_method": method,
        "fromto": fromto.tolist(),
    }


def _collision_geom_ids_for_body(
    model: mujoco.MjModel, body_id: int
) -> list[int]:
    out: list[int] = []
    for geom_id in range(int(model.ngeom)):
        if int(model.geom_bodyid[geom_id]) != int(body_id):
            continue
        geom_name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name.startswith(GENERATED_PREFIX):
            continue
        if int(model.geom_group[geom_id]) == 3:
            out.append(geom_id)
    return out


def _hand_collision_geom_ids(model: mujoco.MjModel, side: str) -> list[int]:
    prefix = f"collision_hand_{side}_"
    return [
        geom_id
        for geom_id in range(int(model.ngeom))
        if _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).startswith(prefix)
    ]


def _all_hand_collision_names(model: mujoco.MjModel) -> list[str]:
    return [
        name
        for geom_id in range(int(model.ngeom))
        if (name := _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)).startswith(
            "collision_hand_"
        )
    ]


def _primitive_local_points(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    angles = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return np.asarray(
            [
                [sx * size[0], sy * size[1], sz * size[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=float,
        )
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        rings = [
            np.column_stack(
                [circle[:, 0] * size[0], circle[:, 1] * size[0], np.full(64, z)]
            )
            for z in (-size[1], size[1])
        ]
        return np.concatenate(rings, axis=0)
    if geom_type in {
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    }:
        radii = (
            np.repeat(size[0], 3)
            if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE)
            else size[:3]
        )
        latitudes = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 33)
        rows = []
        for latitude in latitudes:
            c = math.cos(float(latitude))
            s = math.sin(float(latitude))
            rows.append(
                np.column_stack(
                    [
                        circle[:, 0] * c * radii[0],
                        circle[:, 1] * c * radii[1],
                        np.full(64, s * radii[2]),
                    ]
                )
            )
        return np.concatenate(rows, axis=0)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius, half_length = float(size[0]), float(size[1])
        rows = []
        latitudes = np.linspace(-0.5 * math.pi, 0.5 * math.pi, 17)
        for center_z in (-half_length, half_length):
            for latitude in latitudes:
                c = math.cos(float(latitude))
                s = math.sin(float(latitude))
                rows.append(
                    np.column_stack(
                        [
                            circle[:, 0] * c * radius,
                            circle[:, 1] * c * radius,
                            np.full(64, center_z + s * radius),
                        ]
                    )
                )
        return np.concatenate(rows, axis=0)
    raise ValueError(
        "unsupported object collision geom for footprint: "
        f"{_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)!r} "
        f"(MuJoCo type {geom_type})"
    )


def _geom_world_points(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> np.ndarray:
    if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0:
            raise ValueError(f"mesh geom {geom_id} has no mesh data")
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(model.mesh_vert[start : start + count], dtype=float)
    else:
        local = _primitive_local_points(model, geom_id)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
    position = np.asarray(data.geom_xpos[geom_id], dtype=float)
    return local @ rotation.T + position


def _object_bottom_footprint(
    world_points: np.ndarray,
    com_world: np.ndarray,
    bottom_slice_m: float,
    min_radius_m: float,
    max_radius_m: float,
) -> tuple[np.ndarray, float, float, str]:
    if world_points.ndim != 2 or world_points.shape[1] != 3 or not len(world_points):
        raise ValueError("object footprint needs non-empty (N,3) world points")
    if not np.isfinite(world_points).all():
        raise ValueError("object footprint contains non-finite vertices")

    bottom_z = float(world_points[:, 2].min())
    bottom_points = world_points[
        world_points[:, 2] <= bottom_z + float(bottom_slice_m)
    ]
    xy_min = bottom_points[:, :2].min(axis=0)
    xy_max = bottom_points[:, :2].max(axis=0)
    bottom_center = (xy_min + xy_max) / 2.0
    bottom_radius = float(
        np.linalg.norm(bottom_points[:, :2] - bottom_center, axis=1).max()
    )
    com_xy = np.asarray(com_world[:2], dtype=float)
    if float(np.linalg.norm(com_xy - bottom_center)) > bottom_radius:
        center_xy = com_xy
        radius = float(
            np.linalg.norm(world_points[:, :2] - center_xy, axis=1).max()
        )
        mode = "com_cover_all_vertices"
    else:
        center_xy = bottom_center
        radius = bottom_radius
        mode = "bottom_contact_patch"
    radius = max(float(min_radius_m), radius)
    if not np.isfinite(radius) or radius > float(max_radius_m):
        raise ValueError(
            f"unsafe support radius {radius:.6f}m; maximum is {max_radius_m:.6f}m"
        )
    return center_xy, radius, bottom_z, mode


def _find_body_element(root: ET.Element, body_name: str) -> ET.Element:
    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise ValueError(f"body {body_name!r} is not materialized in scene XML")
    return body


def _ensure_contact(root: ET.Element) -> ET.Element:
    contact = root.find("contact")
    if contact is not None:
        return contact
    contact = ET.Element("contact")
    children = list(root)
    before = {"equality", "tendon", "actuator", "sensor", "keyframe"}
    index = next(
        (idx for idx, child in enumerate(children) if child.tag in before),
        len(children),
    )
    root.insert(index, contact)
    return contact


def _available_materials(root: ET.Element) -> set[str]:
    return {
        value
        for material in root.findall("./asset/material")
        if (value := material.get("name"))
    }


def _add_support_geometry(
    root: ET.Element,
    side: str,
    endpoint: str,
    body_position: np.ndarray,
    body_rotation: np.ndarray,
    center_xy: np.ndarray,
    radius: float,
    bottom_z: float,
    object_collision_names: list[str],
    all_hand_collision_names: list[str],
    friction_scale: float,
) -> dict[str, Any]:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("scene XML has no worldbody")
    object_body = _find_body_element(root, f"{side}_object")

    pedestal_name = f"{GENERATED_PREFIX}pedestal_{side}_{endpoint}"
    support_name = f"{GENERATED_PREFIX}support_{side}_{endpoint}"
    pedestal_position = np.asarray(
        [
            float(center_xy[0]),
            float(center_xy[1]),
            bottom_z - PEDESTAL_HALF_HEIGHT_M - PEDESTAL_OBJECT_GAP_M,
        ],
        dtype=float,
    )
    support_world_position = np.asarray(
        [float(center_xy[0]), float(center_xy[1]), bottom_z + SUPPORT_HALF_HEIGHT_M],
        dtype=float,
    )
    support_local_position = body_rotation.T @ (
        support_world_position - body_position
    )

    # The desired support orientation is identity in world space.  Its local
    # orientation is therefore the inverse of the object's world orientation.
    body_quat = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(body_quat, np.asarray(body_rotation).reshape(-1))
    support_local_quat = body_quat * np.asarray([1.0, -1.0, -1.0, -1.0])

    pedestal_attrs = {
        "name": pedestal_name,
        "type": "cylinder",
        "size": _vec([radius, PEDESTAL_HALF_HEIGHT_M]),
        "pos": _vec(pedestal_position),
        "group": "3",
    }
    materials = _available_materials(root)
    material = (
        f"{side}_groundplane"
        if f"{side}_groundplane" in materials
        else "groundplane" if "groundplane" in materials else None
    )
    if material:
        pedestal_attrs["material"] = material
    else:
        pedestal_attrs["rgba"] = "0.35 0.35 0.4 1"
    worldbody.append(ET.Element("geom", pedestal_attrs))

    object_body.append(
        ET.Element(
            "geom",
            {
                "name": support_name,
                "type": "cylinder",
                "size": _vec([radius, SUPPORT_HALF_HEIGHT_M]),
                "pos": _vec(support_local_position),
                "quat": _vec(support_local_quat),
                "contype": "0",
                "conaffinity": "0",
                "density": "0",
                "group": "3",
                "rgba": "0.4 0.4 0.8 0.4",
            },
        )
    )

    friction = _vec(
        [1.0 * friction_scale, 1.0 * friction_scale, 0.1 * friction_scale, 0, 0]
    )
    contact = _ensure_contact(root)

    def add_pair(name: str, geom1: str, geom2: str, condim: int) -> None:
        contact.append(
            ET.Element(
                "pair",
                {
                    "name": name,
                    "geom1": geom1,
                    "geom2": geom2,
                    "condim": str(condim),
                    "solref": DEFAULT_SOLREF,
                    "solimp": DEFAULT_SOLIMP,
                    "friction": friction,
                },
            )
        )

    for object_geom_name in object_collision_names:
        add_pair(
            f"{GENERATED_PREFIX}pair_{object_geom_name}_{pedestal_name}",
            object_geom_name,
            pedestal_name,
            6,
        )
    for hand_geom_name in all_hand_collision_names:
        add_pair(
            f"{GENERATED_PREFIX}pair_{hand_geom_name}_{pedestal_name}",
            hand_geom_name,
            pedestal_name,
            3,
        )
    add_pair(
        f"{GENERATED_PREFIX}pair_{support_name}_{pedestal_name}",
        support_name,
        pedestal_name,
        3,
    )

    return {
        "endpoint": endpoint,
        "pedestal_name": pedestal_name,
        "support_name": support_name,
        "center_xy_m": center_xy.tolist(),
        "radius_m": float(radius),
        "object_bottom_z_m": float(bottom_z),
        "pedestal_position_m": pedestal_position.tolist(),
        "pedestal_half_height_m": PEDESTAL_HALF_HEIGHT_M,
        "pedestal_object_gap_m": PEDESTAL_OBJECT_GAP_M,
        "support_world_position_m": support_world_position.tolist(),
        "support_local_position_m": support_local_position.tolist(),
        "support_local_quat_wxyz": support_local_quat.tolist(),
        "support_half_height_m": SUPPORT_HALF_HEIGHT_M,
        "support_attached_to_body": f"{side}_object",
        "support_massless": True,
    }


def _write_validated_scene(
    root: ET.Element, input_scene: Path, output_scene: Path
) -> None:
    temp = _temporary_xml(root, output_scene.parent)
    try:
        # Fail before replacement if the generated MJCF is invalid.
        mujoco.MjModel.from_xml_path(str(temp))
        mode = stat.S_IMODE(input_scene.stat().st_mode)
        os.chmod(temp, mode)
        os.replace(temp, output_scene)
    finally:
        temp.unlink(missing_ok=True)


def resolve_spider_rest_support(
    scene_act_path: str | Path,
    trajectory_path: str | Path,
    output_scene_act_path: str | Path | None = None,
    provenance_path: str | Path | None = None,
    hand_object_distance_m: float = DEFAULT_HAND_OBJECT_DISTANCE_M,
    distance_query_max_m: float = DEFAULT_DISTANCE_QUERY_MAX_M,
    bottom_slice_m: float = DEFAULT_BOTTOM_SLICE_M,
    min_support_radius_m: float = DEFAULT_MIN_SUPPORT_RADIUS_M,
    max_support_radius_m: float = DEFAULT_MAX_SUPPORT_RADIUS_M,
    friction_scale: float = 1.5,
    route_binding: dict[str, Any] | None = None,
    apply_supports: bool = True,
) -> dict[str, Any]:
    scene_act_path = Path(scene_act_path).expanduser().resolve()
    trajectory_path = Path(trajectory_path).expanduser().resolve()
    output_scene = (
        Path(output_scene_act_path).expanduser().resolve()
        if output_scene_act_path is not None
        else scene_act_path
    )
    provenance = (
        Path(provenance_path).expanduser().resolve()
        if provenance_path is not None
        else output_scene.with_name("spider_rest_support_provenance.json")
    )

    if not scene_act_path.is_file():
        raise FileNotFoundError(f"SPIDER scene_act.xml not found: {scene_act_path}")
    if not trajectory_path.is_file():
        raise FileNotFoundError(
            f"SPIDER trajectory_kinematic_act.npz not found: {trajectory_path}"
        )
    if output_scene.parent != scene_act_path.parent:
        raise ValueError(
            "output scene must stay beside input scene so relative MJCF assets remain valid"
        )
    if hand_object_distance_m <= 0 or distance_query_max_m <= hand_object_distance_m:
        raise ValueError("distance thresholds must be positive and query max must exceed gate")
    if not 0 < min_support_radius_m <= max_support_radius_m:
        raise ValueError("invalid support radius bounds")

    input_scene_sha = _sha256(scene_act_path)
    trajectory_sha_before = _sha256(trajectory_path)
    tree = ET.parse(scene_act_path)
    root = tree.getroot()
    stripped_count = _strip_generated(root)

    base_temp = _temporary_xml(root, scene_act_path.parent)
    try:
        base_scene_sha_after_strip = _sha256(base_temp)
        model = mujoco.MjModel.from_xml_path(str(base_temp))
    finally:
        base_temp.unlink(missing_ok=True)

    with np.load(trajectory_path, allow_pickle=False) as archive:
        if "qpos" not in archive:
            raise ValueError(f"trajectory has no qpos array: {trajectory_path}")
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or not qpos.shape[0]:
        raise ValueError(f"trajectory qpos must be non-empty (T,nq), got {qpos.shape}")
    if qpos.shape[1] != int(model.nq):
        raise ValueError(
            f"trajectory/model nq mismatch: qpos={qpos.shape[1]}, model={model.nq}"
        )
    if not np.isfinite(qpos).all():
        raise ValueError("trajectory qpos contains non-finite values")

    data = mujoco.MjData(model)

    all_hand_names = _all_hand_collision_names(model)
    if not all_hand_names:
        raise ValueError("scene contains no collision_hand_* geoms; refusing support decision")

    object_records: list[dict[str, Any]] = []
    support_count = 0
    support_would_be_added_count = 0
    not_in_hand_endpoint_count = 0
    deduplicated_fixture_count = 0
    for side in ("right", "left"):
        body_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_object")
        )
        if body_id < 0:
            continue
        object_geom_ids = _collision_geom_ids_for_body(model, body_id)
        if not object_geom_ids:
            raise ValueError(f"{side}_object has no group=3 collision geometry")
        hand_geom_ids = _hand_collision_geom_ids(model, side)
        if not hand_geom_ids:
            raise ValueError(
                f"{side}_object has no same-side collision_hand_{side}_* geometry"
            )

        object_geom_names = [
            _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in object_geom_ids
        ]
        hand_geom_names = [
            _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            for geom_id in hand_geom_ids
        ]
        placed_for_side: list[dict[str, Any]] = []
        for endpoint, frame_index in (
            ("start", 0),
            ("end", int(qpos.shape[0] - 1)),
        ):
            data.qpos[:] = qpos[frame_index]
            mujoco.mj_forward(model, data)

            distance_records = [
                _geom_distance(
                    model, data, hand_id, object_id, distance_query_max_m
                )
                for hand_id in hand_geom_ids
                for object_id in object_geom_ids
            ]
            best = min(
                distance_records,
                key=lambda distance_record: distance_record["effective_distance_m"],
            )
            min_distance = float(best["effective_distance_m"])
            if not np.isfinite(min_distance):
                raise ValueError(
                    f"non-finite {side} hand-object {endpoint} endpoint distance"
                )
            endpoint_in_hand = bool(min_distance < hand_object_distance_m)
            record: dict[str, Any] = {
                "side": side,
                "object_body": f"{side}_object",
                "endpoint": endpoint,
                "endpoint_frame_index": frame_index,
                "endpoint_in_hand": endpoint_in_hand,
                "hand_object_distance_threshold_m": float(hand_object_distance_m),
                "minimum_hand_object_distance_m": min_distance,
                "minimum_distance_evidence": best,
                "false_zero_corrections_considered": int(
                    sum(
                        bool(item["false_zero_corrected"])
                        for item in distance_records
                    )
                ),
                "object_collision_geoms": object_geom_names,
                "same_side_hand_collision_geoms": hand_geom_names,
                "object_body_world_position_m": np.asarray(
                    data.xpos[body_id]
                ).tolist(),
                "object_body_world_quat_wxyz": np.asarray(
                    data.xquat[body_id]
                ).tolist(),
            }
            if endpoint_in_hand:
                record["action"] = "skipped_endpoint_in_hand"
                object_records.append(record)
                continue

            not_in_hand_endpoint_count += 1
            world_points = np.concatenate(
                [
                    _geom_world_points(model, data, geom_id)
                    for geom_id in object_geom_ids
                ],
                axis=0,
            )
            center_xy, radius, bottom_z, footprint_mode = _object_bottom_footprint(
                world_points,
                np.asarray(data.xipos[body_id], dtype=float),
                bottom_slice_m=bottom_slice_m,
                min_radius_m=min_support_radius_m,
                max_radius_m=max_support_radius_m,
            )

            # Same de-duplication used by Do-as-I-Do: if endpoint fixtures on
            # one side overlap in XY and have effectively the same top height,
            # one static pedestal/object-attached support serves both.
            reused = next(
                (
                    previous
                    for previous in placed_for_side
                    if float(
                        np.linalg.norm(
                            np.asarray(center_xy) - previous["center_xy"]
                        )
                    )
                    < min(float(radius), float(previous["radius_m"]))
                    and abs(float(bottom_z) - float(previous["bottom_z_m"]))
                    < 2.0 * PEDESTAL_HALF_HEIGHT_M
                ),
                None,
            )
            if reused is not None:
                record["action"] = (
                    "reused_overlapping_rest_support"
                    if apply_supports
                    else "diagnostic_overlapping_support_not_applied"
                )
                record["fixture_reused_from_endpoint"] = reused["endpoint"]
                record["placement"] = reused["placement"]
                record["candidate_footprint"] = {
                    "center_xy_m": np.asarray(center_xy).tolist(),
                    "radius_m": float(radius),
                    "object_bottom_z_m": float(bottom_z),
                    "footprint_mode": footprint_mode,
                    "bottom_slice_m": float(bottom_slice_m),
                }
                object_records.append(record)
                deduplicated_fixture_count += 1
                continue

            body_rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
            if not apply_supports:
                candidate = {
                    "center_xy_m": np.asarray(center_xy).tolist(),
                    "radius_m": float(radius),
                    "object_bottom_z_m": float(bottom_z),
                    "footprint_mode": footprint_mode,
                    "bottom_slice_m": float(bottom_slice_m),
                }
                placed_for_side.append(
                    {
                        "endpoint": endpoint,
                        "center_xy": np.asarray(center_xy, dtype=float),
                        "radius_m": float(radius),
                        "bottom_z_m": float(bottom_z),
                        "placement": candidate,
                    }
                )
                record["action"] = "diagnostic_support_not_applied"
                record["candidate_footprint"] = candidate
                object_records.append(record)
                support_would_be_added_count += 1
                continue

            placement = _add_support_geometry(
                root,
                side=side,
                endpoint=endpoint,
                body_position=np.asarray(data.xpos[body_id], dtype=float),
                body_rotation=body_rotation,
                center_xy=center_xy,
                radius=radius,
                bottom_z=bottom_z,
                object_collision_names=object_geom_names,
                all_hand_collision_names=all_hand_names,
                friction_scale=friction_scale,
            )
            placement["footprint_mode"] = footprint_mode
            placement["bottom_slice_m"] = float(bottom_slice_m)
            placed_for_side.append(
                {
                    "endpoint": endpoint,
                    "center_xy": np.asarray(center_xy, dtype=float),
                    "radius_m": float(radius),
                    "bottom_z_m": float(bottom_z),
                    "placement": placement,
                }
            )
            record["action"] = "added_rest_support"
            record["placement"] = placement
            object_records.append(record)
            support_count += 1

    if not object_records:
        raise ValueError("scene contains neither left_object nor right_object body")

    if support_count or stripped_count or output_scene != scene_act_path:
        _write_validated_scene(root, scene_act_path, output_scene)
    output_scene_sha = _sha256(output_scene)
    trajectory_sha_after = _sha256(trajectory_path)
    if trajectory_sha_after != trajectory_sha_before:
        raise RuntimeError("trajectory changed during read-only rest-support resolution")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "stage": "spider_post_ik_rest_support",
        "status": "ok",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "scene_act_xml": str(scene_act_path),
            "scene_act_sha256_before": input_scene_sha,
            "trajectory_kinematic_act_npz": str(trajectory_path),
            "trajectory_sha256_before": trajectory_sha_before,
        },
        "outputs": {
            "scene_act_xml": str(output_scene),
            "scene_act_sha256_after": output_scene_sha,
            "provenance_json": str(provenance),
        },
        "route_binding": route_binding or {},
        "base_scene_sha256_after_stripping_generated_support": base_scene_sha_after_strip,
        "policy": {
            "endpoints": ["start", "end"],
            "support_only_when_endpoint_not_in_hand": True,
            "overlapping_same_side_endpoint_fixtures_deduplicated": True,
            "trajectory_write_allowed": False,
            "trajectory_modified": False,
            "hand_object_distance_threshold_m": float(hand_object_distance_m),
            "distance_query_max_m": float(distance_query_max_m),
            "mujoco_false_zero_guard": True,
            "generated_xml_prefix": GENERATED_PREFIX,
            "pedestal_static_world_geom": True,
            "support_object_body_attached": True,
            "support_massless": True,
            "max_support_radius_m": float(max_support_radius_m),
            "support_application_mode": (
                "explicit_diagnostic_apply" if apply_supports else "audit_only"
            ),
            "diagnostic_only": not apply_supports,
            "production_scene_geometry_modified": bool(support_count),
        },
        "trajectory_integrity": {
            "sha256_after": trajectory_sha_after,
            "unchanged": True,
        },
        "previous_generated_elements_removed": int(stripped_count),
        "not_in_hand_endpoints": int(not_in_hand_endpoint_count),
        "supports_added": int(support_count),
        "supports_would_be_added": int(support_would_be_added_count),
        "overlapping_endpoint_fixtures_reused": int(deduplicated_fixture_count),
        "objects": object_records,
        "endpoint_accounting": {
            "records": int(len(object_records)),
            "expected_records": int(
                2 * len({str(item.get("side")) for item in object_records})
            ),
            "complete": bool(
                len(object_records)
                == 2 * len({str(item.get("side")) for item in object_records})
            ),
        },
    }
    _write_json_atomic(provenance, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-act", required=True, type=Path)
    parser.add_argument("--trajectory-kinematic-act", required=True, type=Path)
    parser.add_argument(
        "--output-scene-act",
        type=Path,
        help="Defaults to atomic in-place replacement of --scene-act.",
    )
    parser.add_argument("--provenance-json", type=Path)
    parser.add_argument(
        "--hand-object-distance-m",
        type=float,
        default=DEFAULT_HAND_OBJECT_DISTANCE_M,
    )
    parser.add_argument(
        "--distance-query-max-m",
        type=float,
        default=DEFAULT_DISTANCE_QUERY_MAX_M,
    )
    parser.add_argument("--bottom-slice-m", type=float, default=DEFAULT_BOTTOM_SLICE_M)
    parser.add_argument(
        "--min-support-radius-m", type=float, default=DEFAULT_MIN_SUPPORT_RADIUS_M
    )
    parser.add_argument(
        "--max-support-radius-m", type=float, default=DEFAULT_MAX_SUPPORT_RADIUS_M
    )
    parser.add_argument("--friction-scale", type=float, default=1.5)
    parser.add_argument("--cell-key", default="")
    parser.add_argument("--trajectory-6dof", default="")
    parser.add_argument("--hand-source", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--hand-type", default="")
    parser.add_argument("--data-id", type=int, default=-1)
    parser.add_argument("--adapter-manifest", type=Path)
    parser.add_argument("--source-keypoints", type=Path)
    parser.add_argument(
        "--apply-supports",
        action="store_true",
        help=(
            "Diagnostic only: add generated support geometry. Production runs "
            "must omit this flag so the scene remains unmodified."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_scene = args.output_scene_act or args.scene_act
    provenance = args.provenance_json or output_scene.with_name(
        "spider_rest_support_provenance.json"
    )
    route_binding = {
        "cell_key": args.cell_key,
        "trajectory_6dof": args.trajectory_6dof,
        "hand_source": args.hand_source,
        "task": args.task,
        "hand_type": args.hand_type,
        "data_id": int(args.data_id),
    }
    for key, value in (
        ("adapter_manifest", args.adapter_manifest),
        ("source_keypoints", args.source_keypoints),
    ):
        if value is not None:
            resolved = value.expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"route binding file missing: {resolved}")
            route_binding[key] = str(resolved)
            route_binding[f"{key}_sha256"] = _sha256(resolved)
    try:
        payload = resolve_spider_rest_support(
            scene_act_path=args.scene_act,
            trajectory_path=args.trajectory_kinematic_act,
            output_scene_act_path=output_scene,
            provenance_path=provenance,
            hand_object_distance_m=args.hand_object_distance_m,
            distance_query_max_m=args.distance_query_max_m,
            bottom_slice_m=args.bottom_slice_m,
            min_support_radius_m=args.min_support_radius_m,
            max_support_radius_m=args.max_support_radius_m,
            friction_scale=args.friction_scale,
            route_binding=route_binding,
            apply_supports=args.apply_supports,
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "stage": "spider_post_ik_rest_support",
            "status": "invalid",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
            "inputs": {
                "scene_act_xml": str(Path(args.scene_act).expanduser().resolve()),
                "trajectory_kinematic_act_npz": str(
                    Path(args.trajectory_kinematic_act).expanduser().resolve()
                ),
            },
            "route_binding": route_binding,
            "policy": {
                "trajectory_write_allowed": False,
                "support_only_when_endpoint_not_in_hand": True,
            },
        }
        _write_json_atomic(Path(provenance).expanduser().resolve(), failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
