from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from aoe_retarget_lab.robot_asset_binding import (
    RobotAssetBindingError,
    bind_robot_asset_pair,
    bind_robot_asset_tree,
    robot_asset_manifest_errors,
)


def make_robot_tree(robots_root: Path, robot_type: str = "sharpa") -> Path:
    root = robots_root / robot_type
    meshes = root / "meshes"
    meshes.mkdir(parents=True)
    (meshes / "hand.stl").write_bytes(b"solid hand\nendsolid hand\n")
    (root / "left.xml").write_text(
        '<mujoco><compiler meshdir="meshes"/><asset>'
        '<mesh name="hand" file="hand.stl"/></asset></mujoco>',
        encoding="utf-8",
    )
    (root / "right.xml").write_text(
        '<mujoco><compiler meshdir="meshes"/><asset>'
        '<mesh name="hand" file="hand.stl"/></asset></mujoco>',
        encoding="utf-8",
    )
    (root / "bimanual.xml").write_text(
        '<mujoco><include file="left.xml"/><include file="right.xml"/></mujoco>',
        encoding="utf-8",
    )
    return root


def test_pair_binding_and_live_manifest_revalidation(tmp_path: Path) -> None:
    source_robots = tmp_path / "source" / "assets" / "robots"
    processed_robots = tmp_path / "processed" / "assets" / "robots"
    source = make_robot_tree(source_robots)
    processed = processed_robots / "sharpa"
    binding = bind_robot_asset_pair(
        source,
        processed,
        source_robots_root=source_robots,
        processed_robots_root=processed_robots,
        robot_type="sharpa",
        hand_type="left",
        copy_source=True,
    )
    assert binding["production_eligible"] is True
    assert binding["copy_matches_source"] is True
    assert binding["source"]["aggregate_sha256"] == binding["processed"]["aggregate_sha256"]
    assert binding["source"]["xml_dependencies"][0]["path"] == "meshes/hand.stl"

    errors, live = robot_asset_manifest_errors(
        {"robot_assets": binding},
        {"robot_assets": binding},
        expected_source_root=source,
        expected_processed_root=processed,
        robot_type="sharpa",
        hand_type="left",
    )
    assert errors == []
    assert live == binding

    (processed / "meshes" / "hand.stl").write_bytes(b"mutated")
    errors, _ = robot_asset_manifest_errors(
        {"robot_assets": binding},
        {"robot_assets": binding},
        expected_source_root=source,
        expected_processed_root=processed,
        robot_type="sharpa",
        hand_type="left",
    )
    assert any("live robot_assets invalid" in error for error in errors)


def test_missing_output_binding_fails_closed(tmp_path: Path) -> None:
    source_robots = tmp_path / "source" / "robots"
    processed_robots = tmp_path / "processed" / "robots"
    source = make_robot_tree(source_robots)
    processed = processed_robots / "sharpa"
    binding = bind_robot_asset_pair(
        source,
        processed,
        source_robots_root=source_robots,
        processed_robots_root=processed_robots,
        robot_type="sharpa",
        hand_type="left",
        copy_source=True,
    )
    errors, _ = robot_asset_manifest_errors(
        {"robot_assets": binding},
        {},
        expected_source_root=source,
        expected_processed_root=processed,
        robot_type="sharpa",
        hand_type="left",
    )
    assert "missing SPIDER output robot_assets binding" in errors


def test_rejects_symlink_and_path_escape(tmp_path: Path) -> None:
    robots = tmp_path / "assets" / "robots"
    root = make_robot_tree(robots)
    outside = tmp_path / "outside.stl"
    outside.write_bytes(b"outside")
    (root / "left.xml").write_text(
        '<mujoco><compiler meshdir="meshes"/><asset>'
        '<mesh name="bad" file="../../../../outside.stl"/></asset></mujoco>',
        encoding="utf-8",
    )
    with pytest.raises(RobotAssetBindingError, match="escapes robot asset root"):
        bind_robot_asset_tree(
            root,
            allowed_robots_root=robots,
            robot_type="sharpa",
            hand_type="left",
        )

    (root / "left.xml").write_text(
        '<mujoco><compiler meshdir="meshes"/><asset>'
        '<mesh name="hand" file="hand.stl"/></asset></mujoco>',
        encoding="utf-8",
    )
    (root / "meshes" / "link.stl").symlink_to(outside)
    with pytest.raises(RobotAssetBindingError, match="contains a symlink"):
        bind_robot_asset_tree(
            root,
            allowed_robots_root=robots,
            robot_type="sharpa",
            hand_type="left",
        )


def test_rejects_unsupported_robot_and_empty_mesh_tree(tmp_path: Path) -> None:
    robots = tmp_path / "assets" / "robots"
    root = make_robot_tree(robots)
    with pytest.raises(RobotAssetBindingError, match="unsupported"):
        bind_robot_asset_tree(
            root,
            allowed_robots_root=robots,
            robot_type="../sharpa",
            hand_type="left",
        )
    shutil.rmtree(root / "meshes")
    (root / "meshes").mkdir()
    with pytest.raises(RobotAssetBindingError, match="mesh tree is empty"):
        bind_robot_asset_tree(
            root,
            allowed_robots_root=robots,
            robot_type="sharpa",
            hand_type="left",
        )


def test_bimanual_include_closure_is_hashed(tmp_path: Path) -> None:
    robots = tmp_path / "assets" / "robots"
    root = make_robot_tree(robots)
    binding = bind_robot_asset_tree(
        root,
        allowed_robots_root=robots,
        robot_type="sharpa",
        hand_type="bimanual",
    )
    dependencies = {item["path"] for item in binding["xml_dependencies"]}
    assert dependencies == {"left.xml", "right.xml", "meshes/hand.stl"}
