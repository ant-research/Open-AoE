from __future__ import annotations

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_SPIDER_ROBOT_TYPES = frozenset({"sharpa", "xhand"})
SUPPORTED_HAND_TYPES = frozenset({"left", "right", "bimanual"})


class RobotAssetBindingError(ValueError):
    """Raised when a SPIDER robot asset tree is incomplete or escapes its route."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_supported(robot_type: str, hand_type: str) -> None:
    if robot_type not in SUPPORTED_SPIDER_ROBOT_TYPES:
        raise RobotAssetBindingError(f"unsupported SPIDER robot_type: {robot_type!r}")
    if hand_type not in SUPPORTED_HAND_TYPES:
        raise RobotAssetBindingError(f"unsupported SPIDER hand_type: {hand_type!r}")


def _require_within(path: Path, root: Path, *, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise RobotAssetBindingError(
            f"{label} is missing or escapes robot asset root: {path}"
        ) from exc
    return resolved


def _xml_asset_dependencies(root: Path, selected_xml: Path) -> list[dict[str, Any]]:
    root = root.resolve(strict=True)
    pending = [selected_xml.resolve(strict=True)]
    visited: set[Path] = set()
    dependencies: dict[str, dict[str, Any]] = {}

    while pending:
        xml_path = pending.pop()
        if xml_path in visited:
            continue
        visited.add(xml_path)
        _require_within(xml_path, root, label="robot XML")
        try:
            tree = ET.parse(xml_path)
        except (ET.ParseError, OSError) as exc:
            raise RobotAssetBindingError(f"cannot parse robot XML: {xml_path}") from exc

        xml_root = tree.getroot()
        compiler = xml_root.find("compiler")
        compiler_dirs: dict[str, Path | None] = {
            "assetdir": None,
            "meshdir": None,
            "texturedir": None,
        }
        if compiler is not None:
            assetdir_value = compiler.get("assetdir")
            assetdir = xml_path.parent / assetdir_value if assetdir_value else None
            for key in compiler_dirs:
                value = compiler.get(key)
                if value:
                    compiler_dirs[key] = xml_path.parent / value
                elif key != "assetdir":
                    compiler_dirs[key] = assetdir

        for element in xml_root.iter():
            file_value = element.get("file")
            if not file_value:
                continue
            if element.tag == "include":
                base = xml_path.parent
                kind = "include"
            elif element.tag == "mesh":
                base = compiler_dirs["meshdir"] or compiler_dirs["assetdir"] or xml_path.parent
                kind = "mesh"
            elif element.tag == "texture":
                base = compiler_dirs["texturedir"] or compiler_dirs["assetdir"] or xml_path.parent
                kind = "texture"
            else:
                base = compiler_dirs["assetdir"] or xml_path.parent
                kind = element.tag
            dependency = _require_within(base / file_value, root, label=f"{kind} dependency")
            relative = dependency.relative_to(root).as_posix()
            dependencies[relative] = {
                "path": relative,
                "kind": kind,
                "size_bytes": dependency.stat().st_size,
                "sha256": file_sha256(dependency),
            }
            if kind == "include":
                pending.append(dependency)

    return [dependencies[key] for key in sorted(dependencies)]


def bind_robot_asset_tree(
    root: Path,
    *,
    allowed_robots_root: Path,
    robot_type: str,
    hand_type: str,
) -> dict[str, Any]:
    """Hash and validate one exact route's immutable SPIDER robot tree."""

    _require_supported(robot_type, hand_type)
    root = Path(root)
    allowed_robots_root = Path(allowed_robots_root)
    if root.is_symlink() or allowed_robots_root.is_symlink():
        raise RobotAssetBindingError("robot asset root or robots parent is a symlink")
    try:
        allowed_resolved = allowed_robots_root.resolve(strict=True)
        root_resolved = root.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise RobotAssetBindingError(f"robot asset root is missing: {root}") from exc
    expected_root = allowed_resolved / robot_type
    if root_resolved != expected_root:
        raise RobotAssetBindingError(
            f"robot asset root is not the exact {robot_type!r} child: {root_resolved}"
        )
    if not root_resolved.is_dir():
        raise RobotAssetBindingError(f"robot asset root is not a directory: {root_resolved}")

    mesh_root = root_resolved / "meshes"
    if mesh_root.is_symlink() or not mesh_root.is_dir():
        raise RobotAssetBindingError(f"robot mesh tree is missing or a symlink: {mesh_root}")
    selected_xml = root_resolved / f"{hand_type}.xml"
    if selected_xml.is_symlink() or not selected_xml.is_file():
        raise RobotAssetBindingError(f"selected robot XML is missing or a symlink: {selected_xml}")

    entries: list[dict[str, Any]] = []
    mesh_file_count = 0
    for path in sorted(
        root_resolved.rglob("*"), key=lambda value: value.relative_to(root_resolved).as_posix()
    ):
        relative = path.relative_to(root_resolved).as_posix()
        if path.is_symlink():
            raise RobotAssetBindingError(f"robot asset tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RobotAssetBindingError(f"robot asset tree contains a non-file entry: {path}")
        if path.is_relative_to(mesh_root):
            mesh_file_count += 1
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not entries:
        raise RobotAssetBindingError(f"robot asset tree is empty: {root_resolved}")
    if mesh_file_count <= 0:
        raise RobotAssetBindingError(f"robot mesh tree is empty: {mesh_root}")

    dependencies = _xml_asset_dependencies(root_resolved, selected_xml)
    aggregate = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "status": "ok",
        "production_eligible": True,
        "root": str(root_resolved),
        "robots_root": str(allowed_resolved),
        "robot_type": robot_type,
        "hand_type": hand_type,
        "file_count": len(entries),
        "mesh_file_count": mesh_file_count,
        "aggregate_sha256": aggregate,
        "entries": entries,
        "selected_xml": {
            "path": selected_xml.relative_to(root_resolved).as_posix(),
            "sha256": file_sha256(selected_xml),
        },
        "xml_dependencies": dependencies,
    }


def robot_asset_trees_match(source: Mapping[str, Any], processed: Mapping[str, Any]) -> bool:
    fields = (
        "robot_type",
        "hand_type",
        "file_count",
        "mesh_file_count",
        "aggregate_sha256",
        "entries",
        "selected_xml",
        "xml_dependencies",
    )
    return all(source.get(field) == processed.get(field) for field in fields)


def bind_robot_asset_pair(
    source_root: Path,
    processed_root: Path,
    *,
    source_robots_root: Path,
    processed_robots_root: Path,
    robot_type: str,
    hand_type: str,
    copy_source: bool = False,
) -> dict[str, Any]:
    source = bind_robot_asset_tree(
        source_root,
        allowed_robots_root=source_robots_root,
        robot_type=robot_type,
        hand_type=hand_type,
    )
    if copy_source:
        if processed_root.exists() or processed_root.is_symlink():
            raise RobotAssetBindingError(
                f"processed robot asset destination already exists: {processed_root}"
            )
        shutil.copytree(source_root, processed_root, symlinks=True)
    processed = bind_robot_asset_tree(
        processed_root,
        allowed_robots_root=processed_robots_root,
        robot_type=robot_type,
        hand_type=hand_type,
    )
    copy_matches_source = robot_asset_trees_match(source, processed)
    if not copy_matches_source:
        raise RobotAssetBindingError("processed SPIDER robot asset tree differs from exact source")
    return {
        "schema_version": 1,
        "status": "ok",
        "production_eligible": True,
        "copy_matches_source": True,
        "source": source,
        "processed": processed,
    }


def robot_asset_manifest_errors(
    input_manifest: object,
    output_manifest: object,
    *,
    expected_source_root: Path,
    expected_processed_root: Path,
    robot_type: str,
    hand_type: str,
) -> tuple[list[str], dict[str, Any] | None]:
    errors: list[str] = []
    input_binding = input_manifest.get("robot_assets") if isinstance(input_manifest, Mapping) else None
    output_binding = output_manifest.get("robot_assets") if isinstance(output_manifest, Mapping) else None
    for label, binding in (("input", input_binding), ("output", output_binding)):
        if not isinstance(binding, Mapping):
            errors.append(f"missing SPIDER {label} robot_assets binding")
            continue
        if binding.get("status") != "ok" or binding.get("production_eligible") is not True:
            errors.append(f"SPIDER {label} robot_assets binding is production-ineligible")
        if binding.get("copy_matches_source") is not True:
            errors.append(f"SPIDER {label} robot_assets source/copy mismatch")

    if isinstance(input_binding, Mapping) and isinstance(output_binding, Mapping):
        for field in ("source", "processed"):
            if input_binding.get(field) != output_binding.get(field):
                errors.append(f"SPIDER robot_assets {field} changed between input and output")

    live: dict[str, Any] | None = None
    try:
        live = bind_robot_asset_pair(
            expected_source_root,
            expected_processed_root,
            source_robots_root=expected_source_root.parent,
            processed_robots_root=expected_processed_root.parent,
            robot_type=robot_type,
            hand_type=hand_type,
        )
    except RobotAssetBindingError as exc:
        errors.append(f"SPIDER live robot_assets invalid: {exc}")
    if live is not None:
        for label, binding in (("input", input_binding), ("output", output_binding)):
            if not isinstance(binding, Mapping):
                continue
            for field in ("source", "processed"):
                if binding.get(field) != live.get(field):
                    errors.append(f"SPIDER {label} robot_assets {field} is stale or cross-route")
    return errors, live
