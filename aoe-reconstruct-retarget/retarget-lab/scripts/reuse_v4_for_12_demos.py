#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import trimesh
except Exception:  # pragma: no cover - optional dependency in some envs
    trimesh = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aoe_retarget_lab.matrix import MatrixCell
from aoe_retarget_lab.admission_evidence import (  # noqa: E402
    authoritative_spider_aligned_tracking,
    raw_to_processed_invariance_errors,
    spider_rest_support_errors,
)
from aoe_retarget_lab.alignment import (  # noqa: E402
    spider_alignment_manifest,
    valid_dai_alignment as _valid_dai_alignment,
    valid_spider_alignment as _valid_spider_alignment,
    valid_spider_tracking,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORIES = ("egoinfinity", "do_as_i_do")
HAND_SOURCES = ("aoe", "estimated")
RETARGETERS = ("egoinfinity", "do_as_i_do", "spider")


def cell_key(trajectory: str, hand_source: str, retargeting: str) -> str:
    return MatrixCell(trajectory, hand_source, retargeting).key


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def first_existing(paths: list[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def path_contains(path: Path | None, needle: str) -> bool:
    if path is None:
        return False
    text = str(path)
    try:
        text = f"{text} {path.resolve()}"
    except OSError:
        pass
    return needle in text


def load_json_optional(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def obj_extent_summary(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {"available": False, "reason": "missing"}
    vertices: list[list[float]] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except Exception as exc:
        return {"available": False, "reason": f"read_error:{exc}"}
    if not vertices:
        return {"available": False, "reason": "no_vertices"}
    arr = np.asarray(vertices, dtype=np.float64)
    extent = np.ptp(arr, axis=0)
    return {
        "available": True,
        "path": str(path),
        "vertices": int(arr.shape[0]),
        "extent": [float(x) for x in extent],
        "diag": float(np.linalg.norm(extent)),
        "center": [float(x) for x in arr.mean(axis=0)],
    }


def remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def materialize(src: Path | None, dst: Path, mode: str) -> Path | None:
    if src is None or not src.exists():
        remove_path(dst)
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        remove_path(dst)
    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    elif mode == "hardlink" and src.is_file():
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src, target_is_directory=src.is_dir())
    return dst


def egoinfinity_clip_dir(source: Path) -> Path:
    return first_existing([
        source / "intermediates" / "trajectory_6dof" / "egoinfinity" / "reconstruction" / "clip",
        source / "intermediates" / "egoinfinity" / "clip",
    ]) or source / "intermediates" / "trajectory_6dof" / "egoinfinity" / "reconstruction" / "clip"


def egoinfinity_mesh_from_manifest(manifest: dict, ego_clip: Path) -> Path | None:
    replacement = manifest.get("object_mesh_replacement") or {}
    ego_mesh = replacement.get("ego_mesh")
    if ego_mesh:
        path = Path(str(ego_mesh))
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if path.exists():
            return path
    ego_object_id = manifest.get("ego_object_id")
    if ego_object_id is not None:
        try:
            candidate = ego_clip / "sam3_meshes" / f"obj_{int(ego_object_id)}.ply"
        except (TypeError, ValueError):
            candidate = None
        if candidate is not None and candidate.exists():
            return candidate
    return None


def egoinfinity_visual_from_manifest(manifest: dict | None) -> Path | None:
    if manifest is None:
        return None
    replacement = manifest.get("object_mesh_replacement") or {}
    for item in replacement.get("targets") or []:
        path = Path(str(item))
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        if path.exists():
            return path
    return None


def egoinfinity_adapter_quality(manifest: dict, mesh_path: Path | None) -> tuple[int, int]:
    if mesh_path is None:
        return (-10, 0)
    score = 0
    replacement = manifest.get("object_mesh_replacement") or {}
    object_name = " ".join(
        str(manifest.get(key) or "")
        for key in ("target_prompt", "ego_prompt", "dai_object_id")
    ).lower()
    if "box" in object_name:
        if replacement.get("boxlike_fallback"):
            score += 4
        stats = (replacement.get("stats") or [{}])[0]
        extents = np.asarray(stats.get("extents") or [], dtype=np.float64)
        if extents.size >= 3 and np.isfinite(extents[:3]).all() and extents[:3].max() > 1e-8:
            ratios = np.sort(extents[:3] / extents[:3].max())
            volumetric = min(1.0, float(ratios[1]) / 0.35) * min(1.0, float(ratios[0]) / 0.18)
            score += int(round(4.0 * volumetric))
            if float(ratios[0]) > 0.75 and float(ratios[1]) > 0.85 and not replacement.get("boxlike_fallback"):
                score -= 5
            if float(ratios[0]) < 0.16:
                score -= 2
    geom = manifest.get("geometry_interaction_hand") or {}
    geom_hand = geom.get("selected_hand")
    selected_hand = manifest.get("selected_hand")
    if selected_hand and geom_hand:
        if selected_hand == geom_hand or selected_hand == "bimanual":
            score += 3
        else:
            score -= 3
    if manifest.get("ego_object_id") is not None:
        score += 2
    if manifest.get("mesh_scale"):
        score += 1
    qc = manifest.get("object_selection_qc") or {}
    if qc.get("available") is True:
        score += 1
    return (score, 1)


def load_latest_egoinfinity_adapter(source: Path, ego_clip: Path | None = None) -> dict | None:
    ego_clip = ego_clip or egoinfinity_clip_dir(source)
    manifests = sorted(
        (
            source
            / "intermediates"
            / "trajectory_6dof"
            / "egoinfinity"
        ).glob("do_as_i_do_raw_dir*/adapter_manifest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    best: tuple[tuple[int, int, float], dict] | None = None
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        manifest["_manifest_path"] = str(manifest_path)
        mesh_path = egoinfinity_mesh_from_manifest(manifest, ego_clip)
        manifest["_resolved_ego_mesh"] = str(mesh_path) if mesh_path is not None else None
        quality = egoinfinity_adapter_quality(manifest, mesh_path)
        rank = (quality[0], quality[1], manifest_path.stat().st_mtime)
        if best is None or rank > best[0]:
            best = (rank, manifest)
    return best[1] if best is not None else None


def selected_egoinfinity_mesh_ply(source: Path, ego_clip: Path) -> Path | None:
    manifest = load_latest_egoinfinity_adapter(source, ego_clip)
    if manifest is not None:
        mesh_path = egoinfinity_mesh_from_manifest(manifest, ego_clip)
        if mesh_path is not None:
            return mesh_path
    return first_existing([
        ego_clip / "sam3_meshes" / "obj_0.ply",
        ego_clip / "sam3_meshes" / "obj_1.ply",
    ])


def selected_egoinfinity_mesh_scale(source: Path, ego_clip: Path | None = None) -> float:
    manifest = load_latest_egoinfinity_adapter(source, ego_clip)
    if manifest is None:
        return 1.0
    try:
        scale = float(manifest.get("mesh_scale") or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_text(cmd: list[str]) -> str | None:
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def git_metadata() -> dict[str, object]:
    diff_hash = None
    try:
        proc = subprocess.run(["git", "diff", "--no-ext-diff", "--binary"], cwd=REPO_ROOT, check=False, capture_output=True)
        if proc.returncode == 0:
            diff_hash = hashlib.sha256(proc.stdout).hexdigest()
    except Exception:
        diff_hash = None
    status = run_text(["git", "status", "--short"])
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "dirty_diff_sha256": diff_hash,
        "status_short": status,
    }


def gpu_mapping_from_env() -> dict[str, str | None]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "MAIN_CUDA",
        "DAI_CUDA_VISIBLE_DEVICES",
        "SAM3_WORKER_CUDA",
        "SAM3D_WORKER_CUDA",
        "SPIDER_CUDA_VISIBLE_DEVICES",
        "SPIDER_DEVICE",
    ]
    return {key: os.environ.get(key) for key in keys}


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or path.is_dir():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths(
    source: Path,
    task: str,
    hand_type: str,
    dai_task: str,
    dai_hand_type: str,
    dai_source_task: str,
    dai_source_hand_type: str,
) -> dict[str, Path | None]:
    dai_key = cell_key("do_as_i_do", "estimated", "do_as_i_do")
    ego_dai_key = cell_key("egoinfinity", "estimated", "do_as_i_do")
    dai_root = (
        source
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / dai_key
        / "retargeting_outputs"
    )
    dai_robot = dai_root / "sharpa" / dai_source_hand_type / dai_source_task / "0"
    dai_mano = dai_root / "mano" / dai_source_hand_type / dai_source_task
    dai_obj = dai_root / "assets" / "objects" / dai_source_task
    ego_clip = egoinfinity_clip_dir(source)
    ego_dai_assets = source / "assets" / "cells" / ego_dai_key / "retargeting" / "do_as_i_do"
    ego_dai_root = (
        source
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / ego_dai_key
        / "retargeting_outputs"
    )
    ego_task_assets = source / "assets" / "trajectory_6dof" / "egoinfinity" / dai_task
    ego_display_assets = source / "assets" / "trajectory_6dof" / "egoinfinity" / task
    ego_selected_ply = selected_egoinfinity_mesh_ply(source, ego_clip)
    ego_selected_scale = selected_egoinfinity_mesh_scale(source, ego_clip)
    ego_adapter = load_latest_egoinfinity_adapter(source, ego_clip)
    ego_adapter_manifest = None
    ego_adapter_visual = egoinfinity_visual_from_manifest(ego_adapter)
    if ego_adapter is not None and ego_adapter.get("_manifest_path"):
        candidate_manifest = Path(str(ego_adapter["_manifest_path"]))
        if candidate_manifest.exists():
            ego_adapter_manifest = candidate_manifest
    ego_meshes = first_existing([
        ego_task_assets / "object_meshes",
        ego_display_assets / "object_meshes",
        ego_clip / "sam3_meshes",
        ego_clip / "sam3d_objects",
        ego_clip / "objects",
    ])
    return {
        "ego_overlay": first_existing([
            ego_clip / "rgb_mesh_overlay_selected.mp4",
            ego_clip / "rgb_mesh_overlay_fix02.mp4",
            ego_clip / "rgb_mesh_overlay.mp4",
            ego_clip / "retarget" / "g1" / "input_viz.mp4",
            ego_clip / "retarget_g1" / "input_viz.mp4",
        ]),
        "ego_depth": first_existing([
            ego_clip / "retarget_samples" / "depth.mp4",
            ego_clip / "retarget" / "g1" / "input_viz.mp4",
            ego_clip / "retarget_g1" / "input_viz.mp4",
        ]),
        "ego_robot": first_existing([ego_clip / "retarget" / "g1" / "robot_sim.mp4", ego_clip / "retarget_g1" / "robot_sim.mp4"]),
        "ego_pipeline": first_existing([
            ego_clip / "pipeline_result_selected.pkl.gz",
            ego_clip / "pipeline_result.pkl.gz",
        ]) or (ego_clip / "pipeline_result_selected.pkl.gz"),
        "ego_object_selection_report": first_existing([
            ego_clip / "object_selection_report.json",
            ego_clip / "object_filter_report.json",
        ]),
        "ego_meshes": ego_meshes,
        "ego_hands": first_existing([ego_clip / "retarget_samples" / "hand_joints.bin", ego_clip / "hand_joints.bin"]),
        "ego_spider_keypoints": first_existing([
            ego_dai_root / "mano" / dai_hand_type / dai_task / "0" / "trajectory_keypoints.npz",
            ego_dai_assets / "trajectories" / "trajectory_keypoints.npz",
        ]),
        "ego_spider_object_visual": None if ego_selected_ply is not None else first_existing([
            ego_adapter_visual,
            source / "assets" / "trajectory_6dof" / "egoinfinity" / dai_task / "object_meshes" / "visual.obj",
            source / "assets" / "trajectory_6dof" / "egoinfinity" / task / "object_meshes" / "visual.obj",
        ]),
        "ego_spider_adapter_visual": ego_adapter_visual,
        "ego_spider_object_ply": ego_selected_ply,
        "ego_spider_object_scale": ego_selected_scale,
        "ego_spider_object_convex": first_existing([
            source / "assets" / "trajectory_6dof" / "egoinfinity" / dai_task / "object_meshes" / "convex",
            source / "assets" / "trajectory_6dof" / "egoinfinity" / task / "object_meshes" / "convex",
        ]),
        "ego_adapter_manifest": ego_adapter_manifest,
        "dai_native_adapter_manifest": dai_adapter_manifest_for_robot(dai_robot),
        "dai_overlay": first_existing([
            source / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction" / "mesh_overlay.mp4",
            source / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction" / "overlay.mp4",
        ]),
        "dai_depth": first_existing([
            source / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction" / "depth.mp4",
            source / "intermediates" / "trajectory_6dof" / "do_as_i_do" / "reconstruction" / "moge_depth.mp4",
        ]),
        "dai_robot": first_existing([
            dai_robot / "visualization_mjwp_act_aligned.mp4",
        ]),
        "dai_robot_front": first_existing([
            dai_robot / "visualization_mjwp_act_aligned.mp4",
        ]),
        "dai_scene": dai_robot / "scene.xml",
        "dai_traj": first_existing([
            dai_robot / "trajectory_mjwp_aligned.npz",
            dai_robot / "trajectory_mjwp_act_aligned.npz",
        ]),
        "dai_keypoints": dai_mano / "0" / "trajectory_keypoints.npz",
        "dai_task_info": dai_mano / "task_info.json",
        "dai_object_visual": dai_obj / "visual.obj",
        "dai_object_convex": dai_obj / "convex",
    }


def load_ply_xyz(src: Path) -> np.ndarray | None:
    try:
        with src.open("rb") as handle:
            header_lines: list[str] = []
            while True:
                line = handle.readline()
                if not line:
                    return None
                text = line.decode("ascii", errors="replace").strip()
                header_lines.append(text)
                if text == "end_header":
                    break
            if not header_lines or header_lines[0] != "ply":
                return None
            fmt = next((line for line in header_lines if line.startswith("format ")), "")
            vertex_count = 0
            properties: list[str] = []
            in_vertex = False
            for line in header_lines:
                if line.startswith("element "):
                    parts = line.split()
                    in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                    if in_vertex:
                        vertex_count = int(parts[2])
                    continue
                if in_vertex and line.startswith("property "):
                    parts = line.split()
                    if len(parts) >= 3:
                        properties.append(parts[-1])
            if vertex_count <= 0 or not {"x", "y", "z"}.issubset(set(properties)):
                return None
            if "binary_little_endian" in fmt:
                dtype = np.dtype([(name, "<f4") for name in properties])
                vertices = np.frombuffer(handle.read(vertex_count * dtype.itemsize), dtype=dtype, count=vertex_count)
                xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
            elif "ascii" in fmt:
                x_idx = properties.index("x")
                y_idx = properties.index("y")
                z_idx = properties.index("z")
                rows = []
                for _ in range(vertex_count):
                    vals = handle.readline().decode("ascii", errors="replace").split()
                    if len(vals) >= len(properties):
                        rows.append([float(vals[x_idx]), float(vals[y_idx]), float(vals[z_idx])])
                xyz = np.asarray(rows, dtype=np.float32)
            else:
                return None
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        return xyz if xyz.shape[0] >= 4 else None
    except Exception:
        return None


def write_bbox_obj_from_xyz(xyz: np.ndarray, src: Path, dst: Path, scale: float = 1.0) -> Path | None:
    if xyz.shape[0] < 8:
        return None
    xyz = np.asarray(xyz, dtype=np.float32) * float(scale)
    lo = np.percentile(xyz, 2.0, axis=0)
    hi = np.percentile(xyz, 98.0, axis=0)
    extent = np.maximum(hi - lo, 1e-4)
    pad = np.maximum(extent * 0.03, 1e-3)
    lo = lo - pad
    hi = hi + pad
    x0, y0, z0 = lo.tolist()
    x1, y1, z1 = hi.tolist()
    verts = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4),
        (5, 8, 7), (5, 7, 6),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 8), (3, 8, 4),
        (4, 8, 5), (4, 5, 1),
    ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as out:
        out.write(f"# bbox converted from {src}\n")
        for x, y, z in verts:
            out.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for a, b, c in faces:
            out.write(f"f {a} {b} {c}\n")
    return dst


def write_visual_obj_from_ply(src: Path | None, dst: Path, scale: float = 1.0) -> Path | None:
    if src is None or not src.exists():
        return None
    scale = float(scale) if np.isfinite(float(scale)) and float(scale) > 0 else 1.0
    try:
        xyz = load_ply_xyz(src)
        if xyz is None:
            return None
        if trimesh is not None:
            try:
                loaded = trimesh.load(src, process=False)
                if hasattr(loaded, "vertices") and len(getattr(loaded, "vertices", [])):
                    vertices = np.asarray(loaded.vertices, dtype=np.float64) * scale
                    faces = np.asarray(getattr(loaded, "faces", []), dtype=np.int64)
                    if faces.size:
                        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
                    else:
                        mesh = trimesh.PointCloud(vertices).convex_hull
                elif hasattr(loaded, "geometry"):
                    chunks = []
                    face_chunks = []
                    offset = 0
                    for geom in loaded.geometry.values():
                        if not hasattr(geom, "vertices") or not len(geom.vertices):
                            continue
                        verts = np.asarray(geom.vertices, dtype=np.float64) * scale
                        chunks.append(verts)
                        geom_faces = np.asarray(getattr(geom, "faces", []), dtype=np.int64)
                        if geom_faces.size:
                            face_chunks.append(geom_faces + offset)
                        offset += len(verts)
                    if chunks:
                        vertices = np.concatenate(chunks, axis=0)
                        faces = np.concatenate(face_chunks, axis=0) if face_chunks else np.zeros((0, 3), dtype=np.int64)
                        mesh = (
                            trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
                            if faces.size
                            else trimesh.PointCloud(vertices).convex_hull
                        )
                    else:
                        mesh = trimesh.PointCloud(xyz).convex_hull
                else:
                    mesh = trimesh.PointCloud(xyz).convex_hull
                dst.parent.mkdir(parents=True, exist_ok=True)
                mesh.export(dst)
                return dst
            except Exception as exc:
                print(f"[warn] trimesh EgoInfinity PLY->OBJ failed, falling back to bbox: {src}: {exc}")
        return write_bbox_obj_from_xyz(xyz, src, dst, scale)
    except Exception as exc:
        print(f"[warn] could not convert EgoInfinity PLY to OBJ for SPIDER: {src}: {exc}")
        return None


def write_visual_mesh_obj(src: Path | None, dst: Path, scale: float = 1.0) -> Path | None:
    if src is None or not src.exists():
        return None
    if src.suffix.lower() == ".ply":
        return write_visual_obj_from_ply(src, dst, scale)
    if trimesh is None:
        if src.suffix.lower() == ".obj" and float(scale) != 1.0:
            dst.parent.mkdir(parents=True, exist_ok=True)
            lines: list[str] = []
            with src.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("v "):
                        parts = line.rstrip("\n").split()
                        xyz = [float(parts[i]) * float(scale) for i in range(1, 4)]
                        suffix = " " + " ".join(parts[4:]) if len(parts) > 4 else ""
                        lines.append(f"v {xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g}{suffix}\n")
                    else:
                        lines.append(line)
            dst.write_text("".join(lines), encoding="utf-8")
            return dst
        return materialize(src, dst, "copy")
    try:
        loaded = trimesh.load(src, process=False)
        if hasattr(loaded, "geometry"):
            loaded = trimesh.util.concatenate([geom for geom in loaded.geometry.values() if hasattr(geom, "vertices")])
        if not hasattr(loaded, "vertices") or len(loaded.vertices) == 0:
            return materialize(src, dst, "copy")
        mesh = loaded.copy()
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(scale)
        dst.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(dst)
        return dst
    except Exception as exc:
        print(f"[warn] could not convert visual mesh to OBJ for SPIDER: {src}: {exc}", file=sys.stderr)
        return materialize(src, dst, "copy")


def write_dai_hand_ego_object_keypoints(dai_src: Path | None, ego_src: Path | None, dst: Path) -> Path | None:
    if dai_src is None or ego_src is None or not dai_src.exists() or not ego_src.exists():
        return None
    try:
        dai = np.load(dai_src, allow_pickle=True)
        ego = np.load(ego_src, allow_pickle=True)
        arrays = {key: dai[key] for key in dai.files}
        replaced: list[str] = []
        for side in ("left", "right"):
            object_key = f"qpos_obj_{side}"
            if object_key in arrays and object_key in ego and np.asarray(arrays[object_key]).shape == np.asarray(ego[object_key]).shape:
                arrays[object_key] = np.asarray(ego[object_key])
                replaced.append(object_key)
            contact_key = f"contact_{side}"
            if contact_key in arrays:
                arrays[contact_key] = np.zeros_like(np.asarray(arrays[contact_key]))
            contact_pos_key = f"contact_pos_{side}"
            if contact_pos_key in arrays:
                arrays[contact_pos_key] = np.zeros_like(np.asarray(arrays[contact_pos_key]))
        if not replaced:
            return None
        if dst.exists() or dst.is_symlink():
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **arrays)
        return dst
    except Exception as exc:
        print(f"[warn] failed to fuse DAI hand with Ego object keypoints: {exc}", file=sys.stderr)
        return None


def summarize_keypoint_quality(path: Path | None) -> dict[str, object]:
    summary: dict[str, object] = {"available": False}
    if path is None or not path.exists():
        summary["reason"] = "missing"
        return summary
    try:
        with np.load(path, allow_pickle=False) as data:
            sides: dict[str, dict[str, object]] = {}
            best_side = None
            best_min = float("inf")
            for side in ("left", "right"):
                finger_key = f"qpos_finger_{side}"
                object_key = f"qpos_obj_{side}"
                if finger_key not in data or object_key not in data:
                    continue
                fingers = np.asarray(data[finger_key], dtype=np.float64)
                obj = np.asarray(data[object_key], dtype=np.float64)
                if fingers.size == 0 or obj.size == 0 or fingers.shape[0] == 0 or obj.shape[0] == 0:
                    continue
                frames = min(fingers.shape[0], obj.shape[0])
                tips = fingers[:frames, :, :3].reshape(frames, -1, 3)
                centers = obj[:frames, :3]
                if not np.isfinite(tips).any() or not np.isfinite(centers).any():
                    continue
                dist = np.linalg.norm(tips - centers[:, None, :], axis=-1)
                if float(np.nanmax(np.abs(tips))) < 1e-8 and float(np.nanmax(np.abs(centers))) < 1e-8:
                    continue
                contact_key = f"contact_{side}"
                contact_active = 0
                contact_mean = 0.0
                if contact_key in data:
                    contact = np.asarray(data[contact_key], dtype=np.float64)
                    contact_active = int(np.count_nonzero(contact > 0.5))
                    contact_mean = float(np.nanmean(contact)) if contact.size else 0.0
                wrist_key = f"qpos_wrist_{side}"
                wrist_motion = None
                if wrist_key in data:
                    wrist = np.asarray(data[wrist_key], dtype=np.float64)
                    if wrist.size:
                        wrist_motion = float(np.nanmax(np.linalg.norm(wrist[:, :3] - wrist[:1, :3], axis=-1)))
                min_dist = float(np.nanmin(dist))
                sides[side] = {
                    "frames": int(frames),
                    "finger_object_distance_median": float(np.nanmedian(dist)),
                    "finger_object_distance_min": min_dist,
                    "finger_object_distance_p10": float(np.nanpercentile(dist, 10)),
                    "contact_active": contact_active,
                    "contact_mean": contact_mean,
                    "wrist_motion": wrist_motion,
                }
                if min_dist < best_min:
                    best_min = min_dist
                    best_side = side
            summary.update({"available": bool(sides), "sides": sides, "best_side": best_side})
            if not sides:
                summary["reason"] = "no_hand_object_keys"
            return summary
    except Exception as exc:
        return {"available": False, "reason": f"load_failed: {exc}"}


def quality_for_hand(summary: dict[str, object], hand_type: str) -> dict[str, object]:
    sides = summary.get("sides")
    if not isinstance(sides, dict):
        return {}
    if hand_type in sides and isinstance(sides[hand_type], dict):
        return sides[hand_type]
    best_side = summary.get("best_side")
    if isinstance(best_side, str) and best_side in sides and isinstance(sides[best_side], dict):
        return sides[best_side]
    return {}


def spider_single_object_hand_selection(
    requested_hand_type: str,
    source_hand_type: str,
    quality: dict[str, object],
) -> dict[str, object]:
    if requested_hand_type in {"left", "right", "bimanual"}:
        return {
            "requested_hand_type": requested_hand_type,
            "resolved_hand_type": requested_hand_type,
            "reason": "explicit_spider_hand_type",
            "quality": quality,
        }
    if source_hand_type in {"left", "right"}:
        return {
            "requested_hand_type": requested_hand_type,
            "resolved_hand_type": source_hand_type,
            "reason": "explicit_source_task_side",
            "quality": quality,
        }
    sides = quality.get("sides") if isinstance(quality, dict) else None
    if not isinstance(sides, dict) or not sides:
        return {
            "requested_hand_type": requested_hand_type,
            "resolved_hand_type": "auto",
            "reason": "insufficient_keypoint_quality",
            "quality": quality,
        }

    def rank(side: str) -> tuple[float, float, float, float, float]:
        row = sides.get(side) if isinstance(sides.get(side), dict) else {}
        contact_active = float(row.get("contact_active", 0) or 0)
        contact_mean = float(row.get("contact_mean", 0.0) or 0.0)
        median = float(row.get("finger_object_distance_median", float("inf")))
        minimum = float(row.get("finger_object_distance_min", float("inf")))
        return (
            1.0 if contact_active > 0 else 0.0,
            contact_active,
            contact_mean,
            -median,
            -minimum,
        )

    available = [side for side in ("left", "right") if side in sides]
    selected = max(available, key=rank)
    return {
        "requested_hand_type": requested_hand_type,
        "resolved_hand_type": selected,
        "reason": "single_object_best_interaction_hand",
        "quality": quality,
        "ranks": {side: list(rank(side)) for side in available},
    }


def dai_interaction_hand_types(hand_type: str, *, auto_select: bool) -> list[str]:
    hand_types = [hand_type]
    if auto_select:
        for candidate in ("left", "right", "bimanual"):
            if candidate and candidate not in hand_types:
                hand_types.append(candidate)
    return hand_types


def dai_quality_score(
    quality: dict[str, object],
    hand_type: str,
    requested_hand_type: str,
    *,
    valid: bool,
    trajectory: str,
) -> tuple[float, float, float, float, float, float]:
    hand_quality = quality_for_hand(quality, hand_type)
    median = float(hand_quality.get("finger_object_distance_median", float("inf")))
    min_dist = float(hand_quality.get("finger_object_distance_min", float("inf")))
    contact_active = int(hand_quality.get("contact_active", 0))
    return (
        1.0 if valid else 0.0,
        1.0 if contact_active > 0 else 0.0,
        float(contact_active),
        -median - 0.1 * min_dist,
        1.0 if trajectory == "do_as_i_do" else 0.0,
        1.0 if hand_type == requested_hand_type else 0.0,
    )


def is_keypoint_quality_valid(
    summary: dict[str, object],
    hand_type: str,
    *,
    min_distance_threshold: float,
    median_distance_threshold: float,
    no_contact_min_distance_threshold: float,
    no_contact_median_distance_threshold: float,
    require_contact: bool,
) -> bool:
    if not summary.get("available"):
        return False
    quality = quality_for_hand(summary, hand_type)
    if not quality:
        return False
    min_dist = float(quality.get("finger_object_distance_min", float("inf")))
    med_dist = float(quality.get("finger_object_distance_median", float("inf")))
    contact_active = int(quality.get("contact_active", 0))
    if min_dist > min_distance_threshold and med_dist > median_distance_threshold:
        return False
    if require_contact and contact_active <= 0:
        if min_dist > no_contact_min_distance_threshold:
            return False
        if med_dist > no_contact_median_distance_threshold:
            return False
    return True


def valid_dai_alignment(robot_root: Path) -> bool:
    return _valid_dai_alignment(robot_root, REPO_ROOT)


def dai_raw_to_processed_hoi_for_robot(robot: Path | None) -> Path | None:
    output_root = dai_retarget_output_root(robot)
    if output_root is None or robot is None:
        return None
    robot_dir = robot.parent
    return (
        output_root
        / "mano"
        / robot_dir.parent.parent.name
        / robot_dir.parent.name
        / "0"
        / "raw_to_processed_hoi_invariance.json"
    )


def find_dai_robot(
    source: Path,
    trajectory: str,
    hand_source: str,
    task: str,
    hand_type: str,
) -> Path | None:
    key = cell_key(trajectory, hand_source, "do_as_i_do")
    root = (
        source
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / key
        / "retargeting_outputs"
        / "sharpa"
        / hand_type
        / task
        / "0"
    )
    if not valid_dai_alignment(root):
        return None
    return first_existing([
        root / "visualization_mjwp_act_aligned.mp4",
    ])


def find_any_dai_robot(
    source: Path,
    trajectory: str,
    hand_source: str,
    hand_type: str,
) -> tuple[Path | None, str | None]:
    key = cell_key(trajectory, hand_source, "do_as_i_do")
    root = (
        source
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / key
        / "retargeting_outputs"
        / "sharpa"
        / hand_type
    )
    for name in ["visualization_mjwp_act_aligned.mp4"]:
        for path in sorted(root.glob(f"*/0/{name}")):
            robot_root = path.parent
            if valid_dai_alignment(robot_root):
                return path, path.parents[1].name
    return None, None


def find_dai_robot_candidates(
    source: Path,
    trajectory: str,
    hand_source: str,
    task: str,
    hand_type: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    hand_types = dai_interaction_hand_types(
        hand_type,
        auto_select=bool(args.dai_auto_select_interaction_hand),
    )
    hand_sources = [hand_source]
    if hand_source != "estimated" and args.allow_dai_hand_fallback:
        hand_sources.append("estimated")

    for search_hand_source in hand_sources:
        for search_hand_type in hand_types:
            bundles = find_dai_asset_bundles(
                source,
                trajectory,
                search_hand_source,
                task,
                search_hand_type,
                allow_glob=True,
            )
            if not bundles:
                robot, matched_task = find_any_dai_robot(source, trajectory, search_hand_source, search_hand_type)
                if robot is not None and matched_task is not None:
                    candidates.append({
                        "robot": robot,
                        "task": matched_task,
                        "trajectory": trajectory,
                        "hand_source": search_hand_source,
                        "hand_type": search_hand_type,
                        "quality": {"available": False, "reason": "missing_keypoints"},
                        "quality_valid": False,
                        "score": (-1.0, 0.0, 0.0, 0.0, 0.0, float("-inf")),
                    })
                continue
            for bundle in bundles:
                robot = find_dai_robot(
                    source,
                    trajectory,
                    search_hand_source,
                    str(bundle["task"]),
                    search_hand_type,
                )
                if robot is None:
                    continue
                quality = summarize_keypoint_quality(Path(str(bundle["keypoints"])))
                keypoint_valid = is_keypoint_quality_valid(
                    quality,
                    search_hand_type,
                    min_distance_threshold=args.dai_asset_min_distance_threshold,
                    median_distance_threshold=args.dai_asset_median_distance_threshold,
                    no_contact_min_distance_threshold=args.dai_asset_no_contact_min_distance_threshold,
                    no_contact_median_distance_threshold=args.dai_asset_no_contact_median_distance_threshold,
                    require_contact=args.dai_asset_require_contact,
                )
                retarget_quality = load_json_optional(Path(str(bundle["retarget_quality"])))
                retarget_valid = bool(retarget_quality and retarget_quality.get("status") == "ok")
                raw_to_processed_path = dai_raw_to_processed_hoi_for_robot(robot)
                raw_to_processed = load_json_optional(raw_to_processed_path)
                raw_to_processed_errors = raw_to_processed_invariance_errors(
                    raw_to_processed,
                    label="exact-route raw_to_processed_hoi_invariance",
                    trajectory_6dof=trajectory,
                    hand_source=search_hand_source,
                    hand_type=search_hand_type,
                    adapter_manifest=Path(str(bundle["adapter_manifest"])),
                    processed_keypoints=Path(str(bundle["keypoints"])),
                )
                raw_to_processed_valid = not raw_to_processed_errors
                valid = keypoint_valid and retarget_valid and raw_to_processed_valid
                candidates.append({
                    "robot": robot,
                    "task": str(bundle["task"]),
                    "trajectory": trajectory,
                    "hand_source": search_hand_source,
                    "hand_type": search_hand_type,
                    "quality": quality,
                    "quality_valid": valid,
                    "keypoint_quality_valid": keypoint_valid,
                    "retarget_quality": retarget_quality,
                    "retarget_quality_valid": retarget_valid,
                    "raw_to_processed_hoi_invariance": raw_to_processed,
                    "raw_to_processed_hoi_invariance_path": (
                        str(raw_to_processed_path) if raw_to_processed_path else None
                    ),
                    "raw_to_processed_hoi_invariance_valid": raw_to_processed_valid,
                    "raw_to_processed_hoi_invariance_errors": raw_to_processed_errors,
                    "score": dai_quality_score(
                        quality,
                        search_hand_type,
                        hand_type,
                        valid=valid,
                        trajectory=trajectory,
                    ),
                })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def find_dai_robot_in_overrides(
    overrides: list[Path],
    trajectory: str,
    hand_source: str,
    task: str,
    hand_type: str,
    args: argparse.Namespace,
) -> tuple[Path | None, str | None]:
    all_candidates: list[dict[str, object]] = []
    search_specs: list[tuple[str, str, str]] = [(trajectory, task, hand_type)]
    if args.allow_cross_trajectory_dai_robot_fallback:
        alt_trajectory = "egoinfinity" if trajectory == "do_as_i_do" else "do_as_i_do"
        alt_task = args.dai_task if alt_trajectory == "egoinfinity" else args.dai_source_task
        alt_hand_type = args.dai_hand_type if alt_trajectory == "egoinfinity" else args.dai_source_hand_type
        spec = (alt_trajectory, alt_task, alt_hand_type)
        if spec not in search_specs:
            search_specs.append(spec)
    for override in overrides:
        for search_trajectory, search_task, search_hand_type in search_specs:
            candidates = find_dai_robot_candidates(
                override,
                search_trajectory,
                hand_source,
                search_task,
                search_hand_type,
                args,
            )
            for item in candidates:
                item = dict(item)
                item["override_name"] = override.name
                item["requested_trajectory"] = trajectory
                item["is_requested_trajectory"] = search_trajectory == trajectory
                all_candidates.append(item)
    if all_candidates:
        requested_exact = [
            item
            for item in all_candidates
            if item.get("is_requested_trajectory") and item.get("hand_source") == hand_source
        ]
        fallback_candidates = [item for item in all_candidates if item not in requested_exact]
        candidate_pool = requested_exact or fallback_candidates
        valid_candidates = [item for item in candidate_pool if item.get("quality_valid")]
        if not valid_candidates and not getattr(
            args, "include_numerically_invalid_aligned_robot", False
        ):
            return None, None
        ranked_candidates = valid_candidates or candidate_pool
        ranked_candidates.sort(key=lambda item: item["score"], reverse=True)
        item = ranked_candidates[0]
        detail = "exact" if item["hand_source"] == hand_source else "estimated-hand fallback"
        trajectory_detail = (
            f"{item['trajectory']} trajectory"
            if item["trajectory"] == item.get("requested_trajectory")
            else f"{item['trajectory']} trajectory fallback for {item.get('requested_trajectory')}"
        )
        hand_detail = (
            f"{item['hand_type']} interaction-hand"
            if item["hand_type"] != hand_type
            else f"{hand_type} hand"
        )
        valid_detail = "quality-valid" if item["quality_valid"] else "quality-unverified"
        return (
            Path(str(item["robot"])),
            (
                f"{detail} {trajectory_detail} {hand_detail} task {item['task']} Do-as-I-Do/Sharpa cell "
                f"from override run {item['override_name']} ({valid_detail})"
            ),
        )
    return None, None


def find_quality_dai_robot_in_source(
    source: Path,
    trajectory: str,
    hand_source: str,
    task: str,
    hand_type: str,
    args: argparse.Namespace,
) -> tuple[Path | None, str | None]:
    candidates = find_dai_robot_candidates(
        source,
        trajectory,
        hand_source,
        task,
        hand_type,
        args,
    )
    if not candidates:
        return None, None
    exact_candidates = [item for item in candidates if item.get("hand_source") == hand_source]
    fallback_candidates = [item for item in candidates if item.get("hand_source") != hand_source]
    candidate_pool = exact_candidates or fallback_candidates
    valid_candidates = [item for item in candidate_pool if item.get("quality_valid")]
    if not valid_candidates and not getattr(
        args, "include_numerically_invalid_aligned_robot", False
    ):
        return None, None
    ranked_candidates = valid_candidates or candidate_pool
    ranked_candidates.sort(key=lambda item: item["score"], reverse=True)
    item = ranked_candidates[0]
    hand_detail = (
        f"{item['hand_type']} interaction-hand"
        if item["hand_type"] != hand_type
        else f"{hand_type} hand"
    )
    source_detail = "exact" if item["hand_source"] == hand_source else f"{item['hand_source']} hand-source fallback"
    valid_detail = "quality-valid" if item["quality_valid"] else "quality-unverified"
    return (
        Path(str(item["robot"])),
        (
            f"{source_detail} source {trajectory} trajectory {hand_detail} task {item['task']} "
            f"Do-as-I-Do/Sharpa cell ({valid_detail})"
        ),
    )


def find_exact_spider_input_bundle(
    source: Path,
    trajectory: str,
    hand_source: str,
    task: str,
    hand_type: str,
    args: argparse.Namespace,
) -> dict[str, Path | str] | None:
    candidates: list[dict[str, object]] = []
    for search_hand_type in dai_interaction_hand_types(
        hand_type,
        auto_select=bool(args.dai_auto_select_interaction_hand),
    ):
        for bundle in find_dai_asset_bundles(
            source,
            trajectory,
            hand_source,
            task,
            search_hand_type,
            allow_glob=True,
        ):
            if trajectory == "egoinfinity":
                adapter = load_json_optional(Path(str(bundle["adapter_manifest"])))
                input_qc = (adapter or {}).get("retarget_input_qc") or {}
                if input_qc.get("status") != "ok":
                    continue
            quality = summarize_keypoint_quality(Path(str(bundle["keypoints"])))
            valid = is_keypoint_quality_valid(
                quality,
                search_hand_type,
                min_distance_threshold=args.dai_asset_min_distance_threshold,
                median_distance_threshold=args.dai_asset_median_distance_threshold,
                no_contact_min_distance_threshold=args.dai_asset_no_contact_min_distance_threshold,
                no_contact_median_distance_threshold=args.dai_asset_no_contact_median_distance_threshold,
                require_contact=args.dai_asset_require_contact,
            )
            candidates.append({
                "bundle": bundle,
                "score": dai_quality_score(
                    quality,
                    search_hand_type,
                    hand_type,
                    valid=valid,
                    trajectory=trajectory,
                ),
            })
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[0]["bundle"]  # type: ignore[return-value]


def find_dai_asset_bundles(
    source: Path,
    trajectory: str,
    hand_source: str,
    task: str | None,
    hand_type: str,
    allow_glob: bool = False,
) -> list[dict[str, Path | str]]:
    key = cell_key(trajectory, hand_source, "do_as_i_do")
    root = (
        source
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / key
        / "retargeting_outputs"
    )
    task_names = [task] if task else []
    if allow_glob:
        mano_root = root / "mano" / hand_type
        task_names += sorted(p.parents[1].name for p in mano_root.glob("*/0/trajectory_keypoints.npz"))

    seen: set[str] = set()
    bundles: list[dict[str, Path | str]] = []
    for task_name in task_names:
        if not task_name or task_name in seen:
            continue
        seen.add(task_name)
        keypoints = root / "mano" / hand_type / task_name / "0" / "trajectory_keypoints.npz"
        if not keypoints.exists():
            continue
        bundles.append({
            "task": task_name,
            "trajectory": trajectory,
            "hand_source": hand_source,
            "hand_type": hand_type,
            "keypoints": keypoints,
            "task_info": root / "mano" / hand_type / task_name / "task_info.json",
            "object_visual": root / "assets" / "objects" / task_name / "visual.obj",
            "object_convex": root / "assets" / "objects" / task_name / "convex",
            "robot_root": root / "sharpa" / hand_type / task_name / "0",
            "retarget_quality": root / "sharpa" / hand_type / task_name / "0" / "mjwp_object_tracking_quality.json",
            "adapter_manifest": root.parent / "raw_dir" / "adapter_manifest.json",
        })
    return bundles


def find_dai_asset_bundle(
    source: Path,
    trajectory: str,
    hand_source: str,
    task: str | None,
    hand_type: str,
    allow_glob: bool = False,
) -> dict[str, Path | str] | None:
    bundles = find_dai_asset_bundles(source, trajectory, hand_source, task, hand_type, allow_glob)
    return bundles[0] if bundles else None


def find_dai_keypoints(
    source: Path,
    trajectory: str,
    hand_source: str,
    task: str,
    hand_type: str,
) -> Path | None:
    key = cell_key(trajectory, hand_source, "do_as_i_do")
    return first_existing([
        source
        / "intermediates"
        / "retargeting"
        / "do_as_i_do"
        / key
        / "retargeting_outputs"
        / "mano"
        / hand_type
        / task
        / "0"
        / "trajectory_keypoints.npz",
        source / "assets" / "cells" / key / "retargeting" / "do_as_i_do" / "trajectories" / "trajectory_keypoints.npz",
    ])


def apply_dai_override_trajectory_assets(
    srcs: dict[str, Path | None],
    overrides: list[Path],
    args: argparse.Namespace,
) -> dict[str, object]:
    report: dict[str, object] = {}
    candidates: list[dict[str, object]] = []
    for override in overrides:
        for trajectory, task, requested_hand_type in (
            ("do_as_i_do", args.dai_source_task, args.dai_source_hand_type),
            ("egoinfinity", args.dai_task, args.dai_hand_type),
        ):
            for hand_type in dai_interaction_hand_types(
                requested_hand_type,
                auto_select=bool(args.dai_auto_select_interaction_hand),
            ):
                for bundle in find_dai_asset_bundles(
                    override,
                    trajectory,
                    "estimated",
                    task,
                    hand_type,
                    allow_glob=True,
                ):
                    quality = summarize_keypoint_quality(Path(str(bundle["keypoints"])))
                    keypoint_valid = is_keypoint_quality_valid(
                        quality,
                        hand_type,
                        min_distance_threshold=args.dai_asset_min_distance_threshold,
                        median_distance_threshold=args.dai_asset_median_distance_threshold,
                        no_contact_min_distance_threshold=args.dai_asset_no_contact_min_distance_threshold,
                        no_contact_median_distance_threshold=args.dai_asset_no_contact_median_distance_threshold,
                        require_contact=args.dai_asset_require_contact,
                    )
                    retarget_quality = load_json_optional(Path(str(bundle["retarget_quality"])))
                    retarget_valid = bool(retarget_quality and retarget_quality.get("status") == "ok")
                    valid = keypoint_valid and retarget_valid
                    candidates.append({
                        "override": override,
                        "bundle": bundle,
                        "quality": quality,
                        "quality_valid": valid,
                        "keypoint_quality_valid": keypoint_valid,
                        "retarget_quality": retarget_quality,
                        "retarget_quality_valid": retarget_valid,
                        "score": dai_quality_score(
                            quality,
                            hand_type,
                            requested_hand_type,
                            valid=valid,
                            trajectory=trajectory,
                        ),
                    })
    if candidates:
        candidates.sort(key=lambda item: item["score"], reverse=True)
        report["dai_asset_candidates"] = [
            {
                "source": Path(str(item["override"])).name,
                "trajectory": str(item["bundle"]["trajectory"]),
                "hand_source": str(item["bundle"]["hand_source"]),
                "hand_type": str(item["bundle"]["hand_type"]),
                "task": str(item["bundle"]["task"]),
                "keypoints": str(item["bundle"]["keypoints"]),
                "quality_valid": bool(item["quality_valid"]),
                "keypoint_quality_valid": bool(item.get("keypoint_quality_valid")),
                "retarget_quality_valid": bool(item.get("retarget_quality_valid")),
                "retarget_quality": item.get("retarget_quality"),
                "quality": item["quality"],
            }
            for item in candidates
        ]
    if args.dai_override_ego_spider_keypoints:
        ego_candidates = [
            item
            for item in candidates
            if str(item["bundle"]["trajectory"]) == "egoinfinity" and bool(item["quality_valid"])
        ]
        if ego_candidates:
            item = ego_candidates[0]
            bundle = item["bundle"]
            srcs["ego_spider_keypoints"] = bundle["keypoints"]
            report["ego_spider_keypoints"] = str(bundle["keypoints"])
            report["ego_spider_keypoints_source"] = Path(str(item["override"])).name
            report["ego_spider_keypoints_hand_type"] = str(bundle["hand_type"])
            report["ego_spider_keypoints_task"] = str(bundle["task"])
            report["ego_spider_keypoints_quality"] = item["quality"]
        else:
            report["ego_spider_keypoints"] = None
            report["ego_spider_keypoints_reason"] = "no_quality_valid_egoinfinity_override_bundle"
    for item in candidates:
        bundle = item["bundle"]
        if str(bundle["trajectory"]) != "do_as_i_do":
            continue
        srcs["dai_keypoints"] = bundle["keypoints"]
        srcs["dai_task_info"] = bundle["task_info"]
        srcs["dai_object_visual"] = bundle["object_visual"]
        srcs["dai_object_convex"] = bundle["object_convex"]
        report["dai_native_source"] = Path(str(item["override"])).name
        report["dai_native_trajectory"] = str(bundle["trajectory"])
        report["dai_native_hand_source"] = str(bundle["hand_source"])
        report["dai_native_hand_type"] = str(bundle["hand_type"])
        report["dai_native_task"] = str(bundle["task"])
        report["dai_native_keypoints"] = str(bundle["keypoints"])
        report["dai_native_quality_valid"] = bool(item["quality_valid"])
        report["dai_native_quality"] = item["quality"]
        report["dai_native_retarget_quality_valid"] = bool(item.get("retarget_quality_valid"))
        report["dai_native_retarget_quality"] = item.get("retarget_quality")
        if not item["quality_valid"]:
            print(
                f"[warn] selected DAI asset has weak hand-object quality: {bundle['keypoints']}",
                file=sys.stderr,
            )
        break
    else:
        if candidates:
            report["dai_native_override_reason"] = "no_do_as_i_do_override_bundle"
    return report


def validate_trajectory_assets(
    dst: Path,
    task: str,
    *,
    dai_task: str,
    dai_hand_type: str,
    dai_source_task: str,
    dai_source_hand_type: str,
    spider_fuse_ego_object_for_dai: bool,
    dai_override_asset_report: dict[str, object],
    dai_asset_min_distance_threshold: float,
    dai_asset_median_distance_threshold: float,
    dai_asset_no_contact_min_distance_threshold: float,
    dai_asset_no_contact_median_distance_threshold: float,
    dai_asset_require_contact: bool,
) -> dict[str, object]:
    source_link = dst / "reuse" / "source_run"
    routes: dict[str, object] = {}
    for trajectory, route_task, route_hand_type in (
        ("egoinfinity", dai_task, dai_hand_type),
        ("do_as_i_do", dai_source_task, dai_source_hand_type),
    ):
        for hand_source in ("aoe", "estimated"):
            route_key = f"{trajectory}__{hand_source}"
            bundle = find_dai_asset_bundle(
                source_link,
                trajectory,
                hand_source,
                route_task,
                route_hand_type,
                allow_glob=True,
            )
            if bundle is None:
                routes[route_key] = {
                    "status": "missing",
                    "trajectory_6dof": trajectory,
                    "hand_source": hand_source,
                    "expected_cell_key": cell_key(trajectory, hand_source, "do_as_i_do"),
                    "errors": [f"missing exact-cell keypoints: {route_key}"],
                }
                continue
            keypoints = Path(str(bundle["keypoints"]))
            quality = summarize_keypoint_quality(keypoints)
            keypoint_valid = is_keypoint_quality_valid(
                quality,
                str(bundle["hand_type"]),
                min_distance_threshold=dai_asset_min_distance_threshold,
                median_distance_threshold=dai_asset_median_distance_threshold,
                no_contact_min_distance_threshold=dai_asset_no_contact_min_distance_threshold,
                no_contact_median_distance_threshold=dai_asset_no_contact_median_distance_threshold,
                require_contact=dai_asset_require_contact,
            )
            retarget_quality_path = Path(str(bundle["retarget_quality"]))
            retarget_quality = load_json_optional(retarget_quality_path)
            retarget_valid = bool(
                isinstance(retarget_quality, dict) and retarget_quality.get("status") == "ok"
            )
            valid = bool(keypoint_valid and retarget_valid)
            route_errors: list[str] = []
            if not keypoint_valid:
                route_errors.append("source keypoint quality failed numerical QC")
            if not retarget_valid:
                route_errors.append("retarget quality is missing or invalid")
            routes[route_key] = {
                "status": "ok" if valid else "invalid",
                "trajectory_6dof": trajectory,
                "hand_source": hand_source,
                "hand_type": str(bundle["hand_type"]),
                "task": str(bundle["task"]),
                "expected_cell_key": cell_key(trajectory, hand_source, "do_as_i_do"),
                "source_keypoints": str(keypoints),
                "source_keypoints_sha256": file_sha256(keypoints),
                "adapter_manifest": str(bundle["adapter_manifest"]),
                "keypoint_quality": quality,
                "keypoint_quality_valid": keypoint_valid,
                "retarget_quality": retarget_quality,
                "retarget_quality_path": str(retarget_quality_path),
                "retarget_quality_valid": retarget_valid,
                "errors": route_errors,
            }

    def rows_for(trajectory: str) -> dict[str, dict[str, object]]:
        prefix = f"{trajectory}__"
        return {
            name: row
            for name, row in routes.items()
            if name.startswith(prefix) and isinstance(row, dict)
        }

    def trajectory_summary(trajectory: str) -> dict[str, object]:
        trajectory_rows = rows_for(trajectory)
        ok_routes = sorted(
            name for name, row in trajectory_rows.items() if row.get("status") == "ok"
        )
        invalid_routes = sorted(
            name for name, row in trajectory_rows.items() if row.get("status") != "ok"
        )
        return {
            # A trajectory remains usable when at least one exact hand-source route
            # passes. Failed siblings remain explicit below and must not be used as
            # fallback inputs for the successful route.
            "quality_valid": bool(ok_routes),
            "all_routes_quality_valid": bool(trajectory_rows) and not invalid_routes,
            "quality_valid_by_hand_source": {
                str(row.get("hand_source")): row.get("status") == "ok"
                for row in trajectory_rows.values()
            },
            "route_statuses": {
                name: str(row.get("status")) for name, row in trajectory_rows.items()
            },
            "ok_routes": ok_routes,
            "invalid_routes": invalid_routes,
        }

    ego_summary = trajectory_summary("egoinfinity")
    dai_summary = trajectory_summary("do_as_i_do")
    report: dict[str, object] = {
        "comparison_scope": "exact_route_and_hand_source_only",
        "cross_route_comparison_performed": False,
        "unified_trajectory_keypoints_authoritative": False,
        "routes": routes,
        "spider_fuse_ego_object_for_dai": spider_fuse_ego_object_for_dai,
        "egoinfinity_quality_valid": ego_summary["quality_valid"],
        "egoinfinity_all_routes_quality_valid": ego_summary["all_routes_quality_valid"],
        "egoinfinity_quality_valid_by_hand_source": ego_summary["quality_valid_by_hand_source"],
        "do_as_i_do_quality_valid": dai_summary["quality_valid"],
        "do_as_i_do_all_routes_quality_valid": dai_summary["all_routes_quality_valid"],
        "do_as_i_do_quality_valid_by_hand_source": dai_summary["quality_valid_by_hand_source"],
        "invalid_routes": sorted(
            list(ego_summary["invalid_routes"]) + list(dai_summary["invalid_routes"])
        ),
        "object_provenance": {
            "egoinfinity": {
                "object_track_source": "egoinfinity",
                "object_mesh_source": "egoinfinity",
                "retarget_object_source": "egoinfinity",
                "status": "ok" if ego_summary["quality_valid"] else "invalid",
                "recommended_route": True,
                "route_statuses": ego_summary["route_statuses"],
            },
            "do_as_i_do": {
                "object_track_source": "fused_ego_for_spider" if spider_fuse_ego_object_for_dai else "dai_native",
                "object_mesh_source": "egoinfinity" if spider_fuse_ego_object_for_dai else "dai_native",
                "retarget_object_source": "fused_ego_for_spider" if spider_fuse_ego_object_for_dai else "dai_native",
                "status": "ok" if dai_summary["quality_valid"] else "invalid",
                "recommended_route": False,
                "baseline_only": not spider_fuse_ego_object_for_dai,
                "route_statuses": dai_summary["route_statuses"],
            },
        },
        "quality_thresholds": {
            "min_distance": dai_asset_min_distance_threshold,
            "median_distance": dai_asset_median_distance_threshold,
            "no_contact_min_distance": dai_asset_no_contact_min_distance_threshold,
            "no_contact_median_distance": dai_asset_no_contact_median_distance_threshold,
            "require_contact": dai_asset_require_contact,
        },
    }
    if spider_fuse_ego_object_for_dai:
        errors = ["object-only DAI-hand/Ego-object fusion is forbidden in production"]
        report["errors"] = errors
        report["object_provenance"]["do_as_i_do"]["status"] = "invalid"
        report["object_provenance"]["do_as_i_do"]["invalid_reasons"] = errors

    for trajectory, summary in (
        ("egoinfinity", ego_summary),
        ("do_as_i_do", dai_summary),
    ):
        if not summary["quality_valid"]:
            report["object_provenance"][trajectory]["invalid_reasons"] = [
                "no exact trajectory/hand-source route passed numerical QC"
            ]

    return report


def prepare_trajectory_assets(
    dst: Path,
    srcs: dict[str, Path | None],
    task: str,
    mode: str,
    *,
    spider_fuse_ego_object_for_dai: bool = False,
) -> dict:
    report: dict[str, dict[str, str | None]] = {}
    for trajectory in TRAJECTORIES:
        root = ensure_dir(dst / "assets" / "trajectory_6dof" / trajectory / task)
        if trajectory == "egoinfinity":
            provenance = {
                "object_track_source": "egoinfinity",
                "object_mesh_source": "egoinfinity",
                "retarget_object_source": "egoinfinity",
                "display_keypoints_hand_source": "estimated",
                "keypoints_authoritative_for_qc": False,
                "object_selection_source": "bbox_target_point_best",
                "recommended_route": True,
            }
            ego_visual = write_visual_mesh_obj(
                srcs.get("ego_spider_adapter_visual"),
                root / "object_meshes" / "visual.obj",
                float(srcs.get("ego_spider_object_scale") or 1.0),
            ) or write_visual_obj_from_ply(
                srcs["ego_spider_object_ply"],
                root / "object_meshes" / "visual.obj",
                float(srcs.get("ego_spider_object_scale") or 1.0),
            )
            entries = {
                "overlay": materialize(srcs["ego_overlay"], root / "overlay.mp4", mode),
                "depth": materialize(srcs["ego_depth"], root / "depth.mp4", mode),
                "pipeline_result": materialize(srcs["ego_pipeline"], root / "object_6dof_native.pkl.gz", mode),
                "object_selection_report": materialize(
                    srcs.get("ego_object_selection_report"),
                    root / "object_selection_report.json",
                    "copy",
                ),
                "native_object_meshes": materialize(srcs["ego_meshes"], root / "native_object_meshes", mode),
                "estimated_hands": materialize(srcs["ego_hands"], root / "estimated_hands", mode),
                "source_keypoints": materialize(srcs["ego_spider_keypoints"], root / "source_trajectory_keypoints.npz", mode),
                "object_visual": ego_visual
                or materialize(srcs["ego_spider_object_visual"], root / "object_meshes" / "visual.obj", mode),
                "object_convex": materialize(srcs["ego_spider_object_convex"], root / "object_meshes" / "convex", mode),
                "adapter_manifest": materialize(
                    srcs.get("ego_adapter_manifest"),
                    root / "egoinfinity_adapter_manifest.json",
                    "copy",
                ),
            }
        else:
            fused_keypoints = None
            fused_visual = None
            repaired_visual = None
            repaired_visual_manifest = None
            provenance = {
                "object_track_source": "dai_native",
                "object_mesh_source": "dai_native",
                "retarget_object_source": "dai_native",
                "display_keypoints_hand_source": "estimated",
                "keypoints_authoritative_for_qc": False,
                "recommended_route": False,
                "baseline_only": True,
            }
            native_adapter = load_json_optional(srcs.get("dai_native_adapter_manifest"))
            if native_adapter is not None:
                provenance["retarget_object_source"] = str(
                    native_adapter.get("retarget_object_source") or "dai_native"
                )
                provenance["hoi_contact_alignment"] = native_adapter.get("hoi_contact_alignment")
            if spider_fuse_ego_object_for_dai:
                provenance = {
                    "object_track_source": "fused_ego_for_spider",
                    "object_mesh_source": "egoinfinity",
                    "retarget_object_source": "fused_ego_for_spider",
                    "recommended_route": False,
                    "baseline_only": False,
                    "explicit_fusion": True,
                }
                fused_keypoints = write_dai_hand_ego_object_keypoints(
                    srcs["dai_keypoints"],
                    srcs["ego_spider_keypoints"],
                    root / "source_trajectory_keypoints.npz",
                )
                fused_visual = write_visual_obj_from_ply(
                    srcs["ego_spider_object_ply"],
                    root / "object_meshes" / "visual.obj",
                    float(srcs.get("ego_spider_object_scale") or 1.0),
                )
            entries = {
                "overlay": materialize(srcs["dai_overlay"], root / "overlay.mp4", mode),
                "depth": materialize(srcs["dai_depth"], root / "depth.mp4", mode),
                "source_keypoints": fused_keypoints
                or materialize(srcs["dai_keypoints"], root / "source_trajectory_keypoints.npz", mode),
                "task_info": materialize(srcs["dai_task_info"], root / "task_info.json", mode),
                "object_visual": fused_visual
                or repaired_visual
                or materialize(srcs["dai_object_visual"], root / "object_meshes" / "visual.obj", mode),
                "object_convex": materialize(
                    srcs["ego_spider_object_convex"] if spider_fuse_ego_object_for_dai else None,
                    root / "object_meshes" / "convex",
                    mode,
                )
                or materialize(srcs["dai_object_convex"], root / "object_meshes" / "convex", mode),
                "spider_fused_ego_object": fused_visual,
                "spider_repaired_object_visual": repaired_visual,
                "adapter_manifest": materialize(
                    srcs.get("dai_native_adapter_manifest"),
                    root / "dai_native_adapter_manifest.json",
                    "copy",
                ),
            }
            if repaired_visual_manifest is not None:
                write_json(root / "dai_spider_object_visual_repair.json", repaired_visual_manifest)
                entries["spider_repaired_object_visual_manifest"] = root / "dai_spider_object_visual_repair.json"
        entry_report = {k: rel(v, dst) for k, v in entries.items()}
        entry_report["provenance"] = provenance
        write_json(root / "reuse_source_manifest.json", entry_report)
        report[trajectory] = entry_report
    return report


def assess_dai_native_spider_input(adapter_manifest: dict | None, hand_source: str) -> dict:
    if not isinstance(adapter_manifest, dict):
        return {"status": "legacy_unknown", "errors": []}
    errors: list[str] = []
    object_pose_quality = adapter_manifest.get("object_pose_quality")
    retarget_input_qc = adapter_manifest.get("retarget_input_qc")
    if isinstance(object_pose_quality, dict) and object_pose_quality.get("status") != "ok":
        errors.extend(
            f"source_{error}"
            for error in (object_pose_quality.get("errors") or ["object_pose_invalid"])
        )
    if (
        hand_source == "aoe"
        and isinstance(retarget_input_qc, dict)
        and retarget_input_qc.get("status") != "ok"
    ):
        errors.extend(
            f"aoe_hand_{error}"
            for error in (retarget_input_qc.get("errors") or ["retarget_input_invalid"])
        )
    return {
        "status": "invalid" if errors else "ok",
        "errors": sorted(set(errors)),
        "hand_source": hand_source,
        "object_pose_quality": object_pose_quality,
        "retarget_input_qc": retarget_input_qc,
    }


def dai_native_spider_input_preflight(
    input_bundle: dict[str, Path | str] | None,
    hand_source: str,
) -> dict:
    path = Path(str(input_bundle["adapter_manifest"])) if input_bundle is not None else None
    result = assess_dai_native_spider_input(load_json_optional(path), hand_source)
    if input_bundle is None:
        result = {
            "status": "invalid",
            "errors": ["missing exact DAI hand-source input bundle"],
            "hand_source": hand_source,
        }
    result["adapter_manifest"] = str(path) if path is not None and path.exists() else None
    result["provenance_scope"] = "exact_dai_cell"
    return result


def spider_source_input_preflight_payload(
    *,
    route: str,
    trajectory_6dof: str,
    hand_source: str,
    input_bundle: dict[str, Path | str] | None,
    assessment: dict,
    input_rc: int = 4,
) -> dict:
    adapter_path = (
        Path(str(input_bundle["adapter_manifest"])).resolve()
        if input_bundle is not None and input_bundle.get("adapter_manifest")
        else None
    )
    adapter = load_json_optional(adapter_path)
    binding_errors: list[str] = []
    if adapter_path is not None and adapter_path.is_file():
        if adapter is None:
            binding_errors.append("underlying_adapter_unreadable")
        elif adapter.get("hand_source") != hand_source:
            binding_errors.append(
                f"underlying_adapter_hand_source={adapter.get('hand_source')},expected={hand_source}"
            )
    elif adapter_path is not None:
        binding_errors.append("underlying_adapter_missing")
    return {
        "schema_version": 1,
        "route": route,
        "trajectory_6dof": trajectory_6dof,
        "hand_source": hand_source,
        "retargeting": "spider",
        "status": "invalid_input",
        "input_rc": int(input_rc),
        "input_assessment": assessment,
        "underlying_adapter_manifest": {
            "path": str(adapter_path) if adapter_path is not None else None,
            "path_base": "matrix_root",
            "sha256": file_sha256(adapter_path),
            "exists": bool(adapter_path is not None and adapter_path.is_file()),
            "adapter_schema_version": (
                adapter.get("adapter_schema_version") if isinstance(adapter, dict) else None
            ),
            "hand_source": adapter.get("hand_source") if isinstance(adapter, dict) else None,
            "retarget_input_qc_status": (
                (adapter.get("retarget_input_qc") or {}).get("status")
                if isinstance(adapter, dict)
                else None
            ),
            "adapter_rigid_invariance_status": (
                (adapter.get("adapter_rigid_invariance") or {}).get("status")
                if isinstance(adapter, dict)
                else None
            ),
        },
        "evidence_validation": {
            "status": "invalid" if binding_errors else "ok",
            "errors": binding_errors,
        },
    }


def audited_spider_input_timebase(
    input_root: Path,
    *,
    task: str,
    hand_type: str,
    data_id: int,
    adapter_manifest: Path,
) -> dict[str, object]:
    """Resolve ref_dt only from agreeing, route-bound input evidence."""
    errors: list[str] = []
    evidence: list[dict[str, object]] = []
    task_info_path = input_root / "mano" / hand_type / task / "task_info.json"
    task_info = load_json_optional(task_info_path)
    if isinstance(task_info, dict):
        try:
            value = float(task_info["ref_dt"])
            if value > 0:
                evidence.append({"source": str(task_info_path), "ref_dt": value})
        except (KeyError, TypeError, ValueError):
            errors.append("spider_task_info_ref_dt_invalid")
    adapter = load_json_optional(adapter_manifest)
    clock = (
        ((adapter or {}).get("temporal_alignment") or {}).get("target_clock") or {}
        if isinstance(adapter, dict)
        else {}
    )
    try:
        fps = float(clock["fps"])
        if fps > 0:
            evidence.append({"source": str(adapter_manifest), "ref_dt": 1.0 / fps})
    except (KeyError, TypeError, ValueError):
        errors.append("spider_adapter_target_clock_invalid")
    values = [float(item["ref_dt"]) for item in evidence]
    if not values:
        errors.append("spider_input_timebase_no_audited_ref_dt")
        ref_dt = None
    elif any(abs(value - values[0]) > 1e-12 for value in values[1:]):
        errors.append("spider_input_timebase_evidence_conflict")
        ref_dt = None
    else:
        ref_dt = values[0]
    return {
        "status": "ok" if not errors else "invalid",
        "ref_dt": ref_dt,
        "task": task,
        "hand_type": hand_type,
        "data_id": data_id,
        "evidence": evidence,
        "errors": errors,
        "backend_tuning": False,
    }


def _native_spider_binding(route_root: Path) -> dict | None:
    if (route_root / "spider_output_manifest.json").exists():
        return None
    return load_json_optional(route_root / "native_spider_run_binding.json")


def write_native_spider_run_binding(
    *,
    route_root: Path,
    route: str,
    trajectory_6dof: str,
    hand_source: str,
    hand_type: str,
    task: str,
    robot_type: str,
    data_id: int,
    ref_dt_evidence: dict[str, object],
    input_root: Path,
    native_run_dir: Path,
    preflight_path: Path,
    log_path: Path,
    returncode: int,
) -> dict[str, object]:
    run_manifest_path = native_run_dir / "vanilla_spider_run_manifest.json"
    outcome_path = native_run_dir / "vanilla_spider_outcome.json"
    run_manifest = load_json_optional(run_manifest_path) or {}
    outcome = load_json_optional(outcome_path) or {}
    native_rc = int(outcome.get("returncode", returncode))
    videos = sorted(native_run_dir.rglob("visualization_mjwp.mp4"))
    robot = videos[0] if len(videos) == 1 else None
    native_log = log_path if log_path.is_file() else None
    metrics: dict[str, float] = {}
    if native_log is not None:
        match = re.search(
            r"Final object tracking error:\s*pos=([0-9.eE+-]+),\s*quat=([0-9.eE+-]+)",
            native_log.read_text(encoding="utf-8", errors="replace"),
        )
        if match:
            metrics = {
                "final_object_position_error_m": float(match.group(1)),
                "final_object_quaternion_error": float(match.group(2)),
            }
    payload: dict[str, object] = {
        "schema_version": 1,
        "route": route,
        "trajectory_6dof": trajectory_6dof,
        "hand_source": hand_source,
        "resolved_hand_type": hand_type,
        "resolved_task": task,
        "robot_type": robot_type,
        "data_id": data_id,
        "input_root": str(input_root.resolve()),
        "ref_dt_evidence": ref_dt_evidence,
        "native_run_dir": rel(native_run_dir, route_root),
        "native_run_manifest": rel(run_manifest_path, route_root),
        "native_outcome": rel(outcome_path, route_root),
        "native_returncode": native_rc,
        "native_returncode_authoritative": True,
        "native_robot_video": rel(robot, route_root) if robot is not None else None,
        "native_metrics": metrics,
        "external_qc": {"role": "diagnostic_only"},
        "preflight": rel(preflight_path, route_root),
        "log": rel(log_path, route_root),
        "backend_commit": ((run_manifest.get("backend") or {}).get("commit")),
        "backend_clean_after": outcome.get("backend_clean_after"),
    }
    write_json(route_root / "native_spider_run_binding.json", payload)
    return payload


def existing_spider_input_binding_errors(
    route_root: Path,
    *,
    route: str,
    trajectory_6dof: str,
    hand_source: str,
    resolved_hand_type: str,
    resolved_task: str,
    preflight_payload: dict,
) -> list[str]:
    binding = _native_spider_binding(route_root)
    if binding is None:
        return ["missing_native_spider_run_binding"]
    expected = {
        "route": route,
        "trajectory_6dof": trajectory_6dof,
        "hand_source": hand_source,
        "resolved_hand_type": resolved_hand_type,
        "resolved_task": resolved_task,
    }
    errors = [
        f"{key}_mismatch"
        for key, value in expected.items()
        if binding.get(key) != value
    ]
    if preflight_payload.get("status") not in {"ok", "passed"}:
        errors.append("preflight_not_ok")
    return errors


def run_spider_cells(args: argparse.Namespace, dst: Path) -> list[dict]:
    if args.run_spider == "none":
        return collect_existing_spider_report(args, dst)
    cells = spider_cell_order(args.run_spider)

    results = []
    env = os.environ.copy()
    ensure_dir(dst / "logs")

    for trajectory, hand_source in cells:
        key = cell_key(trajectory, hand_source, "spider")
        robot = find_spider_robot(
            dst,
            key,
            args.prefer_spider_ik_video,
            allow_mjwp_fallback=args.allow_spider_mjwp_fallback_video,
            allow_invalid_tracking_for_review=args.include_numerically_invalid_aligned_robot,
        )
        if args.skip_existing_spider and robot is not None and robot.exists():
            item = spider_report_item(args, dst, key, returncode=0)
            if item is not None:
                item["skipped_existing"] = True
                results.append(item)
            else:
                results.append({"cell": key, "status": "ok", "robot": rel(robot, dst), "skipped_existing": True})
            continue
        source_task = args.dai_task if trajectory == "egoinfinity" else args.dai_source_task
        source_hand_type = args.dai_hand_type if trajectory == "egoinfinity" else args.dai_source_hand_type
        input_bundle = find_exact_spider_input_bundle(
            REPO_ROOT / "experiments" / args.source_run,
            trajectory,
            hand_source,
            source_task,
            source_hand_type,
            args,
        )
        preflight_path = (
            dst
            / "intermediates"
            / "retargeting"
            / "spider"
            / key
            / "spider_source_input_preflight.json"
        )
        remove_path(preflight_path)
        if trajectory == "do_as_i_do":
            input_preflight = dai_native_spider_input_preflight(input_bundle, hand_source)
            if input_preflight.get("status") == "invalid":
                preflight_payload = spider_source_input_preflight_payload(
                    route=key,
                    trajectory_6dof=trajectory,
                    hand_source=hand_source,
                    input_bundle=input_bundle,
                    assessment=input_preflight,
                    input_rc=4,
                )
                write_json(preflight_path, preflight_payload)
                preflight_reference = {
                    "path": rel(preflight_path, dst),
                    "path_base": "matrix_root",
                    "sha256": file_sha256(preflight_path),
                    "trajectory_6dof": trajectory,
                    "hand_source": hand_source,
                    "status": "invalid_input",
                    "input_rc": 4,
                }
                results.append(
                    {
                        "cell": key,
                        "status": "invalid_input",
                        "input_status": "invalid_input",
                        "input_rc": 4,
                        "returncode": 4,
                        "robot": None,
                        "input_preflight": input_preflight,
                        "preflight_evidence": preflight_reference,
                    }
                )
                continue
        cell_env = env.copy()
        if input_bundle is not None:
            cell_env["SPIDER_SOURCE_KEYPOINTS"] = str(input_bundle["keypoints"])
            cell_env["SPIDER_OBJECT_VISUAL"] = str(input_bundle["object_visual"])
            cell_env["SPIDER_SOURCE_ADAPTER_MANIFEST"] = str(input_bundle["adapter_manifest"])
        keypoint_quality = summarize_keypoint_quality(
            Path(str(input_bundle["keypoints"])) if input_bundle is not None else None
        )
        hand_selection = spider_single_object_hand_selection(
            args.spider_hand_type,
            source_hand_type,
            keypoint_quality,
        )
        resolved_spider_hand_type = str(hand_selection["resolved_hand_type"])
        selection_path = (
            dst
            / "intermediates"
            / "retargeting"
            / "spider"
            / key
            / "spider_matrix_hand_selection.json"
        )
        write_json(selection_path, hand_selection)
        route_root = selection_path.parent
        if input_bundle is None:
            results.append({"cell": key, "status": "invalid_input", "input_rc": 4})
            continue
        input_root = Path(str(input_bundle["keypoints"])).parents[4]
        resolved_task = str(input_bundle["task"])
        timebase = audited_spider_input_timebase(
            input_root,
            task=resolved_task,
            hand_type=resolved_spider_hand_type,
            data_id=args.spider_data_id,
            adapter_manifest=Path(str(input_bundle["adapter_manifest"])),
        )
        preflight_payload = {
            "schema_version": 1,
            "route": key,
            "trajectory_6dof": trajectory,
            "hand_source": hand_source,
            "status": timebase["status"],
            "input_rc": 0 if timebase["status"] == "ok" else 4,
            "input_assessment": timebase,
            "resolved_hand_type": resolved_spider_hand_type,
            "resolved_task": resolved_task,
        }
        write_json(preflight_path, preflight_payload)
        if timebase["status"] != "ok":
            results.append({"cell": key, "status": "invalid_input", "input_rc": 4})
            continue
        attempt = 1
        while (route_root / "native_runs" / f"attempt{attempt:02d}").exists():
            attempt += 1
        native_run = route_root / "native_runs" / f"attempt{attempt:02d}"
        cmd = [
            str(REPO_ROOT / "scripts" / "run_spider_retarget.sh"),
            "--spider-root",
            args.spider_root,
            "--python",
            args.spider_python,
            "--input-root",
            str(input_root),
            "--output-dir",
            str(native_run),
            "--task",
            resolved_task,
            "--robot-type",
            args.spider_robot_type,
            "--embodiment-type",
            resolved_spider_hand_type,
            "--data-id",
            str(args.spider_data_id),
            "--ref-dt",
            str(timebase["ref_dt"]),
            "--cuda-visible-devices",
            args.spider_cuda_visible_devices,
            "--egl-device-id",
            args.spider_egl_device_id,
        ]
        log_path = dst / "logs" / f"reuse12_spider_{key}.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=cell_env, stdout=log, stderr=subprocess.STDOUT)
        write_native_spider_run_binding(
            route_root=route_root,
            route=key,
            trajectory_6dof=trajectory,
            hand_source=hand_source,
            hand_type=resolved_spider_hand_type,
            task=resolved_task,
            robot_type=args.spider_robot_type,
            data_id=args.spider_data_id,
            ref_dt_evidence=timebase,
            input_root=input_root,
            native_run_dir=native_run,
            preflight_path=preflight_path,
            log_path=log_path,
            returncode=proc.returncode,
        )
        robot_after = find_spider_robot(
            dst,
            key,
            args.prefer_spider_ik_video,
            allow_mjwp_fallback=args.allow_spider_mjwp_fallback_video,
            allow_invalid_tracking_for_review=args.include_numerically_invalid_aligned_robot,
        )
        item = spider_report_item(args, dst, key, returncode=proc.returncode, log_path=log_path)
        if item is None:
            item = {
                "cell": key,
                "status": "failed",
                "returncode": proc.returncode,
                "log": rel(log_path, dst),
                "robot": rel(robot_after, dst) if robot_after is not None and robot_after.exists() else None,
            }
        item["hand_selection"] = {
            **hand_selection,
            "path": rel(selection_path, dst),
        }
        results.append(item)
        if proc.returncode != 0 and not args.keep_going:
            raise SystemExit(f"SPIDER failed for {key}; see {log_path}")
    return results


def spider_cell_order(run_spider: str) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    if run_spider == "all":
        cells += [("egoinfinity", hand) for hand in HAND_SOURCES]
    if run_spider in {"do_as_i_do", "all"}:
        cells += [("do_as_i_do", hand) for hand in HAND_SOURCES]
    return cells


def collect_existing_spider_report(args: argparse.Namespace, dst: Path) -> list[dict]:
    results: list[dict] = []
    for trajectory in TRAJECTORIES:
        for hand_source in HAND_SOURCES:
            key = cell_key(trajectory, hand_source, "spider")
            item = spider_report_item(args, dst, key)
            if item is not None:
                item["reindexed_existing"] = True
                results.append(item)
    return results


def spider_report_item(
    args: argparse.Namespace,
    dst: Path,
    key: str,
    *,
    returncode: int | None = None,
    log_path: Path | None = None,
) -> dict | None:
    root = dst / "intermediates" / "retargeting" / "spider" / key
    native = _native_spider_binding(root)
    if native is not None:
        native_rc = int(native.get("native_returncode", 1))
        robot = _recorded_manifest_path(
            native.get("native_robot_video"), relative_to=root
        )
        ok = native_rc == 0 and robot is not None and robot.is_file()
        return {
            "cell": key,
            "status": "ok" if ok else "failed",
            "robot": rel(robot, dst) if ok else None,
            "returncode": native_rc,
            "native_returncode_authoritative": True,
            "external_qc": native.get("external_qc"),
            "native_metrics": native.get("native_metrics"),
            "native_binding": rel(root / "native_spider_run_binding.json", dst),
        }
    input_manifest = load_json_optional(root / "spider_input_manifest.json")
    output_manifest = load_json_optional(root / "spider_output_manifest.json")
    preflight_path = root / "spider_source_input_preflight.json"
    input_preflight = load_json_optional(preflight_path)
    robot = find_spider_robot(
        dst,
        key,
        args.prefer_spider_ik_video,
        allow_mjwp_fallback=args.allow_spider_mjwp_fallback_video,
        allow_invalid_tracking_for_review=args.include_numerically_invalid_aligned_robot,
    )
    if (
        input_manifest is None
        and output_manifest is None
        and robot is None
        and input_preflight is None
    ):
        return None
    if (
        input_preflight is not None
        and input_preflight.get("status") == "invalid_input"
        and robot is None
        and output_manifest is None
    ):
        return {
            "cell": key,
            "status": "invalid_input",
            "input_status": "invalid_input",
            "input_rc": input_preflight.get("input_rc"),
            "returncode": input_preflight.get("input_rc"),
            "robot": None,
            "input_preflight": input_preflight.get("input_assessment"),
            "preflight_evidence": {
                "path": rel(preflight_path, dst),
                "path_base": "matrix_root",
                "sha256": file_sha256(preflight_path),
                "trajectory_6dof": input_preflight.get("trajectory_6dof"),
                "hand_source": input_preflight.get("hand_source"),
                "status": input_preflight.get("status"),
                "input_rc": input_preflight.get("input_rc"),
            },
        }
    mjwp_fallback = output_manifest.get("mjwp_fallback") if output_manifest else None
    alignment = spider_alignment_manifest(root, output_manifest)
    fallback_applied = bool((mjwp_fallback or {}).get("applied"))
    tracking_valid = valid_spider_tracking(output_manifest)
    production_evidence_errors, production_evidence = spider_production_evidence(
        root,
        key,
        input_manifest,
        output_manifest,
    )
    weak_input = bool(input_manifest and not input_manifest.get("quality_valid", True))
    if robot is not None and robot.exists():
        if fallback_applied:
            status = "fallback"
        elif tracking_valid and not weak_input and not production_evidence_errors:
            status = "ok"
        elif weak_input:
            status = "reviewable_weak_input"
        elif production_evidence_errors:
            status = "reviewable_invalid_production_evidence"
        else:
            status = "reviewable_invalid_tracking"
    else:
        status = "failed"
    item = {
        "cell": key,
        "status": status,
        "robot": rel(robot, dst) if robot is not None and robot.exists() else None,
        "input_status": input_manifest.get("status") if input_manifest else None,
        "quality": input_manifest.get("quality") if input_manifest else None,
        "quality_errors": input_manifest.get("quality_errors") if input_manifest else None,
        "input_quality_valid": input_manifest.get("quality_valid") if input_manifest else None,
        "tracking_quality_valid": tracking_valid,
        "production_evidence_valid": not production_evidence_errors,
        "production_evidence_errors": production_evidence_errors,
        "production_evidence": production_evidence,
        "alignment": alignment,
        "mjwp_fallback": mjwp_fallback,
    }
    if returncode is not None:
        item["returncode"] = returncode
    if log_path is not None:
        item["log"] = rel(log_path, dst)
    return item


def valid_spider_alignment(root: Path, output_manifest: dict[str, object] | None = None) -> bool:
    return _valid_spider_alignment(root, REPO_ROOT, output_manifest)


def valid_spider_review_alignment(root: Path, output_manifest: dict[str, object] | None = None) -> bool:
    # Review may ignore numerical tracking failure, but never artifact identity,
    # timeline, or byte bindings.  This is the same validator as production
    # admission with only the tracking-status predicate disabled.
    return _valid_spider_alignment(
        root,
        REPO_ROOT,
        output_manifest,
        require_tracking=False,
    )


def spider_production_evidence(
    root: Path,
    key: str,
    input_manifest: dict[str, object] | None,
    output_manifest: dict[str, object] | None,
) -> tuple[list[str], dict[str, object]]:
    input_payload = input_manifest or {}
    output_payload = output_manifest or {}
    errors = raw_to_processed_invariance_errors(
        input_payload.get("raw_to_processed_hoi_invariance"),
        label="exact-route raw_to_processed_hoi_invariance",
        trajectory_6dof=str(input_payload.get("trajectory_6dof") or ""),
        hand_source=str(input_payload.get("hand_source") or ""),
        hand_type=str(output_payload.get("resolved_hand_type") or ""),
        adapter_manifest=_recorded_manifest_path(
            input_payload.get("adapter_manifest"), relative_to=root
        ),
        processed_keypoints=_recorded_manifest_path(
            input_payload.get("source_keypoints"), relative_to=root
        ),
    )
    rest_support = output_payload.get("rest_support_provenance")
    errors.extend(
        spider_rest_support_errors(
            rest_support,
            cell_key=key,
            trajectory_6dof=str(output_payload.get("trajectory_6dof") or ""),
            hand_source=str(output_payload.get("hand_source") or ""),
            task=str(output_payload.get("task") or ""),
            hand_type=str(output_payload.get("resolved_hand_type") or ""),
            data_id=output_payload.get("data_id"),
            input_manifest=input_payload,
            root=root,
            repo_root=REPO_ROOT,
        )
    )
    tracking, tracking_evidence, tracking_errors = (
        authoritative_spider_aligned_tracking(
            output_payload,
            root=root,
            cell_key=key,
            repo_root=REPO_ROOT,
        )
    )
    errors.extend(tracking_errors)
    return errors, {
        "raw_to_processed_hoi_invariance": input_payload.get(
            "raw_to_processed_hoi_invariance"
        ),
        "rest_support_provenance": rest_support,
        "authoritative_aligned_tracking": tracking,
        "authoritative_aligned_tracking_evidence": tracking_evidence,
    }


def find_spider_robot(
    dst: Path,
    key: str,
    prefer_ik: bool = False,
    *,
    allow_mjwp_fallback: bool = False,
    allow_invalid_tracking_for_review: bool = False,
) -> Path | None:
    del prefer_ik
    root = dst / "intermediates" / "retargeting" / "spider" / key
    native = _native_spider_binding(root)
    if native is not None:
        if int(native.get("native_returncode", 1)) != 0:
            return None
        robot = _recorded_manifest_path(
            native.get("native_robot_video"), relative_to=root
        )
        return robot if robot is not None and robot.is_file() else None
    input_manifest = load_json_optional(root / "spider_input_manifest.json")
    if (
        input_manifest
        and input_manifest.get("status") == "invalid_weak_interaction"
        and not allow_invalid_tracking_for_review
    ):
        return None
    output_manifest = load_json_optional(root / "spider_output_manifest.json")
    fallback = (output_manifest or {}).get("mjwp_fallback") or {}
    fallback_applied = bool(fallback.get("applied"))
    if fallback_applied and not allow_mjwp_fallback:
        return None
    if (
        not fallback_applied
        and not allow_invalid_tracking_for_review
        and not valid_spider_tracking(output_manifest)
    ):
        return None
    if not allow_invalid_tracking_for_review:
        production_errors, _ = spider_production_evidence(
            root,
            key,
            input_manifest,
            output_manifest,
        )
        if production_errors:
            return None
    alignment_valid = (
        valid_spider_review_alignment(root, output_manifest)
        if allow_invalid_tracking_for_review
        else valid_spider_alignment(root, output_manifest)
    )
    if not alignment_valid:
        return None
    alignment = spider_alignment_manifest(root, output_manifest)
    if not isinstance(alignment, dict):
        return None
    mjwp_video = _recorded_manifest_path(
        alignment.get("aligned_video"),
        relative_to=root,
    )
    return mjwp_video if mjwp_video is not None and mjwp_video.is_file() else None


def compose_cell(dst: Path, key: str, duration: float) -> Path:
    out = dst / "videos" / f"{key}__triptych.mp4"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "review" / "compose_triptych.py"),
        "--overlay",
        str(dst / "cells" / key / "overlay.mp4"),
        "--depth",
        str(dst / "cells" / key / "depth.mp4"),
        "--robot",
        str(dst / "cells" / key / "robot.mp4"),
        "--output",
        str(out),
        "--layout",
        "vertical",
        "--labels",
        "reconstruction overlay|depth / input geometry|retargeting robot",
    ]
    if duration > 0:
        cmd += ["--duration", str(duration)]
    subprocess.run(cmd, check=True)
    return out


def dai_retarget_output_root(robot_src: Path | None) -> Path | None:
    if robot_src is None:
        return None
    for parent in robot_src.parents:
        if parent.name == "retargeting_outputs":
            return parent
    return None


def dai_adapter_manifest_for_robot(robot_src: Path | None) -> Path | None:
    output_root = dai_retarget_output_root(robot_src)
    if output_root is None:
        return None
    candidate = output_root.parent / "raw_dir" / "adapter_manifest.json"
    return candidate if candidate.exists() else None


ROUTE_PROVENANCE_FIELDS = (
    "hand_source",
    "object_track_source",
    "object_mesh_source",
    "retarget_object_source",
    "canonical_transform",
    "hoi_refinement",
    "hoi_contact_alignment",
    "adapter_rigid_invariance",
    "route_input_qc",
)


def dai_route_input_qc_for_robot(robot_src: Path | None, key: str) -> dict[str, object] | None:
    output_root = dai_retarget_output_root(robot_src)
    candidates: list[Path] = []
    if output_root is not None:
        candidates.append(output_root.parent / "route_input_qc.json")
    if robot_src is not None:
        for ancestor in robot_src.parents:
            candidates.append(
                ancestor
                / "assets"
                / "cells"
                / key
                / "retargeting"
                / "do_as_i_do"
                / "metadata"
                / "route_input_qc.json"
            )
            for manifest_path in (
                ancestor / "assets" / "cells" / key / "asset_manifest.json",
                ancestor / "cells" / key / "manifest.json",
            ):
                manifest = load_json_optional(manifest_path)
                route_qc = ((manifest or {}).get("provenance") or {}).get("route_input_qc")
                if isinstance(route_qc, dict):
                    return route_qc
    for candidate in candidates:
        route_qc = load_json_optional(candidate)
        if route_qc is not None:
            return route_qc
    return None


def _recorded_manifest_path(value: object, *, relative_to: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    for path in (relative_to / candidate, REPO_ROOT / candidate):
        if path.exists():
            return path
    return relative_to / candidate


def exact_cell_route_provenance(
    *,
    dst: Path,
    trajectory: str,
    retargeting: str,
    hand_source: str,
    robot_src: Path | None,
) -> tuple[dict[str, object], dict[str, object]]:
    provenance: dict[str, object] = {field: None for field in ROUTE_PROVENANCE_FIELDS}
    evidence: dict[str, object] = {}
    key = cell_key(trajectory, hand_source, retargeting)

    if retargeting == "do_as_i_do":
        adapter_path = dai_adapter_manifest_for_robot(robot_src)
        adapter = load_json_optional(adapter_path)
        evidence["adapter_manifest"] = str(adapter_path) if adapter_path else None
        evidence["adapter"] = adapter
        if adapter is not None:
            for field in ROUTE_PROVENANCE_FIELDS:
                if field != "route_input_qc":
                    provenance[field] = adapter.get(field)
        provenance["route_input_qc"] = dai_route_input_qc_for_robot(robot_src, key)
        return provenance, evidence

    if retargeting == "spider":
        root = dst / "intermediates" / "retargeting" / "spider" / key
        input_path = root / "spider_input_manifest.json"
        output_path = root / "spider_output_manifest.json"
        input_manifest = load_json_optional(input_path)
        output_manifest = load_json_optional(output_path)
        evidence.update({
            "spider_input_manifest_path": str(input_path) if input_path.exists() else None,
            "spider_output_manifest_path": str(output_path) if output_path.exists() else None,
            "spider_input_manifest": input_manifest,
            "spider_output_manifest": output_manifest,
        })
        adapter_ref = None
        for manifest in (output_manifest, input_manifest):
            if isinstance(manifest, dict) and manifest.get("adapter_manifest"):
                adapter_ref = manifest.get("adapter_manifest")
                break
        adapter_path = _recorded_manifest_path(adapter_ref, relative_to=root)
        adapter = load_json_optional(adapter_path)
        evidence["adapter_manifest"] = (
            str(adapter_path) if adapter_path is not None and adapter_path.exists() else None
        )
        evidence["adapter"] = adapter
        for field in ROUTE_PROVENANCE_FIELDS:
            for manifest in (output_manifest, input_manifest, adapter):
                if isinstance(manifest, dict) and manifest.get(field) is not None:
                    provenance[field] = manifest.get(field)
                    break
        if provenance["route_input_qc"] is None:
            provenance["route_input_qc"] = load_json_optional(root / "route_input_qc.json")
        return provenance, evidence

    # Native retarget cells do not consume a DAI adapter. Keep the absence of
    # adapter-specific metadata explicit instead of borrowing a trajectory-wide
    # convenience manifest.
    provenance.update({
        "hand_source": hand_source,
        "object_track_source": "egoinfinity" if trajectory == "egoinfinity" else None,
        "object_mesh_source": "egoinfinity" if trajectory == "egoinfinity" else None,
        "retarget_object_source": "egoinfinity" if trajectory == "egoinfinity" else None,
    })
    evidence.update({"adapter_manifest": None, "source_kind": "native_retarget_without_dai_adapter"})
    return provenance, evidence


def validate_route_provenance_payload(
    provenance: dict[str, object],
    *,
    trajectory: str,
    hand_source: str,
) -> list[str]:
    errors: list[str] = []
    if provenance.get("hand_source") != hand_source:
        errors.append(
            f"route hand_source mismatch: expected={hand_source} actual={provenance.get('hand_source')}"
        )
    expected_object_source = "egoinfinity" if trajectory == "egoinfinity" else "dai_native"
    for field in ("object_track_source", "object_mesh_source", "retarget_object_source"):
        if provenance.get(field) != expected_object_source:
            errors.append(
                f"route {field} mismatch: expected={expected_object_source} actual={provenance.get(field)}"
            )
    if not isinstance(provenance.get("canonical_transform"), dict):
        errors.append("missing canonical_transform provenance")
    refinement = provenance.get("hoi_refinement")
    if not isinstance(refinement, dict):
        errors.append("missing hoi_refinement provenance")
    elif refinement.get("applied") is not False:
        errors.append("HOI refinement was applied or is not explicitly disabled")
    contact_alignment = provenance.get("hoi_contact_alignment")
    if not isinstance(contact_alignment, dict):
        errors.append("missing hoi_contact_alignment provenance")
    elif contact_alignment.get("applied") is not False or contact_alignment.get("transforms_modified") is True:
        errors.append("HOI contact alignment modified production transforms")
    invariance = provenance.get("adapter_rigid_invariance")
    if not isinstance(invariance, dict):
        errors.append("missing adapter_rigid_invariance provenance")
    elif invariance.get("status") != "ok":
        errors.append("adapter_rigid_invariance is not ok")
    route_qc = provenance.get("route_input_qc")
    if not isinstance(route_qc, dict):
        errors.append("missing route_input_qc provenance")
    else:
        if route_qc.get("status") != "ok":
            errors.append("route_input_qc is not ok")
        if route_qc.get("hand_source") != hand_source:
            errors.append("route_input_qc hand_source does not match cell")
        if route_qc.get("expected_cell_key") != cell_key(trajectory, hand_source, "do_as_i_do"):
            errors.append("route_input_qc does not name the exact DAI source cell")
    return errors


def dai_hand_provenance_for_robot(robot_src: Path | None) -> Path | None:
    output_root = dai_retarget_output_root(robot_src)
    if output_root is None:
        return None
    candidate = output_root.parent / "input_hand_provenance.json"
    return candidate if candidate.exists() else None


def dai_object_mesh_for_robot(robot_src: Path | None, task: str) -> Path | None:
    output_root = dai_retarget_output_root(robot_src)
    if output_root is None:
        return None
    candidate = output_root / "assets" / "objects" / task / "visual.obj"
    return candidate if candidate.exists() else None


def validate_cell_provenance(
    *,
    dst: Path,
    source: Path,
    trajectory: str,
    retargeting: str,
    hand_source: str,
    robot_src: Path | None,
    srcs: dict[str, Path | None],
    args: argparse.Namespace,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    evidence: dict[str, object] = {
        "robot_src": rel(robot_src, dst) if robot_src else None,
        "trajectory_6dof": trajectory,
        "retargeting": retargeting,
    }

    if retargeting in {"do_as_i_do", "spider"}:
        route_provenance, route_evidence = exact_cell_route_provenance(
            dst=dst,
            trajectory=trajectory,
            retargeting=retargeting,
            hand_source=hand_source,
            robot_src=robot_src,
        )
        evidence["exact_route_provenance"] = route_provenance
        evidence["exact_route_sources"] = route_evidence
        errors.extend(
            validate_route_provenance_payload(
                route_provenance,
                trajectory=trajectory,
                hand_source=hand_source,
            )
        )

    if retargeting == "do_as_i_do":
        expected_cell = cell_key(trajectory, hand_source, "do_as_i_do")
        if robot_src is None:
            errors.append(f"missing Do-as-I-Do robot for exact cell {expected_cell}")
        elif expected_cell not in str(robot_src):
            errors.append(f"Do-as-I-Do robot does not come from exact hand-source cell {expected_cell}: {robot_src}")
        hand_provenance_path = dai_hand_provenance_for_robot(robot_src)
        hand_provenance = load_json_optional(hand_provenance_path)
        evidence["input_hand_provenance"] = hand_provenance
        evidence["input_hand_provenance_path"] = str(hand_provenance_path) if hand_provenance_path else None
        if hand_provenance is None:
            errors.append(f"missing input_hand_provenance.json for {expected_cell}")
        else:
            if hand_provenance.get("status") != "ok":
                errors.append(f"invalid hand provenance for {expected_cell}: {hand_provenance.get('errors')}")
            if hand_provenance.get("requested_hand_source") != hand_source:
                errors.append(
                    f"hand provenance label mismatch for {expected_cell}: "
                    f"{hand_provenance.get('requested_hand_source')}"
                )
        raw_to_processed_path = dai_raw_to_processed_hoi_for_robot(robot_src)
        raw_to_processed = load_json_optional(raw_to_processed_path)
        evidence["raw_to_processed_hoi_invariance_path"] = (
            str(raw_to_processed_path) if raw_to_processed_path else None
        )
        evidence["raw_to_processed_hoi_invariance"] = raw_to_processed
        errors.extend(
            raw_to_processed_invariance_errors(
                raw_to_processed,
                label="exact-route raw_to_processed_hoi_invariance",
                trajectory_6dof=trajectory,
                hand_source=hand_source,
                hand_type=robot_src.parent.parent.parent.name if robot_src else None,
                adapter_manifest=dai_adapter_manifest_for_robot(robot_src),
                processed_keypoints=(
                    raw_to_processed_path.with_name("trajectory_keypoints.npz")
                    if raw_to_processed_path
                    else None
                ),
            )
        )
    elif retargeting == "spider":
        expected_dai_cell = cell_key(trajectory, hand_source, "do_as_i_do")
        root = dst / "intermediates" / "retargeting" / "spider" / cell_key(
            trajectory, hand_source, "spider"
        )
        input_manifest = load_json_optional(root / "spider_input_manifest.json")
        output_manifest = load_json_optional(root / "spider_output_manifest.json")
        if input_manifest is None:
            errors.append("missing SPIDER input manifest")
        if output_manifest is None:
            errors.append("missing SPIDER output manifest")
        for label, manifest in (("SPIDER input", input_manifest), ("SPIDER output", output_manifest)):
            if manifest is None:
                continue
            if manifest.get("trajectory_6dof") != trajectory:
                errors.append(f"{label} trajectory_6dof does not match cell")
            if manifest.get("hand_source") != hand_source:
                errors.append(f"{label} hand_source does not match cell")
            for field in ROUTE_PROVENANCE_FIELDS[:-1]:
                value = manifest.get(field)
                if value is not None and value != route_provenance.get(field):
                    errors.append(f"{label} {field} conflicts with exact route provenance")
        if input_manifest is not None and input_manifest.get("expected_dai_cell_key") != expected_dai_cell:
            errors.append("SPIDER input manifest does not name the exact DAI source cell")
        production_errors, production_evidence = spider_production_evidence(
            root,
            cell_key(trajectory, hand_source, "spider"),
            input_manifest,
            output_manifest,
        )
        errors.extend(production_errors)
        evidence["spider_production_evidence"] = production_evidence
        adapter_path = _recorded_manifest_path(
            (output_manifest or {}).get("adapter_manifest")
            or (input_manifest or {}).get("adapter_manifest"),
            relative_to=root,
        )
        if adapter_path is None or not adapter_path.exists():
            errors.append("missing exact DAI source adapter for SPIDER cell")
        elif expected_dai_cell not in str(adapter_path):
            errors.append("SPIDER source adapter path is not from the exact DAI source cell")
        else:
            adapter = load_json_optional(adapter_path)
            if adapter is None:
                errors.append("invalid exact DAI source adapter for SPIDER cell")
            else:
                for field in ROUTE_PROVENANCE_FIELDS[:-1]:
                    if adapter.get(field) != route_provenance.get(field):
                        errors.append(f"SPIDER {field} does not match exact DAI source adapter")

    if trajectory == "do_as_i_do":
        pure_dai_paths = {
            "dai_keypoints": srcs.get("dai_keypoints"),
            "dai_object_visual": srcs.get("dai_object_visual"),
            "dai_object_convex": srcs.get("dai_object_convex"),
        }
        evidence["pure_dai_paths"] = {name: str(path) if path else None for name, path in pure_dai_paths.items()}
        if not args.spider_fuse_ego_object_for_dai:
            for name, path in pure_dai_paths.items():
                if path_contains(path, "egoinfinity") or path_contains(path, "fused_ego"):
                    errors.append(f"pure DAI asset {name} resolves to an EgoInfinity/fused path: {path}")
            if retargeting == "do_as_i_do" and path_contains(robot_src, "traj_egoinfinity__"):
                errors.append(f"pure DAI retarget robot resolves to an EgoInfinity DAI cell: {robot_src}")
        elif retargeting == "spider":
            warnings.append("DAI->SPIDER uses explicit fused Ego object because --spider-fuse-ego-object-for-dai is enabled")

    if trajectory == "egoinfinity" and retargeting == "do_as_i_do":
        if robot_src is None:
            errors.append("missing EgoInfinity->Do-as-I-Do retarget robot")
        elif not path_contains(robot_src, "traj_egoinfinity__"):
            errors.append(f"EgoInfinity->Do-as-I-Do robot does not come from traj_egoinfinity cell: {robot_src}")
        adapter_path = dai_adapter_manifest_for_robot(robot_src)
        adapter = load_json_optional(adapter_path)
        evidence["adapter_manifest"] = str(adapter_path) if adapter_path else None
        if adapter is None:
            errors.append("missing EgoInfinity adapter_manifest.json for Ego->DAI retarget raw_dir")
        else:
            replacement = adapter.get("object_mesh_replacement") or {}
            evidence["adapter"] = {
                "ego_object_id": adapter.get("ego_object_id"),
                "ego_prompt": adapter.get("ego_prompt"),
                "mesh_scale": adapter.get("mesh_scale"),
                "mesh_scale_source": adapter.get("mesh_scale_source"),
                "selected_hand": adapter.get("selected_hand"),
                "anchor_hand": adapter.get("anchor_hand"),
                "object_mesh_replacement": {
                    "ego_mesh": replacement.get("ego_mesh") if isinstance(replacement, dict) else None,
                    "targets": replacement.get("targets") if isinstance(replacement, dict) else None,
                    "boxlike_fallback": replacement.get("boxlike_fallback") if isinstance(replacement, dict) else None,
                },
                "geometry_interaction_hand": adapter.get("geometry_interaction_hand"),
                "visual_interaction_hand": adapter.get("visual_interaction_hand"),
            }
            if not isinstance(replacement, dict) or not replacement.get("ego_mesh") or not replacement.get("targets"):
                errors.append("Ego->DAI adapter did not record an Ego object mesh replacement")
            geometry = adapter.get("geometry_interaction_hand") or {}
            if isinstance(geometry, dict) and geometry.get("available") and not geometry.get("selected_hand"):
                warnings.append("Ego object track has no geometric hand-object contact; treat this scene as weak display material")
        mesh_path = dai_object_mesh_for_robot(robot_src, args.dai_task if trajectory == "egoinfinity" else args.task)
        extent = obj_extent_summary(mesh_path)
        evidence["retarget_object_mesh"] = extent
        if extent.get("available") and float(extent.get("diag", 0.0)) < 0.04:
            warnings.append(f"Ego object mesh is very small after scale baking: diag={extent.get('diag'):.4f}m")

    status = "invalid" if errors else "ok"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "evidence": evidence,
    }


def cell_object_provenance(
    dst: Path,
    trajectory: str,
    retargeting: str,
    hand_source: str,
    robot_src: Path | None,
    args: argparse.Namespace,
    srcs: dict[str, Path | None],
) -> dict[str, object]:
    del args, srcs
    route_provenance, route_evidence = exact_cell_route_provenance(
        dst=dst,
        trajectory=trajectory,
        retargeting=retargeting,
        hand_source=hand_source,
        robot_src=robot_src,
    )
    object_source = "egoinfinity" if trajectory == "egoinfinity" else "dai_native"
    object_source_verified = all(
        route_provenance.get(field) == object_source
        for field in ("object_track_source", "object_mesh_source", "retarget_object_source")
    )
    route_provenance.update({
        "route_provenance_sources": route_evidence,
        "route": (
            f"{object_source}_object_tracking__{retargeting}_retargeting"
            if object_source_verified
            else f"unverified_object_source__{retargeting}_retargeting"
        ),
        "recommended_route": trajectory == "egoinfinity",
        "baseline_only": trajectory == "do_as_i_do",
    })
    if trajectory == "egoinfinity" and object_source_verified:
        route_provenance["object_selection_source"] = "bbox_target_point_best"
        route_provenance["route"] = f"egoinfinity_object_tracking__{retargeting}_retargeting"
    return route_provenance


def build_cells(dst: Path, source: Path, srcs: dict[str, Path | None], args: argparse.Namespace) -> list[dict]:
    reports = []
    fallback_spider = None
    if args.allow_spider_fallback:
        fallback_spider = first_existing(
            [
                find_spider_robot(
                    dst,
                    cell_key("do_as_i_do", "estimated", "spider"),
                    args.prefer_spider_ik_video,
                    allow_mjwp_fallback=args.allow_spider_mjwp_fallback_video,
                    allow_invalid_tracking_for_review=args.include_numerically_invalid_aligned_robot,
                )
                or Path("__missing__"),
                find_spider_robot(
                    dst,
                    cell_key("do_as_i_do", "aoe", "spider"),
                    args.prefer_spider_ik_video,
                    allow_mjwp_fallback=args.allow_spider_mjwp_fallback_video,
                    allow_invalid_tracking_for_review=args.include_numerically_invalid_aligned_robot,
                )
                or Path("__missing__"),
            ]
        )

    for trajectory in TRAJECTORIES:
        overlay = dst / "assets" / "trajectory_6dof" / trajectory / args.task / "overlay.mp4"
        depth = dst / "assets" / "trajectory_6dof" / trajectory / args.task / "depth.mp4"
        for hand_source in HAND_SOURCES:
            for retargeting in RETARGETERS:
                key = cell_key(trajectory, hand_source, retargeting)
                cell_dir = ensure_dir(dst / "cells" / key)
                assets_dir = ensure_dir(dst / "assets" / "cells" / key)
                if retargeting == "egoinfinity":
                    robot_src = srcs["ego_robot"]
                    robot_note = "reused from source EgoInfinity/G1 full run"
                elif retargeting == "do_as_i_do":
                    robot_task = args.dai_task if trajectory == "egoinfinity" else args.dai_source_task
                    robot_hand_type = args.dai_hand_type if trajectory == "egoinfinity" else args.dai_source_hand_type
                    robot_src, robot_note = find_dai_robot_in_overrides(
                        args.dai_override_sources,
                        trajectory,
                        hand_source,
                        robot_task,
                        robot_hand_type,
                        args,
                    )
                    if robot_src is None:
                        robot_src, robot_note = find_quality_dai_robot_in_source(
                            source,
                            trajectory,
                            hand_source,
                            robot_task,
                            robot_hand_type,
                            args,
                        )
                    if robot_src is None and (robot_task != args.task or robot_hand_type != args.hand_type):
                        robot_src, robot_note = find_quality_dai_robot_in_source(
                            source,
                            trajectory,
                            hand_source,
                            args.task,
                            args.hand_type,
                            args,
                        )
                        robot_note = (
                            robot_note or "fallback Do-as-I-Do/Sharpa robot from source trajectory task"
                            if robot_src is not None
                            else "missing source Do-as-I-Do/Sharpa cell"
                        )
                    if robot_src is None and hand_source != "estimated" and args.allow_dai_hand_fallback:
                        robot_src, robot_note = find_quality_dai_robot_in_source(
                            source,
                            trajectory,
                            "estimated",
                            robot_task,
                            robot_hand_type,
                            args,
                        )
                        if robot_src is None and (robot_task != args.task or robot_hand_type != args.hand_type):
                            robot_src, robot_note = find_quality_dai_robot_in_source(
                                source,
                                trajectory,
                                "estimated",
                                args.task,
                                args.hand_type,
                                args,
                            )
                        robot_note = (
                            robot_note or "fallback Do-as-I-Do/Sharpa robot from same trajectory estimated-hand cell"
                            if robot_src is not None
                            else "missing source Do-as-I-Do/Sharpa cell"
                        )
                    elif robot_src is None:
                        robot_note = "missing source Do-as-I-Do/Sharpa cell"
                else:
                    exact = find_spider_robot(
                        dst,
                        key,
                        args.prefer_spider_ik_video,
                        allow_mjwp_fallback=args.allow_spider_mjwp_fallback_video,
                        allow_invalid_tracking_for_review=args.include_numerically_invalid_aligned_robot,
                    )
                    robot_src = exact or fallback_spider
                    if exact:
                        robot_note = "exact SPIDER aligned MJWP cell"
                    elif fallback_spider is not None:
                        robot_note = "fallback SPIDER robot reused from Do-as-I-Do trajectory cell"
                    else:
                        robot_note = "missing source SPIDER cell"

                overlay_dst = materialize(overlay, cell_dir / "overlay.mp4", args.mode)
                depth_dst = materialize(depth, cell_dir / "depth.mp4", args.mode)
                robot_dst = materialize(robot_src, cell_dir / "robot.mp4", args.mode)
                provenance_validation = validate_cell_provenance(
                    dst=dst,
                    source=source,
                    trajectory=trajectory,
                    retargeting=retargeting,
                    hand_source=hand_source,
                    robot_src=robot_src,
                    srcs=srcs,
                    args=args,
                )
                missing = [
                    name
                    for name, value in {
                        "overlay": overlay_dst,
                        "depth": depth_dst,
                        "robot": robot_dst,
                    }.items()
                    if value is None
                ]
                video = None
                if args.compose and not missing:
                    video = compose_cell(dst, key, args.duration)
                elif args.compose:
                    remove_path(dst / "videos" / f"{key}__triptych.mp4")
                manifest = {
                    "cell_key": key,
                    "settings": {
                        "trajectory_6dof": trajectory,
                        "hand_source": hand_source,
                        "retargeting": retargeting,
                        "task": args.task,
                        "hand_type": args.hand_type,
                        "spider_hand_type": args.spider_hand_type,
                        "dai_task": args.dai_task,
                        "dai_hand_type": args.dai_hand_type,
                        "dai_source_task": args.dai_source_task,
                        "dai_source_hand_type": args.dai_source_hand_type,
                    },
                    "reuse_policy": {
                        "source_run": args.source_run,
                        "overlay_depth": f"from {trajectory} trajectory_6dof full run",
                        "robot": robot_note,
                    },
                    "provenance": cell_object_provenance(
                        dst,
                        trajectory,
                        retargeting,
                        hand_source,
                        robot_src,
                        args,
                        srcs,
                    ),
                    "provenance_validation": provenance_validation,
                    "assets": {
                        "cell.overlay": rel(overlay_dst, dst),
                        "cell.depth": rel(depth_dst, dst),
                        "cell.robot": rel(robot_dst, dst),
                        "source.overlay": rel(overlay, dst),
                        "source.depth": rel(depth, dst),
                        "source.robot": rel(robot_src, dst) if robot_src else None,
                        "video.triptych": rel(video, dst) if video else None,
                    },
                    "missing": missing,
                }
                write_json(cell_dir / "manifest.json", manifest)
                write_json(assets_dir / "asset_manifest.json", manifest)
                reports.append(manifest)
    return reports


def collect_route_availability(
    source: Path,
    spider_report: list[dict],
) -> dict[str, dict]:
    source_manifest = load_json_optional(source / "reuse" / "reuse_manifest.json") or {}
    source_availability = source_manifest.get("route_availability") or {}
    result = {
        str(route): dict(entry)
        for route, entry in source_availability.items()
        if isinstance(entry, dict)
    }
    for item in spider_report:
        if not isinstance(item, dict) or not item.get("cell"):
            continue
        if item.get("status") != "invalid_input":
            continue
        result[str(item["cell"])] = {
            "input_rc": item.get("input_rc"),
            "post_retarget_rc": None,
            "status": "invalid_input",
            "available": False,
            "preflight_evidence": item.get("preflight_evidence"),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse one EgoInfinity full run and one Do-as-I-Do full run to expand a 12-cell AoE retargeting demo matrix."
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--task", default="foundation_jar_bimanual_leftscale")
    parser.add_argument("--hand-type", default="bimanual")
    parser.add_argument(
        "--spider-hand-type",
        default=None,
        help=(
            "Hand type for SPIDER retargeting. Auto selects one interaction hand per trajectory because "
            "SPIDER's bimanual model creates two independent object bodies for a single-object scene. "
            "Pass bimanual explicitly only when that two-object model is intended."
        ),
    )
    parser.add_argument(
        "--dai-task",
        default=None,
        help="Do-as-I-Do retargeting task to use for DAI robot cells. Defaults to --task.",
    )
    parser.add_argument(
        "--dai-hand-type",
        default=None,
        help="Do-as-I-Do hand type to use for DAI robot cells. Defaults to --hand-type.",
    )
    parser.add_argument(
        "--dai-source-task",
        default=None,
        help="Do-as-I-Do source trajectory asset task. Defaults to --task.",
    )
    parser.add_argument(
        "--dai-source-hand-type",
        default=None,
        help="Do-as-I-Do source trajectory asset hand type. Defaults to --hand-type.",
    )
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--compose", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--run-spider", choices=["none", "do_as_i_do", "all"], default="do_as_i_do")
    parser.add_argument("--allow-spider-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--allow-spider-mjwp-fallback-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow SPIDER cells whose MJWP output was replaced by the IK/object-reference "
            "fallback to enter composed videos. Default false so failed SPIDER dynamics "
            "remain visibly missing instead of being presented as successful retargeting."
        ),
    )
    parser.add_argument("--allow-dai-hand-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--include-numerically-invalid-aligned-robot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic opt-in to expose original, finite, aligned DAI/SPIDER MuJoCo videos for plain-video "
            "review even when numerical quality thresholds fail. Default false: numerical-invalid routes "
            "cannot enter matrix videos or demo admission. Provenance errors, fallback videos, and unaligned "
            "outputs remain forbidden even when explicitly enabled."
        ),
    )
    parser.add_argument(
        "--dai-auto-select-interaction-hand",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When searching DAI override runs for EgoInfinity->DAI cells, also score bimanual/opposite-hand "
            "candidates and pick the hand with better hand-object interaction."
        ),
    )
    parser.add_argument(
        "--allow-cross-trajectory-dai-robot-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When an exact Do-as-I-Do robot cell has weak hand-object quality, allow a quality-valid "
            "robot cell from the other trajectory to be used as the visible DAI retarget result."
        ),
    )
    parser.add_argument(
        "--dai-override-run",
        action="append",
        default=[],
        help=(
            "Experiment run to search before --source-run for Do-as-I-Do robot cells. "
            "May be passed multiple times; useful for fresh repaired DAI retarget probes."
        ),
    )
    parser.add_argument(
        "--dai-override-ego-spider-keypoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also let DAI override runs replace EgoInfinity->SPIDER keypoints. "
            "Only quality-valid EgoInfinity override bundles are used; object meshes still come from the "
            "matrix trajectory assets."
        ),
    )
    parser.add_argument(
        "--dai-asset-min-distance-threshold",
        type=float,
        default=0.35,
        help="Reject/flag DAI source keypoints when every fingertip stays farther than this from the object center.",
    )
    parser.add_argument(
        "--dai-asset-median-distance-threshold",
        type=float,
        default=0.45,
        help="Reject/flag DAI source keypoints when median fingertip-object distance is above this threshold.",
    )
    parser.add_argument(
        "--dai-asset-no-contact-min-distance-threshold",
        type=float,
        default=0.08,
        help=(
            "When contact labels are absent or inactive, require at least one fingertip to be this close "
            "to the object center before accepting the DAI source asset."
        ),
    )
    parser.add_argument(
        "--dai-asset-no-contact-median-distance-threshold",
        type=float,
        default=0.18,
        help=(
            "When contact labels are absent or inactive, reject DAI source assets whose median "
            "fingertip-object distance exceeds this tighter threshold."
        ),
    )
    parser.add_argument(
        "--dai-asset-require-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require nonzero contact for DAI assets unless fingertip-object distance is already plausible.",
    )
    parser.add_argument("--skip-existing-spider", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-going", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--spider-root",
        default=os.environ.get(
            "SPIDER_ROOT", str(REPO_ROOT / "third_party" / "SPIDER")
        ),
    )
    parser.add_argument("--spider-python", default=os.environ.get("SPIDER_PYTHON"))
    parser.add_argument("--spider-cuda-visible-devices", default="")
    parser.add_argument("--spider-egl-device-id", default="0")
    parser.add_argument("--spider-device", default="cuda:0")
    parser.add_argument("--spider-robot-type", default="xhand")
    parser.add_argument("--spider-dataset-name", default="do_as_i_do")
    parser.add_argument("--spider-data-id", type=int, default=0)
    parser.add_argument("--spider-max-sim-steps", default="-1")
    parser.add_argument("--spider-num-samples", default="1024")
    parser.add_argument("--spider-max-num-iterations", default="16")
    parser.add_argument("--spider-skip-mjwp", action="store_true")
    parser.add_argument("--prefer-spider-ik-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--spider-fuse-ego-object-for-dai",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly use DAI hand keypoints with EgoInfinity object pose/mesh for do_as_i_do->SPIDER cells. "
            "Default is false so pure DAI cells stay DAI-native or are marked invalid."
        ),
    )
    args = parser.parse_args()
    if not args.run_name:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.source_run}__reuse12_{stamp}"
    if args.dai_task is None:
        args.dai_task = args.task
    if args.dai_hand_type is None:
        args.dai_hand_type = args.hand_type
    if args.dai_source_task is None:
        args.dai_source_task = args.task
    if args.dai_source_hand_type is None:
        args.dai_source_hand_type = args.hand_type
    if args.spider_hand_type is None:
        args.spider_hand_type = "auto"
    args.dai_override_sources = [REPO_ROOT / "experiments" / run for run in args.dai_override_run]
    return args


def main() -> int:
    args = parse_args()
    if args.spider_fuse_ego_object_for_dai:
        raise SystemExit(
            "--spider-fuse-ego-object-for-dai is disabled: production routes may not replace only the "
            "object trajectory while keeping a different hand source"
        )
    source = REPO_ROOT / "experiments" / args.source_run
    if not source.exists():
        raise SystemExit(f"source run does not exist: {source}")
    for override in args.dai_override_sources:
        if not override.exists():
            raise SystemExit(f"DAI override run does not exist: {override}")
    dst = REPO_ROOT / "experiments" / args.run_name
    ensure_dir(dst / "logs")
    ensure_dir(dst / "videos")
    srcs = source_paths(
        source,
        args.task,
        args.hand_type,
        args.dai_task,
        args.dai_hand_type,
        args.dai_source_task,
        args.dai_source_hand_type,
    )
    dai_override_asset_report = apply_dai_override_trajectory_assets(srcs, args.dai_override_sources, args)
    materialize(source, dst / "reuse" / "source_run", args.mode)
    traj_report = prepare_trajectory_assets(
        dst,
        srcs,
        args.task,
        args.mode,
        spider_fuse_ego_object_for_dai=args.spider_fuse_ego_object_for_dai,
    )
    trajectory_asset_validation = validate_trajectory_assets(
        dst,
        args.task,
        dai_task=args.dai_task,
        dai_hand_type=args.dai_hand_type,
        dai_source_task=args.dai_source_task,
        dai_source_hand_type=args.dai_source_hand_type,
        spider_fuse_ego_object_for_dai=args.spider_fuse_ego_object_for_dai,
        dai_override_asset_report=dai_override_asset_report,
        dai_asset_min_distance_threshold=args.dai_asset_min_distance_threshold,
        dai_asset_median_distance_threshold=args.dai_asset_median_distance_threshold,
        dai_asset_no_contact_min_distance_threshold=args.dai_asset_no_contact_min_distance_threshold,
        dai_asset_no_contact_median_distance_threshold=args.dai_asset_no_contact_median_distance_threshold,
        dai_asset_require_contact=args.dai_asset_require_contact,
    )
    spider_report = run_spider_cells(args, dst)
    route_availability = collect_route_availability(source, spider_report)
    cell_reports = build_cells(dst, source, srcs, args)
    provenance_validation = {
        item["cell_key"]: item.get("provenance_validation")
        for item in cell_reports
        if (item.get("provenance_validation") or {}).get("status") != "ok"
        or (item.get("provenance_validation") or {}).get("warnings")
    }
    summary = {
        "run_name": args.run_name,
        "source_run": args.source_run,
        "experiment_root": str(dst),
        "git": git_metadata(),
        "command": {
            "argv": sys.argv,
            "cwd": str(REPO_ROOT),
        },
        "gpu_mapping": gpu_mapping_from_env(),
        "recommendation": {
            "object_reconstruction_and_6dof_tracking": "egoinfinity",
            "dai_native_reconstruction": "baseline_only",
            "note": "Pure DAI cells do not use EgoInfinity object assets unless an explicit fused-Ego option is enabled.",
        },
        "demo_admission_policy": {
            "policy": "retarget_only_plain_mujoco_numerical_and_visual_v3",
            "criterion": "route_specific_numerical_qc_and_provenance_and_plain_visual_review",
            "matrix_completeness_required": False,
            "numerical_qc_is_diagnostic": False,
            "numerical_invalid_can_be_visual_success": False,
            "failed_routes_must_remain_explicit": True,
        },
        "mode": args.mode,
        "task": args.task,
        "hand_type": args.hand_type,
        "spider_hand_type": args.spider_hand_type,
        "dai_task": args.dai_task,
        "dai_hand_type": args.dai_hand_type,
        "dai_source_task": args.dai_source_task,
        "dai_source_hand_type": args.dai_source_hand_type,
        "spider_fuse_ego_object_for_dai": args.spider_fuse_ego_object_for_dai,
        "include_numerically_invalid_aligned_robot": args.include_numerically_invalid_aligned_robot,
        "dai_override_runs": args.dai_override_run,
        "dai_override_assets": dai_override_asset_report,
        "trajectory_assets": traj_report,
        "trajectory_asset_validation": trajectory_asset_validation,
        "provenance_validation": provenance_validation,
        "spider": spider_report,
        "route_availability": route_availability,
        "cells": len(cell_reports),
        "composed": sum(1 for item in cell_reports if item["assets"].get("video.triptych")),
        "missing_cells": {item["cell_key"]: item["missing"] for item in cell_reports if item["missing"]},
        "fallback_cells": [
            item["cell_key"]
            for item in cell_reports
            if item["reuse_policy"]["robot"].startswith("fallback")
        ],
        "dai_hand_fallback_cells": [
            item["cell_key"]
            for item in cell_reports
            if item["reuse_policy"]["robot"].startswith("fallback Do-as-I-Do")
        ],
    }
    write_json(dst / "reuse_12_demo_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
