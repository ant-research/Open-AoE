#!/usr/bin/env python3
"""Create a Do-as-I-Do override run with a corrected object mesh scale.

The script keeps the official Do-as-I-Do retargeting trajectory intact, but
copies the MuJoCo scene into a new override run, replaces the object visual and
convex meshes with a uniformly scaled/axis-aligned version, and records the
scale audit.
The plain MuJoCo robot video is rendered from the corrected scene by default so
matrix composition does not reuse a stale video with the old object scale.
RGB robot overlays are optional diagnostics only and are disabled by default.
This is intended for cases where reconstruction produced a plausible hand/pose
trajectory but retargeting assets used a visibly wrong object scale.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


CELL_KEY_TEMPLATE = "traj_{trajectory}__hand_{hand_source}__retarget_do_as_i_do"
SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def cell_key(trajectory: str, hand_source: str) -> str:
    return CELL_KEY_TEMPLATE.format(trajectory=trajectory, hand_source=hand_source)


def validate_path_component(value: str, name: str) -> str:
    if not SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{name} is not a safe path component: {value!r}")
    return value


def resolve_under(root: Path, path: Path) -> Path:
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def load_json_optional(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def extents(path: Path, scale: float = 1.0) -> tuple[float, float, float]:
    mesh = as_trimesh(path)
    if mesh.vertices.size == 0:
        raise ValueError(f"mesh has no vertices: {path}")
    return tuple(float(v) * float(scale) for v in mesh.extents)


def choose_axis_order(
    source_extents: tuple[float, float, float],
    target_extents: tuple[float, float, float],
    scale: float,
) -> tuple[int, int, int]:
    def score(order: tuple[int, int, int]) -> float:
        fixed = [source_extents[order[i]] * scale for i in range(3)]
        return sum(abs(math.log(max(fixed[i], 1e-12) / max(target_extents[i], 1e-12))) for i in range(3))

    return min(itertools.permutations(range(3)), key=score)


def parse_axis_order(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("axis order must have three comma-separated indices, e.g. 0,2,1")
    try:
        order = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("axis order entries must be integers") from exc
    if sorted(order) != [0, 1, 2]:
        raise argparse.ArgumentTypeError(f"axis order must be a permutation of 0,1,2: {value}")
    return order


def parse_vec3(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("vector must have three comma-separated values, e.g. 0,0,1")
    try:
        vec = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("vector entries must be numeric") from exc
    norm = math.sqrt(sum(part * part for part in vec))
    if norm <= 1e-8 or not math.isfinite(norm):
        raise argparse.ArgumentTypeError(f"vector must be non-zero: {value}")
    return vec


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    valid = norm[:, 0] > 1e-8
    quat = quat.copy()
    quat[valid] /= norm[valid]
    w, x, y, z = quat.T
    mats = np.empty((len(quat), 3, 3), dtype=np.float64)
    mats[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    mats[:, 0, 1] = 2.0 * (x * y - z * w)
    mats[:, 0, 2] = 2.0 * (x * z + y * w)
    mats[:, 1, 0] = 2.0 * (x * y + z * w)
    mats[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    mats[:, 1, 2] = 2.0 * (y * z - x * w)
    mats[:, 2, 0] = 2.0 * (x * z - y * w)
    mats[:, 2, 1] = 2.0 * (y * z + x * w)
    mats[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    mats[~valid] = np.eye(3)
    return mats


def qpos_local_axis_alignment(qpos_path: Path, up_vector: tuple[float, float, float]) -> dict[str, object] | None:
    qpos = load_qpos(qpos_path)
    if qpos is None or len(qpos) == 0:
        return None
    width = object_qpos_width(qpos)
    if width != 7:
        return None
    quat = qpos[:, -4:] if width == 7 else qpos[:, -width + 3 : -width + 7]
    if quat.shape[1] != 4:
        return None
    up = np.asarray(up_vector, dtype=np.float64)
    up /= np.linalg.norm(up)
    rot = quat_wxyz_to_matrix(quat)
    dots = np.abs(np.einsum("nji,j->ni", rot, up))
    med = np.nanmedian(dots, axis=0)
    if not np.all(np.isfinite(med)):
        return None
    return {
        "trajectory": str(qpos_path),
        "frames": int(len(qpos)),
        "object_qpos_width": int(width),
        "up_vector": [float(v) for v in up],
        "median_abs_axis_dot_up": [float(v) for v in med],
        "local_up_axis": int(np.argmax(med)),
    }


def choose_axis_order_align_long_to_qpos_up(
    source_extents: tuple[float, float, float],
    target_extents: tuple[float, float, float],
    scale: float,
    local_up_axis: int,
) -> tuple[int, int, int]:
    source_long_axis = int(np.argmax(np.asarray(source_extents, dtype=np.float64)))
    target_ext = list(float(v) for v in target_extents)
    target_long_axis = int(np.argmax(np.asarray(target_ext, dtype=np.float64)))
    target_long = target_ext[target_long_axis]
    target_cross = [target_ext[i] for i in range(3) if i != target_long_axis]
    fixed_cross_axes = [i for i in range(3) if i != local_up_axis]

    desired_extents: list[tuple[float, float, float]] = []
    for cross in itertools.permutations(target_cross):
        desired = [0.0, 0.0, 0.0]
        desired[local_up_axis] = target_long
        for axis, value in zip(fixed_cross_axes, cross):
            desired[axis] = value
        desired_extents.append(tuple(desired))

    candidates = [order for order in itertools.permutations(range(3)) if order[local_up_axis] == source_long_axis]

    def score(order: tuple[int, int, int]) -> float:
        fixed = [source_extents[order[i]] * scale for i in range(3)]
        return min(
            sum(abs(math.log(max(fixed[i], 1e-12) / max(desired[i], 1e-12))) for i in range(3))
            for desired in desired_extents
        )

    return min(candidates, key=score)


def transform_obj(src: Path, dst: Path, scale: float, axis_order: tuple[int, int, int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    with src.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.rstrip("\n").split()
                source_xyz = [float(parts[i]) for i in range(1, 4)]
                xyz = [source_xyz[axis_order[i]] * scale for i in range(3)]
                suffix = " " + " ".join(parts[4:]) if len(parts) > 4 else ""
                out.append(f"v {xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g}{suffix}\n")
            else:
                out.append(line)
    dst.write_text("".join(out), encoding="utf-8")


def as_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [geom for geom in mesh.geometry.values() if hasattr(geom, "vertices")]
        )
    if not hasattr(mesh, "vertices"):
        raise ValueError(f"mesh has no vertices: {path}")
    return mesh


def export_mesh_as_obj(src: Path, dst: Path, scale: float = 1.0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".obj" and float(scale) == 1.0:
        copy_file(src, dst)
        return
    mesh = as_trimesh(src)
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        mesh = trimesh.PointCloud(np.asarray(mesh.vertices, dtype=np.float64)).convex_hull
    if float(scale) != 1.0:
        mesh = mesh.copy()
        mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * float(scale)
    mesh.export(dst)


def rel_symlink(src: Path, dst: Path, *, is_dir: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve(), target_is_directory=is_dir)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_qpos(path: Path) -> np.ndarray | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "qpos" not in data:
                return None
            qpos = np.asarray(data["qpos"], dtype=np.float64)
    except Exception:
        return None
    if qpos.ndim < 2:
        return None
    return qpos.reshape(-1, qpos.shape[-1])


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def resample_rows(arr: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(arr)
    if len(arr) == n:
        return arr
    idx = np.rint(np.linspace(0, len(arr) - 1, n)).astype(int)
    return arr[idx]


def object_qpos_width(qpos: np.ndarray) -> int:
    """Return 7 for freejoint object qpos, otherwise 6 for act slide/hinge qpos."""
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 6:
        raise ValueError(f"qpos is too short for object state: shape={qpos.shape}")
    if qpos.shape[1] >= 7:
        quat = qpos[:, -4:]
        quat_norm = np.linalg.norm(quat, axis=1)
        finite = quat_norm[np.isfinite(quat_norm)]
        if finite.size and float(np.nanmedian(np.abs(finite - 1.0))) < 0.05:
            return 7
    return 6


def object_tracking_error(mjwp_path: Path, ref_path: Path) -> dict[str, float | int | bool | str]:
    mjwp = load_qpos(mjwp_path)
    ref = load_qpos(ref_path)
    if mjwp is None or ref is None or len(mjwp) == 0 or len(ref) == 0:
        return {"available": False, "reason": "missing_qpos"}
    if mjwp.shape[1] < 6 or ref.shape[1] < 6:
        return {"available": False, "reason": "qpos_too_short"}
    n = len(mjwp)
    ref = resample_rows(ref, n)
    mjwp_width = object_qpos_width(mjwp)
    ref_width = object_qpos_width(ref)
    if mjwp_width != ref_width:
        return {
            "available": False,
            "reason": "object_qpos_width_mismatch",
            "mjwp_width": int(mjwp_width),
            "ref_width": int(ref_width),
        }
    mjwp_obj = mjwp[:, -mjwp_width:]
    ref_obj = ref[:, -ref_width:]
    pos = np.linalg.norm(mjwp_obj[:, :3] - ref_obj[:, :3], axis=1)
    if mjwp_width == 7:
        mj_quat = mjwp_obj[:, 3:7]
        ref_quat = ref_obj[:, 3:7]
        mj_norm = np.linalg.norm(mj_quat, axis=1, keepdims=True)
        ref_norm = np.linalg.norm(ref_quat, axis=1, keepdims=True)
        valid = (mj_norm[:, 0] > 1e-8) & (ref_norm[:, 0] > 1e-8)
        rot = np.full(n, np.nan, dtype=np.float64)
        if np.any(valid):
            dots = np.abs(
                np.sum((mj_quat[valid] / mj_norm[valid]) * (ref_quat[valid] / ref_norm[valid]), axis=1)
            )
            rot[valid] = 2.0 * np.arccos(np.clip(dots, -1.0, 1.0))
    else:
        rot = np.linalg.norm(mjwp_obj[:, 3:6] - ref_obj[:, 3:6], axis=1)
    return {
        "available": True,
        "frames": int(n),
        "object_qpos_width": int(mjwp_width),
        "pos_median": float(np.nanmedian(pos)),
        "pos_max": float(np.nanmax(pos)),
        "rot_median": float(np.nanmedian(rot)) if np.any(np.isfinite(rot)) else 0.0,
        "rot_max": float(np.nanmax(rot)) if np.any(np.isfinite(rot)) else 0.0,
    }


def object_error_exceeds_thresholds(
    error: dict[str, float | int | bool | str],
    *,
    pos_median: float,
    rot_median: float,
    pos_max: float,
    rot_max: float,
) -> bool:
    if not error.get("available"):
        return False
    return (
        float(error.get("pos_median", 0.0)) > pos_median
        or float(error.get("rot_median", 0.0)) > rot_median
        or float(error.get("pos_max", 0.0)) > pos_max
        or float(error.get("rot_max", 0.0)) > rot_max
    )


def write_hybrid_object_ref(base_path: Path, ref_path: Path, output_path: Path) -> Path:
    base_arrays = load_npz_arrays(base_path)
    ref_qpos = load_qpos(ref_path)
    if ref_qpos is None:
        raise ValueError(f"{ref_path} does not contain usable qpos")
    qpos = np.asarray(base_arrays["qpos"], dtype=np.float64).copy()
    original_shape = qpos.shape
    flat = qpos.reshape(-1, qpos.shape[-1])
    ref = resample_rows(ref_qpos, len(flat))
    base_width = object_qpos_width(flat)
    ref_width = object_qpos_width(ref)
    if base_width != ref_width:
        raise ValueError(f"object qpos width mismatch: base={flat.shape}, ref={ref.shape}")
    flat[:, -base_width:] = ref[:, -ref_width:]
    base_arrays["qpos"] = flat.reshape(original_shape)
    if "qvel" in base_arrays:
        qvel = np.asarray(base_arrays["qvel"], dtype=np.float64).copy()
        flat_qvel = qvel.reshape(-1, qvel.shape[-1])
        if len(flat_qvel) == len(flat) and flat_qvel.shape[1] >= 6:
            flat_qvel[:, -6:] = 0.0
            base_arrays["qvel"] = flat_qvel.reshape(qvel.shape)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **base_arrays)
    return output_path


def write_visual_scene_hide_geom_groups(scene: Path, output: Path, groups: set[int]) -> Path:
    tree = ET.parse(scene)
    root = tree.getroot()
    parent = {child: node for node in root.iter() for child in list(node)}
    removed_geom_names: set[str] = set()
    for geom in list(root.iter("geom")):
        group = int(geom.get("group", "0"))
        if group not in groups:
            continue
        name = geom.get("name")
        if name:
            removed_geom_names.add(name)
        parent[geom].remove(geom)
    if removed_geom_names:
        for node in list(root.iter()):
            if node.tag == "geom":
                continue
            refs = {node.get("geom"), node.get("geom1"), node.get("geom2")}
            if refs & removed_geom_names:
                parent_node = parent.get(node)
                if parent_node is not None:
                    parent_node.remove(node)
    for body in root.iter("body"):
        has_moving_joint = any(
            child.tag == "joint" and child.get("type") in {"free", "slide", "hinge"} for child in list(body)
        )
        has_inertial = any(child.tag == "inertial" for child in list(body))
        if has_moving_joint and not has_inertial:
            inertial = ET.Element("inertial", pos="0 0 0", mass="0.1", diaginertia="0.001 0.001 0.001")
            body.insert(0, inertial)
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="unicode")
    return output


def referenced_convex_names(scene_path: Path, task: str) -> set[str]:
    text = scene_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(rf"objects/{re.escape(task)}/convex/([^\"']+\.obj)")
    return set(pattern.findall(text))


def build_paths(args: argparse.Namespace) -> dict[str, Path]:
    source_run = args.repo_root / "experiments" / args.source_run
    override_run = args.repo_root / "experiments" / args.override_run
    key = cell_key(args.trajectory_6dof, args.hand_source)
    source_root = source_run / "intermediates" / "retargeting" / "do_as_i_do" / key / "retargeting_outputs"
    override_root = override_run / "intermediates" / "retargeting" / "do_as_i_do" / key / "retargeting_outputs"
    source_robot = source_root / "sharpa" / args.hand_type / args.task / "0"
    override_robot = override_root / "sharpa" / args.hand_type / args.task / "0"
    source_mano = source_root / "mano" / args.hand_type / args.task
    override_mano = override_root / "mano" / args.hand_type / args.task
    source_object = source_root / "assets" / "objects" / args.task
    override_object = override_root / "assets" / "objects" / args.task
    return {
        "source_run": source_run,
        "override_run": override_run,
        "source_root": source_root,
        "override_root": override_root,
        "source_robot": source_robot,
        "override_robot": override_robot,
        "source_mano": source_mano,
        "override_mano": override_mano,
        "source_object": source_object,
        "override_object": override_object,
    }


def inherited_mjwp_alignment(paths: dict[str, Path], args: argparse.Namespace) -> dict[str, object]:
    manifest_path = paths["source_robot"] / "mjwp_alignment_manifest.json"
    source_manifest = load_json_optional(manifest_path)
    if source_manifest is None:
        raise FileNotFoundError(manifest_path)
    source_frame_count = int(source_manifest.get("source_frame_count") or source_manifest.get("target_count") or 0)
    target_count = int(source_manifest.get("target_count") or source_frame_count)
    target_fps = float(source_manifest.get("target_fps") or 0.0)
    start_index = int(source_manifest.get("start_index", 0))
    warmup_steps = int(source_manifest.get("warmup_steps", start_index))
    if source_frame_count <= 0 or target_count <= 0:
        frame_count = len(sorted((args.raw_dir / "all_frames").glob("*.png")))
        source_frame_count = source_frame_count or frame_count
        target_count = target_count or frame_count
    if target_fps <= 0:
        metadata = load_json_optional(args.raw_dir / "metadata.json") or {}
        target_fps = float(metadata.get("source_fps") or 25.0)
    if source_frame_count <= 0 or target_count <= 0:
        raise ValueError("could not infer source frame count for repaired MJWP alignment")
    return {
        "source_manifest": str(manifest_path),
        "alignment_kind": source_manifest.get(
            "alignment_kind",
            "drop_warmup_and_sample_exact_source_timestamps",
        ),
        "start_index": start_index,
        "warmup_steps": warmup_steps,
        "source_frame_count": source_frame_count,
        "target_count": target_count,
        "target_fps": target_fps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--diagnostic-override-only",
        action="store_true",
        required=True,
        help=(
            "Explicit diagnostic opt-in. Outputs are ineligible for pristine "
            "backend success and Demo admission."
        ),
    )
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--override-run", required=True)
    parser.add_argument("--trajectory-6dof", choices=["do_as_i_do", "egoinfinity"], required=True)
    parser.add_argument("--hand-source", choices=["aoe", "estimated"], default="estimated")
    parser.add_argument("--hand-type", choices=["left", "right"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--target-object-visual", type=Path, required=True)
    parser.add_argument(
        "--target-object-scale",
        type=float,
        default=1.0,
        help="Scale applied to the target visual mesh before auditing extents or replacing the object.",
    )
    parser.add_argument("--target-object-convex-dir", type=Path)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--qpos-time-map", choices=["source-frame", "sim-time"], default="sim-time")
    parser.add_argument(
        "--no-axis-align",
        action="store_true",
        help="Only scale the source mesh; do not permute axes to match the target mesh extents.",
    )
    parser.add_argument(
        "--axis-order",
        type=parse_axis_order,
        help=(
            "Explicit source-to-fixed axis permutation, formatted as i,j,k where "
            "fixed_xyz = source_xyz[i,j,k]. Overrides automatic extent matching."
        ),
    )
    parser.add_argument(
        "--align-long-axis-to-qpos-up",
        action="store_true",
        help=(
            "Choose the object mesh axis order so the source mesh longest axis maps to the "
            "local object axis that is most aligned with the MuJoCo up vector in qpos."
        ),
    )
    parser.add_argument(
        "--qpos-up-vector",
        type=parse_vec3,
        default=(0.0, 0.0, 1.0),
        help="World up vector used by --align-long-axis-to-qpos-up. Default: 0,0,1.",
    )
    parser.add_argument(
        "--replace-with-target-object",
        action="store_true",
        help="Replace the retargeting object mesh with the target mesh instead of transforming the source mesh.",
    )
    parser.add_argument(
        "--no-render-robot-video",
        action="store_true",
        help="Skip rendering corrected plain MuJoCo visualization_mjwp.mp4.",
    )
    parser.add_argument(
        "--object-ref-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When MJWP object tracking drifts from the kinematic reference, render the stable robot video "
            "from MJWP hand qpos with kinematic object qpos."
        ),
    )
    parser.add_argument("--object-ref-pos-threshold", type=float, default=0.12)
    parser.add_argument("--object-ref-rot-threshold", type=float, default=0.45)
    parser.add_argument("--object-ref-pos-max-threshold", type=float, default=0.25)
    parser.add_argument("--object-ref-rot-max-threshold", type=float, default=0.75)
    args = parser.parse_args()

    args.source_run = validate_path_component(args.source_run, "source-run")
    args.override_run = validate_path_component(args.override_run, "override-run")
    args.task = validate_path_component(args.task, "task")
    args.repo_root = args.repo_root.expanduser().resolve()
    args.target_object_visual = resolve_under(args.repo_root, args.target_object_visual)
    if args.target_object_convex_dir is not None:
        args.target_object_convex_dir = resolve_under(args.repo_root, args.target_object_convex_dir)
    args.raw_dir = resolve_under(args.repo_root, args.raw_dir)

    paths = build_paths(args)
    source_visual = paths["source_object"] / "visual.obj"
    if not source_visual.exists():
        raise FileNotFoundError(source_visual)
    if not args.target_object_visual.exists():
        raise FileNotFoundError(args.target_object_visual)

    source_ext = extents(source_visual)
    if args.target_object_scale <= 0 or not math.isfinite(args.target_object_scale):
        raise ValueError(f"target object scale must be positive: {args.target_object_scale}")
    target_ext = extents(args.target_object_visual, args.target_object_scale)
    source_max = max(source_ext)
    target_max = max(target_ext)
    if source_max <= 0 or target_max <= 0:
        raise ValueError(f"degenerate object extents: source={source_max}, target={target_max}")
    scale = target_max / source_max
    qpos_up_axis_stats = None
    axis_order_source = "auto_extent"
    if args.axis_order is not None:
        axis_order = args.axis_order
        axis_order_source = "explicit"
    elif args.align_long_axis_to_qpos_up:
        qpos_candidates = [
            paths["source_robot"] / "trajectory_kinematic_act.npz",
            paths["source_robot"] / "trajectory_kinematic.npz",
            paths["source_robot"] / "trajectory_mjwp_act.npz",
            paths["source_robot"] / "trajectory_mjwp.npz",
        ]
        for candidate in qpos_candidates:
            if not candidate.exists():
                continue
            qpos_up_axis_stats = qpos_local_axis_alignment(candidate, args.qpos_up_vector)
            if qpos_up_axis_stats is not None:
                break
        if qpos_up_axis_stats is None:
            raise ValueError("could not infer object local up axis from source qpos")
        axis_order = choose_axis_order_align_long_to_qpos_up(
            source_ext,
            target_ext,
            scale,
            int(qpos_up_axis_stats["local_up_axis"]),
        )
        axis_order_source = "qpos_up"
    elif args.no_axis_align:
        axis_order = (0, 1, 2)
        axis_order_source = "identity"
    else:
        axis_order = choose_axis_order(source_ext, target_ext, scale)

    # Use real scene XML files so MuJoCo resolves meshdir relative to the
    # override run, while large trajectory/config files can remain symlinks.
    for name in ("scene.xml", "scene_eq.xml", "scene_ik.xml", "scene_act.xml"):
        src = paths["source_robot"] / name
        if src.exists():
            copy_file(src, paths["override_robot"] / name)
    for name in (
        "config.yaml",
        "config_act.yaml",
        "trajectory_mjwp.npz",
        "trajectory_mjwp_act.npz",
        "trajectory_ikrollout.npz",
        "trajectory_ikrollout_act.npz",
        "trajectory_kinematic.npz",
        "trajectory_kinematic_act.npz",
    ):
        src = paths["source_robot"] / name
        if src.exists():
            rel_symlink(src, paths["override_robot"] / name)
    rel_symlink(paths["source_root"] / "assets" / "robots", paths["override_root"] / "assets" / "robots", is_dir=True)

    rel_symlink(paths["source_mano"] / "0" / "trajectory_keypoints.npz", paths["override_mano"] / "0" / "trajectory_keypoints.npz")
    if (paths["source_mano"] / "task_info.json").exists():
        rel_symlink(paths["source_mano"] / "task_info.json", paths["override_mano"] / "task_info.json")

    if args.replace_with_target_object:
        export_mesh_as_obj(args.target_object_visual, paths["override_object"] / "visual.obj", args.target_object_scale)
    else:
        transform_obj(source_visual, paths["override_object"] / "visual.obj", scale, axis_order)
    source_convex = paths["source_object"] / "convex"
    override_convex = paths["override_object"] / "convex"
    override_convex.mkdir(parents=True, exist_ok=True)
    copied_convex: list[Path] = []
    if args.replace_with_target_object and args.target_object_convex_dir is not None and args.target_object_convex_dir.exists():
        for src in sorted(args.target_object_convex_dir.glob("*.obj")):
            dst = override_convex / src.name
            copy_file(src, dst)
            copied_convex.append(dst)
    elif args.replace_with_target_object and source_convex.exists():
        for src in sorted(source_convex.glob("*.obj")):
            dst = override_convex / src.name
            export_mesh_as_obj(args.target_object_visual, dst, args.target_object_scale)
            copied_convex.append(dst)
    elif source_convex.exists():
        for src in sorted(source_convex.glob("*.obj")):
            dst = override_convex / src.name
            transform_obj(src, dst, scale, axis_order)
            copied_convex.append(dst)
    scene_for_refs = paths["override_robot"] / "scene.xml"
    ref_names = referenced_convex_names(scene_for_refs, args.task) if scene_for_refs.exists() else set()
    fallback = copied_convex[0] if copied_convex else paths["override_object"] / "visual.obj"
    for name in ref_names:
        dst = override_convex / name
        if not dst.exists():
            shutil.copy2(fallback, dst)

    rendered_robot_video = None
    render_scene: Path | None = None
    aligned_robot_trajectory: Path | None = None
    alignment: dict[str, object] | None = None
    object_ref_fallback: dict[str, object] = {
        "enabled": bool(args.object_ref_fallback),
        "applied": False,
        "reason": "disabled_or_not_needed",
        "pos_threshold": float(args.object_ref_pos_threshold),
        "rot_threshold": float(args.object_ref_rot_threshold),
        "pos_max_threshold": float(args.object_ref_pos_max_threshold),
        "rot_max_threshold": float(args.object_ref_rot_max_threshold),
    }
    if not args.no_render_robot_video:
        robot_renderer = args.repo_root / "scripts" / "diagnostics" / "render_mujoco_trajectory.py"
        scene = first_existing([paths["override_robot"] / "scene_act.xml", paths["override_robot"] / "scene.xml"])
        render_scene = scene
        trajectory = first_existing(
            [paths["override_robot"] / "trajectory_mjwp_act.npz", paths["override_robot"] / "trajectory_mjwp.npz"]
        )
        ref_trajectory = first_existing(
            [paths["override_robot"] / "trajectory_kinematic_act.npz", paths["override_robot"] / "trajectory_kinematic.npz"]
        )
        if (
            args.object_ref_fallback
            and trajectory is not None
            and ref_trajectory is not None
            and trajectory.exists()
            and ref_trajectory.exists()
        ):
            err = object_tracking_error(trajectory, ref_trajectory)
            object_ref_fallback["tracking_error"] = err
            if object_error_exceeds_thresholds(
                err,
                pos_median=args.object_ref_pos_threshold,
                rot_median=args.object_ref_rot_threshold,
                pos_max=args.object_ref_pos_max_threshold,
                rot_max=args.object_ref_rot_max_threshold,
            ):
                hybrid = paths["override_robot"] / "trajectory_hybrid_object_ref.npz"
                write_hybrid_object_ref(trajectory, ref_trajectory, hybrid)
                trajectory = hybrid
                object_ref_fallback["applied"] = True
                object_ref_fallback["reason"] = "mjwp_object_tracking_error_exceeded"
                object_ref_fallback["stable_source"] = "hybrid_mjwp_hand_object_ref"
                object_ref_fallback["trajectory"] = str(hybrid)
        elif not args.object_ref_fallback:
            object_ref_fallback["reason"] = "disabled"
        if scene is not None and trajectory is not None and scene.exists() and trajectory.exists():
            alignment = inherited_mjwp_alignment(paths, args)
            aligned_stem = "mjwp_act" if trajectory.name.startswith("trajectory_mjwp_act") else "mjwp"
            aligned_robot_trajectory = paths["override_robot"] / f"trajectory_{aligned_stem}_aligned.npz"
            output = (
                paths["override_robot"]
                / f"visualization_diagnostic_scale_override_{aligned_stem}.mp4"
            )
            trimmer = args.repo_root / "scripts" / "diagnostics" / "trim_mujoco_trajectory.py"
            subprocess.run(
                [
                    args.python_bin,
                    str(trimmer),
                    "--input",
                    str(trajectory),
                    "--output",
                    str(aligned_robot_trajectory),
                    "--start-index",
                    str(alignment["start_index"]),
                    "--alignment-kind",
                    str(alignment["alignment_kind"]),
                    "--source-frame-count",
                    str(alignment["source_frame_count"]),
                    "--target-fps",
                    str(alignment["target_fps"]),
                    "--warmup-steps",
                    str(alignment["warmup_steps"]),
                    "--target-count",
                    str(alignment["target_count"]),
                    "--metadata-json",
                    str(paths["override_robot"] / "mjwp_alignment_manifest.json"),
                ],
                cwd=args.repo_root,
                check=True,
            )
            render_scene = write_visual_scene_hide_geom_groups(scene, paths["override_robot"] / "scene_visual.xml", {3})
            cmd = [
                args.python_bin,
                str(robot_renderer),
                "--scene",
                str(render_scene),
                "--trajectory",
                str(aligned_robot_trajectory),
                "--output",
                str(output),
                "--width",
                "640",
                "--height",
                "480",
                "--fps",
                str(alignment["target_fps"]),
                "--stride",
                "8",
                "--target-frames",
                str(alignment["target_count"]),
                "--hide-geom-groups",
                "3",
                "--disable-shadows",
            ]
            subprocess.run(cmd, cwd=args.repo_root, check=True)
            rendered_robot_video = str(output)
            alignment_manifest_path = paths["override_robot"] / "mjwp_alignment_manifest.json"
            alignment_manifest = load_json_optional(alignment_manifest_path) or {}
            alignment_manifest["applied"] = True
            alignment_manifest.setdefault("reason", "repaired_object_scale_and_resampled_to_source_clip")
            alignment_manifest["aligned_video"] = str(output)
            alignment_manifest["rendered_video"] = str(output)
            alignment_manifest_path.write_text(json.dumps(alignment_manifest, indent=2), encoding="utf-8")

    manifest = {
        "scope": "diagnostic_only_posthoc_object_scale_override",
        "eligible_for_pristine_backend_success": False,
        "eligible_for_demo_admission": False,
        "source_run": args.source_run,
        "override_run": args.override_run,
        "trajectory_6dof": args.trajectory_6dof,
        "hand_source": args.hand_source,
        "hand_type": args.hand_type,
        "task": args.task,
        "source_visual": str(source_visual),
        "target_object_visual": str(args.target_object_visual),
        "target_object_scale": float(args.target_object_scale),
        "target_object_convex_dir": str(args.target_object_convex_dir) if args.target_object_convex_dir is not None else None,
        "fixed_visual": str(paths["override_object"] / "visual.obj"),
        "source_extents": source_ext,
        "target_extents": target_ext,
        "fixed_extents": extents(paths["override_object"] / "visual.obj"),
        "scale": scale,
        "axis_order": axis_order,
        "axis_order_source": axis_order_source,
        "axis_order_note": "fixed_xyz = source_xyz[axis_order] * scale",
        "qpos_up_axis_stats": qpos_up_axis_stats,
        "replace_with_target_object": bool(args.replace_with_target_object),
        "raw_dir": str(args.raw_dir),
        "rendered_robot_video": rendered_robot_video,
        "rendered_robot_trajectory": str(aligned_robot_trajectory) if aligned_robot_trajectory is not None else None,
        "rendered_scene": str(render_scene) if rendered_robot_video else None,
        "alignment": alignment,
        "object_ref_fallback": object_ref_fallback,
        "robot_rgb_overlay_policy": "prohibited",
    }
    (paths["override_root"] / "scale_fix_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
