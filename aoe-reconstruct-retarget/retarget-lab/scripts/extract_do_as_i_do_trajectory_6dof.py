#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src, target_is_directory=src.is_dir())


def export_scaled_obj(src: Path, dst: Path, scale: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    out = []
    with src.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.rstrip("\n").split()
                xyz = [float(parts[i]) * scale for i in range(1, 4)]
                suffix = " " + " ".join(parts[4:]) if len(parts) > 4 else ""
                out.append(f"v {xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g}{suffix}\n")
            else:
                out.append(line)
    dst.write_text("".join(out), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Index Do-as-I-Do object 6DoF tracking assets from a reconstruction raw-dir.")
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument(
        "--allow-diagnostic-shim-scale",
        action="store_true",
        help="Allow legacy layouts whose metric mesh_scale was shimmed from local_to_scene.scale. "
        "Default is to fail because this is diagnostic-only and can create non-physical object scale.",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.expanduser().resolve()
    exp = args.experiment_root.expanduser().resolve()
    assets = exp / "assets" / "trajectory_6dof" / "do_as_i_do" / args.task
    assets.mkdir(parents=True, exist_ok=True)

    config = load_json(raw_dir / "config.json")
    object_id = config["object_names"][0]
    layout_path = raw_dir / "obj_tracking_out" / object_id / "combined_visualization" / "layout_camera_frame_optimized.json"
    layout = load_json(layout_path)
    scale_meta = layout.get("translation_scale_optimization") or {}
    if (
        scale_meta.get("method") == "shim_from_local_to_scene_scale"
        and not args.allow_diagnostic_shim_scale
    ):
        raise RuntimeError(
            "Refusing diagnostic shim scale in Do-as-I-Do 6DoF assets: "
            f"{layout_path} has translation_scale_optimization.method="
            "'shim_from_local_to_scene_scale'. Re-run reconstruction with real "
            "SAM3 masks + hand_anchored_pointmap scale optimization, or pass "
            "--allow-diagnostic-shim-scale only for debugging."
        )
    mesh_scale = float(scale_meta.get("mesh_scale") or 1.0)
    if mesh_scale <= 0:
        raise RuntimeError(f"invalid Do-as-I-Do mesh_scale={mesh_scale} in {layout_path}")

    max_idx = -1
    for obj in layout.get("objects", []):
        idx = int(obj.get("frame_idx", obj.get("frame_index", -1)))
        max_idx = max(max_idx, idx)
    if max_idx < 0:
        raise RuntimeError(f"No objects found in {layout_path}")

    qpos = np.zeros((max_idx + 1, 7), dtype=np.float32)
    valid = np.zeros((max_idx + 1,), dtype=bool)
    qpos[:, 3] = 1.0
    for obj in layout.get("objects", []):
        idx = int(obj.get("frame_idx", obj.get("frame_index")))
        ls = obj["local_to_scene"]
        qpos[idx, :3] = np.asarray(ls["translation_camera_frame"], dtype=np.float32)
        qpos[idx, 3:] = np.asarray(ls["quat_wxyz_camera_frame"], dtype=np.float32)
        valid[idx] = True

    np.savez(
        assets / "object_6dof.npz",
        qpos_obj=qpos,
        valid=valid,
        object_id=object_id,
        layout_path=str(layout_path),
        mesh_scale=np.asarray(mesh_scale, dtype=np.float32),
        scale_meta_json=np.asarray(json.dumps(scale_meta), dtype="<U2048"),
    )
    link_or_copy(layout_path, assets / "source_layout_camera_frame_optimized.json", args.mode)

    mesh_candidates = sorted(
        raw_dir.glob(f"video_segmentation/masks/frame_*_masks/{object_id}/{object_id}.obj")
    )
    if mesh_candidates:
        object_mesh = assets / "object_meshes" / "visual.obj"
        if abs(mesh_scale - 1.0) > 1e-6:
            export_scaled_obj(mesh_candidates[0], object_mesh, mesh_scale)
        else:
            link_or_copy(mesh_candidates[0], object_mesh, args.mode)

    hand_candidates = [
        raw_dir / "raw" / "all_hand_meshes.npz",
        raw_dir / "all_hand_meshes.npz",
        raw_dir / args.task / "all_hand_meshes.npz",
    ]
    for candidate in hand_candidates:
        if candidate.exists():
            link_or_copy(candidate, assets / "hand_meshes" / "all_hand_meshes.npz", args.mode)
            break

    mask_video_candidates = sorted((raw_dir / "video_segmentation").glob("tracked_*.mp4"))
    if mask_video_candidates:
        link_or_copy(mask_video_candidates[0], assets / "pre_sam3d_mask.mp4", args.mode)
    mask_qc = raw_dir / "video_segmentation" / "mask_qc_summary.json"
    if mask_qc.exists():
        link_or_copy(mask_qc, assets / "mask_qc_summary.json", args.mode)

    overlay_candidates = [
        raw_dir / f"output_tapir_{object_id}_overlay.mp4",
        raw_dir / f"output_tapir_{object_id}.mp4",
        raw_dir / "obj_tracking_out" / object_id / "combined_visualization" / "projected_hand_object_optimized" / "video.mp4",
        raw_dir / "obj_tracking_out" / object_id / "combined_visualization" / "projected" / "video.mp4",
        raw_dir / "output_tapir_foundation_jar.mp4",
        raw_dir / "raw.mp4",
    ]
    for candidate in overlay_candidates:
        if candidate.exists():
            link_or_copy(candidate, assets / "overlay.mp4", args.mode)
            break

    depth_candidates = [raw_dir / "moge_depth.mp4", raw_dir / "pointmap_depth.mp4", raw_dir / "raw.mp4"]
    for candidate in depth_candidates:
        if candidate.exists():
            link_or_copy(candidate, assets / "depth.mp4", args.mode)
            break

    manifest = {
        "trajectory_6dof": "do_as_i_do",
        "task": args.task,
        "raw_dir": str(raw_dir),
        "object_id": object_id,
        "object_6dof_npz": rel(assets / "object_6dof.npz", exp),
        "layout": rel(assets / "source_layout_camera_frame_optimized.json", exp),
        "object_mesh": rel(assets / "object_meshes" / "visual.obj", exp) if (assets / "object_meshes" / "visual.obj").exists() else None,
        "mesh_scale_baked_into_object_mesh": mesh_scale,
        "scale_meta": scale_meta,
        "pre_sam3d_mask": rel(assets / "pre_sam3d_mask.mp4", exp) if (assets / "pre_sam3d_mask.mp4").exists() else None,
        "mask_qc": rel(assets / "mask_qc_summary.json", exp) if (assets / "mask_qc_summary.json").exists() else None,
        "asset_root": rel(assets, exp),
    }
    (assets / "trajectory_6dof_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
