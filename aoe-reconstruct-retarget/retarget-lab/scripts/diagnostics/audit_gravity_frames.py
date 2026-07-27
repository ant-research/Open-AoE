#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

EXPECTED_CAMERA_UP = np.array([0.0, -1.0, 0.0], dtype=np.float64)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    au = unit(a.astype(np.float64))
    bu = unit(b.astype(np.float64))
    return math.degrees(math.acos(float(np.clip(np.dot(au, bu), -1.0, 1.0))))


def npz_summary(path: Path, keys: list[str]) -> dict:
    out: dict[str, object] = {"path": str(path)}
    if not path.exists():
        out["exists"] = False
        return out
    out["exists"] = True
    data = np.load(path, allow_pickle=False)
    for key in keys:
        if key not in data.files:
            continue
        arr = np.asarray(data[key])
        entry: dict[str, object] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        if arr.size and np.issubdtype(arr.dtype, np.number):
            xyz = arr[..., :3] if arr.ndim > 1 and arr.shape[-1] >= 3 else arr
            entry.update({
                "min_xyz": np.nanmin(xyz.reshape(-1, xyz.shape[-1]), axis=0).tolist() if xyz.ndim > 1 else [float(np.nanmin(xyz))],
                "max_xyz": np.nanmax(xyz.reshape(-1, xyz.shape[-1]), axis=0).tolist() if xyz.ndim > 1 else [float(np.nanmax(xyz))],
                "first": arr.reshape(-1, arr.shape[-1])[0].tolist() if arr.ndim and arr.shape[-1] else arr.reshape(-1)[:8].tolist(),
            })
        out[key] = entry
    return out


def scene_summary(path: Path) -> dict:
    out: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return out
    root = ET.parse(path).getroot()
    option = root.find("option")
    gravity = option.get("gravity") if option is not None else None
    out["gravity"] = gravity or "0 0 -9.81 (MuJoCo default)"
    cameras = []
    for cam in root.findall(".//camera"):
        cameras.append({"name": cam.get("name"), "pos": cam.get("pos"), "quat": cam.get("quat"), "mode": cam.get("mode")})
    out["cameras"] = cameras[:8]
    floor = root.find(".//geom[@name='floor']")
    out["has_floor"] = floor is not None
    return out


def infer_paths(run_root: Path, source_run_root: Path | None, task: str, hand_type: str, cell: str | None) -> dict[str, Path | None]:
    src = source_run_root or run_root
    clip_gravity = src / "intermediates/trajectory_6dof/do_as_i_do/reconstruction/clip/gravity.json"
    raw_gravity = src / "intermediates/trajectory_6dof/do_as_i_do/reconstruction/raw_dir/gravity.json"
    gravity = clip_gravity if clip_gravity.exists() else raw_gravity
    paths: dict[str, Path | None] = {
        "gravity": gravity,
        "clip_gravity": clip_gravity,
        "raw_gravity": raw_gravity,
        "layout": None,
        "dai_scene": src / f"intermediates/retargeting/do_as_i_do/traj_do_as_i_do__hand_estimated__retarget_do_as_i_do/retargeting_outputs/sharpa/{hand_type}/{task}/0/scene.xml",
        "dai_traj": src / f"intermediates/retargeting/do_as_i_do/traj_do_as_i_do__hand_estimated__retarget_do_as_i_do/retargeting_outputs/sharpa/{hand_type}/{task}/0/trajectory_mjwp.npz",
        "spider_scene": None,
        "spider_traj": None,
        "spider_task_info": None,
        "source_keypoints": run_root / f"assets/trajectory_6dof/do_as_i_do/{task}/source_trajectory_keypoints.npz",
    }
    layout_matches = sorted((src / "intermediates/trajectory_6dof/do_as_i_do/reconstruction/clip/obj_tracking_out").glob("*/combined_visualization/layout_camera_frame_optimized.json"))
    if layout_matches:
        paths["layout"] = layout_matches[0]
    if cell:
        spider = run_root / "intermediates/retargeting/spider" / cell
        paths["spider_scene"] = spider / "robot/scene.xml"
        paths["spider_traj"] = spider / "robot/trajectory_mjwp.npz"
        paths["spider_task_info"] = spider / f"dataset/processed/do_as_i_do/xhand/{hand_type}/{task}/task_info.json"
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit camera-frame gravity and MuJoCo retarget frame consistency.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--source-run-root", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--hand-type", default="left")
    parser.add_argument("--spider-cell")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    source_root = args.source_run_root.resolve() if args.source_run_root else None
    paths = infer_paths(run_root, source_root, args.task, args.hand_type, args.spider_cell)
    report: dict[str, object] = {"run_root": str(run_root), "source_run_root": str(source_root) if source_root else None, "paths": {k: str(v) if v else None for k, v in paths.items()}}

    gravity_path = paths["gravity"]
    gravity_report: dict[str, object] = {"exists": bool(gravity_path and gravity_path.exists())}
    if gravity_path and gravity_path.exists():
        gravity = load_json(gravity_path)
        vec = np.asarray(gravity.get("vec3d"), dtype=np.float64)
        gravity_report.update({
            "vec3d": vec.tolist(),
            "frame": gravity.get("frame"),
            "semantics": gravity.get("semantics", "world_up_direction_in_camera_frame (Do-as-I-Do process_dataset expectation)"),
            "angle_to_expected_upright_camera_up_deg": angle_deg(vec, EXPECTED_CAMERA_UP),
            "legacy_z_forward_identity": bool(np.allclose(unit(vec), np.array([0.0, 0.0, 1.0]))),
            "note": gravity.get("note"),
        })
    report["gravity"] = gravity_report

    layout_path = paths["layout"]
    layout_report: dict[str, object] = {"exists": bool(layout_path and layout_path.exists())}
    if layout_path and layout_path.exists():
        layout = load_json(layout_path)
        objs = layout.get("objects", [])
        translations = [obj.get("local_to_scene", {}).get("translation_camera_frame") for obj in objs]
        translations = [t for t in translations if t is not None]
        layout_report.update({"frame": layout.get("frame"), "note": layout.get("note"), "num_objects": len(objs)})
        if translations:
            arr = np.asarray(translations, dtype=np.float64)
            layout_report["translation_camera_frame_min"] = arr.min(axis=0).tolist()
            layout_report["translation_camera_frame_max"] = arr.max(axis=0).tolist()
    report["layout"] = layout_report

    report["dai_scene"] = scene_summary(paths["dai_scene"]) if paths["dai_scene"] else None
    report["spider_scene"] = scene_summary(paths["spider_scene"]) if paths["spider_scene"] else None
    report["dai_traj"] = npz_summary(paths["dai_traj"], ["qpos", "trace_ref"]) if paths["dai_traj"] else None
    report["spider_traj"] = npz_summary(paths["spider_traj"], ["qpos", "trace_ref"]) if paths["spider_traj"] else None
    report["source_keypoints"] = npz_summary(paths["source_keypoints"], [f"qpos_obj_{args.hand_type}", f"qpos_wrist_{args.hand_type}", f"mano_verts_{args.hand_type}"])

    issues: list[str] = []
    if gravity_report.get("legacy_z_forward_identity"):
        issues.append("gravity.json uses legacy [0,0,1] camera-forward vector as up; upright camera fallback should be [0,-1,0].")
    if layout_report.get("frame") not in (None, "camera_frame"):
        issues.append(f"layout frame is {layout_report.get('frame')!r}, expected camera_frame before gravity alignment.")
    report["issues"] = issues

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"run_root: {run_root}")
        print(f"gravity: {gravity_report}")
        print(f"layout: {layout_report}")
        print(f"dai_scene_gravity: {report['dai_scene']['gravity'] if report['dai_scene'] else None}")
        if report.get("spider_scene"):
            print(f"spider_scene_gravity: {report['spider_scene']['gravity']}")
        if issues:
            print("issues:")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("issues: none")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
