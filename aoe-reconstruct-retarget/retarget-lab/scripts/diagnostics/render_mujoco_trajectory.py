#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mujoco
import numpy as np


def ffmpeg_writer(path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def load_qpos(
    path: Path,
    nq: int,
    qpos0: np.ndarray,
    *,
    allow_shape_coercion: bool = False,
) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "qpos" not in data:
        raise KeyError(f"{path} does not contain qpos")
    qpos = np.asarray(data["qpos"], dtype=np.float64).reshape(-1, data["qpos"].shape[-1])
    if qpos.shape[1] != nq and not allow_shape_coercion:
        raise ValueError(
            f"qpos width={qpos.shape[1]} does not match model.nq={nq}; "
            "select the trajectory's matching scene instead of padding or truncating it"
        )
    if qpos.shape[1] < nq:
        padded = np.broadcast_to(np.asarray(qpos0, dtype=np.float64), (len(qpos), nq)).copy()
        padded[:, : qpos.shape[1]] = qpos
        qpos = padded
    if qpos.shape[1] > nq:
        qpos = qpos[:, :nq]
    return qpos


def resample_rows(arr: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or len(arr) == n:
        return arr
    if len(arr) == 0:
        return arr
    idx = np.rint(np.linspace(0, len(arr) - 1, n)).astype(int)
    return arr[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a MuJoCo qpos trajectory to mp4.")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument(
        "--target-frames",
        type=int,
        default=0,
        help="When >0, resample the qpos trajectory to exactly this many rendered frames. Overrides --stride.",
    )
    parser.add_argument("--camera", default=None)
    parser.add_argument(
        "--hide-geom-groups",
        default="",
        help="Comma-separated MuJoCo geom group ids to hide while rendering, e.g. 3 for collision/support geoms.",
    )
    parser.add_argument("--disable-shadows", action="store_true")
    parser.add_argument(
        "--allow-qpos-shape-coercion",
        action="store_true",
        help="Legacy diagnostics only: pad/truncate qpos to model.nq instead of failing.",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    if args.disable_shadows:
        model.vis.quality.shadowsize = 0
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    qpos = load_qpos(
        args.trajectory,
        model.nq,
        model.qpos0,
        allow_shape_coercion=args.allow_qpos_shape_coercion,
    )
    if args.target_frames > 0:
        qpos_to_render = resample_rows(qpos, args.target_frames)
    else:
        qpos_to_render = qpos[:: max(args.stride, 1)]
    scene_option = mujoco.MjvOption()
    if args.hide_geom_groups:
        for item in args.hide_geom_groups.split(","):
            item = item.strip()
            if not item:
                continue
            group = int(item)
            if group < 0 or group >= len(scene_option.geomgroup):
                raise ValueError(f"geom group out of range: {group}")
            scene_option.geomgroup[group] = 0

    camera = args.camera
    if camera is None and model.ncam > 0:
        camera = 0

    proc = ffmpeg_writer(args.output, args.width, args.height, args.fps)
    assert proc.stdin is not None
    try:
        for frame_qpos in qpos_to_render:
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            if camera is None:
                renderer.update_scene(data, scene_option=scene_option)
            else:
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
            frame = renderer.render()
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
        renderer.close()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {rc}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
