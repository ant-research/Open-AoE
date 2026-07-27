#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_probe(path: Path) -> dict[str, float | int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = (json.loads(result.stdout) or {}).get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream: {path}")
    stream = streams[0]
    frame_count = int(stream["nb_read_frames"])
    fps = next(
        float(Fraction(str(stream[key])))
        for key in ("avg_frame_rate", "r_frame_rate")
        if str(stream.get(key) or "") not in {"", "0/0", "N/A"}
    )
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"invalid source video grid: frames={frame_count}, fps={fps}")
    return {"frame_count": frame_count, "fps": fps}


def flattened_time_series(value: np.ndarray, count: int) -> np.ndarray | None:
    array = np.asarray(value)
    if array.ndim >= 1 and array.shape[0] == count:
        return array
    if array.ndim >= 2 and int(np.prod(array.shape[:2])) == count:
        return array.reshape((count,) + array.shape[2:])
    return None


def interpolate_rows(
    value: np.ndarray,
    source_time: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fci":
        indices = np.searchsorted(source_time, target_time, side="left")
        indices = np.clip(indices, 0, len(source_time) - 1)
        return array[indices]
    right = np.searchsorted(source_time, target_time, side="left")
    right = np.clip(right, 0, len(source_time) - 1)
    left = np.maximum(right - 1, 0)
    denominator = source_time[right] - source_time[left]
    alpha = np.divide(
        target_time - source_time[left],
        denominator,
        out=np.zeros_like(target_time),
        where=denominator > 0,
    )
    shape = (len(alpha),) + (1,) * (array.ndim - 1)
    result = array[left] * (1.0 - alpha.reshape(shape)) + array[right] * alpha.reshape(shape)
    return result.astype(array.dtype, copy=False)


def normalize_qpos_quaternions(scene: Path, qpos: np.ndarray) -> np.ndarray:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene))
    if qpos.shape[1] != model.nq:
        raise RuntimeError(f"qpos width={qpos.shape[1]} does not match scene nq={model.nq}")
    normalized = np.asarray(qpos, dtype=np.float64).copy()
    for row in normalized:
        mujoco.mj_normalizeQuat(model, row)
    return normalized.astype(qpos.dtype, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize strict plain review artifacts from an untouched pristine DAI rollout."
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--hand-type", required=True, choices=("left", "right", "bimanual"))
    parser.add_argument("--trajectory-6dof", required=True, choices=("do_as_i_do", "egoinfinity"))
    parser.add_argument("--hand-source", required=True, choices=("aoe", "estimated"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--robot", default="sharpa")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    raw_dir = args.raw_dir.resolve()
    canonical_raw_dir = output_root.parent / "raw_dir"
    if canonical_raw_dir.is_symlink():
        if canonical_raw_dir.resolve() != raw_dir:
            canonical_raw_dir.unlink()
    elif canonical_raw_dir.exists() and canonical_raw_dir.resolve() != raw_dir:
        raise SystemExit(
            f"canonical raw-dir exists with a different binding: {canonical_raw_dir}"
        )
    if not canonical_raw_dir.exists():
        canonical_raw_dir.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(raw_dir, canonical_raw_dir, target_is_directory=True)
    run_dir = output_root / args.robot / args.hand_type / args.task / "0"
    source_video = raw_dir / "raw.mp4"
    source_trajectory = run_dir / "trajectory_mjwp.npz"
    scene = run_dir / "scene_act.xml"
    legacy_scene = run_dir / "scene.xml"
    if not scene.is_file() and legacy_scene.is_file():
        # Pinned pristine DAI writes the actuator-capable scene under the
        # historical scene.xml name.  Keep the native byte stream untouched and
        # install the canonical review alias expected by strict Lab admission.
        shutil.copy2(legacy_scene, scene)
    config_path = run_dir / "config.yaml"
    for label, path in (
        ("source video", source_video),
        ("source trajectory", source_trajectory),
        ("actuator scene", scene),
        ("DAI config", config_path),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")

    probe = video_probe(source_video)
    target_count = int(probe["frame_count"])
    target_fps = float(probe["fps"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    warmup_steps = int(config.get("warmup_steps") or 0)
    sim_dt = float(config.get("sim_dt") or 0.0)
    if warmup_steps <= 0 or sim_dt <= 0:
        raise SystemExit(f"invalid pristine DAI timing: warmup_steps={warmup_steps}, sim_dt={sim_dt}")

    source = np.load(source_trajectory, allow_pickle=True)
    qpos_raw = np.asarray(source["qpos"])
    qpos_count = int(np.prod(qpos_raw.shape[:-1]))
    raw_time_series = flattened_time_series(np.asarray(source["time"]), qpos_count)
    if raw_time_series is None:
        raise SystemExit("DAI trajectory time does not align with qpos")
    source_time = np.asarray(raw_time_series, dtype=np.float64).reshape(-1)
    if len(source_time) != qpos_count or np.any(np.diff(source_time) <= 0):
        raise SystemExit("DAI trajectory time is not a strict increasing grid")

    # Upstream records post-step states and omits the initial state.  The source
    # t=0 pose is therefore the final warmup state at raw index warmup_steps-1.
    start_index = warmup_steps - 1
    start_time = float(source_time[start_index])
    target_time = start_time + np.arange(target_count, dtype=np.float64) / target_fps
    tolerance = max(sim_dt * 1.0e-6, 1.0e-9)
    if target_time[-1] > source_time[-1] + tolerance:
        raise SystemExit(
            f"pristine rollout does not cover source endpoint: target={target_time[-1]}, "
            f"available={source_time[-1]}"
        )

    payload: dict[str, np.ndarray] = {}
    interpolated_keys: list[str] = []
    for key in source.files:
        series = flattened_time_series(np.asarray(source[key]), qpos_count)
        if series is None:
            payload[key] = np.asarray(source[key])
            continue
        payload[key] = interpolate_rows(series, source_time, target_time)
        interpolated_keys.append(key)
    payload["time"] = np.arange(target_count, dtype=np.float64) / target_fps
    payload["qpos"] = normalize_qpos_quaternions(scene, np.asarray(payload["qpos"]))

    aligned_trajectory = run_dir / "trajectory_mjwp_act_aligned.npz"
    aligned_video = run_dir / "visualization_mjwp_act_aligned.mp4"
    alignment_manifest = run_dir / "mjwp_alignment_manifest.json"
    for path in (aligned_trajectory, aligned_video, alignment_manifest):
        path.unlink(missing_ok=True)
    np.savez(aligned_trajectory, **payload)

    subprocess.run(
        [
            args.python_bin,
            str(repo_root / "scripts" / "diagnostics" / "render_mujoco_trajectory.py"),
            "--scene",
            str(scene),
            "--trajectory",
            str(aligned_trajectory),
            "--output",
            str(aligned_video),
            "--width",
            "640",
            "--height",
            "480",
            "--fps",
            str(target_fps),
            "--target-frames",
            str(target_count),
        ],
        cwd=repo_root,
        check=True,
    )
    rendered = video_probe(aligned_video)
    if rendered != probe:
        raise SystemExit(f"aligned video grid mismatch: source={probe}, rendered={rendered}")

    nearest = np.minimum(
        np.searchsorted(source_time, target_time, side="left"),
        len(source_time) - 1,
    )
    nearest_error = np.abs(source_time[nearest] - target_time)
    manifest = {
        "schema_version": 2,
        "applied": True,
        "reason": "drop_warmup_and_interpolate_exact_source_timestamps",
        "alignment_kind": "drop_warmup_and_sample_exact_source_timestamps",
        "sampling_semantics": "exact_source_timestamp_interpolation",
        "exact_source_timestamps": True,
        "complete_source_coverage": True,
        "start_index": start_index,
        "warmup_steps": warmup_steps,
        "source_frame_count": target_count,
        "target_count": target_count,
        "target_frames": target_count,
        "target_fps": target_fps,
        "render_fps": target_fps,
        "robot_render_fps": target_fps,
        "render_dt": 1.0 / target_fps,
        "trace_dt": 1.0 / target_fps,
        "sim_dt": sim_dt,
        "source_trajectory": str(source_trajectory),
        "aligned_trajectory": str(aligned_trajectory),
        "aligned_video": str(aligned_video),
        "rendered_video": str(aligned_video),
        "render_scene": str(scene),
        "interpolated_keys": interpolated_keys,
        "max_nearest_sim_sample_error_sec": float(nearest_error.max()),
        "artifact_bindings": {
            "aligned_video": {
                "path": str(aligned_video),
                "sha256": sha256(aligned_video),
                "frame_count": int(rendered["frame_count"]),
                "fps": float(rendered["fps"]),
            },
            "aligned_trajectory": {
                "path": str(aligned_trajectory),
                "sha256": sha256(aligned_trajectory),
            },
            "render_scene": {"path": str(scene), "sha256": sha256(scene)},
            "source_trajectory": {
                "path": str(source_trajectory),
                "sha256": sha256(source_trajectory),
            },
            "source_video": {
                "path": str(source_video),
                "sha256": sha256(source_video),
                "frame_count": target_count,
                "fps": target_fps,
            },
        },
    }
    alignment_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
