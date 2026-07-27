#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np


FRAME_KEYS = (
    "qpos_wrist_left",
    "qpos_wrist_right",
    "qpos_obj_left",
    "qpos_obj_right",
    "mano_verts_left",
    "mano_verts_right",
)


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def keypoint_frame_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        for key in FRAME_KEYS:
            if key in data:
                value = np.asarray(data[key])
                if value.ndim > 0 and value.shape[0] > 0:
                    return int(value.shape[0])
    raise ValueError(f"cannot determine source frame count from {path}")


def source_run_from_matrix(experiment_root: Path) -> str | None:
    manifest = experiment_root / "reuse_12_demo_manifest.json"
    if not manifest.exists():
        return None
    value = str(load_json(manifest).get("source_run") or "").strip()
    return value or None


def experiment_from_keypoints(repo_root: Path, keypoints: Path) -> Path | None:
    try:
        relative = keypoints.resolve().relative_to((repo_root / "experiments").resolve())
    except (OSError, ValueError):
        return None
    return repo_root / "experiments" / relative.parts[0] if relative.parts else None


def metadata_candidates(repo_root: Path, experiment_root: Path, keypoints: Path) -> list[Path]:
    roots: list[Path] = []
    source_run = source_run_from_matrix(experiment_root)
    if source_run:
        roots.append(repo_root / "experiments" / source_run)
    keypoint_experiment = experiment_from_keypoints(repo_root, keypoints)
    if keypoint_experiment is not None:
        roots.append(keypoint_experiment)
    roots.append(experiment_root)

    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        path = root / "intermediates/trajectory_6dof/do_as_i_do/reconstruction/clip/metadata.json"
        if path not in seen:
            candidates.append(path)
            seen.add(path)
    return candidates


def exact_route_video_candidates(keypoints: Path) -> list[Path]:
    """Return raw videos bound to the DAI cell that owns ``keypoints``."""

    resolved = keypoints.expanduser().resolve()
    candidates: list[Path] = []
    for parent in resolved.parents:
        if parent.name != "retargeting_outputs":
            continue
        cell_root = parent.parent
        candidates.extend(
            [
                cell_root / "raw_dir" / "raw.mp4",
                cell_root / "raw_dir" / "raw" / "raw.mp4",
            ]
        )
        break
    seen: set[Path] = set()
    return [path for path in candidates if not (path in seen or seen.add(path))]


def video_fps(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError(f"ffprobe did not return one video stream for {path}")
    stream = streams[0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = str(stream.get(key) or "").strip()
        if not value or value in {"0/0", "N/A"}:
            continue
        fps = float(Fraction(value))
        if np.isfinite(fps) and fps > 0:
            return fps
    raise ValueError(f"ffprobe returned no positive frame rate for {path}")


def required_video_fps(path: Path) -> float:
    """Return a verified video FPS without a guessed/default fallback."""

    if not path.is_file():
        raise FileNotFoundError(f"required source video is missing: {path}")
    return video_fps(path)


def resolve_source_fps(
    repo_root: Path,
    experiment_root: Path,
    keypoints: Path,
    explicit_source_fps: float = 0.0,
) -> tuple[float, str]:
    if explicit_source_fps > 0:
        return float(explicit_source_fps), "explicit"
    for path in metadata_candidates(repo_root, experiment_root, keypoints):
        if not path.exists():
            continue
        fps = float(load_json(path).get("source_fps") or 0.0)
        if fps > 0:
            return fps, str(path)
    for path in exact_route_video_candidates(keypoints):
        if path.is_file():
            return required_video_fps(path), f"ffprobe_exact_route_raw_video:{path.resolve()}"
    raise ValueError(
        "cannot resolve SPIDER source FPS from fresh clip metadata or the "
        "exact-route DAI raw video"
    )


def build_timing(
    *,
    repo_root: Path,
    experiment_root: Path,
    keypoints: Path,
    sim_dt: float,
    requested_max_sim_steps: int,
    explicit_source_fps: float = 0.0,
    explicit_ref_dt: float = 0.0,
    substeps_per_frame: int = 6,
) -> dict:
    if substeps_per_frame <= 0:
        raise ValueError(
            f"substeps_per_frame must be positive, got {substeps_per_frame}"
        )
    frames = keypoint_frame_count(keypoints)
    source_fps, source_fps_provenance = resolve_source_fps(
        repo_root,
        experiment_root,
        keypoints,
        explicit_source_fps,
    )
    ref_dt = float(explicit_ref_dt) if explicit_ref_dt > 0 else 1.0 / source_fps
    source_ref_dt = 1.0 / source_fps
    if not np.isclose(ref_dt, source_ref_dt, atol=1.0e-9, rtol=1.0e-7):
        raise ValueError(
            "SPIDER ref_dt must match the exact source FPS: "
            f"ref_dt={ref_dt}, source_ref_dt={source_ref_dt}"
        )
    if sim_dt <= 0:
        sim_dt = ref_dt / int(substeps_per_frame)
        sim_dt_provenance = "ref_dt_divided_by_integer_substeps"
    else:
        ratio = ref_dt / float(sim_dt)
        rounded = int(round(ratio))
        if rounded <= 0 or not np.isclose(ratio, rounded, atol=1.0e-7, rtol=0.0):
            raise ValueError(
                "SPIDER ref_dt/sim_dt must be an integer to preserve the source "
                f"timeline exactly: ratio={ratio}"
            )
        substeps_per_frame = rounded
        sim_dt_provenance = "explicit_integer_ratio"
    target_duration_sec = max(frames - 1, 0) / source_fps
    reference_duration_sec = max(frames - 1, 0) * ref_dt
    # MJWP max_sim_steps counts integration intervals.  The patched SPIDER
    # saver additionally records the initial t=0 state, so K intervals produce
    # K+1 samples and the exact final source timestamp needs no extra step.
    required_sim_steps = max(frames - 1, 0) * int(substeps_per_frame)
    reference_sample_count = required_sim_steps + 1
    simulation_duration_sec = required_sim_steps * float(sim_dt)
    if requested_max_sim_steps > 0 and int(requested_max_sim_steps) != required_sim_steps:
        raise ValueError(
            "SPIDER max_sim_steps must equal the exact source-timeline budget: "
            f"requested={requested_max_sim_steps}, required={required_sim_steps}"
        )
    effective_max_sim_steps = required_sim_steps
    return {
        "source_frame_count": frames,
        "source_fps": source_fps,
        "source_fps_provenance": source_fps_provenance,
        "ref_dt": ref_dt,
        "ref_dt_provenance": "explicit" if explicit_ref_dt > 0 else "inverse_source_fps",
        "sim_dt": float(sim_dt),
        "sim_dt_provenance": sim_dt_provenance,
        "substeps_per_frame": int(substeps_per_frame),
        "target_duration_sec": target_duration_sec,
        "reference_duration_sec": reference_duration_sec,
        "simulation_duration_sec": simulation_duration_sec,
        "reference_duration_coverage": (
            reference_duration_sec / target_duration_sec if target_duration_sec > 0 else 1.0
        ),
        "required_sim_steps": required_sim_steps,
        "reference_sample_count": reference_sample_count,
        "requested_max_sim_steps": int(requested_max_sim_steps),
        "effective_max_sim_steps": effective_max_sim_steps,
        "max_sim_steps_expanded": int(requested_max_sim_steps) <= 0,
        "exact_endpoint_timeline": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve source-timeline timing for SPIDER IK and MJWP.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--source-keypoints", type=Path, required=True)
    parser.add_argument("--sim-dt", type=float, required=True)
    parser.add_argument("--max-sim-steps", type=int, default=-1)
    parser.add_argument("--source-fps", type=float, default=0.0)
    parser.add_argument("--ref-dt", type=float, default=0.0)
    parser.add_argument("--substeps-per-frame", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    timing = build_timing(
        repo_root=args.repo_root,
        experiment_root=args.experiment_root,
        keypoints=args.source_keypoints,
        sim_dt=args.sim_dt,
        requested_max_sim_steps=args.max_sim_steps,
        explicit_source_fps=args.source_fps,
        explicit_ref_dt=args.ref_dt,
        substeps_per_frame=args.substeps_per_frame,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(timing, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
