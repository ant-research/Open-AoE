#!/usr/bin/env python3
"""Materialize a probe-local raw video from an exact reconstruction frame grid.

This utility is for retarget-only probes whose old ``raw.mp4`` was cleaned up
while the lossless ``all_frames/%06d.png`` reconstruction input remained.  It
never runs reconstruction and never edits the source reconstruction directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_frame_grid(frames_dir: Path) -> list[Path]:
    frames = sorted(
        path
        for path in frames_dir.glob("*.png")
        if path.stem.isdigit() and len(path.stem) == 6
    )
    if not frames:
        raise ValueError(f"no six-digit PNG frames found: {frames_dir}")
    expected = [f"{index:06d}.png" for index in range(len(frames))]
    actual = [path.name for path in frames]
    if actual != expected:
        raise ValueError(
            "all_frames must be a contiguous zero-based %06d.png grid; "
            f"expected={expected[:3]}...{expected[-3:]}, "
            f"actual={actual[:3]}...{actual[-3:]}"
        )
    return frames


def aggregate_frame_sha256(frames: list[Path]) -> str:
    """Hash ordered names and per-file SHA256 values without loading all frames."""

    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(frame)))
    return digest.hexdigest()


def parse_fps(value: str) -> float:
    return float(Fraction(value)) if value and value != "0/0" else 0.0


def probe_video(ffprobe_bin: str, path: Path) -> dict[str, Any]:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_read_frames,nb_frames,width,height,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    frame_count = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    fps = parse_fps(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0"))
    return {
        "command": command,
        "frame_count": frame_count,
        "fps": fps,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration_sec": float(stream.get("duration") or 0.0),
    }


def git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        status = run("status", "--short")
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(status),
            "status_short": status,
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status_short": None}


def materialize(
    source_dir: Path,
    output_dir: Path,
    *,
    fps: float | None = None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    force: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    source_dir = source_dir.expanduser().resolve()
    output_dir = Path(os.path.abspath(str(output_dir.expanduser())))
    if output_dir.is_symlink():
        raise ValueError("probe-local output-dir must not be a symlink")
    resolved_output = output_dir.resolve(strict=False)
    if (
        source_dir == resolved_output
        or resolved_output.is_relative_to(source_dir)
        or source_dir.is_relative_to(resolved_output)
    ):
        raise ValueError(
            "probe-local output-dir must be disjoint from source-dir "
            "(neither may contain the other)"
        )
    if not (source_dir / "config.json").is_file():
        raise FileNotFoundError(source_dir / "config.json")
    frames = exact_frame_grid(source_dir / "all_frames")
    input_manifest_path = source_dir / "fresh_input_manifest.json"
    input_manifest = (
        json.loads(input_manifest_path.read_text(encoding="utf-8"))
        if input_manifest_path.is_file()
        else {}
    )
    resolved_fps = float(fps or input_manifest.get("source_fps") or 0.0)
    if resolved_fps <= 0.0:
        duration = float(input_manifest.get("duration") or 0.0)
        if duration > 0.0:
            resolved_fps = len(frames) / duration
    if resolved_fps <= 0.0:
        raise ValueError("unable to determine a positive source FPS")

    if output_dir.exists() and not force:
        raise FileExistsError(f"output-dir exists: {output_dir}; pass --force")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output-dir exists and is not a directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.materialize.",
            dir=str(output_dir.parent),
        )
    )
    backup_dir: Path | None = None
    try:
        for child in sorted(source_dir.iterdir()):
            if child.name in {"raw.mp4", "raw_video_materialization.json"}:
                continue
            os.symlink(
                child.resolve(),
                stage_dir / child.name,
                target_is_directory=child.is_dir(),
            )

        staged_raw_video = stage_dir / "raw.mp4"
        temporary_video = stage_dir / "raw.tmp.mp4"
        final_raw_video = output_dir / "raw.mp4"
        ffmpeg_command = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            f"{resolved_fps:.12g}",
            "-start_number",
            "0",
            "-i",
            str(stage_dir / "all_frames" / "%06d.png"),
            "-frames:v",
            str(len(frames)),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(temporary_video),
        ]
        subprocess.run(ffmpeg_command, check=True)
        temporary_video.replace(staged_raw_video)
        video_probe = probe_video(ffprobe_bin, staged_raw_video)
        if video_probe["frame_count"] != len(frames):
            raise ValueError(
                "materialized frame count mismatch: "
                f"{video_probe['frame_count']} != {len(frames)}"
            )
        if abs(video_probe["fps"] - resolved_fps) > 1e-6:
            raise ValueError(
                f"materialized FPS mismatch: {video_probe['fps']} != {resolved_fps}"
            )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "kind": "materialized_exact_all_frames",
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_reconstruction_dir": str(source_dir),
            "source_input_manifest": (
                {
                    "path": str(input_manifest_path.resolve()),
                    "sha256": sha256_file(input_manifest_path),
                }
                if input_manifest_path.is_file()
                else None
            ),
            "source_rgb": input_manifest.get("source_video"),
            "source_interval": {
                "start_frame": input_manifest.get("start_frame"),
                "end_frame": input_manifest.get("end_frame"),
                "start_sec": input_manifest.get("start_sec"),
                "duration_sec": input_manifest.get("duration"),
            },
            "all_frames": {
                "path": str((source_dir / "all_frames").resolve()),
                "frame_count": len(frames),
                "first": frames[0].name,
                "last": frames[-1].name,
                "aggregate_sha256": aggregate_frame_sha256(frames),
                "aggregate_hash_policy": (
                    "sha256(ordered basename NUL raw-file-sha256-bytes)"
                ),
            },
            "materialization": {
                "method": "ffmpeg_exact_png_grid",
                "command": ffmpeg_command,
                "fps": resolved_fps,
                "original_manifest_ffmpeg_command": input_manifest.get(
                    "ffmpeg_command"
                ),
            },
            "output": {
                "path": str(final_raw_video),
                "sha256": sha256_file(staged_raw_video),
                "frame_count": video_probe["frame_count"],
                "fps": video_probe["fps"],
                "width": video_probe["width"],
                "height": video_probe["height"],
                "duration_sec": video_probe["duration_sec"],
            },
            "ffprobe": video_probe,
            "git": git_state(repo_root or Path(__file__).resolve().parents[1]),
        }
        manifest_path = stage_dir / "raw_video_materialization.json"
        temporary_manifest = stage_dir / ".raw_video_materialization.json.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary_manifest.replace(manifest_path)

        if output_dir.exists():
            backup_dir = output_dir.parent / (
                f".{output_dir.name}.backup.{uuid.uuid4().hex}"
            )
            os.replace(output_dir, backup_dir)
        try:
            os.replace(stage_dir, output_dir)
        except BaseException:
            if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
                os.replace(backup_dir, output_dir)
                backup_dir = None
            raise
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir)
            backup_dir = None
        return manifest
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            os.replace(backup_dir, output_dir)


def attest_existing(
    source_dir: Path,
    *,
    ffprobe_bin: str = "ffprobe",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Bind a fresh reconstruction's existing raw video to its exact frame grid."""

    source_dir = source_dir.expanduser().resolve()
    raw_video = source_dir / "raw.mp4"
    input_manifest_path = source_dir / "fresh_input_manifest.json"
    if not raw_video.is_file():
        raise FileNotFoundError(raw_video)
    if not input_manifest_path.is_file():
        raise FileNotFoundError(input_manifest_path)
    frames = exact_frame_grid(source_dir / "all_frames")
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    resolved_fps = float(input_manifest.get("source_fps") or 0.0)
    if resolved_fps <= 0.0:
        raise ValueError("fresh input manifest must declare a positive source_fps")
    video_probe = probe_video(ffprobe_bin, raw_video)
    if video_probe["frame_count"] != len(frames):
        raise ValueError(
            "fresh raw video/frame-grid count mismatch: "
            f"{video_probe['frame_count']} != {len(frames)}"
        )
    if abs(video_probe["fps"] - resolved_fps) > 1e-6:
        raise ValueError(
            f"fresh raw video FPS mismatch: {video_probe['fps']} != {resolved_fps}"
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "kind": "materialized_exact_all_frames",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_reconstruction_dir": str(source_dir),
        "source_input_manifest": {
            "path": str(input_manifest_path),
            "sha256": sha256_file(input_manifest_path),
        },
        "source_rgb": input_manifest.get("source_video"),
        "source_interval": {
            "start_frame": input_manifest.get("start_frame"),
            "end_frame": input_manifest.get("end_frame"),
            "start_sec": input_manifest.get("start_sec"),
            "duration_sec": input_manifest.get("duration"),
        },
        "all_frames": {
            "path": str((source_dir / "all_frames").resolve()),
            "frame_count": len(frames),
            "first": frames[0].name,
            "last": frames[-1].name,
            "aggregate_sha256": aggregate_frame_sha256(frames),
            "aggregate_hash_policy": "sha256(ordered basename NUL raw-file-sha256-bytes)",
        },
        "materialization": {
            "method": "fresh_raw_and_exact_png_grid_attestation",
            "command": input_manifest.get("ffmpeg_command"),
            "fps": resolved_fps,
            "original_manifest_ffmpeg_command": input_manifest.get("ffmpeg_command"),
        },
        "output": {
            "path": str(raw_video),
            "sha256": sha256_file(raw_video),
            "frame_count": video_probe["frame_count"],
            "fps": video_probe["fps"],
            "width": video_probe["width"],
            "height": video_probe["height"],
            "duration_sec": video_probe["duration_sec"],
        },
        "ffprobe": video_probe,
        "git": git_state(repo_root or Path(__file__).resolve().parents[1]),
    }
    manifest_path = source_dir / "raw_video_materialization.json"
    temporary_manifest = source_dir / ".raw_video_materialization.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--attest-existing", action="store_true")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.attest_existing:
        if args.output_dir is not None:
            parser.error("--output-dir cannot be used with --attest-existing")
        if args.fps is not None or args.force:
            parser.error("--fps/--force cannot be used with --attest-existing")
        manifest = attest_existing(args.source_dir, ffprobe_bin=args.ffprobe_bin)
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required unless --attest-existing is used")
        manifest = materialize(
            args.source_dir,
            args.output_dir,
            fps=args.fps,
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
            force=args.force,
        )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
