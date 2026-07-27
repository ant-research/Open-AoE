#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def probe_duration(path: Path) -> float | None:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        value = float(proc.stdout.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument("--robot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--labels", default="RGB overlay|Depth/input|Robot retarget")
    parser.add_argument("--layout", choices=["vertical", "horizontal"], default="vertical")
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels = args.labels.split("|")
    labels = (labels + [""] * 3)[:3]
    durations = [probe_duration(path) for path in [args.overlay, args.depth, args.robot]]
    available_durations = [duration for duration in durations if duration is not None]
    effective_duration = args.duration if args.duration > 0 else (min(available_durations) if available_durations else 0)
    duration_args = []
    output_duration_args = []
    if effective_duration > 0:
        duration_args = ["-t", f"{effective_duration:.6f}"]
        output_duration_args = ["-frames:v", str(max(1, int(round(effective_duration * 15))))]
        trim_prefix = f"trim=duration={effective_duration:.6f},setpts=PTS-STARTPTS,fps=15,"
    else:
        trim_prefix = "fps=15,"

    stack = "vstack=inputs=3" if args.layout == "vertical" else "hstack=inputs=3"
    filtergraph = (
        f"[0:v]{trim_prefix}scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{labels[0]}':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.45[v0];"
        f"[1:v]{trim_prefix}scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{labels[1]}':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.45[v1];"
        f"[2:v]{trim_prefix}scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,"
        f"drawtext=text='{labels[2]}':x=12:y=12:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.45[v2];"
        f"[v0][v1][v2]{stack}[v]"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        *duration_args, "-i", str(args.overlay),
        *duration_args, "-i", str(args.depth),
        *duration_args, "-i", str(args.robot),
        "-filter_complex", filtergraph,
        "-map", "[v]", "-an", "-shortest", *output_duration_args,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-threads", "4",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    subprocess.run(cmd, check=True)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
