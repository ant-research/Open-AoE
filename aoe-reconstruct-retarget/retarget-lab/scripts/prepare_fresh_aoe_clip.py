#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(os.environ["AOE_DATA_ROOT"]) if os.environ.get("AOE_DATA_ROOT") else None
DEFAULT_BBOX_FORMAT = "xyxy"


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def object_id(name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", name.strip().lower()).strip("_")
    if not value:
        raise ValueError(f"cannot derive object id from {name!r}")
    return value


def first_action(annotation: dict) -> dict:
    actions = annotation.get("atomic_action") or []
    if not actions:
        raise ValueError(f"annotation has no atomic_action: {annotation}")
    return actions[0]


def find_annotation(segment_dir: Path, annotation_id: int) -> dict:
    path = segment_dir / "ego_annotation" / "ego_action_annotation.json"
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"expected annotation list: {path}")
    for row in rows:
        if int(row.get("id")) == annotation_id:
            return row
    raise KeyError(f"annotation id {annotation_id} not found in {path}")


def ffprobe_fps(video: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    if "/" in out:
        num, den = out.split("/", 1)
        return float(num) / float(den)
    return float(out)


def ffprobe_size(video: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=p=0:s=x",
        str(video),
    ]
    width, height = subprocess.check_output(cmd, text=True).strip().split("x", 1)
    return int(width), int(height)


def ffprobe_frame_count_and_duration(video: Path) -> tuple[int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,duration",
        "-of",
        "json",
        str(video),
    ]
    payload = json.loads(subprocess.check_output(cmd, text=True))
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"expected one video stream in {video}, got {len(streams)}")
    stream = streams[0]
    raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if raw_count in {None, "N/A"}:
        raise ValueError(f"ffprobe did not report a frame count for {video}")
    frame_count = int(raw_count)
    duration = float(stream.get("duration") or 0.0)
    if frame_count <= 0 or duration <= 0.0:
        raise ValueError(
            f"invalid encoded clip metadata for {video}: "
            f"frame_count={frame_count}, duration={duration}"
        )
    return frame_count, duration


def select_reference_frame(
    *,
    actual_num_frames: int,
    start_frame: int,
    ref_source_frame: int,
    ref_frame_fraction: float,
    verb: str,
) -> tuple[int, float, str]:
    if actual_num_frames <= 0:
        raise ValueError("actual output frame count must be positive")
    if ref_source_frame >= 0:
        ref_frame_in_clip = ref_source_frame - start_frame
        if not 0 <= ref_frame_in_clip < actual_num_frames:
            raise ValueError(
                f"ref source frame {ref_source_frame} is outside actual encoded clip "
                f"[{start_frame}, {start_frame + actual_num_frames})"
            )
        fraction = ref_frame_in_clip / max(actual_num_frames - 1, 1)
        return ref_frame_in_clip, fraction, "explicit_audited_source_frame"
    if ref_frame_fraction < 0:
        fraction, strategy = automatic_ref_frame_fraction(verb)
    else:
        fraction = min(max(float(ref_frame_fraction), 0.0), 1.0)
        strategy = "explicit_fraction"
    ref_frame_in_clip = max(
        0,
        min(actual_num_frames - 1, int(round((actual_num_frames - 1) * fraction))),
    )
    return ref_frame_in_clip, fraction, strategy


def annotation_size(
    segment_dir: Path,
    fallback: tuple[int, int],
    mode: str,
) -> tuple[int, int, str]:
    if mode == "normalized-1000":
        return 1000, 1000, "normalized_0_1000"
    if mode == "source-video":
        return fallback[0], fallback[1], "ffprobe_source_video"
    info_path = segment_dir / "video_info.json"
    try:
        info = load_json(info_path)
        camera = info.get("cameraParams") if isinstance(info, dict) else None
        resolution = str((camera or {}).get("resolution") or "")
        match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", resolution)
        if match:
            return int(match.group(1)), int(match.group(2)), str(info_path)
    except Exception:
        pass
    return fallback[0], fallback[1], "ffprobe_source_video_fallback"


def transform_bbox(
    bbox: object,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    bbox_format: str = "xywh",
) -> list[float] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    values = [float(value) for value in bbox]
    scale_x = target_size[0] / max(float(source_size[0]), 1.0)
    scale_y = target_size[1] / max(float(source_size[1]), 1.0)
    if bbox_format == "xywh":
        x1, y1, width, height = values
        values = [x1, y1, x1 + width, y1 + height]
    elif bbox_format != "xyxy":
        raise ValueError(f"unsupported bbox format: {bbox_format}")
    return [values[0] * scale_x, values[1] * scale_y, values[2] * scale_x, values[3] * scale_y]


def quantize_shared_duration(duration: float, ego_fps: float) -> float:
    if duration <= 0.0 or ego_fps <= 0.0:
        raise ValueError("duration and Ego fps must be positive")
    return max(1, int(round(duration * ego_fps))) / ego_fps


def automatic_ref_frame_fraction(verb: str) -> tuple[float, str]:
    normalized = verb.strip().lower()
    if normalized in {"uncap", "open", "remove"}:
        return 0.10, f"verb_{normalized}_early_visibility"
    if normalized in {"place", "put_down", "set_down"}:
        return 0.25, f"verb_{normalized}_pre_placement_visibility"
    if normalized in {
        "cap",
        "close",
        "grasp",
        "hold",
        "insert",
        "lift",
        "move",
        "put",
    }:
        return 0.85, f"verb_{normalized}_late_visibility"
    return 0.50, "verb_default_midpoint"


def clip_window(
    annotation_start_sec: float,
    annotation_end_sec: float,
    start_offset_sec: float,
    duration_sec: float,
    extra_sec_after: float,
) -> tuple[float, float]:
    if start_offset_sec < 0:
        raise ValueError("start offset must be non-negative")
    start_sec = annotation_start_sec + start_offset_sec
    if duration_sec > 0:
        duration = duration_sec
    else:
        duration = max(0.0, annotation_end_sec - start_sec)
    duration += extra_sec_after
    if duration <= 0:
        raise ValueError(
            f"invalid clip window after {start_offset_sec:.3f}s start offset: duration={duration:.3f}s"
        )
    return start_sec, duration


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a clean AoE RGB clip for a fresh 12-demo source run.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--annotation-id", required=True, type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--object-name")
    parser.add_argument("--anchor-hand", required=True, choices=["left", "right", "bimanual"])
    parser.add_argument("--duration-sec", type=float, default=0.0, help="Override annotation duration when > 0.")
    parser.add_argument(
        "--start-offset-sec",
        type=float,
        default=0.0,
        help="Skip this many seconds from the annotation start before extracting the fresh clip.",
    )
    parser.add_argument("--extra-sec-after", type=float, default=0.0)
    parser.add_argument("--scale-width", type=int, default=1280, help="Set 0 to keep original width.")
    parser.add_argument(
        "--bbox-coordinate-space",
        choices=["normalized-1000", "video-info", "source-video"],
        default="normalized-1000",
        help="AoE 3-hour action bboxes use normalized 0..1000 coordinates; other modes are explicit compatibility options.",
    )
    parser.add_argument(
        "--bbox-format",
        choices=["xywh", "xyxy"],
        default=DEFAULT_BBOX_FORMAT,
        help="AoE annotations use [x1,y1,x2,y2]; xywh is an explicit compatibility mode.",
    )
    parser.add_argument("--ego-fps", type=float, default=15.0)
    parser.add_argument(
        "--ref-frame-fraction",
        type=float,
        default=-1.0,
        help="Reference position in [0,1]. Negative selects a deterministic verb-aware visibility heuristic.",
    )
    parser.add_argument(
        "--ref-source-frame",
        type=int,
        default=-1,
        help="Audited absolute source-video frame; overrides --ref-frame-fraction.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.dataset_root is None:
        parser.error("set AOE_DATA_ROOT or pass --dataset-root")
    dataset_root = args.dataset_root.expanduser().resolve()
    segment_dir = dataset_root / args.segment
    source_video = segment_dir / "ego_process" / "ego_undistorted_video" / "raw_video_undistorted.mp4"
    hands_npz = segment_dir / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
    if not source_video.exists():
        raise FileNotFoundError(source_video)
    if not hands_npz.exists():
        raise FileNotFoundError(hands_npz)

    annotation = find_annotation(segment_dir, args.annotation_id)
    annotation_action = first_action(annotation)
    action = dict(annotation_action)
    obj_name = args.object_name or str(action.get("object") or "")
    obj_id = object_id(obj_name)
    annotation_start_sec = float(annotation.get("start_ts", 0.0))
    annotation_end_sec = float(annotation.get("end_ts", annotation_start_sec))
    start_sec, requested_duration = clip_window(
        annotation_start_sec,
        annotation_end_sec,
        args.start_offset_sec,
        args.duration_sec,
        args.extra_sec_after,
    )
    duration = quantize_shared_duration(requested_duration, args.ego_fps)

    source_fps = ffprobe_fps(source_video)
    source_video_size = ffprobe_size(source_video)
    annotation_width, annotation_height, annotation_size_source = annotation_size(
        segment_dir,
        source_video_size,
        args.bbox_coordinate_space,
    )
    start_frame = int(round(start_sec * source_fps))
    num_frames = int(round(duration * source_fps))
    requested_end_frame = start_frame + num_frames

    out_dir = args.output_dir or (
        REPO_ROOT
        / "experiments"
        / args.run_name
        / "intermediates"
        / "trajectory_6dof"
        / "do_as_i_do"
        / "reconstruction"
        / "clip"
    )
    out_dir = out_dir.expanduser().resolve()
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"{out_dir} exists; pass --force to recreate only for a fresh run")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_mp4 = out_dir / "raw.mp4"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration:.6f}",
    ]
    if args.scale_width > 0:
        ffmpeg_cmd += ["-vf", f"scale={args.scale_width}:-2"]
    ffmpeg_cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(raw_mp4)]
    run(ffmpeg_cmd)
    output_video_size = ffprobe_size(raw_mp4)
    actual_num_frames, actual_duration = ffprobe_frame_count_and_duration(raw_mp4)
    ref_frame_in_clip, ref_frame_fraction, ref_frame_strategy = select_reference_frame(
        actual_num_frames=actual_num_frames,
        start_frame=start_frame,
        ref_source_frame=args.ref_source_frame,
        ref_frame_fraction=args.ref_frame_fraction,
        verb=str(action.get("verb") or ""),
    )
    end_frame = start_frame + actual_num_frames
    original_bbox = annotation_action.get("bbox")
    transformed_bbox = transform_bbox(
        original_bbox,
        (annotation_width, annotation_height),
        output_video_size,
        args.bbox_format,
    )
    if transformed_bbox is not None:
        action["bbox"] = transformed_bbox

    config = {
        "frame_number": ref_frame_in_clip,
        "object_names": [obj_id],
        "anchor_hand": args.anchor_hand,
        "prompt": obj_name,
    }
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "dataset_root": str(dataset_root),
        "segment": args.segment,
        "segment_dir": str(segment_dir),
        "annotation_id": args.annotation_id,
        "annotation": annotation,
        "action": action,
        "annotation_action_original": annotation_action,
        "bbox_coordinate_transform": {
            "source": args.bbox_coordinate_space,
            "bbox_format": args.bbox_format,
            "annotation_size": [annotation_width, annotation_height],
            "annotation_size_source": annotation_size_source,
            "source_video_size": list(source_video_size),
            "output_video_size": list(output_video_size),
            "scale": [
                output_video_size[0] / max(float(annotation_width), 1.0),
                output_video_size[1] / max(float(annotation_height), 1.0),
            ],
            "original_bbox": original_bbox,
            "transformed_bbox": transformed_bbox,
        },
        "source_video": str(source_video),
        "source_fps": source_fps,
        "annotation_start_sec": annotation_start_sec,
        "annotation_end_sec": annotation_end_sec,
        "start_offset_sec": args.start_offset_sec,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "requested_end_frame": requested_end_frame,
        "start_sec": start_sec,
        "requested_duration": requested_duration,
        "requested_quantized_duration": duration,
        "duration": actual_duration,
        "actual_frame_count": actual_num_frames,
        "duration_quantization": {
            "clock": "egoinfinity",
            "fps": args.ego_fps,
            "frame_count": int(round(actual_duration * args.ego_fps)),
        },
        "object_name": obj_name,
        "object_id": obj_id,
        "anchor_hand": args.anchor_hand,
        "ref_source_frame": start_frame + ref_frame_in_clip,
        "ref_frame_in_clip": ref_frame_in_clip,
        "ref_frame_fraction": ref_frame_fraction,
        "ref_frame_strategy": ref_frame_strategy,
        "hands_npz": str(hands_npz),
        "trimmed_video": str(raw_mp4),
        "ffmpeg_command": ffmpeg_cmd,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "fresh_input_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"clip_dir": str(out_dir), "raw_mp4": str(raw_mp4), "metadata": metadata}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
