#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DATASET_ROOT = Path(os.environ.get("AOE_DATA_ROOT", "/path/to/open-aoe/extracted"))


def read_frame(video_path: Path, frame_id: int, fps: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_id, 0))
        ok, frame = cap.read()
    finally:
        cap.release()
    if ok and frame is not None:
        return frame

    # Some long-GOP AoE videos report the right frame count but OpenCV cannot
    # seek into their later keyframe ranges. ffmpeg handles those files
    # reliably and keeps a decoder quirk from becoming a false data rejection.
    timestamp = max(frame_id, 0) / max(float(fps), 1e-6)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    fallback = cv2.imdecode(np.frombuffer(proc.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if proc.returncode != 0 or fallback is None:
        error = proc.stderr.decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"failed to read frame {frame_id} from {video_path}: {error}")
    return fallback


def fit_tile(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def draw_text(frame: np.ndarray, text: str, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)


def draw_bbox(frame: np.ndarray, bbox: list[float] | None, screen_width: int, screen_height: int) -> None:
    if not bbox:
        return
    scale_x = frame.shape[1] / max(float(screen_width), 1.0)
    scale_y = frame.shape[0] / max(float(screen_height), 1.0)
    x1, y1, x2, y2 = bbox
    p1 = (int(round(x1 * scale_x)), int(round(y1 * scale_y)))
    p2 = (int(round(x2 * scale_x)), int(round(y2 * scale_y)))
    cv2.rectangle(frame, p1, p2, (0, 220, 255), 3, cv2.LINE_AA)


def projected_hands(
    hands: object | None,
    frame_id: int,
    annotation_width: int,
    annotation_height: int,
    annotation_cx: float,
    annotation_cy: float,
    screen_width: int,
    screen_height: int,
) -> list[tuple[int, float, float]]:
    if hands is None:
        return []
    pred_trans_cam = np.asarray(hands["pred_trans_cam"])
    pred_valid = np.asarray(hands["pred_valid"])
    focal = float(np.asarray(hands["focal"]).reshape(()))
    source_frame_id = min(max(frame_id, 0), pred_trans_cam.shape[1] - 1)
    scale_x = screen_width / max(float(annotation_width), 1.0)
    scale_y = screen_height / max(float(annotation_height), 1.0)
    points: list[tuple[int, float, float]] = []
    for slot in range(min(pred_trans_cam.shape[0], pred_valid.shape[0])):
        if float(pred_valid[slot, source_frame_id]) <= 0.5:
            continue
        x, y, z = [float(v) for v in pred_trans_cam[slot, source_frame_id]]
        if not math.isfinite(z) or z <= 1e-6:
            continue
        px = (focal * x / z + annotation_cx) * scale_x
        py = (focal * y / z + annotation_cy) * scale_y
        if math.isfinite(px) and math.isfinite(py):
            points.append((slot, px, py))
    return points


def draw_hands(
    frame: np.ndarray,
    points: list[tuple[int, float, float]],
    screen_width: int,
    screen_height: int,
) -> None:
    colors = [(255, 170, 0), (255, 80, 255)]
    scale_x = frame.shape[1] / max(float(screen_width), 1.0)
    scale_y = frame.shape[0] / max(float(screen_height), 1.0)
    for slot, px, py in points:
        point = (int(round(px * scale_x)), int(round(py * scale_y)))
        color = colors[slot % len(colors)]
        cv2.circle(frame, point, 10, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.circle(frame, point, 8, color, 3, cv2.LINE_AA)
        cv2.putText(frame, f"H{slot}", (point[0] + 10, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def select_candidates(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    selected: list[dict] = []
    seen_objects: set[str] = set()
    seen_segments: set[str] = set()
    for row in rows:
        obj = str(row.get("object_name") or "").strip().lower()
        segment = str(row.get("segment") or "").strip()
        if args.unique_object and obj in seen_objects:
            continue
        if args.unique_segment and segment in seen_segments:
            continue
        seen_objects.add(obj)
        seen_segments.add(segment)
        selected.append(row)
        if len(selected) >= args.max_candidates:
            break
    return selected


def render_candidate(row: dict, dataset_root: Path, args: argparse.Namespace) -> list[np.ndarray]:
    segment = str(row["segment"])
    segment_dir = dataset_root / segment
    video_path = segment_dir / "ego_process" / "ego_undistorted_video" / "raw_video_undistorted.mp4"
    hands_path = segment_dir / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
    hands = np.load(hands_path, allow_pickle=False) if hands_path.exists() else None
    start = float(row.get("start_ts") or 0.0)
    end = float(row.get("end_ts") or start)
    fps = float(row.get("video_fps") or 30.0)
    video_width = int(row.get("video_width") or 1280)
    video_height = int(row.get("video_height") or 720)
    annotation_width = int(row.get("annotation_width") or video_width)
    annotation_height = int(row.get("annotation_height") or video_height)
    annotation_cx = float(row.get("annotation_cx") or annotation_width * 0.5)
    annotation_cy = float(row.get("annotation_cy") or annotation_height * 0.5)
    screen_width = int(row.get("screen_width") or 1280)
    screen_height = int(row.get("screen_height") or round(screen_width * video_height / max(video_width, 1)))
    fractions = np.linspace(0.1, 0.9, args.samples)
    tiles: list[np.ndarray] = []
    try:
        for fraction in fractions:
            timestamp = start + (end - start) * float(fraction)
            frame_id = int(round(timestamp * fps))
            frame = read_frame(video_path, frame_id, fps)
            frame = cv2.resize(frame, (screen_width, screen_height), interpolation=cv2.INTER_AREA)
            draw_bbox(frame, row.get("bbox"), screen_width, screen_height)
            draw_hands(
                frame,
                projected_hands(
                    hands,
                    frame_id,
                    annotation_width,
                    annotation_height,
                    annotation_cx,
                    annotation_cy,
                    screen_width,
                    screen_height,
                ),
                screen_width,
                screen_height,
            )
            frame = fit_tile(frame, args.tile_width, args.tile_height)
            probe = row.get("hand_track_probe") or {}
            draw_text(
                frame,
                f"{row.get('object_name')} | {row.get('verb')} | {row.get('hand_type')} | ann {row.get('annotation_id')}",
                25,
                (255, 255, 255),
            )
            draw_text(frame, f"{segment} | t={timestamp:.2f}s", 49, (210, 255, 210))
            draw_text(
                frame,
                f"score={row.get('score')} near={probe.get('near_object_fraction')} valid={probe.get('valid_fraction')}",
                73,
                (210, 255, 255),
            )
            tiles.append(frame)
    finally:
        if hands is not None:
            hands.close()
    return tiles


def main() -> int:
    parser = argparse.ArgumentParser(description="Render visual review sheets for screened AoE retarget candidates.")
    parser.add_argument("--screen-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--section", default="accepted_candidates")
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--rows-per-page", type=int, default=6)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=360)
    parser.add_argument("--unique-object", action="store_true")
    parser.add_argument("--unique-segment", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.screen_json.read_text(encoding="utf-8"))
    rows = payload.get(args.section) or []
    candidates = select_candidates(rows, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict] = []
    page_rows: list[np.ndarray] = []
    page_paths: list[str] = []
    page_id = 1
    for index, row in enumerate(candidates):
        try:
            tiles = render_candidate(row, args.dataset_root, args)
            page_rows.append(cv2.hconcat(tiles))
            rendered.append(
                {
                    "rank": index + 1,
                    "segment": row.get("segment"),
                    "annotation_id": row.get("annotation_id"),
                    "object_name": row.get("object_name"),
                    "source_run": row.get("source_run"),
                    "matrix_run": row.get("matrix_run"),
                    "command": row.get("command"),
                    "status": "rendered",
                }
            )
        except Exception as exc:
            rendered.append(
                {
                    "rank": index + 1,
                    "segment": row.get("segment"),
                    "annotation_id": row.get("annotation_id"),
                    "object_name": row.get("object_name"),
                    "status": "failed",
                    "error": str(exc),
                }
            )
        if len(page_rows) >= args.rows_per_page or index == len(candidates) - 1:
            if page_rows:
                page = cv2.vconcat(page_rows)
                page_path = args.output_dir / f"candidate_contact_sheet_{page_id:03d}.jpg"
                if not cv2.imwrite(str(page_path), page, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                    raise RuntimeError(f"failed to write {page_path}")
                page_paths.append(str(page_path))
                page_rows = []
                page_id += 1

    manifest = {
        "screen_json": str(args.screen_json.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "section": args.section,
        "candidate_count": len(candidates),
        "rendered_count": sum(row["status"] == "rendered" for row in rendered),
        "failed_count": sum(row["status"] == "failed" for row in rendered),
        "pages": page_paths,
        "candidates": rendered,
    }
    manifest_path = args.output_dir / "contact_sheet_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
