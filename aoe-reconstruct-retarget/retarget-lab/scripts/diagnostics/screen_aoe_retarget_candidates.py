#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from validate_sam3_prompt_gate import validate_prompt_qc


DEFAULT_DATASET_ROOT = Path(os.environ["AOE_DATA_ROOT"]) if os.environ.get("AOE_DATA_ROOT") else None

GOOD_VERBS = {
    "grasp",
    "grab",
    "pick_up",
    "pick",
    "hold",
    "lift",
    "place",
    "put_down",
    "put",
    "insert",
    "remove",
    "rotate",
    "open",
    "close",
    "cap",
    "uncap",
    "pour",
    "shake",
    "move",
    "take",
    "set_down",
}
WEAK_VERBS = {"read", "look", "point", "press", "tap", "wipe", "fold", "unfold", "tear", "scoop"}
ARTICULATED_ACTION_VERBS = {"open", "close", "cap", "uncap"}
RIGID_TOKENS = {
    "bottle",
    "jar",
    "cup",
    "mug",
    "box",
    "container",
    "case",
    "lid",
    "pot",
    "pan",
    "can",
    "tube",
    "vape",
    "brush",
    "spoon",
    "fork",
    "knife",
    "remote",
    "phone",
    "bowl",
    "plate",
    "pen",
    "tool",
    "stick",
    "tablet",
    "device",
    "inhaler",
    "applicator",
    "scoop",
    "lamp",
    "scraper",
    "thermos",
    "basin",
    "cleaver",
    "colander",
    "comb",
    "hairbrush",
    "kettle",
    "peeler",
    "ruler",
    "scissors",
    "tin",
}
RIGID_OBJECT_NAMES = {
    "water bottle",
    "water gun",
    "water pump",
}
SCENE_ID_ALIASES = {
    "bottle of cooking wine": "cooking_wine_bottle",
    "bottle of vinegar": "vinegar_bottle",
    "water bottle": "water_bottle",
}
DEFORMABLE_TOKENS = {
    "bag",
    "cloth",
    "towel",
    "paper",
    "book",
    "booklet",
    "card",
    "napkin",
    "tissue",
    "wrapper",
    "packet",
    "packets",
    "package",
    "pouch",
    "food",
    "leaf",
    "flower",
    "cable",
    "sock",
    "sponge",
    "manual",
    "instruction",
    "magazine",
    "magazines",
    "pants",
    "envelope",
    "wrap",
    "onion",
    "dumpling",
    "pepper",
    "peppers",
    "water",
    "cucumber",
    "potato",
    "sneaker",
    "trash",
}


def normalized_rigid_object_name(name: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", name.lower()))


def is_supported_rigid_object(name: str) -> bool:
    normalized = normalized_rigid_object_name(name)
    tokens = set(normalized.split())
    return normalized in RIGID_OBJECT_NAMES or bool(tokens & RIGID_TOKENS)


def is_likely_deformable_object(name: str) -> bool:
    normalized = normalized_rigid_object_name(name)
    if normalized in RIGID_OBJECT_NAMES:
        return False
    return bool(set(normalized.split()) & DEFORMABLE_TOKENS)


VISUAL_RISK_TOKENS = {"clear", "transparent", "translucent", "reflective", "glass"}
GEOMETRY_RISK_TOKENS = {
    "bowl",
    "brush",
    "cap",
    "fork",
    "knife",
    "lid",
    "pan",
    "pen",
    "plate",
    "scraper",
    "scoop",
    "spoon",
    "stick",
    "basin",
    "cleaver",
    "colander",
    "comb",
    "hairbrush",
    "peeler",
    "ruler",
    "scissors",
}
DETACHABLE_DEVICE_TOKENS = {"gun", "pump", "revolver"}
ATTACHED_ENVIRONMENT_TOKENS = {"drawer", "door", "handle"}
MULTI_OBJECT_PATTERNS = [
    re.compile(r"\b(and|with)\b", re.IGNORECASE),
    re.compile(r"\b(two|three|pair|stack)\b", re.IGNORECASE),
    re.compile(r"\binto\b", re.IGNORECASE),
    re.compile(r"\b(items?|objects?|things?)\s+in\b", re.IGNORECASE),
    re.compile(r"\b(several|multiple|assorted)\b", re.IGNORECASE),
]
OBJECT_PART_TOKENS = {"handle", "cap", "mouth", "rim", "edge"}


@dataclass
class VideoInfo:
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    available: bool = False
    error: str | None = None


@dataclass
class HandTrackProbe:
    available: bool = False
    valid_fraction: float | None = None
    near_object_fraction: float | None = None
    median_distance_px: float | None = None
    near_threshold_px: float | None = None
    sample_count: int = 0
    error: str | None = None


@dataclass
class AnnotationSpace:
    width: int
    height: int
    cx: float
    cy: float
    source: str


@dataclass
class BboxSpace:
    width: int
    height: int
    source: str


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def object_id(name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", name.strip().lower()).strip("_")
    return value or "object"


def canonical_scene_id(name: str) -> str:
    normalized = normalized_rigid_object_name(name)
    return SCENE_ID_ALIASES.get(normalized, object_id(normalized))


def existing_scene_ids(experiments_root: Path) -> set[str]:
    if not experiments_root.exists():
        return set()
    return {
        path.name.split("_full_source_", 1)[0]
        for path in experiments_root.glob("*_full_source_*")
        if path.is_dir() and "_full_source_" in path.name
    }


def run_token(segment: str, annotation_id: object, obj: str) -> str:
    digest = hashlib.sha1(f"{segment}:{annotation_id}:{obj}".encode("utf-8")).hexdigest()[:8]
    ann = re.sub(r"[^0-9A-Za-z]+", "", str(annotation_id or "na")) or "na"
    return f"{digest}_ann{ann}"


def normalized_object_name(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def current_prompt_gate(mask_qc: dict) -> dict:
    text = str(mask_qc.get("text") or "").strip()
    bbox = mask_qc.get("target_bbox") or []
    if not text or not isinstance(bbox, list) or len(bbox) != 4:
        return {
            "status": "failed",
            "errors": ["missing semantic text or transformed target bbox"],
            "warnings": [],
        }
    return validate_prompt_qc(mask_qc, text, [float(value) for value in bbox])


def existing_run_records(
    experiments_root: Path,
) -> dict[tuple[str, str, str], list[dict]]:
    records: dict[tuple[str, str, str], list[dict]] = {}
    if not experiments_root.exists():
        return records
    for path in sorted(experiments_root.glob("*_full_source_*/fresh_run_metadata.json")):
        try:
            metadata = load_json(path)
        except Exception:
            continue
        if not isinstance(metadata, dict):
            continue
        dataset = metadata.get("dataset") or {}
        task = metadata.get("task") or {}
        if not isinstance(dataset, dict) or not isinstance(task, dict):
            continue
        segment = str(dataset.get("segment") or "").strip()
        annotation_id = str(dataset.get("annotation_id") or "").strip()
        obj = normalized_object_name(task.get("object_name"))
        if segment and annotation_id and obj:
            run_root = path.parent
            run_command = str(metadata.get("run_command") or "")
            preflight_manifest = run_root / "sam3_preflight_manifest.json"
            is_preflight = "--sam3-preflight-only" in run_command or preflight_manifest.is_file()
            mask_qc_path = (
                run_root
                / "intermediates/trajectory_6dof/do_as_i_do/reconstruction/clip"
                / "video_segmentation/mask_qc_summary.json"
            )
            mask_qc: dict = {}
            outcome: dict = {}
            try:
                loaded = load_json(mask_qc_path)
                mask_qc = loaded if isinstance(loaded, dict) else {}
            except Exception:
                pass
            prompt_gate = current_prompt_gate(mask_qc) if mask_qc else {
                "status": "failed",
                "errors": ["missing mask_qc_summary.json"],
                "warnings": [],
            }
            try:
                loaded = load_json(run_root / "fresh_run_outcome.json")
                outcome = loaded if isinstance(loaded, dict) else {}
            except Exception:
                pass
            runs = metadata.get("runs") or {}
            matrix_name = str(runs.get("matrix_run") or "") if isinstance(runs, dict) else ""
            matrix_manifest = experiments_root / matrix_name / "reuse_12_demo_manifest.json"
            if is_preflight:
                stage = (
                    "preflight_passed"
                    if mask_qc.get("qa_pass") is True and prompt_gate.get("status") == "ok"
                    else "preflight_failed"
                )
            elif matrix_manifest.is_file():
                stage = "full_run_completed"
            elif outcome.get("status") in {"failed", "aborted"}:
                stage = "full_run_failed"
            else:
                stage = "full_run_in_progress"
            records.setdefault((segment, annotation_id, obj), []).append(
                {
                    "source_run": run_root.name,
                    "matrix_run": matrix_name or None,
                    "stage": stage,
                    "mask_qa_pass": mask_qc.get("qa_pass"),
                    "mask_valid_ratio": mask_qc.get("valid_ratio"),
                    "current_prompt_gate_status": prompt_gate.get("status"),
                    "current_prompt_gate_errors": prompt_gate.get("errors") or [],
                    "current_prompt_gate_warnings": prompt_gate.get("warnings") or [],
                    "outcome_status": outcome.get("status"),
                    "outcome_stage": outcome.get("stage"),
                    "git": metadata.get("git") or {},
                    "gpu_mapping": metadata.get("gpu_mapping") or {},
                }
            )
    return records


def candidate_history_summary(history: list[dict]) -> dict:
    stages = {str(item.get("stage") or "") for item in history}
    full_attempted = bool(
        stages & {"full_run_completed", "full_run_failed", "full_run_in_progress"}
    )
    preflight_passed = "preflight_passed" in stages
    preflight_failed = "preflight_failed" in stages
    promotion_eligible = preflight_passed and not full_attempted
    if "full_run_completed" in stages:
        next_action = "review_existing_full_run"
    elif "full_run_in_progress" in stages:
        next_action = "wait_for_existing_full_run"
    elif "full_run_failed" in stages:
        next_action = "inspect_full_run_failure"
    elif promotion_eligible:
        next_action = "launch_fresh_full_run"
    elif preflight_failed:
        next_action = "skip_known_preflight_failure"
    else:
        next_action = "launch_fresh_preflight"
    return {
        "already_run": bool(history),
        "existing_stages": sorted(stage for stage in stages if stage),
        "promotion_eligible": promotion_eligible,
        "recommended_next_action": next_action,
        "existing_runs": history,
    }


def candidate_run_suffix(args: argparse.Namespace, segment: str, annotation_id: object, obj: str) -> str:
    if args.run_name_mode == "scene":
        return args.run_suffix
    return f"{args.run_suffix}_{run_token(segment, annotation_id, obj)}"


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bbox(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def ffprobe_video(path: Path) -> VideoInfo:
    if not path.exists():
        return VideoInfo(available=False, error="missing_video")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
        data = json.loads(out)
        stream = (data.get("streams") or [{}])[0]
        fps_text = str(stream.get("r_frame_rate") or "30/1")
        if "/" in fps_text:
            num, den = fps_text.split("/", 1)
            fps = float(num) / max(float(den), 1e-9)
        else:
            fps = float(fps_text)
        return VideoInfo(
            width=int(stream.get("width") or 1280),
            height=int(stream.get("height") or 720),
            fps=fps,
            available=True,
        )
    except Exception as exc:
        return VideoInfo(available=False, error=str(exc))


def screening_size(video: VideoInfo, width: int) -> tuple[int, int]:
    if width <= 0:
        return video.width, video.height
    height = int(round(width * video.height / max(video.width, 1)))
    if height % 2:
        height += 1
    return width, height


def load_annotation_space(segment_dir: Path, video: VideoInfo) -> AnnotationSpace:
    info_path = segment_dir / "video_info.json"
    try:
        info = load_json(info_path)
        camera = info.get("cameraParams") if isinstance(info, dict) else None
        camera = camera if isinstance(camera, dict) else {}
        resolution = str(camera.get("resolution") or "")
        match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", resolution)
        if match:
            width, height = int(match.group(1)), int(match.group(2))
            return AnnotationSpace(
                width=width,
                height=height,
                cx=parse_float(camera.get("cx_pixels"), width * 0.5),
                cy=parse_float(camera.get("cy_pixels"), height * 0.5),
                source=str(info_path),
            )
    except Exception:
        pass
    return AnnotationSpace(
        width=video.width,
        height=video.height,
        cx=video.width * 0.5,
        cy=video.height * 0.5,
        source="ffprobe_source_video_fallback",
    )


def load_bbox_space(
    segment_dir: Path,
    video: VideoInfo,
    camera_space: AnnotationSpace,
    mode: str,
) -> BboxSpace:
    if mode == "normalized-1000":
        return BboxSpace(width=1000, height=1000, source="normalized_0_1000")
    if mode == "video-info":
        return BboxSpace(width=camera_space.width, height=camera_space.height, source=camera_space.source)
    return BboxSpace(width=video.width, height=video.height, source="ffprobe_source_video")


def scale_bbox(
    bbox: list[float] | None,
    source: BboxSpace,
    target_width: int,
    target_height: int,
) -> list[float] | None:
    if bbox is None:
        return None
    scale_x = target_width / max(float(source.width), 1.0)
    scale_y = target_height / max(float(source.height), 1.0)
    return [bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y]


def point_bbox_distance(x: float, y: float, bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def probe_hand_track(
    hands: object | None,
    video: VideoInfo,
    start: float,
    end: float,
    bbox: list[float] | None,
    annotation_space: AnnotationSpace,
    screen_width: int,
    screen_height: int,
    args: argparse.Namespace,
) -> HandTrackProbe:
    if hands is None:
        return HandTrackProbe(error="missing_hands_npz")
    if bbox is None or not video.available:
        return HandTrackProbe(error="missing_bbox_or_video")
    try:
        import numpy as np

        pred_trans_cam = np.asarray(hands["pred_trans_cam"])
        pred_valid = np.asarray(hands["pred_valid"])
        focal = float(np.asarray(hands["focal"]).reshape(()))
        if pred_trans_cam.ndim != 3 or pred_trans_cam.shape[-1] != 3:
            raise ValueError(f"unexpected pred_trans_cam shape {pred_trans_cam.shape}")
        frame_count = pred_trans_cam.shape[1]
        sample_count = max(int(args.hand_probe_samples), 1)
        times = np.linspace(start, end, sample_count + 2, dtype=np.float64)[1:-1]
        frame_ids = np.clip(np.rint(times * video.fps).astype(np.int64), 0, frame_count - 1)
        scale_x = screen_width / max(float(annotation_space.width), 1.0)
        scale_y = screen_height / max(float(annotation_space.height), 1.0)
        near_threshold = max(
            float(args.hand_near_px),
            float(args.hand_near_bbox_scale) * max(bbox[2] - bbox[0], bbox[3] - bbox[1]),
        )
        valid_frames = 0
        near_frames = 0
        distances: list[float] = []
        for frame_id in frame_ids:
            frame_distances: list[float] = []
            for slot in range(min(pred_trans_cam.shape[0], pred_valid.shape[0])):
                if float(pred_valid[slot, frame_id]) <= 0.5:
                    continue
                x, y, z = [float(v) for v in pred_trans_cam[slot, frame_id]]
                if not math.isfinite(z) or z <= 1e-6:
                    continue
                px = (focal * x / z + annotation_space.cx) * scale_x
                py = (focal * y / z + annotation_space.cy) * scale_y
                if math.isfinite(px) and math.isfinite(py):
                    frame_distances.append(point_bbox_distance(px, py, bbox))
            if not frame_distances:
                continue
            valid_frames += 1
            distance = min(frame_distances)
            distances.append(distance)
            if distance <= near_threshold:
                near_frames += 1
        return HandTrackProbe(
            available=True,
            valid_fraction=valid_frames / max(len(frame_ids), 1),
            near_object_fraction=near_frames / max(valid_frames, 1),
            median_distance_px=float(np.median(distances)) if distances else None,
            near_threshold_px=near_threshold,
            sample_count=len(frame_ids),
        )
    except Exception as exc:
        return HandTrackProbe(error=f"hand_probe_failed:{exc}")


def first_action(row: dict) -> dict:
    actions = row.get("atomic_action") or []
    return actions[0] if actions and isinstance(actions[0], dict) else {}


def hand_type(action: dict) -> str | None:
    hand = str(action.get("hand") or "").strip().lower()
    if hand in {"left", "right"}:
        return hand
    if hand in {"both", "bimanual", "two", "two_hands"}:
        return "bimanual"
    return None


def classify_suitability(
    recommendation: str,
    hand: str | None,
    name_tokens: set[str],
    warnings: list[str],
    verb: str | None = None,
) -> tuple[str, list[str]]:
    flags: list[str] = []
    if hand == "bimanual":
        flags.append("bimanual_interaction")
    if name_tokens & GEOMETRY_RISK_TOKENS:
        flags.append("thin_open_or_articulated_geometry")
    if name_tokens & DETACHABLE_DEVICE_TOKENS:
        flags.append("detachable_or_articulated_device")
    if name_tokens & VISUAL_RISK_TOKENS:
        flags.append("transparent_or_reflective_appearance")
    if verb in ARTICULATED_ACTION_VERBS:
        flags.append("articulated_action")
    if warnings:
        flags.append("manual_review_required")
    flags = sorted(set(flags))
    if recommendation == "reject":
        return "reject", flags
    if any(
        flag in flags
        for flag in (
            "thin_open_or_articulated_geometry",
            "transparent_or_reflective_appearance",
            "articulated_action",
            "detachable_or_articulated_device",
        )
    ):
        return "stress", flags
    if flags or recommendation == "borderline":
        return "conditional", flags
    return "primary", flags


def score_candidate(
    segment: str,
    row: dict,
    action: dict,
    video: VideoInfo,
    has_hands: bool,
    hands: object | None,
    annotation_space: AnnotationSpace,
    bbox_space: BboxSpace,
    args: argparse.Namespace,
) -> dict:
    reasons: list[str] = []
    warnings: list[str] = []
    score = 0.0

    start = parse_float(row.get("start_ts"))
    end = parse_float(row.get("end_ts"), start)
    duration = max(0.0, end - start)
    verb = str(action.get("verb") or "").strip().lower()
    obj = str(action.get("object") or "").strip()
    hand = hand_type(action)
    bbox_original = parse_bbox(action.get("bbox"))
    confidence = parse_float(action.get("confidence"), 0.0)
    width, height = screening_size(video, args.screen_width)
    bbox = scale_bbox(bbox_original, bbox_space, width, height)
    hand_probe = probe_hand_track(
        hands,
        video,
        start,
        end,
        bbox,
        annotation_space,
        width,
        height,
        args,
    )

    if video.available:
        score += 1.0
    else:
        reasons.append(video.error or "missing_video")
    if has_hands:
        score += 1.0
    else:
        reasons.append("missing_hands_npz")
    if obj:
        score += 1.0
    else:
        reasons.append("missing_object_name")
    if hand:
        score += 1.0
    else:
        reasons.append("unknown_hand_type")

    if args.min_duration <= duration <= args.max_duration:
        score += 1.0
    else:
        reasons.append(f"duration_outside_range:{duration:.2f}s")
    if 2.0 <= duration <= 6.0:
        score += 0.5

    if verb in GOOD_VERBS:
        score += 2.0
    elif verb in WEAK_VERBS:
        score -= 1.0
        reasons.append(f"weak_or_deformable_action:{verb}")
    else:
        reasons.append(f"non_retargetable_action_verb:{verb}")

    name_tokens = set(re.findall(r"[a-z0-9]+", obj.lower()))
    if any(pattern.search(obj) for pattern in MULTI_OBJECT_PATTERNS):
        reasons.append("compound_or_collection_object")
        score -= 3.0
    if name_tokens & OBJECT_PART_TOKENS:
        score -= 1.0
        warnings.append("object_name_may_be_part_not_whole_object")
    if name_tokens & ATTACHED_ENVIRONMENT_TOKENS:
        reasons.append("attached_environment_object")
        score -= 2.0
    if is_supported_rigid_object(obj):
        score += 2.0
    else:
        reasons.append("object_not_in_supported_rigid_class")
    if is_likely_deformable_object(obj):
        score -= 2.5
        reasons.append("likely_deformable_or_nonrigid_object")
    if name_tokens & VISUAL_RISK_TOKENS:
        score -= 1.5
        warnings.append("transparent_or_reflective_object_risk")

    bbox_area_ratio = None
    if bbox is None:
        reasons.append("missing_or_invalid_bbox")
    else:
        x1, y1, x2, y2 = bbox
        bbox_area_ratio = ((x2 - x1) * (y2 - y1)) / max(float(width * height), 1.0)
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5
        if args.min_bbox_area <= bbox_area_ratio <= args.max_bbox_area:
            score += 2.0
        else:
            warnings.append(f"bbox_area_ratio_outside_range:{bbox_area_ratio:.4f}")
        if 0 <= center_x < width and 0 <= center_y < height:
            score += 0.5
        else:
            reasons.append("bbox_center_outside_frame")
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            warnings.append("bbox_partly_outside_frame")
            score -= 1.0

    if hand_probe.available:
        if (hand_probe.valid_fraction or 0.0) >= args.min_hand_valid_fraction:
            score += 1.0
        else:
            warnings.append(f"weak_hand_track_validity:{hand_probe.valid_fraction or 0.0:.2f}")
            score -= 1.0
        if (hand_probe.near_object_fraction or 0.0) >= args.min_hand_near_fraction:
            score += 1.5
        else:
            warnings.append(f"weak_hand_object_proximity:{hand_probe.near_object_fraction or 0.0:.2f}")
            score -= 1.5
            if args.require_hand_object_near:
                reasons.append("hand_track_not_near_object_bbox")
    elif has_hands:
        warnings.append(hand_probe.error or "hand_track_probe_unavailable")

    if confidence >= args.min_confidence:
        score += 0.5
    else:
        warnings.append(f"low_confidence:{confidence:.2f}")

    if reasons:
        recommendation = "reject"
    elif score >= args.accept_score:
        recommendation = "accept"
    elif score >= args.borderline_score:
        recommendation = "borderline"
    else:
        recommendation = "reject"
    suitability_tier, risk_flags = classify_suitability(
        recommendation,
        hand,
        name_tokens,
        warnings,
        verb,
    )

    task = object_id(obj)
    if hand and hand != "bimanual":
        task = f"{task}_{hand}"
    elif hand == "bimanual":
        task = f"{task}_bimanual"
    scene = canonical_scene_id(obj)
    machine_tag = args.machine_tag
    suffix = candidate_run_suffix(args, segment, row.get("id"), obj)
    source_run = f"{scene}_full_source_{args.date_tag}_{machine_tag}_{suffix}"
    matrix_run = f"{scene}_matrix12_{args.date_tag}_{machine_tag}_{suffix}"
    suggested_duration = min(max(duration + args.extra_sec_after, args.min_duration), args.clip_duration_cap)
    command = None
    if recommendation in {"accept", "borderline"} and hand:
        command = (
            "scripts/run_fresh_aoe_scene_12_demos.sh "
            f"--scene {scene} "
            f"--segment {segment} "
            f"--annotation-id {row.get('id')} "
            f"--object-name {json.dumps(obj)} "
            f"--task {task} "
            f"--hand-type {hand} "
            f"--anchor-hand {hand} "
            f"--source-run {source_run} "
            f"--matrix-run {matrix_run} "
            f"--duration-sec {suggested_duration:.3f}"
        )

    return {
        "recommendation": recommendation,
        "score": round(score, 3),
        "segment": segment,
        "annotation_id": row.get("id"),
        "scene": row.get("scene"),
        "scene_id": scene,
        "start_ts": start,
        "end_ts": end,
        "duration": duration,
        "suggested_duration": suggested_duration,
        "verb": verb,
        "object_name": obj,
        "object_id": object_id(obj) if obj else None,
        "hand_type": hand,
        "bbox": bbox,
        "bbox_original": bbox_original,
        "bbox_area_ratio": bbox_area_ratio,
        "annotation_width": annotation_space.width,
        "annotation_height": annotation_space.height,
        "annotation_cx": annotation_space.cx,
        "annotation_cy": annotation_space.cy,
        "annotation_space_source": annotation_space.source,
        "bbox_space_width": bbox_space.width,
        "bbox_space_height": bbox_space.height,
        "bbox_space_source": bbox_space.source,
        "screen_width": width,
        "screen_height": height,
        "video_width": video.width,
        "video_height": video.height,
        "video_fps": video.fps,
        "hand_track_probe": asdict(hand_probe),
        "confidence": confidence,
        "reasons": reasons,
        "warnings": warnings,
        "screening_profile": suitability_tier,
        "suitability_tier": suitability_tier,
        "risk_flags": risk_flags,
        "run_name_mode": args.run_name_mode,
        "run_token": run_token(segment, row.get("id"), obj),
        "source_run": source_run,
        "matrix_run": matrix_run,
        "task": task,
        "command": command,
    }


def iter_segments(dataset_root: Path, limit_segments: int) -> list[Path]:
    segments = sorted(p for p in dataset_root.iterdir() if (p / "ego_annotation" / "ego_action_annotation.json").exists())
    if limit_segments > 0:
        return segments[:limit_segments]
    return segments


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "recommendation",
        "screening_profile",
        "suitability_tier",
        "risk_flags",
        "score",
        "segment",
        "annotation_id",
        "scene",
        "scene_id",
        "verb",
        "object_name",
        "hand_type",
        "duration",
        "bbox_area_ratio",
        "annotation_width",
        "annotation_height",
        "bbox_space_width",
        "bbox_space_height",
        "bbox_space_source",
        "screen_width",
        "screen_height",
        "hand_valid_fraction",
        "hand_near_object_fraction",
        "hand_median_distance_px",
        "confidence",
        "source_run",
        "matrix_run",
        "task",
        "run_token",
        "run_name_mode",
        "scene_already_run",
        "already_run",
        "existing_stages",
        "promotion_eligible",
        "recommended_next_action",
        "warnings",
        "reasons",
        "command",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fieldnames}
            probe = row.get("hand_track_probe") or {}
            out["hand_valid_fraction"] = probe.get("valid_fraction")
            out["hand_near_object_fraction"] = probe.get("near_object_fraction")
            out["hand_median_distance_px"] = probe.get("median_distance_px")
            out["warnings"] = ";".join(row.get("warnings") or [])
            out["reasons"] = ";".join(row.get("reasons") or [])
            out["risk_flags"] = ";".join(row.get("risk_flags") or [])
            out["existing_stages"] = ";".join(row.get("existing_stages") or [])
            writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Screen AoE annotations before launching expensive retarget-lab fresh runs.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--existing-experiments-root",
        type=Path,
        default=Path("experiments"),
        help="Experiments directory used to detect already launched fresh runs.",
    )
    parser.add_argument(
        "--exclude-existing",
        action="store_true",
        help="Exclude candidates whose segment/annotation/object already appears in fresh_run_metadata.json.",
    )
    parser.add_argument(
        "--exclude-existing-scenes",
        action="store_true",
        help=(
            "Exclude every annotation whose canonical scene id already has a source run; "
            "use this when expanding scene diversity rather than revisiting an old scene."
        ),
    )
    parser.add_argument(
        "--include-passed-preflights",
        action="store_true",
        help=(
            "With --exclude-existing, retain exact annotations that passed SAM3 preflight "
            "but have never attempted a full run."
        ),
    )
    parser.add_argument("--limit-segments", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=8.0)
    parser.add_argument("--clip-duration-cap", type=float, default=6.0)
    parser.add_argument("--extra-sec-after", type=float, default=0.25)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--min-bbox-area", type=float, default=0.004)
    parser.add_argument("--max-bbox-area", type=float, default=0.22)
    parser.add_argument(
        "--screen-width",
        type=int,
        default=1280,
        help="Width used by the fresh clip and bbox/hand screening coordinate space; 0 keeps source size.",
    )
    parser.add_argument(
        "--bbox-coordinate-space",
        choices=["normalized-1000", "video-info", "source-video"],
        default="normalized-1000",
        help="AoE 3-hour action bboxes use normalized 0..1000 coordinates; other modes are compatibility options.",
    )
    parser.add_argument("--hand-probe-samples", type=int, default=7)
    parser.add_argument("--min-hand-valid-fraction", type=float, default=0.7)
    parser.add_argument("--min-hand-near-fraction", type=float, default=0.4)
    parser.add_argument("--hand-near-px", type=float, default=120.0)
    parser.add_argument("--hand-near-bbox-scale", type=float, default=0.5)
    parser.add_argument(
        "--require-hand-object-near",
        action="store_true",
        help="Reject candidates whose projected AoE hand roots rarely approach the annotated object bbox.",
    )
    parser.add_argument("--accept-score", type=float, default=8.0)
    parser.add_argument("--borderline-score", type=float, default=6.0)
    parser.add_argument("--date-tag", required=True)
    parser.add_argument("--machine-tag", default="236")
    parser.add_argument("--run-suffix", default="screen01")
    parser.add_argument(
        "--run-name-mode",
        choices=["unique", "scene"],
        default="unique",
        help="unique appends a stable segment/annotation token to avoid collisions across accepted candidates.",
    )
    args = parser.parse_args()

    if args.dataset_root is None:
        parser.error("set AOE_DATA_ROOT or pass --dataset-root")
    dataset_root = args.dataset_root.expanduser().resolve()
    experiments_root = args.existing_experiments_root.expanduser().resolve()
    existing_records = existing_run_records(experiments_root)
    existing_scenes = existing_scene_ids(experiments_root)
    rows: list[dict] = []
    scanned_segments = 0
    for segment_dir in iter_segments(dataset_root, args.limit_segments):
        scanned_segments += 1
        segment = segment_dir.name
        video_path = segment_dir / "ego_process" / "ego_undistorted_video" / "raw_video_undistorted.mp4"
        hands_path = segment_dir / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
        video = ffprobe_video(video_path)
        annotation_space = load_annotation_space(segment_dir, video)
        bbox_space = load_bbox_space(segment_dir, video, annotation_space, args.bbox_coordinate_space)
        has_hands = hands_path.exists()
        hands = None
        if has_hands:
            try:
                import numpy as np

                hands = np.load(hands_path, allow_pickle=False)
            except Exception:
                hands = None
        annotation_path = segment_dir / "ego_annotation" / "ego_action_annotation.json"
        try:
            annotations = load_json(annotation_path)
        except Exception as exc:
            rows.append(
                {
                    "recommendation": "reject",
                    "score": 0.0,
                    "segment": segment,
                    "annotation_id": None,
                    "reasons": [f"annotation_load_failed:{exc}"],
                    "warnings": [],
                }
            )
            if hands is not None:
                hands.close()
            continue
        if not isinstance(annotations, list):
            if hands is not None:
                hands.close()
            continue
        for row in annotations:
            if not isinstance(row, dict):
                continue
            action = first_action(row)
            candidate = score_candidate(
                segment,
                row,
                action,
                video,
                has_hands,
                hands,
                annotation_space,
                bbox_space,
                args,
            )
            key = (
                str(candidate.get("segment") or ""),
                str(candidate.get("annotation_id") or ""),
                normalized_object_name(candidate.get("object_name")),
            )
            candidate.update(candidate_history_summary(existing_records.get(key, [])))
            candidate["scene_already_run"] = candidate.get("scene_id") in existing_scenes
            rows.append(candidate)
        if hands is not None:
            hands.close()

    tier_rank = {"primary": 3, "conditional": 2, "stress": 1, "reject": 0}
    rows.sort(
        key=lambda item: (
            item.get("recommendation") == "accept",
            bool(item.get("promotion_eligible")),
            tier_rank.get(str(item.get("suitability_tier")), 0),
            item.get("score", 0.0),
        ),
        reverse=True,
    )
    ranked_rows = rows
    if args.exclude_existing:
        ranked_rows = [
            row
            for row in rows
            if not row.get("already_run")
            or (args.include_passed_preflights and row.get("promotion_eligible"))
        ]
    if args.exclude_existing_scenes:
        ranked_rows = [row for row in ranked_rows if not row.get("scene_already_run")]
    accept_rows = [row for row in ranked_rows if row.get("recommendation") == "accept"]
    borderline_rows = [row for row in ranked_rows if row.get("recommendation") == "borderline"]
    payload = {
        "dataset_root": str(dataset_root),
        "existing_experiments_root": str(experiments_root),
        "scanned_segments": scanned_segments,
        "total_annotations": len(rows),
        "already_run_annotations": sum(1 for row in rows if row.get("already_run")),
        "scene_already_run_annotations": sum(
            1 for row in rows if row.get("scene_already_run")
        ),
        "exclude_existing": bool(args.exclude_existing),
        "exclude_existing_scenes": bool(args.exclude_existing_scenes),
        "include_passed_preflights": bool(args.include_passed_preflights),
        "accepted": len(accept_rows),
        "borderline": len(borderline_rows),
        "criteria": {
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "clip_duration_cap": args.clip_duration_cap,
            "min_confidence": args.min_confidence,
            "min_bbox_area": args.min_bbox_area,
            "max_bbox_area": args.max_bbox_area,
            "screen_width": args.screen_width,
            "bbox_coordinate_space": args.bbox_coordinate_space,
            "hand_probe_samples": args.hand_probe_samples,
            "min_hand_valid_fraction": args.min_hand_valid_fraction,
            "min_hand_near_fraction": args.min_hand_near_fraction,
            "hand_near_px": args.hand_near_px,
            "hand_near_bbox_scale": args.hand_near_bbox_scale,
            "require_hand_object_near": bool(args.require_hand_object_near),
            "good_verbs": sorted(GOOD_VERBS),
            "weak_verbs": sorted(WEAK_VERBS),
            "articulated_action_verbs": sorted(ARTICULATED_ACTION_VERBS),
            "rigid_tokens": sorted(RIGID_TOKENS),
            "rigid_object_names": sorted(RIGID_OBJECT_NAMES),
            "scene_id_aliases": dict(sorted(SCENE_ID_ALIASES.items())),
            "deformable_tokens": sorted(DEFORMABLE_TOKENS),
            "visual_risk_tokens": sorted(VISUAL_RISK_TOKENS),
            "geometry_risk_tokens": sorted(GEOMETRY_RISK_TOKENS),
            "attached_environment_tokens": sorted(ATTACHED_ENVIRONMENT_TOKENS),
        },
        "top_candidates": ranked_rows[: args.top_k],
        "accepted_candidates": accept_rows[: args.top_k],
        "borderline_candidates": borderline_rows[: args.top_k],
        "video_defaults": asdict(VideoInfo()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        write_csv(args.output_csv, ranked_rows[: max(args.top_k, 1)])
    print(json.dumps({"output_json": str(args.output_json), "output_csv": str(args.output_csv) if args.output_csv else None, "accepted": len(accept_rows), "borderline": len(borderline_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
