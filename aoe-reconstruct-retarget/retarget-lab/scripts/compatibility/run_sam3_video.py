#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Retarget Lab compatibility entry for pristine SAM3 video segmentation.

This file owns prompt-instance selection, temporal QA, and output packaging so
the pinned Do-as-I-Do checkout can remain byte-for-byte clean.  Model loading
and prediction still use the upstream SAM3 implementation and checkpoint.

Output structure (in <video_dir>/video_segmentation/ by default):
    masks/frame_000000_masks/<obj_id>.png  - binary masks per frame per object
    overlays/               - overlay visualizations as frame_000000.png, ...
    tracked_<prompt>.mp4    - overlay video

Usage examples:
    # Text prompt (output: video_segmentation/tracked_person.mp4)
    python run_sam3_video.py --video /path/to/video.mp4 --text "left hand" --obj_id left_hand_0

    # Text prompt with custom output directory
    python run_sam3_video.py --video video.mp4 --text "dog" --output_dir output/

    # Point prompt (x,y in pixels, label 1=positive 0=negative)
    python run_sam3_video.py --video video.mp4 --points 210,350 --point_labels 1 --obj_id 1

    # Interactive click mode - opens a window to click points on a frame
    python run_sam3_video.py --video /path/to/video.mp4 --click --obj_id scooper_0 --frame_idx 6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.append(
    os.environ.get(
        "SAM3_PKG_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modules", "sam3"),
    )
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM3 video segmentation")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video (MP4 file or directory of JPEG frames)",
    )
    parser.add_argument("--text", type=str, default=None, help="Text prompt")
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help="Point prompts as 'x1,y1;x2,y2;...' in pixel coords",
    )
    parser.add_argument(
        "--point_labels",
        type=str,
        default=None,
        help="Point labels as '1;0;1;...' (1=positive, 0=negative)",
    )
    parser.add_argument(
        "--obj_id",
        type=str,
        required=True,
        help="Object ID used for naming saved masks (e.g. 'flower', 'hand')",
    )
    parser.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Frame index to add prompt on (default: 0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save output masks (default: same as video path)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: auto-download from HF)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=os.environ.get("SAM3_VERSION", "sam3"),
        choices=["sam3", "sam3.1"],
        help="SAM3 model family. Do-as-I-Do official reconstruction uses base sam3.",
    )
    parser.add_argument(
        "--bpe_path",
        type=str,
        default=None,
        help="Path to SAM3 BPE vocab. Defaults to the active SAM3 repo asset.",
    )
    parser.add_argument(
        "--target-point",
        type=str,
        default=None,
        help="Optional target point 'x,y' in pixels on --frame_idx used to pick one SAM3 instance.",
    )
    parser.add_argument(
        "--target-bbox",
        type=str,
        default=None,
        help="Optional target bbox 'x1,y1,x2,y2' in pixels on --frame_idx used to pick one SAM3 instance.",
    )
    parser.add_argument(
        "--select-mode",
        choices=["auto", "score", "point", "bbox"],
        default="auto",
        help="How to choose the SAM3 instance on the prompt frame.",
    )
    parser.add_argument(
        "--max-centroid-jump-px",
        type=float,
        default=float(os.environ.get("SAM3_MAX_CENTROID_JUMP_PX", "320")),
        help="Fail QA if selected mask centroid jumps more than this between adjacent valid frames. <=0 disables.",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=float(os.environ.get("SAM3_MIN_VALID_RATIO", "0.70")),
        help="Fail QA if fewer than this fraction of frames have the selected object mask.",
    )
    parser.add_argument(
        "--qa-json",
        type=str,
        default=None,
        help="Optional path for mask QA JSON. Defaults to output_dir/mask_qc_summary.json.",
    )
    parser.add_argument(
        "--click",
        action="store_true",
        help="Interactive mode: open a window to click points on a frame. "
        "Left-click = positive, right-click = negative. Press Enter/Space to confirm, Esc to cancel.",
    )
    return parser.parse_args()


def _parse_pair(value):
    vals = [float(x) for x in value.split(",")]
    if len(vals) != 2:
        raise ValueError(f"expected x,y; got {value!r}")
    return tuple(vals)


def _parse_bbox(value):
    vals = [float(x) for x in value.split(",")]
    if len(vals) != 4:
        raise ValueError(f"expected x1,y1,x2,y2; got {value!r}")
    x1, y1, x2, y2 = vals
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _mask_bbox(mask):
    if mask is None:
        return None
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _mask_centroid(mask):
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)


def _bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _output_arrays(outputs):
    obj_ids = _to_numpy(outputs.get("out_obj_ids", []))
    probs = _to_numpy(outputs.get("out_probs", []))
    masks = _to_numpy(outputs.get("out_binary_masks", []))
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    return obj_ids.astype(int), probs.astype(float), masks.astype(bool)


def _candidate_rows(outputs):
    obj_ids, probs, masks = _output_arrays(outputs)
    rows = []
    for i, obj_id in enumerate(obj_ids):
        mask = masks[i] if i < len(masks) else None
        bbox = _mask_bbox(mask)
        centroid = None
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            centroid = [0.5 * (x1 + x2), 0.5 * (y1 + y2)]
        rows.append(
            {
                "index": int(i),
                "sam3_obj_id": int(obj_id),
                "score": float(probs[i]) if i < len(probs) else 0.0,
                "mask_pixels": int(np.asarray(mask).sum()) if mask is not None else 0,
                "bbox_xyxy": bbox,
                "centroid_xy": centroid,
            }
        )
    return rows


def select_prompt_object(outputs, target_point=None, target_bbox=None, select_mode="auto"):
    obj_ids, probs, masks = _output_arrays(outputs)
    if len(obj_ids) == 0:
        return None, []

    candidates = []
    if select_mode == "auto":
        if target_point is not None:
            select_mode = "point"
        elif target_bbox is not None:
            select_mode = "bbox"
        else:
            select_mode = "score"

    for i, obj_id in enumerate(obj_ids):
        mask = masks[i]
        bbox = _mask_bbox(mask)
        centroid = _mask_centroid(mask)
        score = float(probs[i]) if i < len(probs) else 0.0
        reason = "score"
        if select_mode == "point" and target_point is not None:
            point = np.array(target_point, dtype=np.float32)
            hit_bonus = 1.0
            x, y = int(round(point[0])), int(round(point[1]))
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                hit_bonus = 10.0
            dist = float(np.linalg.norm((centroid if centroid is not None else point) - point))
            score = hit_bonus + float(probs[i]) - 0.002 * dist
            reason = f"point dist={dist:.1f} hit={hit_bonus > 1.0}"
        elif select_mode == "bbox" and target_bbox is not None:
            iou = _bbox_iou(bbox, target_bbox)
            score = 5.0 * iou + float(probs[i])
            reason = f"bbox_iou={iou:.3f}"
        candidates.append((score, int(obj_id), i, reason))
    candidates.sort(reverse=True)
    rows = _candidate_rows(outputs)
    for score, obj_id, i, reason in candidates:
        rows[i]["selection_score"] = float(score)
        rows[i]["selection_reason"] = reason
    return candidates[0][1], rows


def mask_for_obj(outputs, selected_obj_id):
    obj_ids, _probs, masks = _output_arrays(outputs)
    for i, obj_id in enumerate(obj_ids):
        if int(obj_id) == int(selected_obj_id):
            return masks[i]
    return None


def candidate_masks(outputs):
    obj_ids, probs, masks = _output_arrays(outputs)
    rows = []
    for i, obj_id in enumerate(obj_ids):
        if i >= len(masks):
            continue
        mask = masks[i]
        if mask is None or not np.asarray(mask).any():
            continue
        centroid = _mask_centroid(mask)
        rows.append(
            {
                "sam3_obj_id": int(obj_id),
                "score": float(probs[i]) if i < len(probs) else 0.0,
                "mask": mask,
                "bbox_xyxy": _mask_bbox(mask),
                "centroid_xy": centroid,
            }
        )
    return rows


def choose_frame_mask(
    outputs,
    preferred_obj_id=None,
    target_bbox=None,
    target_point=None,
    reference_centroid=None,
):
    candidates = candidate_masks(outputs)
    if not candidates:
        return None, None
    target_point_arr = None if target_point is None else np.array(target_point, dtype=np.float32)
    ref_arr = None if reference_centroid is None else np.array(reference_centroid, dtype=np.float32)
    scored = []
    for row in candidates:
        mask = row["mask"]
        centroid = row["centroid_xy"]
        score = float(row["score"])
        reasons = [f"sam3={score:.3f}"]
        if preferred_obj_id is not None and int(row["sam3_obj_id"]) == int(preferred_obj_id):
            score += 1.0
            reasons.append("preferred_id")
        if target_bbox is not None:
            iou = _bbox_iou(row["bbox_xyxy"], target_bbox)
            score += 2.0 * iou
            reasons.append(f"bbox_iou={iou:.3f}")
        if target_point_arr is not None and centroid is not None:
            x, y = int(round(target_point_arr[0])), int(round(target_point_arr[1]))
            hit = 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]
            dist = float(np.linalg.norm(centroid - target_point_arr))
            score += (1.0 if hit else 0.0) - 0.002 * dist
            reasons.append(f"point_dist={dist:.1f},hit={hit}")
        if ref_arr is not None and centroid is not None:
            dist = float(np.linalg.norm(centroid - ref_arr))
            score -= 0.01 * dist
            reasons.append(f"ref_dist={dist:.1f}")
        scored.append((score, row, ";".join(reasons)))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best, reason = scored[0]
    info = {
        "sam3_obj_id": int(best["sam3_obj_id"]),
        "selection_score": float(best_score),
        "selection_reason": reason,
    }
    return best["mask"], info


def find_cached_checkpoint(version: str) -> str | None:
    if version == "sam3.1":
        repo_dir = "models--facebook--sam3.1"
        ckpt_name = "sam3.1_multiplex.pt"
    else:
        repo_dir = "models--facebook--sam3"
        ckpt_name = "sam3.pt"
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    candidates = sorted((hf_home / "hub" / repo_dir / "snapshots").glob(f"*/{ckpt_name}"))
    return str(candidates[-1]) if candidates else None


def write_mask_qc(path, summary):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_video_frames(video_path):
    """Load video frames for visualization."""
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
    else:
        IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
        frame_paths = [
            os.path.join(video_path, p)
            for p in os.listdir(video_path)
            if os.path.splitext(p)[-1].lower() in IMAGE_EXTS
        ]
        try:
            frame_paths.sort(
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            )
        except ValueError:
            frame_paths.sort()
        return [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in frame_paths]


def get_highest_score_obj(outputs):
    """From a frame output dict, return the obj_id and mask with the highest score."""
    obj_ids = outputs["out_obj_ids"]
    probs = outputs["out_probs"]
    masks = outputs["out_binary_masks"]

    if len(obj_ids) == 0:
        return None, None, None

    best_idx = np.argmax(probs)
    return obj_ids[best_idx], probs[best_idx], masks[best_idx]


def interactive_click_prompt(frame, obj_id=""):
    """
    Open a window showing `frame` (RGB array). The user can:
      - Left-click  to add a positive point (green dot)
      - Right-click to add a negative point (red dot)
      - Press Enter or Space to confirm
      - Press 'u' to undo the last point
      - Press Esc to abort

    Returns (points, labels) as numpy arrays, or (None, None) if cancelled.
    """
    points = []
    labels = []
    window_name = f"SAM3 [{obj_id}] - Click to add points (Enter=confirm, U=undo, Esc=cancel)"

    def _redraw():
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
        for (x, y), lbl in zip(points, labels):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(img, (int(x), int(y)), 6, color, -1)
            cv2.circle(img, (int(x), int(y)), 6, (255, 255, 255), 1)
        return img

    def _on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((x, y))
            labels.append(0)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(frame.shape[1], 1280), min(frame.shape[0], 720))
    cv2.setMouseCallback(window_name, _on_mouse)

    print("Left-click = positive point (green), Right-click = negative point (red)")
    print("Press Enter/Space to confirm, 'u' to undo, Esc to cancel")

    while True:
        cv2.imshow(window_name, _redraw())
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 32):  # Enter or Space
            break
        elif key == 27:  # Esc
            points, labels = None, None
            break
        elif key == ord("u") and points:  # Undo
            points.pop()
            labels.pop()

    cv2.destroyWindow(window_name)

    if points is None or len(points) == 0:
        return None, None

    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)


def overlay_mask_on_frame(frame, mask, color=(30, 144, 255), alpha=0.5):
    """Overlay a binary mask on a frame with the given color and alpha."""
    overlay = frame.copy()
    overlay[mask] = (
        (1 - alpha) * overlay[mask] + alpha * np.array(color, dtype=np.uint8)
    ).astype(np.uint8)
    # Draw contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def main():
    args = parse_args()

    assert args.text is not None or args.points is not None or args.click, (
        "Must provide either --text, --points, or --click prompt"
    )

    if args.output_dir is None:
        if args.video.endswith(".mp4"):
            video_dir = os.path.dirname(args.video) or "."
        else:
            video_dir = args.video
        args.output_dir = os.path.join(video_dir, "video_segmentation")

    # Determine prompt name for the output video filename
    if args.text is not None:
        prompt_name = args.text.replace(" ", "_")
    elif args.click:
        prompt_name = "click"
    else:
        prompt_name = "points"

    masks_base_dir = os.path.join(args.output_dir, "masks")
    overlay_dir = os.path.join(args.output_dir, "overlays")
    os.makedirs(masks_base_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Interactive click mode: collect clicks BEFORE loading the model, because
    # CUDA/PyTorch initialization can corrupt the Qt libraries that cv2 uses
    # for its GUI, causing a segfault.
    click_points = None
    click_labels = None
    if args.click:
        video_frames = load_video_frames(args.video)
        frame = video_frames[args.frame_idx]

        print(f"Opening frame {args.frame_idx} for interactive clicking...")
        click_points, click_labels = interactive_click_prompt(frame, obj_id=args.obj_id)
        if click_points is None:
            print("Cancelled. Exiting.")
            return

    # Import torch and SAM3 after any cv2 GUI work, because sam3's imports
    # load libraries that conflict with cv2's Qt backend and cause segfaults.
    import torch
    import sam3.model_builder as model_builder

    # Build predictor
    print("Loading SAM3 model...")
    checkpoint = args.checkpoint or find_cached_checkpoint(args.version)
    if checkpoint:
        print(f"Using SAM3 checkpoint: {checkpoint}")
    elif os.environ.get("HF_HUB_OFFLINE") == "1":
        raise FileNotFoundError(
            f"No cached checkpoint found for {args.version}; unset HF_HUB_OFFLINE or provide --checkpoint"
        )

    if args.bpe_path is None:
        sam3_root = os.environ.get("SAM3_PKG_DIR")
        if sam3_root:
            bpe = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
            args.bpe_path = bpe if os.path.isfile(bpe) else None

    if hasattr(model_builder, "build_sam3_predictor"):
        predictor = model_builder.build_sam3_predictor(
            checkpoint_path=checkpoint,
            bpe_path=args.bpe_path,
            version=args.version,
            compile=False,
            warm_up=False,
            async_loading_frames=False,
        )
    else:
        if args.version != "sam3":
            raise RuntimeError(
                "The active Do-as-I-Do SAM3 module only exposes "
                "build_sam3_video_predictor(), so it supports base sam3 only. "
                "Use EgoInfinity's SAM3.1 worker for sam3.1 experiments."
            )
        predictor = model_builder.build_sam3_video_predictor(
            checkpoint_path=checkpoint,
            bpe_path=args.bpe_path,
        )

    # Start session
    print(f"Starting session on: {args.video}")
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=args.video,
        )
    )
    session_id = response["session_id"]

    # SAM3 requires an integer obj_id; use 1 internally, keep string for naming
    sam3_obj_id = 1

    # Add prompt
    if args.click:
        h, w = video_frames[0].shape[:2]

        rel_points = [[x / w, y / h] for x, y in click_points.tolist()]
        points_tensor = torch.tensor(rel_points, dtype=torch.float32)
        labels_tensor = torch.tensor(click_labels.tolist(), dtype=torch.int32)

        print(
            f"Adding {len(click_points)} clicked point(s) on frame {args.frame_idx}: "
            f"points={click_points.tolist()}, labels={click_labels.tolist()}"
        )
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=args.frame_idx,
                points=points_tensor,
                point_labels=labels_tensor,
                obj_id=sam3_obj_id,
            )
        )
    elif args.text is not None:
        print(f"Adding text prompt: '{args.text}' on frame {args.frame_idx}")
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=args.frame_idx,
                text=args.text,
                obj_id=sam3_obj_id,
            )
        )
    else:
        # Parse point prompts
        points_list = [
            [float(c) for c in pt.split(",")] for pt in args.points.split(";")
        ]
        labels_list = [int(l) for l in args.point_labels.split(";")]
        assert len(points_list) == len(labels_list), (
            "Number of points must match number of labels"
        )

        # Load a frame to get dimensions for coordinate normalization
        video_frames = load_video_frames(args.video)
        h, w = video_frames[0].shape[:2]
        rel_points = [[x / w, y / h] for x, y in points_list]

        points_tensor = torch.tensor(rel_points, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_list, dtype=torch.int32)

        print(
            f"Adding point prompt on frame {args.frame_idx}: "
            f"points={points_list}, labels={labels_list}"
        )
        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=args.frame_idx,
                points=points_tensor,
                point_labels=labels_tensor,
                obj_id=sam3_obj_id,
            )
        )

    # Show prompt frame results
    prompt_out = response["outputs"]
    target_point = _parse_pair(args.target_point) if args.target_point else None
    target_bbox = _parse_bbox(args.target_bbox) if args.target_bbox else None
    selected_obj_id, prompt_candidates = select_prompt_object(
        prompt_out,
        target_point=target_point,
        target_bbox=target_bbox,
        select_mode=args.select_mode,
    )
    if selected_obj_id is not None:
        best_score = next(
            (row.get("score", 0.0) for row in prompt_candidates if row["sam3_obj_id"] == int(selected_obj_id)),
            0.0,
        )
        print(
            f"Prompt frame {args.frame_idx}: selected obj_id={selected_obj_id}, score={best_score:.4f}"
        )
    else:
        print("No objects detected on the prompt frame.")

    # Propagate through video
    print("Propagating through video...")
    outputs_per_frame = {}
    for resp in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[resp["frame_index"]] = resp["outputs"]

    # Load video frames for overlays and video
    if "video_frames" not in locals():
        video_frames = load_video_frames(args.video)

    # Save masks, overlays, and tracked video
    sorted_frames = sorted(outputs_per_frame.keys())
    h, w = video_frames[0].shape[:2]
    out_video_path = os.path.join(args.output_dir, f"tracked_{args.obj_id}_{prompt_name}.mp4")

    print(f"Saving {len(sorted_frames)} frames to {args.output_dir}/...")
    chosen_masks = {}
    chosen_infos = {}
    frame_count = len(video_frames)
    anchor = min(max(int(args.frame_idx), 0), frame_count - 1)

    # Propagation can omit the prompt frame even though add_prompt returned a
    # valid object. Prefer propagation, then preserve the official prompt
    # response as the anchor fallback. This changes no SAM3 inference result.
    anchor_sources = (
        (outputs_per_frame.get(anchor), "propagation"),
        (prompt_out, "add_prompt"),
    )
    for anchor_outputs, anchor_source in anchor_sources:
        if anchor_outputs is None:
            continue
        mask, info = choose_frame_mask(
            anchor_outputs,
            preferred_obj_id=selected_obj_id,
            target_bbox=target_bbox,
            target_point=target_point,
        )
        if mask is not None:
            chosen_masks[anchor] = mask
            chosen_infos[anchor] = dict(info or {}, anchor_source=anchor_source)
            break

    ref_centroid = _mask_centroid(chosen_masks[anchor]) if anchor in chosen_masks else None
    for frame_idx in range(anchor + 1, frame_count):
        outputs = outputs_per_frame.get(frame_idx)
        if outputs is None:
            continue
        mask, info = choose_frame_mask(
            outputs,
            preferred_obj_id=selected_obj_id,
            target_bbox=target_bbox,
            target_point=target_point,
            reference_centroid=ref_centroid,
        )
        if mask is not None:
            chosen_masks[frame_idx] = mask
            chosen_infos[frame_idx] = info
            ref_centroid = _mask_centroid(mask)

    ref_centroid = _mask_centroid(chosen_masks[anchor]) if anchor in chosen_masks else None
    for frame_idx in range(anchor - 1, -1, -1):
        outputs = outputs_per_frame.get(frame_idx)
        if outputs is None:
            continue
        mask, info = choose_frame_mask(
            outputs,
            preferred_obj_id=selected_obj_id,
            target_bbox=target_bbox,
            target_point=target_point,
            reference_centroid=ref_centroid,
        )
        if mask is not None:
            chosen_masks[frame_idx] = mask
            chosen_infos[frame_idx] = info
            ref_centroid = _mask_centroid(mask)

    qc_rows = []
    valid_centroids = []
    for frame_idx in range(frame_count):
        best_mask = chosen_masks.get(frame_idx)
        chosen_info = chosen_infos.get(frame_idx)

        # Save mask into per-frame directory as <obj_id>.png
        frame_mask_dir = os.path.join(masks_base_dir, f"frame_{frame_idx:06d}_masks")
        os.makedirs(frame_mask_dir, exist_ok=True)
        if best_mask is not None:
            mask_uint8 = (best_mask * 255).astype(np.uint8)
            centroid = _mask_centroid(best_mask)
            bbox = _mask_bbox(best_mask)
        else:
            mask_uint8 = np.zeros((h, w), dtype=np.uint8)
            centroid = None
            bbox = None
        cv2.imwrite(
            os.path.join(frame_mask_dir, f"{args.obj_id}.png"), mask_uint8
        )
        if centroid is not None:
            valid_centroids.append((frame_idx, centroid))
        qc_rows.append(
            {
                "frame_idx": int(frame_idx),
                "valid": bool(best_mask is not None and np.asarray(best_mask).any()),
                "mask_pixels": int((mask_uint8 > 0).sum()),
                "bbox_xyxy": bbox,
                "centroid_xy": None if centroid is None else [float(centroid[0]), float(centroid[1])],
                "selection": chosen_info,
            }
        )

        # Save overlay
        if frame_idx < len(video_frames):
            frame = video_frames[frame_idx]
            if best_mask is not None:
                overlay = overlay_mask_on_frame(frame, best_mask)
            else:
                overlay = frame
            cv2.imwrite(
                os.path.join(overlay_dir, f"frame_{frame_idx:06d}.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
            )

    jumps = []
    for (f0, c0), (f1, c1) in zip(valid_centroids[:-1], valid_centroids[1:]):
        jumps.append({"from": int(f0), "to": int(f1), "jump_px": float(np.linalg.norm(c1 - c0))})
    valid_count = sum(1 for row in qc_rows if row["valid"])
    valid_ratio = valid_count / max(len(qc_rows), 1)
    bbox_set = {
        tuple(row["bbox_xyxy"])
        for row in qc_rows
        if row["bbox_xyxy"] is not None
    }
    max_jump = max([j["jump_px"] for j in jumps], default=0.0)
    fail_reasons = []
    if valid_ratio < args.min_valid_ratio:
        fail_reasons.append(f"valid_ratio {valid_ratio:.3f} < {args.min_valid_ratio:.3f}")
    if args.max_centroid_jump_px > 0 and max_jump > args.max_centroid_jump_px:
        fail_reasons.append(f"max_centroid_jump_px {max_jump:.1f} > {args.max_centroid_jump_px:.1f}")
    if len(bbox_set) <= 1 and len(qc_rows) > 3:
        fail_reasons.append("constant_bbox_across_clip")

    qc = {
        "source": "SAM3 video predictor",
        "video": args.video,
        "text": args.text,
        "obj_id": args.obj_id,
        "sam3_version": args.version,
        "checkpoint": checkpoint,
        "prompt_frame_idx": args.frame_idx,
        "selected_sam3_obj_id": None if selected_obj_id is None else int(selected_obj_id),
        "target_point": None if target_point is None else [float(target_point[0]), float(target_point[1])],
        "target_bbox": None if target_bbox is None else [float(x) for x in target_bbox],
        "prompt_candidates": prompt_candidates,
        "frames": len(qc_rows),
        "valid_frames": int(valid_count),
        "valid_ratio": float(valid_ratio),
        "max_centroid_jump_px": float(max_jump),
        "centroid_jumps": jumps,
        "constant_bbox": len(bbox_set) <= 1,
        "frame_stats": qc_rows,
        "qa_pass": not fail_reasons,
        "fail_reasons": fail_reasons,
        "mask_video": out_video_path,
    }
    qa_path = args.qa_json or os.path.join(args.output_dir, "mask_qc_summary.json")
    write_mask_qc(qa_path, qc)

    # Encode overlay frames to H264 video using ffmpeg
    print(f"Encoding tracked video with H264: {out_video_path}")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", "24",
            "-i", os.path.join(overlay_dir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            out_video_path,
        ],
        check=True,
    )

    # Cleanup
    predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )
    if hasattr(predictor, "shutdown"):
        predictor.shutdown()

    print(f"Done! Output saved to: {args.output_dir}/")
    print(f"  masks/frame_XXXXXX_masks/{args.obj_id}.png")
    print(f"  overlays/    - frame_000000.png, ...")
    print(f"  tracked_{args.obj_id}_{prompt_name}.mp4")
    print(f"  mask_qc_summary.json qa_pass={qc['qa_pass']}")
    if fail_reasons:
        raise SystemExit("SAM3 mask QA failed: " + "; ".join(fail_reasons))


if __name__ == "__main__":
    main()
