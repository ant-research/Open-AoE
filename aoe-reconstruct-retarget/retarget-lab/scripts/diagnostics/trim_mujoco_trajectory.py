#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Drop leading qpos samples from a MuJoCo trajectory npz, optionally resampling to a fixed frame count.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-index", required=True, type=int)
    parser.add_argument(
        "--alignment-kind",
        default="",
        help="Human/machine readable alignment semantics recorded in the metadata JSON.",
    )
    parser.add_argument("--source-frame-count", type=int, default=0)
    parser.add_argument("--target-fps", type=float, default=0.0)
    parser.add_argument("--sim-dt", type=float, default=0.0)
    parser.add_argument("--min-duration-coverage", type=float, default=0.95)
    parser.add_argument("--warmup-steps", type=int, default=-1)
    parser.add_argument(
        "--target-count",
        type=int,
        default=0,
        help="When >0, resample trimmed time-varying arrays to exactly this many rows.",
    )
    parser.add_argument("--metadata-json", type=Path, default=None)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if "qpos" not in data:
        raise KeyError(f"{args.input} does not contain qpos")
    qpos_raw = np.asarray(data["qpos"])
    if qpos_raw.ndim < 2:
        raise ValueError(f"{args.input} qpos must be at least 2-D, got {qpos_raw.shape}")
    qpos = qpos_raw.reshape(-1, qpos_raw.shape[-1])
    qpos_count = int(qpos.shape[0])
    start = int(max(0, args.start_index))
    if start >= qpos_count:
        raise ValueError(f"start-index {start} leaves no qpos samples from {qpos_count}")

    post_warmup_count = int(qpos_count - start)
    sample_indices: np.ndarray | None = None
    sample_stride: int | None = None
    sampling_semantics = "none"
    if args.target_count > 0 and args.target_fps > 0 and args.sim_dt > 0:
        samples_per_frame = 1.0 / (float(args.target_fps) * float(args.sim_dt))
        sample_stride = int(round(samples_per_frame))
        if sample_stride <= 0 or not np.isclose(
            samples_per_frame, sample_stride, rtol=0.0, atol=1.0e-7
        ):
            raise ValueError(
                "target_fps and sim_dt must define an integer source-frame stride; "
                f"got samples_per_frame={samples_per_frame}"
            )
        sample_indices = np.arange(args.target_count, dtype=np.int64) * sample_stride
        if sample_indices.size and int(sample_indices[-1]) >= post_warmup_count:
            raise ValueError(
                "executed trajectory does not cover the final exact source timestamp: "
                f"required_relative_index={int(sample_indices[-1])}, "
                f"available_post_warmup_count={post_warmup_count}"
            )
        ignored_tail = (
            int(post_warmup_count - int(sample_indices[-1]) - 1)
            if sample_indices.size
            else post_warmup_count
        )
        if ignored_tail != 0:
            raise ValueError(
                "exact source-timestamp alignment rejects unaccounted simulator tail: "
                f"ignored_tail_samples={ignored_tail}"
            )
        if "time" not in data.files:
            raise ValueError(
                "exact source-timestamp alignment requires recorded simulator time"
            )
        raw_time = np.asarray(data["time"], dtype=np.float64).reshape(-1)
        if raw_time.shape[0] != qpos_count:
            raise ValueError(
                f"time/qpos count mismatch: time={raw_time.shape[0]} qpos={qpos_count}"
            )
        if not np.all(np.isfinite(raw_time)):
            raise ValueError("recorded simulator time contains non-finite values")
        time_delta = np.diff(raw_time)
        if np.any(time_delta <= 0) or not np.allclose(
            time_delta, float(args.sim_dt), rtol=1.0e-6, atol=1.0e-9
        ):
            raise ValueError(
                "recorded simulator time is not a strict sim_dt grid: "
                f"sim_dt={args.sim_dt}"
            )
        sampling_semantics = "exact_source_timestamp_indices"

    def resample_rows(arr: np.ndarray, n: int) -> np.ndarray:
        if n <= 0 or len(arr) == n:
            return arr
        if len(arr) == 0:
            return arr
        if sample_indices is not None:
            return arr[sample_indices]
        idx = np.rint(np.linspace(0, len(arr) - 1, n)).astype(int)
        return arr[idx]

    def trim_value(value: np.ndarray) -> tuple[np.ndarray, bool]:
        arr = np.asarray(value)
        if arr.shape == qpos_raw.shape:
            return arr.reshape(qpos_count, qpos_raw.shape[-1])[start:], True
        if arr.ndim >= 1 and int(arr.shape[0]) == qpos_count:
            return arr[start:], True
        if arr.ndim >= 2 and int(np.prod(arr.shape[:2])) == qpos_count:
            return arr.reshape((qpos_count,) + arr.shape[2:])[start:], True
        return arr, False

    if qpos.ndim < 2:
        raise ValueError(f"{args.input} qpos must be at least 2-D, got {qpos.shape}")

    trimmed = {}
    trimmed_keys = []
    source_post_warmup_duration_sec = None
    for key in data.files:
        value, did_trim = trim_value(np.asarray(data[key]))
        if did_trim:
            trimmed_keys.append(key)
            if args.target_count > 0:
                value = resample_rows(value, args.target_count)
            if key == "time" and value.size:
                value = value - value.reshape(-1)[0]
                source_post_warmup_duration_sec = float(value.reshape(-1)[-1])
        trimmed[key] = value

    if source_post_warmup_duration_sec is None and args.sim_dt > 0:
        selected_count = (
            int(sample_indices[-1]) + 1
            if sample_indices is not None and sample_indices.size
            else post_warmup_count
        )
        source_post_warmup_duration_sec = float(
            max(selected_count - 1, 0) * args.sim_dt
        )

    aligned_time_semantics = "source_trajectory_time"
    target_duration_sec = None
    if args.target_count > 0 and args.target_fps > 0:
        # The aligned trajectory has one sample per source RGB frame. Keeping
        # the original simulator timestamps after row resampling makes a
        # sim-time renderer freeze early or reuse the final qpos. Rebase the
        # aligned artifact onto the source clip timeline explicitly.
        trimmed["time"] = np.arange(args.target_count, dtype=np.float64) / float(args.target_fps)
        aligned_time_semantics = "source_clip_frame_time"
        target_duration_sec = float(max(args.target_count - 1, 0) / args.target_fps)
    duration_coverage_ratio = (
        float(source_post_warmup_duration_sec / target_duration_sec)
        if source_post_warmup_duration_sec is not None and target_duration_sec and target_duration_sec > 0
        else None
    )
    complete_source_coverage = (
        duration_coverage_ratio is not None
        and duration_coverage_ratio >= float(args.min_duration_coverage)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **trimmed)
    manifest = {
        "applied": True,
        "reason": (
            "sampled_exact_source_timestamps"
            if args.alignment_kind
            in {
                "sample_exact_source_timestamps",
                "drop_warmup_and_sample_exact_source_timestamps",
            }
            else ("trimmed_and_resampled" if args.target_count > 0 else "trimmed")
        ),
        "input": str(args.input),
        "source_trajectory": str(args.input),
        "output": str(args.output),
        "aligned_trajectory": str(args.output),
        "alignment_kind": args.alignment_kind,
        "start_index": start,
        "warmup_steps": int(args.warmup_steps),
        "source_qpos_shape": list(qpos_raw.shape),
        "source_qpos_count": qpos_count,
        "source_frame_count": int(args.source_frame_count),
        "aligned_qpos_count": int(np.asarray(trimmed["qpos"]).shape[0]),
        "target_frames": int(args.target_count),
        "target_count": int(args.target_count),
        "render_fps": float(args.target_fps),
        "target_fps": float(args.target_fps),
        "aligned_time_semantics": aligned_time_semantics,
        "aligned_duration_sec": target_duration_sec,
        "source_post_warmup_qpos_count": post_warmup_count,
        "source_post_warmup_duration_sec": source_post_warmup_duration_sec,
        "sampling_semantics": sampling_semantics,
        "sample_stride": sample_stride,
        "sample_indices_relative_to_post_warmup": (
            sample_indices.tolist() if sample_indices is not None else None
        ),
        "sample_indices_absolute": (
            (sample_indices + start).tolist() if sample_indices is not None else None
        ),
        "selected_last_post_warmup_index": (
            int(sample_indices[-1])
            if sample_indices is not None and sample_indices.size
            else None
        ),
        "ignored_tail_samples": (
            int(post_warmup_count - int(sample_indices[-1]) - 1)
            if sample_indices is not None and sample_indices.size
            else 0
        ),
        "exact_source_timestamps": bool(sample_indices is not None),
        "duration_coverage_ratio": duration_coverage_ratio,
        "min_duration_coverage": float(args.min_duration_coverage),
        "complete_source_coverage": complete_source_coverage,
        "trimmed_keys": trimmed_keys,
    }
    if args.metadata_json is not None:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
