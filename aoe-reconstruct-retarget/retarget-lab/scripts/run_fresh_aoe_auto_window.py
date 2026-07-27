#!/usr/bin/env python3
"""Select one clean AoE clip window with production SAM3 preflights.

This is a thin front-end orchestrator.  It never changes SAM3 gates or any
Ego/DAI/SPIDER setting: it runs the existing fresh wrapper in preflight-only
mode for a deterministic, longest-first list of right-trimmed windows, then
runs the unchanged full wrapper once with the first passing window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CONTROLLED_OPTIONS = {
    "--source-run",
    "--matrix-run",
    "--duration-sec",
    "--sam3-preflight-only",
}
RETRYABLE_TEMPORAL_PHRASES = (
    "temporal centroid jump",
    "mask area changes too abruptly",
    "temporal coverage",
    "valid frame",
    "empty mask",
    "target disappear",
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantize_candidates(values: str, ego_fps: float) -> list[float]:
    if ego_fps <= 0:
        raise ValueError("ego_fps must be positive")
    candidates: set[int] = set()
    for raw in values.split(","):
        raw = raw.strip()
        if not raw:
            continue
        duration = float(raw)
        frames = int(round(duration * ego_fps))
        if frames <= 0:
            raise ValueError("candidate durations must be positive")
        candidates.add(frames)
    if not candidates:
        raise ValueError("at least one candidate duration is required")
    return [frames / ego_fps for frames in sorted(candidates, reverse=True)]


def validate_base_args(values: list[str]) -> None:
    for token in values:
        option = token.split("=", 1)[0]
        if option in CONTROLLED_OPTIONS:
            raise ValueError(
                f"{option} is controlled by the auto-window runner; remove it from arguments after --"
            )


def classify_preflight_failure(prompt_gate: dict | None, mask_qc: dict | None) -> str:
    if not prompt_gate:
        return "non_retryable_missing_prompt_gate"
    messages = [str(value) for value in (prompt_gate.get("errors") or [])]
    messages.extend(str(value) for value in ((mask_qc or {}).get("fail_reasons") or []))
    if not messages:
        return "non_retryable_unclassified"
    lowered = [message.lower() for message in messages]
    if all(any(phrase in message for phrase in RETRYABLE_TEMPORAL_PHRASES) for message in lowered):
        return "retryable_temporal_window_failure"
    return "non_retryable_prompt_or_identity_failure"


def load_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--matrix-run", required=True)
    parser.add_argument(
        "--window-selection-mode",
        choices=("auto", "direct"),
        default="auto",
        help="auto runs longest-first SAM3 preflights; direct launches one full run immediately",
    )
    parser.add_argument("--candidate-durations-sec")
    parser.add_argument("--direct-duration-sec", type=float)
    parser.add_argument("--ego-fps", type=float, default=15.0)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight-prefix")
    parser.add_argument("base_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    base_args = list(args.base_args)
    if base_args[:1] == ["--"]:
        base_args = base_args[1:]
    validate_base_args(base_args)
    if args.window_selection_mode == "auto":
        if not args.candidate_durations_sec:
            parser.error("--candidate-durations-sec is required in auto mode")
        candidates = quantize_candidates(args.candidate_durations_sec, args.ego_fps)
    else:
        if args.direct_duration_sec is None or args.direct_duration_sec <= 0:
            parser.error("--direct-duration-sec must be positive in direct mode")
        candidates = []
    runner = args.runner.resolve()
    if not runner.is_file():
        raise SystemExit(f"fresh runner not found: {runner}")
    prefix = args.preflight_prefix or f"{args.source_run}__window_preflight"
    manifest = {
        "schema_version": 1,
        "status": "running",
        "policy": {
            "mode": args.window_selection_mode,
            "selection": "longest_passing_candidate",
            "boundary_change": "right_trim_only_same_annotation_anchor",
            "candidate_order": "descending_duration",
            "ego_fps": args.ego_fps,
            "thresholds_modified": False,
            "backend_modified": False,
            "formal_reconstruction_count": 0,
        },
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "source_run": args.source_run,
        "matrix_run": args.matrix_run,
        "candidate_durations_sec": candidates,
        "base_args": base_args,
        "candidates": [],
        "selected_duration_sec": None,
        "formal_run": None,
    }
    write_manifest(args.manifest, manifest)

    if args.window_selection_mode == "direct":
        direct_duration = quantize_candidates(str(args.direct_duration_sec), args.ego_fps)[0]
        manifest["policy"].update(
            {
                "selection": "disabled_direct_user_window",
                "boundary_change": "none",
                "formal_reconstruction_count": 1,
            }
        )
        manifest["selected_duration_sec"] = direct_duration
        formal_command = [
            str(runner),
            *base_args,
            "--duration-sec",
            f"{direct_duration:.12g}",
            "--source-run",
            args.source_run,
            "--matrix-run",
            args.matrix_run,
        ]
        manifest["formal_run"] = {
            "command": formal_command,
            "return_code": None,
            "window_preflight_bypassed": True,
        }
        write_manifest(args.manifest, manifest)
        completed = subprocess.run(formal_command, check=False)
        manifest["formal_run"]["return_code"] = completed.returncode
        manifest["status"] = "ok" if completed.returncode == 0 else "formal_run_failed"
        write_manifest(args.manifest, manifest)
        return completed.returncode

    selected: float | None = None
    for index, duration in enumerate(candidates, start=1):
        run_name = f"{prefix}_{index:02d}"
        run_root = args.experiments_root / run_name
        command = [
            str(runner),
            *base_args,
            "--duration-sec",
            f"{duration:.12g}",
            "--source-run",
            run_name,
            "--matrix-run",
            f"{run_name}__unused_matrix",
            "--sam3-preflight-only",
        ]
        completed = subprocess.run(command, check=False)
        clip = run_root / "intermediates/trajectory_6dof/do_as_i_do/reconstruction/clip"
        prompt_path = clip / "video_segmentation/prompt_gate.json"
        mask_path = clip / "video_segmentation/mask_qc_summary.json"
        prompt_gate = load_json(prompt_path)
        mask_qc = load_json(mask_path)
        passed = (
            completed.returncode == 0
            and (prompt_gate or {}).get("status") == "ok"
            and not ((prompt_gate or {}).get("errors") or [])
            and (mask_qc or {}).get("qa_pass") is True
        )
        classification = "passed" if passed else classify_preflight_failure(prompt_gate, mask_qc)
        record = {
            "index": index,
            "duration_sec": duration,
            "source_run": run_name,
            "command": command,
            "return_code": completed.returncode,
            "passed": passed,
            "classification": classification,
            "prompt_gate": str(prompt_path),
            "prompt_gate_sha256": sha256_file(prompt_path),
            "prompt_gate_errors": (prompt_gate or {}).get("errors") or [],
            "mask_qc": str(mask_path),
            "mask_qc_sha256": sha256_file(mask_path),
            "mask_qc_fail_reasons": (mask_qc or {}).get("fail_reasons") or [],
        }
        manifest["candidates"].append(record)
        write_manifest(args.manifest, manifest)
        if passed:
            selected = duration
            break
        if completed.returncode == 86 or not classification.startswith("retryable_"):
            manifest["status"] = "failed_non_retryable_preflight"
            write_manifest(args.manifest, manifest)
            return completed.returncode or 4

    if selected is None:
        manifest["status"] = "failed_no_clean_window"
        write_manifest(args.manifest, manifest)
        return 4

    manifest["selected_duration_sec"] = selected
    formal_command = [
        str(runner),
        *base_args,
        "--duration-sec",
        f"{selected:.12g}",
        "--source-run",
        args.source_run,
        "--matrix-run",
        args.matrix_run,
    ]
    manifest["policy"]["formal_reconstruction_count"] = 1
    manifest["formal_run"] = {"command": formal_command, "return_code": None}
    write_manifest(args.manifest, manifest)
    completed = subprocess.run(formal_command, check=False)
    manifest["formal_run"]["return_code"] = completed.returncode
    manifest["status"] = "ok" if completed.returncode == 0 else "formal_run_failed"
    write_manifest(args.manifest, manifest)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
