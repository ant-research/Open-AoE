#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATE_RE = re.compile(r"(20\d{6})")


def parse_date_token(name: str) -> str | None:
    matches = DATE_RE.findall(name)
    return max(matches) if matches else None


def is_before_cutoff(date_token: str | None, cutoff: str | None) -> bool:
    if not cutoff or not date_token:
        return False
    return date_token < cutoff


def entry_size_kib(path: Path) -> int | None:
    try:
        output = subprocess.check_output(["du", "-sk", "--", str(path)], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    try:
        return int(output.split()[0])
    except Exception:
        return None


def build_report(entry: Path, experiments_root: Path, *, size: bool) -> dict[str, Any]:
    date_token = parse_date_token(entry.name)
    report: dict[str, Any] = {
        "name": entry.name,
        "path": str(entry),
        "relative_path": str(entry.relative_to(experiments_root)),
        "date_token": date_token,
        "kind": "dir" if entry.is_dir() else "file" if entry.is_file() else "other",
        "size_kib": entry_size_kib(entry) if size else None,
    }
    return report


def select_entries(args: argparse.Namespace) -> list[Path]:
    root = args.experiments_root.resolve()
    if root.name != "experiments":
        raise SystemExit(f"refusing to prune a non-experiments root: {root}")
    if not root.exists():
        raise SystemExit(f"experiments root does not exist: {root}")

    include_res = [re.compile(item) for item in args.include_regex]
    exclude_res = [re.compile(item) for item in args.exclude_regex]
    selected: list[Path] = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        name = entry.name
        date_token = parse_date_token(name)
        reasons = []
        if is_before_cutoff(date_token, args.before_date):
            reasons.append("before_date")
        if include_res and any(pattern.search(name) for pattern in include_res):
            reasons.append("include_regex")
        if args.name and name in args.name:
            reasons.append("explicit_name")
        if not reasons:
            continue
        if any(pattern.search(name) for pattern in exclude_res):
            continue
        if args.max_entries is not None and len(selected) >= args.max_entries:
            break
        selected.append(entry)
    if not args.matrix_reuse_only:
        return selected
    reuse_dirs = []
    for entry in selected:
        reuse = entry / "reuse"
        manifest = entry / "reuse_12_demo_manifest.json"
        if "_matrix12_" in entry.name and entry.is_dir() and manifest.is_file() and reuse.is_dir():
            reuse_dirs.append(reuse)
    return reuse_dirs


def delete_entry(entry: Path, timeout_sec: int) -> dict[str, Any]:
    result: dict[str, Any] = {"name": entry.name, "path": str(entry), "status": "pending"}
    process = subprocess.Popen(
        ["rm", "-rf", "--", str(entry)],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + timeout_sec if timeout_sec > 0 else None
    while process.poll() is None:
        if deadline is not None and time.monotonic() >= deadline:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.stderr is not None:
                process.stderr.close()
            result["status"] = "timeout"
            result["timeout_sec"] = timeout_sec
            result["pid"] = process.pid
            return result
        time.sleep(0.2)

    stderr = process.stderr.read() if process.stderr is not None else ""
    if process.stderr is not None:
        process.stderr.close()
    if process.returncode is None:
        result["status"] = "timeout"
        result["timeout_sec"] = timeout_sec
        return result
    result["returncode"] = process.returncode
    result["stderr"] = stderr[-2000:]
    result["status"] = "deleted" if process.returncode == 0 else "failed"
    return result


def write_manifest(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run or prune top-level retarget-lab experiment outputs.")
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--before-date", help="Select names whose embedded YYYYMMDD token is before this date.")
    parser.add_argument("--include-regex", action="append", default=[], help="Select names matching this regex.")
    parser.add_argument("--exclude-regex", action="append", default=[], help="Skip names matching this regex.")
    parser.add_argument("--name", action="append", default=[], help="Select an exact top-level entry name.")
    parser.add_argument("--max-entries", type=int)
    parser.add_argument("--with-size", action="store_true", help="Run du -sk for selected entries.")
    parser.add_argument(
        "--matrix-reuse-only",
        action="store_true",
        help="For selected matrix runs, prune only reuse/ deep copies and preserve manifests/cells/logs.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually delete selected entries.")
    parser.add_argument("--per-entry-timeout-sec", type=int, default=900)
    parser.add_argument("--stop-on-timeout", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    selected = select_entries(args)
    selected_reports = [
        build_report(entry, args.experiments_root.resolve(), size=args.with_size) for entry in selected
    ]
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiments_root": str(args.experiments_root.resolve()),
        "apply": args.apply,
        "criteria": {
            "before_date": args.before_date,
            "include_regex": args.include_regex,
            "exclude_regex": args.exclude_regex,
            "name": args.name,
            "max_entries": args.max_entries,
            "matrix_reuse_only": args.matrix_reuse_only,
        },
        "selected_count": len(selected_reports),
        "selected": selected_reports,
        "deleted": [],
    }

    if args.apply:
        deleted = []
        for entry in selected:
            result = delete_entry(entry, args.per_entry_timeout_sec)
            deleted.append(result)
            if result["status"] == "timeout" and args.stop_on_timeout:
                break
        payload["deleted"] = deleted
        payload["deleted_count"] = sum(1 for item in deleted if item.get("status") == "deleted")
        payload["timeout_count"] = sum(1 for item in deleted if item.get("status") == "timeout")

    write_manifest(args.manifest, payload)
    return 0 if not payload.get("timeout_count") else 124


if __name__ == "__main__":
    raise SystemExit(main())
