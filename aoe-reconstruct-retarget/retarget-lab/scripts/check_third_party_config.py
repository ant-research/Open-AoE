#!/usr/bin/env python3
"""Read-only preflight for the public third-party configuration surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


REQUIRED_EXECUTABLES = (
    "EGOINFINITY_PYTHON",
    "RETARGETING_PYTHON",
    "SPIDER_PYTHON",
    "SAM3_PYTHON",
    "SAM3D_PYTHON",
)
REQUIRED_DIRS = (
    "AOE_DATA_ROOT",
    "EGOINFINITY_ROOT",
    "SAM3_REPO",
    "SAM3D_REPO",
    "DINOV2_LOCAL_REPO",
    "GEOCALIB_DIR",
)

# File sizes from the checkpoint bundle published for the pinned SAM 3D
# Objects runtime.  Checking the complete bundle prevents a partially copied
# gated download from passing the otherwise path-only configuration audit.
SAM3D_CHECKPOINT_FILES = {
    "pipeline.yaml": 3_548,
    "ss_generator.ckpt": 6_690_136_964,
    "ss_generator.yaml": 5_076,
    "slat_generator.ckpt": 4_906_537_684,
    "slat_generator.yaml": 1_986,
    "ss_decoder.ckpt": 147_609_242,
    "ss_decoder.yaml": 244,
    "slat_decoder_gs.ckpt": 171_476_155,
    "slat_decoder_gs.yaml": 576,
    "slat_decoder_gs_4.ckpt": 170_269_801,
    "slat_decoder_gs_4.yaml": 575,
    "slat_decoder_mesh.ckpt": 363_726_862,
    "slat_decoder_mesh.yaml": 300,
}

# Fast-SAM3D consumes the same six official model checkpoints, alongside its
# own configuration YAML files.  Audit these separately because a complete
# SAM3D Objects bundle does not prove that the Fast-SAM3D checkout can load
# its tracking pipeline.
FASTSAM3D_CHECKPOINT_FILES = {
    name: size
    for name, size in SAM3D_CHECKPOINT_FILES.items()
    if name.endswith(".ckpt")
}


def resolve_executable(value: str) -> str | None:
    if "/" in value:
        path = Path(value).expanduser()
        return str(path.resolve()) if path.is_file() and os.access(path, os.X_OK) else None
    found = shutil.which(value)
    return str(Path(found).resolve()) if found else None


def check_executable(
    name: str, value: str, required_root: Path | None = None
) -> dict[str, object]:
    resolved = resolve_executable(value) if value else None
    resolved_path = Path(resolved) if resolved else None
    normalized_root = required_root.expanduser().resolve() if required_root else None
    within_required_root = (
        resolved_path.is_relative_to(normalized_root)
        if resolved_path is not None and normalized_root is not None
        else None
    )
    return {
        "name": name,
        "kind": "executable",
        "value": value,
        "resolved": resolved,
        "required_root": str(normalized_root) if normalized_root else None,
        "resolved_within_required_root": within_required_root,
        "ok": bool(resolved)
        and (within_required_root is not False),
    }


def check_python_package_root(
    name: str, value: str, required_file: str
) -> dict[str, object]:
    root = Path(value).expanduser() if value else None
    required_path = root / required_file if root else None
    return {
        "name": name,
        "kind": "python_package_root",
        "value": value,
        "resolved": str(root.resolve()) if root and root.is_dir() else None,
        "required_file": required_file,
        "ok": bool(required_path and required_path.is_file()),
    }


def git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sha256_manifest(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, filename = line.split(maxsplit=1)
        hashes[filename.lstrip("* ")] = digest.lower()
    return hashes


def check_sam3d_checkpoint_bundle(
    path: Path,
    sha256_manifest: Path | None = None,
    *,
    name: str = "SAM3D_CHECKPOINT_DIR",
    required_files: dict[str, int] | None = None,
) -> dict[str, object]:
    missing: list[str] = []
    size_mismatches: list[dict[str, object]] = []
    hash_mismatches: list[dict[str, object]] = []
    manifest_hashes: dict[str, str] = {}
    manifest_error: str | None = None
    if sha256_manifest is not None:
        try:
            manifest_hashes = load_sha256_manifest(sha256_manifest)
        except (OSError, ValueError) as error:
            manifest_error = str(error)
    files = required_files or SAM3D_CHECKPOINT_FILES
    for relative, expected_size in files.items():
        checkpoint = path / relative
        if not checkpoint.is_file():
            missing.append(relative)
            continue
        actual_size = checkpoint.stat().st_size
        if actual_size != expected_size:
            size_mismatches.append(
                {"path": relative, "expected_size": expected_size, "actual_size": actual_size}
            )
            continue
        if sha256_manifest is not None:
            expected_hash = manifest_hashes.get(relative)
            if expected_hash is None:
                hash_mismatches.append(
                    {"path": relative, "error": "missing from SHA256 manifest"}
                )
                continue
            actual_hash = sha256_file(checkpoint)
            if actual_hash != expected_hash:
                hash_mismatches.append(
                    {
                        "path": relative,
                        "expected_sha256": expected_hash,
                        "actual_sha256": actual_hash,
                    }
                )
    return {
        "name": name,
        "kind": "checkpoint_bundle",
        "value": str(path),
        "resolved": str(path.resolve()) if path.is_dir() else None,
        "missing": missing,
        "size_mismatches": size_mismatches,
        "sha256_manifest": str(sha256_manifest) if sha256_manifest else None,
        "manifest_error": manifest_error,
        "hash_mismatches": hash_mismatches,
        "ok": path.is_dir()
        and not missing
        and not size_mismatches
        and manifest_error is None
        and not hash_mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="write the report to this path")
    parser.add_argument(
        "--required-executable-root",
        type=Path,
        default=(
            Path(os.environ["AOE_REQUIRED_EXECUTABLE_ROOT"])
            if os.environ.get("AOE_REQUIRED_EXECUTABLE_ROOT")
            else None
        ),
        help=(
            "require every resolved Python executable to live under this root; "
            "also configurable with AOE_REQUIRED_EXECUTABLE_ROOT"
        ),
    )
    args = parser.parse_args()

    checks: list[dict[str, object]] = []
    for name in REQUIRED_EXECUTABLES:
        value = os.environ.get(name, "")
        checks.append(check_executable(name, value, args.required_executable_root))

    for name in REQUIRED_DIRS:
        value = os.environ.get(name, "")
        path = Path(value).expanduser() if value else None
        ok = bool(path and path.is_dir())
        checks.append(
            {
                "name": name,
                "kind": "directory",
                "value": value,
                "resolved": str(path.resolve()) if ok and path else None,
                "git_head": git_head(path) if ok and path else None,
                "ok": ok,
            }
        )

    open3d_compat_value = os.environ.get("SAM3D_OPEN3D_COMPAT_ROOT", "")
    if open3d_compat_value:
        checks.append(
            check_python_package_root(
                "SAM3D_OPEN3D_COMPAT_ROOT",
                open3d_compat_value,
                "open3d/__init__.py",
            )
        )

    sam3d_repo = os.environ.get("SAM3D_REPO", "")
    checkpoint_value = os.environ.get("SAM3D_CHECKPOINT_DIR", "")
    checkpoint_dir = (
        Path(checkpoint_value).expanduser()
        if checkpoint_value
        else Path(sam3d_repo).expanduser() / "checkpoints" / "hf"
        if sam3d_repo
        else Path()
    )
    manifest_value = os.environ.get("SAM3D_CHECKPOINT_SHA256_MANIFEST", "")
    manifest_path = Path(manifest_value).expanduser() if manifest_value else None
    checks.append(check_sam3d_checkpoint_bundle(checkpoint_dir, manifest_path))

    fastsam3d_dir_value = os.environ.get("FASTSAM3D_DIR", "")
    fast_checkpoint_value = os.environ.get("FASTSAM3D_CHECKPOINT_DIR", "")
    fast_checkpoint_dir = (
        Path(fast_checkpoint_value).expanduser()
        if fast_checkpoint_value
        else Path(fastsam3d_dir_value).expanduser() / "checkpoints" / "hf"
        if fastsam3d_dir_value
        else Path()
    )
    fast_manifest_value = os.environ.get(
        "FASTSAM3D_CHECKPOINT_SHA256_MANIFEST", manifest_value
    )
    fast_manifest_path = (
        Path(fast_manifest_value).expanduser() if fast_manifest_value else None
    )
    checks.append(
        check_sam3d_checkpoint_bundle(
            fast_checkpoint_dir,
            fast_manifest_path,
            name="FASTSAM3D_CHECKPOINT_DIR",
            required_files=FASTSAM3D_CHECKPOINT_FILES,
        )
    )

    report = {"status": "ok" if all(item["ok"] for item in checks) else "error", "checks": checks}
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
