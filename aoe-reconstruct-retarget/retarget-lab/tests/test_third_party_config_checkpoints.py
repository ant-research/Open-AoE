import hashlib
import os
from pathlib import Path

from scripts.check_third_party_config import (
    FASTSAM3D_CHECKPOINT_FILES,
    SAM3D_CHECKPOINT_FILES,
    check_executable,
    check_python_package_root,
    check_sam3d_checkpoint_bundle,
)


def materialize_bundle(root: Path) -> None:
    root.mkdir(parents=True)
    for relative, size in SAM3D_CHECKPOINT_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        os.truncate(path, size)


def test_complete_sam3d_checkpoint_bundle_passes(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "hf"
    materialize_bundle(checkpoint_dir)

    result = check_sam3d_checkpoint_bundle(checkpoint_dir)

    assert result["ok"] is True
    assert result["missing"] == []
    assert result["size_mismatches"] == []


def test_missing_sam3d_checkpoint_is_reported(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "hf"
    materialize_bundle(checkpoint_dir)
    (checkpoint_dir / "slat_generator.yaml").unlink()

    result = check_sam3d_checkpoint_bundle(checkpoint_dir)

    assert result["ok"] is False
    assert result["missing"] == ["slat_generator.yaml"]


def test_truncated_sam3d_checkpoint_is_reported(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "hf"
    materialize_bundle(checkpoint_dir)
    truncated = checkpoint_dir / "ss_generator.ckpt"
    os.truncate(truncated, SAM3D_CHECKPOINT_FILES["ss_generator.ckpt"] - 1)

    result = check_sam3d_checkpoint_bundle(checkpoint_dir)

    assert result["ok"] is False
    assert result["size_mismatches"] == [
        {
            "path": "ss_generator.ckpt",
            "expected_size": SAM3D_CHECKPOINT_FILES["ss_generator.ckpt"],
            "actual_size": SAM3D_CHECKPOINT_FILES["ss_generator.ckpt"] - 1,
        }
    ]


def test_truncated_fast_sam3d_checkpoint_is_reported(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "fast-checkpoints" / "hf"
    checkpoint_dir.mkdir(parents=True)
    for relative, size in FASTSAM3D_CHECKPOINT_FILES.items():
        checkpoint = checkpoint_dir / relative
        checkpoint.touch()
        os.truncate(checkpoint, size)
    truncated = checkpoint_dir / "ss_generator.ckpt"
    os.truncate(truncated, FASTSAM3D_CHECKPOINT_FILES["ss_generator.ckpt"] - 1)

    result = check_sam3d_checkpoint_bundle(
        checkpoint_dir,
        name="FASTSAM3D_CHECKPOINT_DIR",
        required_files=FASTSAM3D_CHECKPOINT_FILES,
    )

    assert result["name"] == "FASTSAM3D_CHECKPOINT_DIR"
    assert result["ok"] is False
    assert result["size_mismatches"] == [
        {
            "path": "ss_generator.ckpt",
            "expected_size": FASTSAM3D_CHECKPOINT_FILES["ss_generator.ckpt"],
            "actual_size": FASTSAM3D_CHECKPOINT_FILES["ss_generator.ckpt"] - 1,
        }
    ]


def test_same_size_corruption_is_reported_by_sha256_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import check_third_party_config as config

    checkpoint_dir = tmp_path / "checkpoints" / "hf"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "model.ckpt"
    checkpoint.write_bytes(b"good")
    monkeypatch.setattr(config, "SAM3D_CHECKPOINT_FILES", {"model.ckpt": 4})
    expected = hashlib.sha256(b"good").hexdigest()
    manifest = tmp_path / "checkpoints.sha256"
    manifest.write_text(f"{expected}  model.ckpt\n", encoding="utf-8")
    checkpoint.write_bytes(b"evil")

    result = config.check_sam3d_checkpoint_bundle(checkpoint_dir, manifest)

    assert result["ok"] is False
    assert result["hash_mismatches"] == [
        {
            "path": "model.ckpt",
            "expected_sha256": expected,
            "actual_sha256": hashlib.sha256(b"evil").hexdigest(),
        }
    ]


def test_executable_root_audit_uses_resolved_symlink_target(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    python = runtime_root / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    configured_root = tmp_path / "configured"
    configured_root.mkdir()
    configured = configured_root / "python"
    configured.symlink_to(python)

    result = check_executable("SAM3D_PYTHON", str(configured), configured_root)

    assert result["resolved"] == str(python.resolve())
    assert result["required_root"] == str(configured_root.resolve())
    assert result["resolved_within_required_root"] is False
    assert result["ok"] is False


def test_executable_root_audit_accepts_runtime_inside_root(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    python = runtime_root / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)

    result = check_executable("SAM3D_PYTHON", str(python), runtime_root)

    assert result["resolved_within_required_root"] is True
    assert result["ok"] is True


def test_open3d_compat_package_root_requires_importable_package(tmp_path: Path) -> None:
    result = check_python_package_root(
        "SAM3D_OPEN3D_COMPAT_ROOT", str(tmp_path), "open3d/__init__.py"
    )
    assert result["ok"] is False

    package = tmp_path / "open3d"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    result = check_python_package_root(
        "SAM3D_OPEN3D_COMPAT_ROOT", str(tmp_path), "open3d/__init__.py"
    )
    assert result["ok"] is True
    assert result["resolved"] == str(tmp_path.resolve())
