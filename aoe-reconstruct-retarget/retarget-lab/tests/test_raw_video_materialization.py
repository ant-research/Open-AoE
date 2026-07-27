from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


materializer = load_module(
    "materialize_reused_raw_video",
    ROOT / "scripts" / "materialize_reused_raw_video.py",
)
adapter = load_module(
    "prepare_egoinfinity_do_as_i_do_raw_dir_materialization_test",
    ROOT / "scripts" / "prepare_egoinfinity_do_as_i_do_raw_dir.py",
)


def make_source(root: Path) -> Path:
    source = root / "source"
    frames = source / "all_frames"
    frames.mkdir(parents=True)
    (frames / "000000.png").write_bytes(b"frame-zero")
    (frames / "000001.png").write_bytes(b"frame-one")
    (source / "raw").mkdir()
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    (source / "fresh_input_manifest.json").write_text(
        json.dumps(
            {
                "source_fps": 30.0,
                "source_video": "/dataset/source.mp4",
                "start_frame": 10,
                "end_frame": 12,
                "start_sec": 1.0,
                "duration": 2.0 / 30.0,
                "ffmpeg_command": ["ffmpeg", "recorded-original-command"],
            }
        ),
        encoding="utf-8",
    )
    return source


def test_probe_local_materialization_and_adapter_binding(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    output = tmp_path / "probe" / "clip"

    def fake_run(command, **kwargs):
        if command[0] == "fake-ffmpeg":
            Path(command[-1]).write_bytes(b"encoded-exact-frame-grid")
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[0] == "fake-ffprobe":
            payload = {
                "streams": [
                    {
                        "avg_frame_rate": "30/1",
                        "r_frame_rate": "30/1",
                        "nb_frames": "2",
                        "width": 1280,
                        "height": 720,
                        "duration": str(2.0 / 30.0),
                    }
                ]
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        raise AssertionError(command)

    with mock.patch.object(materializer.subprocess, "run", side_effect=fake_run), mock.patch.object(
        materializer,
        "git_state",
        return_value={"commit": "abc", "dirty": False, "status_short": ""},
    ):
        manifest = materializer.materialize(
            source,
            output,
            ffmpeg_bin="fake-ffmpeg",
            ffprobe_bin="fake-ffprobe",
        )

    assert not (source / "raw.mp4").exists()
    assert (output / "all_frames").is_symlink()
    assert manifest["kind"] == "materialized_exact_all_frames"
    assert manifest["all_frames"]["frame_count"] == 2
    assert len(manifest["all_frames"]["aggregate_sha256"]) == 64
    assert manifest["output"]["sha256"] == materializer.sha256_file(output / "raw.mp4")

    binding = adapter.build_raw_video_provenance(output)
    assert binding["status"] == "validated"
    assert binding["kind"] == "materialized_exact_all_frames"
    assert binding["frame_count"] == 2
    assert binding["fps"] == 30.0
    assert binding["raw_video"]["sha256"] == manifest["output"]["sha256"]
    assert binding["materialization_manifest"]["sha256"]


def test_adapter_rejects_materialized_video_sha_mismatch(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    (source / "raw.mp4").write_bytes(b"actual-video")
    (source / "raw_video_materialization.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ok",
                "kind": "materialized_exact_all_frames",
                "output": {
                    "path": str((source / "raw.mp4").resolve()),
                    "sha256": "0" * 64,
                    "frame_count": 2,
                    "fps": 30.0,
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="output_sha256_mismatch"):
        adapter.build_raw_video_provenance(source)


def test_materializer_rejects_noncontiguous_frame_grid(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "000000.png").write_bytes(b"zero")
    (frames / "000002.png").write_bytes(b"two")
    with pytest.raises(ValueError, match="contiguous zero-based"):
        materializer.exact_frame_grid(frames)


def test_materializer_rejects_symlink_output_without_touching_target(
    tmp_path: Path,
) -> None:
    source = make_source(tmp_path)
    target = tmp_path / "unrelated-target"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    output_link = tmp_path / "probe-link"
    output_link.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        materializer.materialize(source, output_link, force=True)

    assert output_link.is_symlink()
    assert marker.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("which", ["source_parent", "source_child"])
def test_materializer_rejects_overlapping_source_and_output(
    tmp_path: Path,
    which: str,
) -> None:
    source = make_source(tmp_path)
    output = tmp_path if which == "source_parent" else source / "probe"
    with pytest.raises(ValueError, match="must be disjoint"):
        materializer.materialize(source, output, force=True)


def test_failed_probe_leaves_no_partial_output_and_preserves_forced_target(
    tmp_path: Path,
) -> None:
    source = make_source(tmp_path)
    output = tmp_path / "probe" / "clip"
    output.mkdir(parents=True)
    marker = output / "keep.txt"
    marker.write_text("old-complete-output", encoding="utf-8")

    def fake_run(command, **kwargs):
        if command[0] == "fake-ffmpeg":
            Path(command[-1]).write_bytes(b"encoded-but-not-yet-admitted")
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[0] == "fake-ffprobe":
            payload = {
                "streams": [
                    {
                        "avg_frame_rate": "30/1",
                        "nb_read_frames": "1",
                        "width": 1280,
                        "height": 720,
                        "duration": str(1.0 / 30.0),
                    }
                ]
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        raise AssertionError(command)

    with mock.patch.object(materializer.subprocess, "run", side_effect=fake_run):
        with pytest.raises(ValueError, match="frame count mismatch"):
            materializer.materialize(
                source,
                output,
                ffmpeg_bin="fake-ffmpeg",
                ffprobe_bin="fake-ffprobe",
                force=True,
            )

    assert marker.read_text(encoding="utf-8") == "old-complete-output"
    assert not (output / "raw.mp4").exists()
    assert not list(output.parent.glob(".clip.materialize.*"))
