from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_native_adapter():
    path = SCRIPTS / "prepare_do_as_i_do_scene_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "prepare_do_as_i_do_scene_adapter_raw_video_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ADAPTER = load_native_adapter()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_raw_dir(root: Path) -> Path:
    raw_dir = root / "raw_dir"
    frames = raw_dir / "all_frames"
    frames.mkdir(parents=True)
    (frames / "000000.png").write_bytes(b"frame-zero")
    (frames / "000001.png").write_bytes(b"frame-one")
    (raw_dir / "raw.mp4").write_bytes(b"native-video")
    (raw_dir / "fresh_input_manifest.json").write_text(
        json.dumps({"source_fps": 30.0}) + "\n", encoding="utf-8"
    )
    return raw_dir


def write_materialization_manifest(raw_dir: Path) -> Path:
    frames = ADAPTER._exact_frame_grid(raw_dir / "all_frames")
    manifest = {
        "schema_version": 1,
        "status": "ok",
        "kind": "materialized_exact_all_frames",
        "source_input_manifest": {
            "path": str((raw_dir / "fresh_input_manifest.json").resolve()),
            "sha256": sha256(raw_dir / "fresh_input_manifest.json"),
        },
        "all_frames": {
            "path": str((raw_dir / "all_frames").resolve()),
            "frame_count": len(frames),
            "first": frames[0].name,
            "last": frames[-1].name,
            "aggregate_sha256": ADAPTER._aggregate_frame_sha256(frames),
        },
        "output": {
            "path": str((raw_dir / "raw.mp4").resolve()),
            "sha256": sha256(raw_dir / "raw.mp4"),
            "frame_count": len(frames),
            "fps": 30.0,
        },
    }
    path = raw_dir / "raw_video_materialization.json"
    path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return path


class NativeRawVideoProvenanceTests(unittest.TestCase):
    def test_adapter_output_rejects_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            target = root / "target"
            target.mkdir()
            sentinel = target / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            output_link = root / "output"
            output_link.symlink_to(target, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                ADAPTER.safe_adapter_output_dir(source, output_link)

            self.assertTrue(output_link.is_symlink())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_adapter_output_rejects_source_ancestor_or_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            source.mkdir()
            for output in (root, source, source / "nested-output"):
                with self.subTest(output=output):
                    with self.assertRaisesRegex(ValueError, "must be disjoint"):
                        ADAPTER.safe_adapter_output_dir(source, output)

    def test_native_reconstruction_video_is_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = make_raw_dir(Path(temp_dir))

            provenance = ADAPTER.build_raw_video_provenance(raw_dir)

            self.assertEqual(provenance["status"], "validated")
            self.assertEqual(provenance["kind"], "reconstruction_raw_video")
            self.assertEqual(
                provenance["raw_video"]["sha256"], sha256(raw_dir / "raw.mp4")
            )
            self.assertEqual(
                provenance["source_input_manifest"]["sha256"],
                sha256(raw_dir / "fresh_input_manifest.json"),
            )

    def test_native_materialized_video_binds_manifest_and_exact_frame_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = make_raw_dir(Path(temp_dir))
            materialization = write_materialization_manifest(raw_dir)

            provenance = ADAPTER.build_raw_video_provenance(raw_dir)

            self.assertEqual(provenance["status"], "validated")
            self.assertEqual(provenance["kind"], "materialized_exact_all_frames")
            self.assertEqual(provenance["frame_count"], 2)
            self.assertEqual(provenance["fps"], 30.0)
            self.assertEqual(
                provenance["materialization_manifest"]["sha256"],
                sha256(materialization),
            )

    def test_native_materialized_video_rejects_changed_same_length_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = make_raw_dir(Path(temp_dir))
            write_materialization_manifest(raw_dir)
            (raw_dir / "all_frames" / "000001.png").write_bytes(
                b"different-scene-frame"
            )

            with self.assertRaisesRegex(
                ValueError, "all_frames_sha256_mismatch"
            ):
                ADAPTER.build_raw_video_provenance(raw_dir)

    def test_native_adapter_propagates_provenance_into_output_manifest(self) -> None:
        source = (SCRIPTS / "prepare_do_as_i_do_scene_adapter.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('"fresh_input_manifest.json",', source)
        self.assertIn('"raw_video_materialization.json",', source)
        self.assertIn(
            "raw_video_provenance = build_raw_video_provenance(output_dir)", source
        )
        self.assertIn('"raw_video_provenance": raw_video_provenance,', source)

    def test_both_adapters_use_common_safe_output_and_video_provenance(self) -> None:
        native = (SCRIPTS / "prepare_do_as_i_do_scene_adapter.py").read_text(
            encoding="utf-8"
        )
        ego = (
            SCRIPTS / "prepare_egoinfinity_do_as_i_do_raw_dir.py"
        ).read_text(encoding="utf-8")

        self.assertIn(
            "output_dir = safe_adapter_output_dir(source_dir, args.output_dir)",
            native,
        )
        self.assertIn(
            "output_dir = safe_adapter_output_dir(source_dir, args.output_dir)",
            ego,
        )
        self.assertNotIn(
            "output_dir = args.output_dir.expanduser().resolve()", native
        )
        self.assertNotIn(
            "output_dir = args.output_dir.expanduser().resolve()", ego
        )
        self.assertIn(
            "return build_common_raw_video_provenance(raw_dir)", ego
        )


if __name__ == "__main__":
    unittest.main()
