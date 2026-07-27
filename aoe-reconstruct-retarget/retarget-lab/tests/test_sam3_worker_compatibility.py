from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_uses_lab_owned_sam3_compatibility_launcher() -> None:
    pipeline = (ROOT / "scripts" / "run_v4_two_full_pipelines.sh").read_text(
        encoding="utf-8"
    )
    assert "scripts/compatibility/run_sam3_worker.py" in pipeline
    assert '--worker "$repo_root/third_party/EgoInfinity/scripts/sam3_worker.py"' in pipeline
    assert '$sam3_repo/assets/bpe_simple_vocab_16e6.txt.gz' in pipeline
    assert '$sam3_repo/sam3/assets/bpe_simple_vocab_16e6.txt.gz' not in pipeline


def test_compatibility_launcher_is_narrow_and_does_not_patch_backend_files() -> None:
    launcher = (ROOT / "scripts" / "compatibility" / "run_sam3_worker.py").read_text(
        encoding="utf-8"
    )
    assert 'worker_args[version_index] != "sam3"' in launcher
    assert "str(checkpoint) if checkpoint is not None else native_download()" in launcher
    assert "runpy.run_path" in launcher
    assert "write_text" not in launcher
    assert "subprocess" not in launcher
