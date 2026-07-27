from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_matrix_reuse_only_preserves_top_level_evidence(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    matrix = experiments / "box_matrix12_20260708_probe"
    reuse = matrix / "reuse"
    reuse.mkdir(parents=True)
    (reuse / "large_copy.bin").write_bytes(b"copy")
    (matrix / "reuse_12_demo_manifest.json").write_text("{}", encoding="utf-8")
    source = experiments / "box_full_source_20260708_probe"
    (source / "reuse").mkdir(parents=True)
    output = tmp_path / "prune.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "maintenance" / "prune_experiments.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--experiments-root",
            str(experiments),
            "--include-regex",
            "box_",
            "--matrix-reuse-only",
            "--manifest",
            str(output),
        ],
        check=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["apply"] is False
    assert report["criteria"]["matrix_reuse_only"] is True
    assert [item["relative_path"] for item in report["selected"]] == [
        "box_matrix12_20260708_probe/reuse"
    ]
    assert matrix.is_dir()
    assert reuse.is_dir()
    assert source.is_dir()
