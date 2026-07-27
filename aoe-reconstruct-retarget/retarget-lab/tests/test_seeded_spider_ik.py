from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_seeded_spider_ik_wrapper_is_syntax_valid() -> None:
    source = (ROOT / "scripts" / "run_seeded_spider_ik.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert 'SPIDER_IK_SEED' in source
    assert "np.random.seed(seed)" in source
    assert 'SPIDER_IK_ENTRYPOINT' in source
    assert "tyro.cli(import_module(module_name).main)" in source
