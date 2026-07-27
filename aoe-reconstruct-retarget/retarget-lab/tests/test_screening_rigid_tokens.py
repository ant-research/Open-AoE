from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "diagnostics"))

from screen_aoe_retarget_candidates import is_supported_rigid_object  # noqa: E402


def test_common_rigid_tool_and_container_names_are_supported() -> None:
    assert is_supported_rigid_object("pink thermos")
    assert is_supported_rigid_object("yellow scraper")
    assert is_supported_rigid_object("white kettle")
    assert is_supported_rigid_object("metal colander")
    assert is_supported_rigid_object("wooden ruler")
    assert is_supported_rigid_object("hairbrush")
    assert is_supported_rigid_object("scissors")


def test_deformable_name_is_not_promoted_as_rigid() -> None:
    assert not is_supported_rigid_object("pink sweater")
