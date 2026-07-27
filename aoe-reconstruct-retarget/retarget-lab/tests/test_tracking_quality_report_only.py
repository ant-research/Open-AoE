from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_backend_runners_do_not_use_tracking_diagnostics_as_exit_gates() -> None:
    for runner in (
        ROOT / "scripts" / "run_do_as_i_do_official_retarget.sh",
        ROOT / "scripts" / "run_spider_retarget.sh",
    ):
        source = runner.read_text(encoding="utf-8")
        assert "evaluate_mjwp_object_tracking.py" not in source
        assert "validate_hard_tracking_admission" not in source
        assert "numerical_invalid" not in source
