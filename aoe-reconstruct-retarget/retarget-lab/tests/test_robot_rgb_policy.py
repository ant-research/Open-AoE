from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_robot_rgb_renderer_is_not_shipped():
    assert not (ROOT / "scripts" / "render_do_as_i_do_robot_rgb_overlay.py").exists()


def test_runtime_scripts_do_not_call_forbidden_robot_rgb_pipeline():
    forbidden = (
        "visualization_mjwp_rgb_overlay",
        "render_do_as_i_do_robot_rgb_overlay",
        "E2FGVI",
        "Phantom Stage 3",
    )
    violations = []
    for path in sorted((ROOT / "scripts").glob("*")):
        if (
            path.name.startswith("._")
            or path.name == "audit_retarget_run.py"
            or path.suffix not in {".py", ".sh"}
        ):
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token.lower() in text.lower():
                violations.append(f"{path.name}: {token}")
    assert violations == []


def test_spider_requires_exact_hand_source_keypoints():
    script = (ROOT / "scripts" / "run_spider_retarget.sh").read_text(encoding="utf-8")
    assert "expected one exact task_info" in script
    assert 'keypoints = task_info_path.parent / str(data_id)' in script
    assert '"source_sha256": binding["keypoints_sha256"]' in script
    assert 'traj_assets / "source_trajectory_keypoints.npz"' not in script
