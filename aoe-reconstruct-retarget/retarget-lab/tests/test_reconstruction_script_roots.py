from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_do_as_i_do_reconstruction_fresh.sh"
COMPAT = ROOT / "scripts" / "compatibility" / "reconstruction"


def test_pristine_reconstruction_root_is_independently_configurable() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert 'recon_root="${DAI_RECON_ROOT:-' in source
    assert 'script_dir="${DAI_RECON_SCRIPTS_DIR:-$recon_root/scripts}"' in source
    assert 'module_root="${DAI_RECON_MODULE_ROOT:-$recon_root/modules}"' in source
    assert 'script_dir="$repo_root/third_party/do-as-i-do/reconstruction/scripts"' not in source


def test_aoe_input_helpers_are_lab_owned_not_backend_patches() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert 'integration_script_dir="$repo_root/scripts/compatibility/reconstruction"' in source
    for name in (
        "export_aoe_hands_to_dai_npz.py",
        "render_aoe_hand_masks_for_dai.py",
    ):
        assert (COMPAT / name).is_file()
        assert f'run_sam3d_py "$integration_script_dir/{name}"' in source
        assert f"run_sam3d_py {name}" not in source


def test_exporter_defaults_to_explicit_hawor_camera_tracks() -> None:
    source = (COMPAT / "export_aoe_hands_to_dai_npz.py").read_text(encoding="utf-8")

    assert 'choices=("pred-cam", "hawor-transform", "auto")' in source
    assert 'default="pred-cam"' in source
