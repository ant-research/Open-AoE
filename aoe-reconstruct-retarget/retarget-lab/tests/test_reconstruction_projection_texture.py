from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_do_as_i_do_reconstruction_fresh.sh"


def test_projection_assigns_deterministic_visualization_color() -> None:
    """Textureless reconstructed OBJ files must remain renderable.

    The pristine projection entrypoint already exposes ``--object-color``.
    Binding a constant color supplies a vertex texture for visualization only;
    it does not alter mesh geometry, poses, intrinsics, or reconstruction data.
    """

    source = RUNNER.read_text()
    projection = source.split('echo "=== [3b] Project mesh ==="', 1)[1].split(
        'echo "=== [3c] Convert layout to camera frame ==="', 1
    )[0]

    assert "run_project_mesh_combined.py" in projection
    assert '--object-color "0.6,0.9,0.6"' in projection
    assert "--mesh-scale" not in projection
    assert "--translation" not in projection
