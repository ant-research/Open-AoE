from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "compatibility" / "run_sam3d_mesh.py"
WRAPPER = ROOT / "scripts" / "run_do_as_i_do_reconstruction_fresh.sh"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("aoe_sam3d_mesh", LAUNCHER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_binds_pristine_notebook_directory_without_writing_backend(tmp_path: Path) -> None:
    module = _load_launcher()
    notebook = tmp_path / "notebook"
    notebook.mkdir()
    inference = notebook / "inference.py"
    inference.write_text("VALUE = 1\n", encoding="utf-8")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert module.bind_local_notebook(tmp_path) == inference
    package = sys.modules["notebook"]
    assert package.__path__ == [str(notebook)]
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert after == before


def test_reconstruction_wrapper_routes_through_lab_launcher() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert 'scripts/compatibility/run_sam3d_mesh.py' in text
    assert '--sam3d-root "$SAM3D_DIR"' in text
    assert "run_sam3d_py generate_mesh_sam3d.py" not in text


def test_reconstruction_prefers_full_sam3d_pipeline_for_mesh_postprocess() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    full = text.index('"checkpoints/hf/pipeline.yaml"')
    mesh_only = text.index('"checkpoints/hf/pipeline_mesh_only.yaml"')
    assert full < mesh_only


def test_launcher_supports_noninteractive_conda_environment() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'os.environ.setdefault(' in text
    assert '"CONDA_PREFIX", os.environ.get("SAM3D_CUDA_HOME", sys.prefix)' in text
    assert 'Path(sys.executable).resolve().parent' in text
    assert 'os.environ["PATH"]' in text


def test_open3d_runtime_root_is_bound_before_open3d_import() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert text.index("bind_open3d_compat_root_from_env()") < text.index(
        "install_open3d_vector_dtype_compat()"
    )


def test_open3d_legacy_vector_inputs_are_normalized_without_value_changes() -> None:
    module = _load_launcher()
    captured: dict[str, np.ndarray] = {}

    def vector3d(values):
        captured["vertices"] = values
        return values

    def vector3i(values):
        captured["faces"] = values
        return values

    utility = types.SimpleNamespace(
        Vector3dVector=vector3d,
        Vector3iVector=vector3i,
    )
    fake_open3d = types.SimpleNamespace(utility=utility)
    module.install_open3d_vector_dtype_compat(fake_open3d, np)

    vertices = np.array([[1.25, 2.5, 3.75]], dtype=np.float32)
    faces = np.array([[0, 2, 1]], dtype=np.int64)
    utility.Vector3dVector(vertices)
    utility.Vector3iVector(faces)

    assert captured["vertices"].dtype == np.float64
    assert captured["vertices"].flags.c_contiguous
    np.testing.assert_array_equal(captured["vertices"], vertices)
    assert captured["faces"].dtype == np.int32
    assert captured["faces"].flags.c_contiguous
    np.testing.assert_array_equal(captured["faces"], faces)


def test_optional_open3d_runtime_root_is_bound_before_import(tmp_path: Path) -> None:
    module = _load_launcher()
    package = tmp_path / "open3d"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    paths = ["existing"]

    resolved = module.bind_open3d_compat_root_from_env(
        {"SAM3D_OPEN3D_COMPAT_ROOT": str(tmp_path)}, paths
    )

    assert resolved == tmp_path.resolve()
    assert paths == [str(tmp_path.resolve()), "existing"]


def test_configured_open3d_runtime_root_fails_closed_when_incomplete(
    tmp_path: Path,
) -> None:
    module = _load_launcher()
    with np.testing.assert_raises(FileNotFoundError):
        module.bind_open3d_compat_root_from_env(
            {"SAM3D_OPEN3D_COMPAT_ROOT": str(tmp_path)}, []
        )
