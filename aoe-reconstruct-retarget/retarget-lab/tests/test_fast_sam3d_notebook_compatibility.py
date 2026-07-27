import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "compatibility" / "run_fast_sam3d_tracker.py"
WRAPPER = ROOT / "scripts" / "run_do_as_i_do_reconstruction_fresh.sh"


def load_launcher():
    spec = importlib.util.spec_from_file_location("fast_sam3d_launcher", LAUNCHER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_binds_pristine_fast_sam3d_notebook_namespace(tmp_path: Path) -> None:
    notebook = tmp_path / "notebook"
    notebook.mkdir()
    inference = notebook / "inference.py"
    inference.write_text("SENTINEL = True\n", encoding="utf-8")

    old = sys.modules.get("notebook")
    try:
        module = load_launcher()
        assert module.bind_local_notebook(tmp_path) == inference
        assert sys.modules["notebook"].__path__ == [str(notebook)]
    finally:
        if old is None:
            sys.modules.pop("notebook", None)
        else:
            sys.modules["notebook"] = old


def test_wrapper_executes_pristine_tracker_without_runtime_patching() -> None:
    source = WRAPPER.read_text(encoding="utf-8")

    assert "scripts/compatibility/run_fast_sam3d_tracker.py" in source
    assert '--fast-sam3d-root "$FASTSAM3D_DIR"' in source
    assert "patch_fast_sam3d_cluster_guard.py" not in source


def test_tracker_installs_shared_open3d_dtype_compatibility() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "from open3d_vector_dtype_compat import" in source
    assert "bind_open3d_compat_root_from_env()" in source
    assert "install_open3d_vector_dtype_compat()" in source
    assert source.index("bind_open3d_compat_root_from_env()") < source.index(
        "install_open3d_vector_dtype_compat()"
    )


def test_cached_dinov2_redirect_avoids_github_validation(
    tmp_path: Path, monkeypatch
) -> None:
    hub_dir = tmp_path / "hub"
    cached_repo = hub_dir / "facebookresearch_dinov2_main"
    cached_repo.mkdir(parents=True)
    (cached_repo / "hubconf.py").write_text("# fixture\n", encoding="utf-8")
    calls = []

    def native_load(repo_or_dir, model, *args, source="github", **kwargs):
        calls.append((repo_or_dir, model, source, args, kwargs))
        return "loaded"

    fake_hub = types.SimpleNamespace(load=native_load, get_dir=lambda: str(hub_dir))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(hub=fake_hub))
    module = load_launcher()

    assert module.install_cached_dinov2_torch_hub_compat() is True
    assert fake_hub.load(
        "facebookresearch/dinov2", "dinov2_vitl14_reg", source="github"
    ) == "loaded"
    assert calls == [
        (
            str(cached_repo),
            "dinov2_vitl14_reg",
            "local",
            (),
            {},
        )
    ]


def test_cached_dinov2_redirect_leaves_other_repositories_native(
    tmp_path: Path, monkeypatch
) -> None:
    hub_dir = tmp_path / "hub"
    cached_repo = hub_dir / "facebookresearch_dinov2_main"
    cached_repo.mkdir(parents=True)
    (cached_repo / "hubconf.py").write_text("# fixture\n", encoding="utf-8")
    calls = []

    def native_load(repo_or_dir, model, *args, source="github", **kwargs):
        calls.append((repo_or_dir, model, source))
        return "native"

    fake_hub = types.SimpleNamespace(load=native_load, get_dir=lambda: str(hub_dir))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(hub=fake_hub))
    module = load_launcher()
    module.install_cached_dinov2_torch_hub_compat()

    assert fake_hub.load("owner/other", "model", source="github") == "native"
    assert calls == [("owner/other", "model", "github")]
