#!/usr/bin/env python3
"""Run pristine Fast-SAM3D tracking without the Jupyter notebook collision.

The pinned Fast-SAM3D repository keeps its inference helpers in a top-level
``notebook/`` directory without ``__init__.py``.  A separately installed
Jupyter ``notebook`` package otherwise wins module resolution.  This launcher
binds only the local namespace in memory and executes the untouched upstream
``track_object.py`` entrypoint.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import types
from pathlib import Path
from typing import Any


COMPATIBILITY_DIR = Path(__file__).resolve().parent
if str(COMPATIBILITY_DIR) not in sys.path:
    sys.path.insert(0, str(COMPATIBILITY_DIR))
from open3d_vector_dtype_compat import (
    bind_open3d_compat_root_from_env,
    install_open3d_vector_dtype_compat,
)


def bind_local_notebook(fast_sam3d_root: Path) -> Path:
    notebook_dir = fast_sam3d_root / "notebook"
    inference = notebook_dir / "inference.py"
    if not inference.is_file():
        raise FileNotFoundError(
            f"missing pristine Fast-SAM3D inference helper: {inference}"
        )
    package = types.ModuleType("notebook")
    package.__file__ = str(notebook_dir)
    package.__package__ = "notebook"
    package.__path__ = [str(notebook_dir)]
    sys.modules["notebook"] = package
    return inference


def install_cached_dinov2_torch_hub_compat() -> bool:
    """Use an already populated Torch Hub DINOv2 checkout without GitHub I/O.

    ``torch.hub.load(..., source="github")`` performs a GitHub repository
    validation request even when both the repository and checkpoint are
    already present under ``TORCH_HOME``.  That request is unnecessary for a
    pinned, populated runtime cache and can fail on an otherwise offline-ready
    machine.  Redirect only the exact upstream DINOv2 repository to Torch
    Hub's existing local checkout; all other calls retain native behavior.
    """
    import torch

    hub = torch.hub
    if getattr(hub.load, "__retarget_lab_cached_dinov2__", False):
        return True
    cached_repo = Path(hub.get_dir()) / "facebookresearch_dinov2_main"
    if not (cached_repo / "hubconf.py").is_file():
        return False
    native_load = hub.load

    def load(
        repo_or_dir: str,
        model: str,
        *args: Any,
        source: str = "github",
        **kwargs: Any,
    ):
        if repo_or_dir == "facebookresearch/dinov2" and source == "github":
            return native_load(
                str(cached_repo), model, *args, source="local", **kwargs
            )
        return native_load(
            repo_or_dir, model, *args, source=source, **kwargs
        )

    load.__retarget_lab_cached_dinov2__ = True
    hub.load = load
    return True


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fast-sam3d-root", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    root = known.fast_sam3d_root.resolve()
    entrypoint = root / "track_object.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(
            f"missing pristine Fast-SAM3D entrypoint: {entrypoint}"
        )

    bind_local_notebook(root)
    bind_open3d_compat_root_from_env()
    install_open3d_vector_dtype_compat()
    install_cached_dinov2_torch_hub_compat()
    os.environ.setdefault(
        "CONDA_PREFIX", os.environ.get("SAM3D_CUDA_HOME", sys.prefix)
    )
    env_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = os.pathsep.join(
        [env_bin, os.environ.get("PATH", "")]
    ).rstrip(os.pathsep)
    sys.path.insert(0, str(root))
    sys.argv = [str(entrypoint), *remaining]
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
