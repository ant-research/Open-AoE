#!/usr/bin/env python3
"""Run pristine SAM3D mesh entrypoint without the Jupyter notebook collision.

The pinned SAM3D repository stores inference helpers in a top-level
``notebook/`` directory without ``__init__.py``.  If the Jupyter ``notebook``
package is installed, Python resolves that regular package first and
``notebook.inference`` becomes unreachable.  Bind only the package namespace
in memory and execute the untouched upstream entrypoint.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import types
from pathlib import Path


COMPATIBILITY_DIR = Path(__file__).resolve().parent
if str(COMPATIBILITY_DIR) not in sys.path:
    sys.path.insert(0, str(COMPATIBILITY_DIR))
from open3d_vector_dtype_compat import (
    bind_open3d_compat_root_from_env,
    install_open3d_vector_dtype_compat,
)


def bind_local_notebook(sam3d_root: Path) -> Path:
    notebook_dir = sam3d_root / "notebook"
    inference = notebook_dir / "inference.py"
    if not inference.is_file():
        raise FileNotFoundError(f"missing pristine SAM3D inference helper: {inference}")
    package = types.ModuleType("notebook")
    package.__file__ = str(notebook_dir)
    package.__package__ = "notebook"
    package.__path__ = [str(notebook_dir)]
    sys.modules["notebook"] = package
    return inference


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sam3d-root", type=Path, required=True)
    known, remaining = parser.parse_known_args()
    sam3d_root = known.sam3d_root.resolve()
    entrypoint = sam3d_root / "generate_mesh_sam3d.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"missing pristine SAM3D entrypoint: {entrypoint}")
    bind_local_notebook(sam3d_root)
    bind_open3d_compat_root_from_env()
    install_open3d_vector_dtype_compat()
    # Some upstream inference helpers assume an activated Conda shell and use
    # CONDA_PREFIX only to derive CUDA_HOME. Non-interactive runners do not
    # necessarily export it, so bind the configured toolkit (or this Python
    # environment) without changing the backend file.
    os.environ.setdefault(
        "CONDA_PREFIX", os.environ.get("SAM3D_CUDA_HOME", sys.prefix)
    )
    # nvdiffrast builds its CUDA extension lazily and locates ``ninja`` via
    # PATH.  Calling this interpreter by absolute path does not activate its
    # Conda environment, so expose the interpreter's own bin directory.
    env_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = os.pathsep.join(
        [env_bin, os.environ.get("PATH", "")]
    ).rstrip(os.pathsep)
    sys.path.insert(0, str(sam3d_root))
    sys.argv = [str(entrypoint), *remaining]
    runpy.run_path(str(entrypoint), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
