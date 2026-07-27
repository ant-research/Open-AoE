"""Open3D package and ABI compatibility for pristine SAM3D entrypoints."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def bind_open3d_compat_root_from_env(
    environ=None, path_entries=None
) -> Path | None:
    """Prepend an isolated Open3D package root before Open3D is imported."""
    environ = os.environ if environ is None else environ
    path_entries = sys.path if path_entries is None else path_entries
    value = environ.get("SAM3D_OPEN3D_COMPAT_ROOT", "").strip()
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    if not (root / "open3d" / "__init__.py").is_file():
        raise FileNotFoundError(
            "SAM3D_OPEN3D_COMPAT_ROOT does not contain open3d/__init__.py: "
            f"{root}"
        )
    root_text = str(root)
    if root_text in path_entries:
        path_entries.remove(root_text)
    path_entries.insert(0, root_text)
    return root


def install_open3d_vector_dtype_compat(open3d_module=None, numpy_module=None) -> None:
    """Normalize NumPy arrays to Open3D's legacy vector ABI contract.

    The pinned Open3D 0.18 CUDA bindings dereference an invalid function
    pointer when ``Vector3dVector`` receives float32 or ``Vector3iVector``
    receives int64.  Upstream SAM3D naturally produces exactly those dtypes.
    Keep both backend checkouts pristine and normalize only the arguments at
    this integration boundary; coordinates and indices are unchanged.
    """
    if open3d_module is None:
        import open3d as open3d_module
    if numpy_module is None:
        import numpy as numpy_module

    utility = open3d_module.utility
    if getattr(utility, "_aoe_vector_dtype_compat", False):
        return

    vector3d = utility.Vector3dVector
    vector3i = utility.Vector3iVector

    def safe_vector3d(values):
        values = numpy_module.ascontiguousarray(values, dtype=numpy_module.float64)
        return vector3d(values)

    def safe_vector3i(values):
        values = numpy_module.ascontiguousarray(values, dtype=numpy_module.int32)
        return vector3i(values)

    utility.Vector3dVector = safe_vector3d
    utility.Vector3iVector = safe_vector3i
    utility._aoe_vector_dtype_compat = True
