#!/usr/bin/env python3
"""Run EgoInfinity's SAM3D worker against the current notebook API.

EgoInfinity's pinned worker passes the historical sampling controls directly to
``notebook.inference.Inference.__call__``.  Current SAM3D keeps the same
controls on ``Inference._pipeline.run`` instead.  This launcher adapts that
interface at process startup without modifying either third-party checkout.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any


_LEGACY_PIPELINE_KEYS = {
    "stage1_inference_steps",
    "stage2_inference_steps",
    "use_stage1_distillation",
    "use_stage2_distillation",
    "decode_formats",
}


class _InferenceCompatibilityProxy:
    def __init__(self, inference: Any):
        self._inference = inference

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inference, name)

    def __call__(self, image: Any, mask: Any, seed: int | None = None, **kwargs: Any) -> dict:
        parameters = inspect.signature(self._inference.__call__).parameters
        if all(key in parameters for key in kwargs):
            return self._inference(image, mask, seed=seed, **kwargs)

        unsupported = set(kwargs) - _LEGACY_PIPELINE_KEYS
        if unsupported:
            raise TypeError(f"unsupported SAM3D compatibility arguments: {sorted(unsupported)}")
        pipeline = getattr(self._inference, "_pipeline", None)
        merge = getattr(self._inference, "merge_mask_to_rgba", None)
        if pipeline is None or not callable(merge) or not callable(getattr(pipeline, "run", None)):
            raise TypeError("SAM3D Inference exposes neither the legacy call API nor _pipeline.run")

        rgba = merge(image, mask)
        return pipeline.run(
            rgba,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=True,
            with_texture_baking=True,
            with_layout_postprocess=True,
            use_vertex_color=False,
            **kwargs,
        )


def _load_worker(path: Path):
    spec = importlib.util.spec_from_file_location("egoinfinity_sam3d_worker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load worker: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", required=True)
    args, worker_args = parser.parse_known_args()
    worker_path = Path(args.worker).resolve()
    if not worker_path.is_file():
        raise FileNotFoundError(worker_path)

    worker = _load_worker(worker_path)
    original = worker._run_reconstruction

    def compatible_run(inference: Any, request: dict) -> dict:
        return original(_InferenceCompatibilityProxy(inference), request)

    worker._run_reconstruction = compatible_run
    sys.argv = [os.fspath(worker_path), *worker_args]
    worker.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
