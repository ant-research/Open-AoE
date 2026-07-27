#!/usr/bin/env python3
"""Launch a pristine EgoInfinity SAM3 worker across the pinned SAM3 API change.

The pinned EgoInfinity worker calls ``download_ckpt_from_hf(version=...)``.
The pinned SAM3 checkout exposes the same operation without that deprecated
keyword.  This launcher adapts only that call in memory and then executes the
unmodified worker source.  It deliberately supports only ``--version sam3``.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    known, worker_args = parser.parse_known_args()
    if "--version" not in worker_args:
        raise SystemExit("the compatibility launcher requires --version sam3")
    version_index = worker_args.index("--version") + 1
    if version_index >= len(worker_args) or worker_args[version_index] != "sam3":
        raise SystemExit("only the pinned SAM3 worker API is supported")

    worker = known.worker.expanduser().resolve(strict=True)
    checkpoint = (
        known.checkpoint.expanduser().resolve(strict=True)
        if known.checkpoint is not None
        else None
    )
    from sam3 import model_builder

    native_download = model_builder.download_ckpt_from_hf

    def download_ckpt_from_hf(*, version: str | None = None):
        if version not in (None, "sam3"):
            raise RuntimeError(f"unsupported SAM3 compatibility version: {version}")
        return str(checkpoint) if checkpoint is not None else native_download()

    model_builder.download_ckpt_from_hf = download_ckpt_from_hf
    sys.argv = [str(worker), *worker_args]
    runpy.run_path(str(worker), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
