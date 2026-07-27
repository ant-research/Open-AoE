#!/usr/bin/env python3
from __future__ import annotations

import os
import random
from importlib import import_module

import numpy as np
import tyro


def seed_process() -> int:
    seed = int(os.environ.get("SPIDER_IK_SEED", "0"))
    random.seed(seed)
    np.random.seed(seed)
    return seed


if __name__ == "__main__":
    seed_process()
    module_name = os.environ.get("SPIDER_IK_ENTRYPOINT", "spider.preprocess.ik")
    tyro.cli(import_module(module_name).main)
