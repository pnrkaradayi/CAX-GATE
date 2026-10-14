import json
import os
import random

import numpy as np


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def load_json(path):
    return json.loads(path.read_text())


def calibration_path(cfg, name, seed, split="grouped"):
    return cfg.sub("calibration", split) / f"{name}_seed{seed}.json"


def load_tau(cfg, name="proposed", seed=0, split="grouped") -> float:
    return float(load_json(calibration_path(cfg, name, seed, split))["tau"])
