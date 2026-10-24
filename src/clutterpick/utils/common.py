import json
import random
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[3]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT / path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def encode(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)

    path.write_text(
        json.dumps(value, indent=2, default=encode, allow_nan=False), encoding="utf-8"
    )


def load_config(path="configs/default.yaml"):
    path = resolve(path)
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML configuration: {path}\n{error}") from error
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config


def object_ids(config):
    return list(
        config["data"].get(
            "object_ids", range(1, config["data"].get("num_objects", 21) + 1)
        )
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
