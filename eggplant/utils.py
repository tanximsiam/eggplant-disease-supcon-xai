"""Shared helpers: logging, hashing, seeding, atomic IO, environment capture."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

_LOG_FILE: Path | None = None


def set_log_file(path: Path | None) -> None:
    global _LOG_FILE
    _LOG_FILE = path
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    """Timestamped foreground print that plays nicely with tqdm bars (and optional file log)."""
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    tqdm.write(line, file=sys.stdout)
    sys.stdout.flush()
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a") as f:
            f.write(line + "\n")


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stable_hash(obj, n: int = 12) -> str:
    """Hash of a JSON-serialisable object (sorted keys)."""
    s = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(s).hexdigest()[:n]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def derive_seed(*parts) -> int:
    """Independent deterministic RNG stream for (seed, purpose, ...)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__, "git_revision": git_revision(),
    }
    import torchvision
    info["torchvision"] = torchvision.__version__
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return info
