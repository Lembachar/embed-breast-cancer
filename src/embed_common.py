"""Paths, timestamped logging, heartbeat, credentials and run manifests."""
from __future__ import annotations

import ctypes
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
BUCKET = "embed-dataset-open"
REGION = "us-east-1"


class Paths:
    def __init__(self, dry_run: bool = False):
        self.tables = ROOT / "data" / "embed"
        run = ROOT / ("dry_run" if dry_run else os.environ.get("EMBED_RUN", "outputs"))
        self.run = run
        self.png = (run / "png") if dry_run or "EMBED_RUN" in os.environ else (ROOT / "data" / "embed" / "png")
        if "EMBED_PNG" in os.environ:
            self.png = ROOT / os.environ["EMBED_PNG"]
        self.logs = run / "logs"
        self.state = run / "state"
        self.ckpt = run / "checkpoints"
        self.preds = run / "predictions"
        self.metrics = run / "metrics"
        self.figs = run / "figures"
        self.proof = run / "proof"
        self.cohort = run / "cohort"
        for p in (self.png, self.logs, self.state, self.ckpt, self.preds,
                  self.metrics, self.figs, self.proof, self.cohort):
            p.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class Log:
    def __init__(self, paths: Paths, name: str):
        self.file = open(paths.logs / "pipeline.log", "a", encoding="utf-8")
        self.name = name
        self.beat_path = paths.state / f"heartbeat_{name}.txt"

    def __call__(self, msg: str):
        line = f"[{now()}] [{self.name}] {msg}"
        print(line, flush=True)
        self.file.write(line + "\n")
        self.file.flush()
        self.beat()

    def beat(self):
        self.beat_path.write_text(now(), encoding="utf-8")


def keep_awake():
    """Hold off idle sleep while this process lives, without changing settings."""
    if platform.system() == "Windows":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def load_aws_env():
    from dotenv import dotenv_values
    vals = dotenv_values(ROOT.parent / ".env")
    os.environ["AWS_ACCESS_KEY_ID"] = vals["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = vals["aws_secret_access_key"]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def env_info() -> dict:
    info = {"written_at": now(), "python": sys.version, "platform": platform.platform(), "seed": SEED}
    try:
        import torch
        info.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except ImportError:
        pass
    return info


def retry(fn, tries: int = 8, base: float = 2.0, log=None, what: str = ""):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # network errors surface under many botocore types
            if i == tries - 1:
                raise
            wait = min(base ** i, 120)
            if log:
                log(f"retry {i + 1}/{tries} {what}: {type(e).__name__}: {str(e)[:150]}; waiting {wait:.0f}s")
            time.sleep(wait)
