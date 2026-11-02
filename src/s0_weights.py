"""Cache ImageNet weights up front so training never needs the network. Retries, then falls back to curl."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import torch
from torchvision.models import ConvNeXt_Tiny_Weights, EfficientNet_V2_S_Weights, ResNet50_Weights

from embed_common import Log, Paths

WEIGHTS = [ResNet50_Weights.IMAGENET1K_V2, EfficientNet_V2_S_Weights.IMAGENET1K_V1, ConvNeXt_Tiny_Weights.IMAGENET1K_V1]


def valid(path: Path) -> bool:
    """torchvision filenames end in the first 8 hex chars of the sha256."""
    want = path.stem.rsplit("-", 1)[-1]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().startswith(want)


def attempt(url: str, tmp: Path, use_curl: bool):
    if use_curl:
        subprocess.run(["curl", "-L", "--fail", "--retry", "3", "--connect-timeout", "30", "-sS", "-o", str(tmp), url], check=True, timeout=1800)
    else:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)


def main(paths: Paths, log: Log):
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    cache.mkdir(parents=True, exist_ok=True)
    for w in WEIGHTS:
        dest = cache / w.url.rsplit("/", 1)[-1]
        if dest.exists() and valid(dest):
            log(f"weights cached: {dest.name}")
            continue
        tmp = dest.with_suffix(".part")
        for i in range(12):
            try:
                attempt(w.url, tmp, use_curl=i % 2 == 1)
                if not valid(tmp):
                    raise ValueError("hash mismatch")
                os.replace(tmp, dest)
                log(f"weights downloaded: {dest.name} {dest.stat().st_size / 1e6:.0f} MB on attempt {i + 1}")
                break
            except Exception as e:
                log(f"weights attempt {i + 1} failed for {dest.name}: {type(e).__name__}: {str(e)[:150]}")
                if i == 11:
                    raise
                time.sleep(min(10 * (i + 1), 60))
    log("STAGE_DONE weights")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    p = Paths(a.dry_run)
    main(p, Log(p, "weights"))
