"""Run settings and shared data access. Editing configs/*.json changes a run without touching code."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from embed_common import ROOT, SEED, Paths

ARCHS = ["resnet50", "efficientnet_v2_s", "convnext_tiny"]
TASKS = {
    "a": {"name": "malignant_vs_benign", "flag": "in_task_a", "label": "label_malignant", "n_classes": 2,
          "classes": ["Benign", "Malignant"]},
    "b": {"name": "birads", "flag": "in_task_b", "label": "label_birads", "n_classes": 6,
          "classes": ["0 Incomplete", "1 Negative", "2 Benign", "3 Probably benign", "4 Suspicious", "5 Highly suggestive"]},
}


def load_config(dry_run: bool) -> dict:
    name = "dry_run.json" if dry_run else os.environ.get("EMBED_CONFIG", "full_run.json")
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


def load_metadata(paths: Paths, dry_run: bool) -> pd.DataFrame:
    full = ROOT / "outputs" / "cohort" / "metadata_images.parquet"  # one cohort for every run; Kaggle copies are trimmed
    d = pd.read_parquet(full)
    d["image_id"] = d.image_id.astype(str)
    if not dry_run:
        return d
    # small subset that still holds every class and every split of both tasks
    rng = np.random.RandomState(SEED)
    keep = []
    a = d[d.in_task_a]
    for (_, _c), g in a.groupby(["split", "label_malignant"]):
        be = g.breast_exam.unique()
        keep.extend(rng.choice(be, min(len(be), 16), replace=False))
    b = d[d.in_task_b]
    for (_, _c), g in b.groupby(["split", "birads_code"]):
        be = g.breast_exam.unique()
        keep.extend(rng.choice(be, min(len(be), 20 if _ == "dev" else 6), replace=False))
    s = d[d.breast_exam.isin(set(keep))].copy()
    # Task A rows selected only through Task B stay labelled in both, which is fine for a smoke test
    return s


def available_images(d: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    """Restrict to images that converted successfully; a failed image must never crash training."""
    fetched = pd.read_csv(paths.state / "fetched.csv", usecols=["image_id"]).image_id.astype(str)
    return d[d.image_id.isin(set(fetched))].copy()
