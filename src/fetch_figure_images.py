"""Fetch only the images the figure galleries show, with the fixed crop. Second-pass images stayed on Kaggle.
About 6.4 MB per image passes through the VPN. --kind test reads test predictions, so run it after the test is opened."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ["EMBED_LONG_SIDE"] = "1536"

import boto3
import numpy as np
import pandas as pd

from embed_common import BUCKET, REGION, Log, Paths, load_aws_env
from pipeline_config import TASKS, available_images, load_config, load_metadata
from s2_fetch_images import png_path, preprocess

SAMPLES_PER_CLASS = 6
CASES_PER_ROW = 3


def gallery_exams(paths, cfg, meta, task, kind):
    """The same case rows s5_figures draws."""
    spec = TASKS[task]
    t = pd.read_parquet(paths.preds / f"{task}_{kind}_breast.parquet")
    if kind == "dev":
        t = t[t.breast_exam.isin(set(meta[meta.fold == cfg["train"][task]["folds"][0]].breast_exam))]
    if task == "a":
        import json
        rep = json.loads((paths.metrics / "a_metrics.json").read_text(encoding="utf-8"))
        yh = (t.stack_p1.values >= rep["thresholds_chosen_on_dev"]["stack"]["max_accuracy"]).astype(int)
        cases = [t[(t.y == y) & (yh == h)] for y in (0, 1) for h in (0, 1)]
    else:
        yh = t[[f"stack_p{i}" for i in range(spec["n_classes"])]].values.argmax(1)
        cases = [t[(t.y == i) & (yh == i)] for i in range(spec["n_classes"])]
    rng = np.random.RandomState(3)
    return [e for s in cases for e in s.breast_exam.values[rng.permutation(len(s))[:CASES_PER_ROW]]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--kind", required=True, choices=["dev", "test"])
    a = ap.parse_args()
    paths, cfg = Paths(False), load_config(False)
    log = Log(paths, "figure_images")
    meta = load_metadata(paths, False)
    spec = TASKS[a.task]
    d = available_images(meta[meta[spec["flag"]]], paths)
    rng = np.random.RandomState(7)
    ids = [d[d.breast_exam == e].image_id.iloc[0] for e in gallery_exams(paths, cfg, meta, a.task, a.kind)]
    for c in range(spec["n_classes"]):
        s = d[d[spec["label"]].astype(int) == c]
        ids.extend(s.image_id.values[rng.permutation(len(s))[:SAMPLES_PER_CLASS]])
    todo = [i for i in dict.fromkeys(ids) if not png_path(paths.png, i).exists()]
    keys = d.set_index("image_id").anon_dicom_path
    load_aws_env()
    s3 = boto3.client("s3", region_name=REGION)
    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for k, i in enumerate(todo, 1):
            f = Path(tmp) / "x.dcm"
            s3.download_file(BUCKET, keys[i], str(f))
            total += f.stat().st_size
            preprocess(str(f), str(png_path(paths.png, i)))
            log.beat()
            if k % 10 == 0:
                print(f"{100 * k // len(todo)}%", flush=True)
    log(f"STAGE_DONE figure images {a.task} {a.kind}: {len(todo)} fetched, {total / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
