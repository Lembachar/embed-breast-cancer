"""Dev-only comparison of stacker inputs that need no new training: mean logits (current), max, and mean+max.
Writes metrics/<task>_stack_input_trial.json. Test predictions are not read."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from embed_common import Log, Paths, write_json
from pipeline_config import ARCHS, TASKS, load_config
from s4_evaluate import mean_proba, nested_oof, primary, to_features


def pooled(paths, task, cfg, how):
    spec = TASKS[task]
    cols = [f"logit_{i}" for i in range(spec["n_classes"])]
    frames = []
    for arch in ARCHS:
        v = pd.concat(pd.read_parquet(paths.preds / f"{task}_{arch}_fold{f}_val.parquet") for f in cfg["train"][task]["folds"])
        v = v.groupby(["image_id", "empi_anon", "breast_exam", spec["label"]], as_index=False)[cols].mean()
        b = v.groupby(["empi_anon", "breast_exam", spec["label"]], as_index=False)[cols].agg(how)
        frames.append(b.rename(columns={spec["label"]: "y"}).astype({"y": int}).sort_values("breast_exam").reset_index(drop=True))
    return frames[0][["empi_anon", "y"]], np.hstack([f[cols].values for f in frames])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    a = ap.parse_args()
    paths, task, cfg = Paths(False), a.task, load_config(False)
    log = Log(paths, "stack_trial")
    n = TASKS[task]["n_classes"]
    res = {}
    tbl, X_mean = pooled(paths, task, cfg, "mean")
    _, X_max = pooled(paths, task, cfg, "max")
    y, g = tbl.y.values, tbl.empi_anon.values
    F_mean, F_max = to_features(X_mean, n), to_features(X_max, n)
    res["mean_of_models|mean_pool"] = primary(task, y, mean_proba(X_mean, n))
    res["mean_of_models|max_pool"] = primary(task, y, mean_proba(X_max, n))
    for name in ("logistic", "lightgbm"):
        res[f"{name}|mean_pool"] = primary(task, y, nested_oof(name, n, F_mean, y, g))
        res[f"{name}|max_pool"] = primary(task, y, nested_oof(name, n, F_max, y, g))
        res[f"{name}|mean_and_max"] = primary(task, y, nested_oof(name, n, np.hstack([F_mean, F_max]), y, g))
    write_json(paths.metrics / f"{task}_stack_input_trial.json", res)
    log(f"{task} stack input trial on dev: { {k: round(v, 4) for k, v in res.items()} }")


if __name__ == "__main__":
    main()
