"""Same model, same folds, same images: baseline ConvNeXt-Tiny against the run with everything outside the breast blacked out.
Dev predictions only. Writes outputs_masked/metrics/masked_vs_baseline.json."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from embed_common import ROOT, SEED, write_json

ARCH = "convnext_tiny"
FOLDS = range(5)


def breast_logodds(run):
    v = pd.concat(pd.read_parquet(ROOT / run / "predictions" / f"a_{ARCH}_fold{f}_val.parquet") for f in FOLDS)
    v["s"] = v.logit_1 - v.logit_0
    return v.groupby(["empi_anon", "breast_exam", "label_malignant"], as_index=False).s.mean().sort_values("breast_exam").reset_index(drop=True)


def main():
    base, masked = breast_logodds("outputs"), breast_logodds("outputs_masked")
    assert (base.breast_exam.values == masked.breast_exam.values).all()
    y = base.label_malignant.values.astype(int)
    idx = {k: np.asarray(v) for k, v in base.groupby("empi_anon").indices.items()}
    keys = list(idx)
    rng = np.random.RandomState(SEED)
    deltas = []
    for _ in range(2000):
        i = np.concatenate([idx[keys[j]] for j in rng.randint(0, len(keys), len(keys))])
        deltas.append(roc_auc_score(y[i], masked.s.values[i]) - roc_auc_score(y[i], base.s.values[i]))
    res = {"n": int(len(y)), "auc_baseline": float(roc_auc_score(y, base.s)), "auc_masked": float(roc_auc_score(y, masked.s)),
           "delta_masked_minus_baseline": float(np.mean(deltas)), "delta_ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]}
    write_json(ROOT / "outputs_masked" / "metrics" / "masked_vs_baseline.json", res)
    print(res)


if __name__ == "__main__":
    main()
