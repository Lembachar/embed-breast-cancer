"""How far do exam type and scanner get with no image at all? Grad-CAM showed attention on skin markers and burned-in marks,
which track the kind of exam, so this measures that shortcut. Dev only. Writes metrics/<task>_metadata_baseline.json."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn import metrics as M

from embed_common import Log, Paths, write_json
from pipeline_config import TASKS, load_config, load_metadata
from s4_evaluate import nested_oof, primary, to_features, load_breast, mean_proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    a = ap.parse_args()
    paths, task, cfg = Paths(False), a.task, load_config(False)
    log = Log(paths, "metadata_baseline")
    spec = TASKS[task]
    n = spec["n_classes"]
    meta = load_metadata(paths, False)
    tbl, _, X = load_breast(paths, task, cfg, "dev")
    be = meta[meta[spec["flag"]]].drop_duplicates("breast_exam").set_index("breast_exam")
    y, g = tbl.y.values, tbl.empi_anon.values
    res = {"n": int(len(y)), "image_stack_mean_of_models": primary(task, y, mean_proba(X, n))}
    sets = {"exam_kind": ["exam_kind"], "exam_kind+scanner": ["exam_kind", "Manufacturer"], "exam_kind+scanner+age": ["exam_kind", "Manufacturer", "age"]}
    Fimg = to_features(X, n)
    for name, cols in sets.items():
        d = be.loc[tbl.breast_exam, cols].reset_index(drop=True)
        F = pd.get_dummies(d, columns=[c for c in cols if c != "age"], dummy_na=True, dtype=float)
        if "age" in F:
            F["age"] = (F.age.fillna(F.age.median()) - 60) / 12
        P = nested_oof("logistic", n, F.values, y, g)
        res[name] = primary(task, y, P)
        res[name + "|acc"] = float((P.argmax(1) == y).mean())
        res[name + "|macro_auc"] = float(np.mean([M.roc_auc_score((y == c).astype(int), P[:, c]) for c in range(n)]))
        res[name + "_plus_images"] = primary(task, y, nested_oof("logistic", n, np.hstack([F.values, Fimg]), y, g))
    write_json(paths.metrics / f"{task}_metadata_baseline.json", res)
    log(f"{task} metadata baseline on dev: { {k: round(v, 4) for k, v in res.items()} }")
    print(be.exam_kind.value_counts().to_dict())
    print(pd.crosstab(be.loc[tbl.breast_exam].exam_kind.values, y))


if __name__ == "__main__":
    main()
