"""Malignant vs benign: average the baseline and second-pass stacks (6 models at 2 resolutions).
Chosen on dev: AUC 0.737 against 0.723, difference +0.014 (95% CI 0.002 to 0.025). Thresholds come from dev only.
Writes outputs_v2/metrics/a_combined.json and predictions/a_<kind>_combined.parquet."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from embed_common import ROOT, now, write_json
from s4_evaluate import binary_metrics, boot_fn_a, bootstrap, pick_thresholds

VARIANTS = {"images": "stack", "images_exam_age": "stack_clinical", "images_exam_age_reading": "stack_reading"}
BOOTS = 2000


def combined(kind, col):
    b = pd.read_parquet(ROOT / "outputs" / "predictions" / f"a_{kind}_breast.parquet").sort_values("breast_exam").reset_index(drop=True)
    v = pd.read_parquet(ROOT / "outputs_v2" / "predictions" / f"a_{kind}_breast.parquet").sort_values("breast_exam").reset_index(drop=True)
    assert (b.breast_exam.values == v.breast_exam.values).all(), "the two runs disagree on the exam list"
    lo = lambda d: logit(np.clip(d[f"{col}_p1"].values, 1e-6, 1 - 1e-6))
    return b[["empi_anon", "breast_exam", "y"]], expit((lo(b) + lo(v)) / 2)


def main():
    out = {"at": now(), "models": "ResNet-50, EfficientNetV2-S, ConvNeXt-Tiny at 768x512 and at 1536x1024", "level": "breast exam"}
    test_open = (ROOT / "outputs" / "predictions" / "a_test_breast.parquet").exists() and (ROOT / "outputs_v2" / "predictions" / "a_test_breast.parquet").exists()
    for name, col in VARIANTS.items():
        tbl, p = combined("dev", col)
        thr = pick_thresholds(tbl.y.values, p)
        res = {"thresholds_on_dev": thr, "dev": binary_metrics(tbl.y.values, p, thr)}
        res["dev"]["ci95"] = bootstrap(boot_fn_a(tbl.y.values, p, thr["max_accuracy"]), tbl, BOOTS)
        tbl.assign(p=p).to_parquet(ROOT / "outputs_v2" / "predictions" / f"a_dev_combined_{name}.parquet", index=False)
        if test_open:
            tt, pt = combined("test", col)
            res["test"] = binary_metrics(tt.y.values, pt, thr)
            res["test"]["ci95"] = bootstrap(boot_fn_a(tt.y.values, pt, thr["max_accuracy"]), tt, BOOTS)
            tt.assign(p=pt).to_parquet(ROOT / "outputs_v2" / "predictions" / f"a_test_combined_{name}.parquet", index=False)
        out[name] = res
    write_json(ROOT / "outputs_v2" / "metrics" / "a_combined.json", out)
    for name in VARIANTS:
        for kind in ("dev", "test"):
            if kind in out[name]:
                r = out[name][kind]
                o, c = r["operating_points"]["max_accuracy"], r["ci95"]["auc_roc"]
                print(f"{name} {kind} n={r['n']} AUC={r['auc_roc']:.3f} ({c['low']:.3f}-{c['high']:.3f}) acc={o['accuracy']:.3f} sens={o['sensitivity_recall']:.3f} spec={o['specificity']:.3f} spec@sens90={r['spec_at_sens90']:.3f} conf0.9={r['confident']['conf_0.9']}")


if __name__ == "__main__":
    main()
