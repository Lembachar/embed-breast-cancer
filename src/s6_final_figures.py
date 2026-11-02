"""Summary figures across runs and the final metrics table. Saved as 300 dpi PNG plus PDF in outputs_v2/figures/final/.
Numbers only, no written report. Run after s4 --unlock-test on both runs and combine_runs.py."""
from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics as M

from embed_common import ROOT
from s5_figures import AXIS, BIN_BEN, BIN_MAL, BLUES, GRID, INK, INK2, MUTED

OUT = ROOT / "outputs_v2" / "figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)
V2, BASE = ROOT / "outputs_v2", ROOT / "outputs"
VARIANTS = [("images", "Images only", INK), ("images_exam_age", "Images + exam type, age", "#2a78d6"), ("images_exam_age_reading", "Images + exam type, age, radiologist's findings", "#eb6834")]


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("figure", name, flush=True)


def preds(kind, name):
    return pd.read_parquet(V2 / "predictions" / f"a_{kind}_combined_{name}.parquet")


def roc_final(comb):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, kind in zip(axs, ("dev", "test")):
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
        for name, label, color in VARIANTS:
            t = preds(kind, name)
            fpr, tpr, _ = M.roc_curve(t.y, t.p)
            r = comb[name][kind]
            ci = r["ci95"]["auc_roc"]
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label}: {r['auc_roc']:.3f} ({ci['low']:.3f}-{ci['high']:.3f})")
        n = comb["images"][kind]
        ax.set(title=f"{kind.capitalize()} set, {n['n']} breast exams, {n['n_pos']} cancers", xlabel="False positive rate", xlim=(0, 1), ylim=(0, 1.01))
        ax.legend(loc="lower right", fontsize=7.5)
    axs[0].set_ylabel("Sensitivity")
    fig.suptitle("Malignant vs benign: ROC, AUC with 95% CI", fontweight="bold")
    fig.tight_layout()
    save(fig, "roc_final")


def coverage_curve():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, kind in zip(axs, ("dev", "test")):
        for name, label, color in VARIANTS:
            t = preds(kind, name)
            conf = np.maximum(t.p, 1 - t.p).values
            right = ((t.p >= 0.5).astype(int) == t.y).values
            o = np.argsort(-conf)
            n = np.arange(1, len(o) + 1)
            start = max(20, len(o) // 20)
            ax.plot(n[start:] / len(o), (np.cumsum(right[o]) / n)[start:], color=color, linewidth=2, label=label)
        ax.axhline(0.9, color=MUTED, linewidth=1, linestyle="--")
        ax.text(0.99, 0.905, "90% right", color=INK2, fontsize=8, ha="right")
        ax.set(title=f"{kind.capitalize()} set", xlabel="Share of exams the model decides (most confident first)", xlim=(0, 1), ylim=(0.6, 1.01))
        ax.legend(loc="lower left", fontsize=7.5)
    axs[0].set_ylabel("Share of those decisions that are right")
    fig.suptitle("Malignant vs benign: accuracy when the model may say \"unsure\"", fontweight="bold")
    fig.tight_layout()
    save(fig, "coverage_vs_accuracy")


def auc_progress(comb, base, v2):
    rows = lambda k: [("PC baseline stack, 768x512", base[k]["stack"]), ("Kaggle stack, 1536x1024", v2[k]["stack"]), ("Both stacks combined", comb["images"][k]),
                      ("Combined + exam type, age", comb["images_exam_age"][k]), ("Combined + exam type, age, findings", comb["images_exam_age_reading"][k])]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, k in zip(axs, ("dev", "test")):
        for i, (label, m) in enumerate(rows(k)):
            y, ci = 4 - i, m["ci95"]["auc_roc"]
            ax.plot([ci["low"], ci["high"]], [y, y], color="#2a78d6", linewidth=2, solid_capstyle="round")
            ax.plot(m["auc_roc"], y, "o", color="#2a78d6", markersize=8)
            ax.text(ci["high"] + 0.006, y, f"{m['auc_roc']:.3f}", va="center", color=INK2, fontsize=9)
        ax.set(title=f"{k.capitalize()} set", xlabel="AUC-ROC with 95% CI", xlim=(0.65, 0.92))
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(5), [r[0] for r in rows("dev")][::-1])
    fig.suptitle("Malignant vs benign: what each step added", fontweight="bold")
    fig.tight_layout()
    save(fig, "auc_progress")


def confusion_final(comb):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (name, label, _) in zip(axs, VARIANTS):
        o = comb[name]["test"]["operating_points"]["max_accuracy"]
        c = np.array([[o["tn"], o["fp"]], [o["fn"], o["tp"]]])
        ax.imshow(c / c.sum(1, keepdims=True), cmap=BLUES, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                share = c[i, j] / c[i].sum()
                ax.text(j, i, f"{c[i, j]}\n{share:.0%}", ha="center", va="center", color="white" if share > 0.6 else INK)
        ax.set_xticks([0, 1], ["Benign", "Malignant"])
        ax.set_yticks([0, 1], ["Benign", "Malignant"])
        ax.set(xlabel="Predicted", title=f"{label}\naccuracy {o['accuracy']:.1%}")
        ax.title.set_fontsize(8.5)
        ax.grid(False)
    axs[0].set_ylabel("Actual")
    fig.suptitle("Test set, threshold fixed on dev", fontweight="bold")
    fig.tight_layout()
    save(fig, "confusion_test_final")


def distributions():
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True)
    for ax, (name, label, _) in zip(axs, VARIANTS):
        t = preds("test", name)
        bins = np.linspace(0, 1, 21)
        ax.hist(t.p[t.y == 0], bins=bins, color=BIN_BEN, alpha=0.65, label="Benign", density=True)
        ax.hist(t.p[t.y == 1], bins=bins, color=BIN_MAL, alpha=0.65, label="Malignant", density=True)
        ax.set(title=label, xlabel="Predicted probability of cancer")
        ax.title.set_fontsize(8.5)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("Density")
    fig.suptitle("Test set score distributions", fontweight="bold")
    fig.tight_layout()
    save(fig, "score_distributions_test")


def calibration():
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, kind in zip(axs, ("dev", "test")):
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
        for name, label, color in VARIANTS:
            t = preds(kind, name)
            q = pd.qcut(t.p, 8, duplicates="drop")
            g = t.groupby(q, observed=True).agg(p=("p", "mean"), y=("y", "mean"))
            ax.plot(g.p, g.y, "-o", color=color, linewidth=2, markersize=5, label=label)
        ax.set(title=f"{kind.capitalize()} set", xlabel="Predicted probability", xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=7.5, loc="upper left")
    axs[0].set_ylabel("Share truly malignant")
    fig.suptitle("Does \"90% sure\" mean 90% right? Calibration", fontweight="bold")
    fig.tight_layout()
    save(fig, "calibration_final")


def birads_shortcut(b):
    mb = load(V2 / "metrics" / "b_metadata_baseline.json")
    img, clin = b["dev"]["stack"], b["dev_clinical"]["stack_clinical"]
    rows = [("Exam type only, no image", mb["exam_kind|acc"], mb["exam_kind|macro_auc"]), ("Images only", img["accuracy"], img["macro_auc_ovr"]), ("Images + exam type, age", clin["accuracy"], clin["macro_auc_ovr"])]
    fig, axs = plt.subplots(1, 2, figsize=(9, 2.8), sharey=True)
    for ax, k, title, xmax in ((axs[0], 1, "Accuracy", 0.7), (axs[1], 2, "Macro AUC", 1.0)):
        for i, r in enumerate(rows):
            ax.plot([0, r[k]], [2 - i, 2 - i], color="#2a78d6", linewidth=6, solid_capstyle="round")
            ax.text(r[k] + xmax * 0.02, 2 - i, f"{r[k]:.3f}", va="center", color=INK2, fontsize=9)
        ax.set(title=title, xlim=(0, xmax))
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks([2, 1, 0], [r[0] for r in rows])
    fig.suptitle("BI-RADS on dev: exam type alone nearly matches the images", fontweight="bold")
    fig.tight_layout()
    save(fig, "birads_exam_type_shortcut")


def marker_test():
    m = load(ROOT / "outputs_masked" / "metrics" / "masked_vs_baseline.json")
    fig, ax = plt.subplots(figsize=(6.5, 1.9))
    lo, hi = m["delta_ci95"]
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")
    ax.plot([lo, hi], [0, 0], color="#2a78d6", linewidth=2, solid_capstyle="round")
    ax.plot(m["delta_masked_minus_baseline"], 0, "o", color="#2a78d6", markersize=8)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set(xlabel=f"AUC change when markers are removed (ConvNeXt-Tiny, dev): {m['auc_baseline']:.3f} to {m['auc_masked']:.3f}", xlim=(-0.06, 0.06))
    ax.set_title("Marker test: change with 95% CI")
    save(fig, "marker_test")


def summary(comb, base, v2, b):
    rows = []

    def add(task, model, kind, m):
        o = m.get("operating_points", {}).get("max_accuracy", {})
        conf = m.get("confident", {}).get("conf_0.9", {})
        ci = m.get("ci95", {}).get("auc_roc", {})
        rows.append({"task": task, "model": model, "set": kind, "n": m["n"], "auc_roc": m.get("auc_roc"), "auc_low": ci.get("low"), "auc_high": ci.get("high"), "auc_pr": m.get("auc_pr"),
                     "accuracy": o.get("accuracy", m.get("accuracy")), "balanced_accuracy": o.get("balanced_accuracy", m.get("balanced_accuracy")), "sensitivity": o.get("sensitivity_recall"), "specificity": o.get("specificity"),
                     "ppv": o.get("ppv_precision"), "npv": o.get("npv"), "f1": o.get("f1", m.get("macro_f1")), "mcc": o.get("mcc", m.get("mcc")), "spec_at_sens90": m.get("spec_at_sens90"), "macro_auc": m.get("macro_auc_ovr"),
                     "conf90_coverage": conf.get("coverage"), "conf90_right": conf.get("right"), "conf90_cancers_said_benign": conf.get("cancers_said_benign")})

    for k in ("dev", "test"):
        for run, rep in (("baseline", base), ("kaggle", v2)):
            for a, m in rep[k].items():
                add("malignant_vs_benign", f"{run}_{a}", k, m)
            add("malignant_vs_benign", f"{run}_stack_exam_age", k, rep[f"{k}_clinical"]["stack_clinical"])
            add("malignant_vs_benign", f"{run}_stack_exam_age_findings", k, rep[f"{k}_reading"]["stack_reading"])
        for name, _, _ in VARIANTS:
            add("malignant_vs_benign", f"combined_{name}", k, comb[name][k])
        for a, m in b[k].items():
            add("birads", a, k, m)
        add("birads", "stack_exam_age", k, b[f"{k}_clinical"]["stack_clinical"])
    with open(V2 / "metrics" / "final_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("summary rows", len(rows))


def main():
    comb, base, v2, b = load(V2 / "metrics" / "a_combined.json"), load(BASE / "metrics" / "a_metrics.json"), load(V2 / "metrics" / "a_metrics.json"), load(V2 / "metrics" / "b_metrics.json")
    assert "test" in comb["images"], "open the test set and run combine_runs.py first"
    roc_final(comb)
    coverage_curve()
    auc_progress(comb, base, v2)
    confusion_final(comb)
    distributions()
    calibration()
    birads_shortcut(b)
    marker_test()
    summary(comb, base, v2, b)


if __name__ == "__main__":
    main()
