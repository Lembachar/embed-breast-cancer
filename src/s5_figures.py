"""All figures for one task. Saved as 300 dpi PNG plus PDF under figures/<task>/."""
from __future__ import annotations

import argparse
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn import metrics as M

from embed_common import Log, Paths, load_aws_env, now, write_json
from pipeline_config import ARCHS, TASKS, available_images, load_config, load_metadata
from s2_fetch_images import png_path
from s3_train import MEAN, STD, Mammo, build_model

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = {"resnet50": "#2a78d6", "efficientnet_v2_s": "#eb6834", "convnext_tiny": "#1baf7a", "stack": INK}
NAMES = {"resnet50": "ResNet-50", "efficientnet_v2_s": "EfficientNetV2-S", "convnext_tiny": "ConvNeXt-Tiny", "stack": "Stacked ensemble"}
BLUES = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
BIN_BEN, BIN_MAL = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "legend.frameon": False, "axes.axisbelow": True})


class Fig:
    def __init__(self, paths, task, log):
        self.dir = paths.figs / task
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log, self.made = log, []

    def save(self, fig, name):
        fig.savefig(self.dir / f"{name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(self.dir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
        self.made.append(name)
        self.log(f"figure {name}")


def ci_txt(m, k):
    ci = m.get("ci95", {}).get(k)
    return f"{m[k]:.3f} ({ci['low']:.3f}-{ci['high']:.3f})" if ci else f"{m[k]:.3f}"


def spec_labels(task):
    return TASKS[task]["classes"]


def on_disk(d, paths):
    """Second-pass results keep only the few images the galleries show; the rest stayed on Kaggle."""
    return d[[png_path(paths.png, i).exists() for i in d.image_id]]


# ---------- cohort
def cohort_figs(F, paths, meta, task):
    spec = TASKS[task]
    d = available_images(meta[meta[spec["flag"]]], paths)
    be = d.drop_duplicates("breast_exam")
    label = spec["label"]
    classes = spec["classes"]

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    c = be[label].astype(int).value_counts().reindex(range(len(classes)), fill_value=0)
    t = be[be.split == "test"][label].astype(int).value_counts().reindex(range(len(classes)), fill_value=0)
    y = np.arange(len(classes))[::-1]
    ax.barh(y, c.values - t.values, color="#2a78d6", label="Development", height=0.62)
    ax.barh(y, t.values, left=c.values - t.values + c.max() * 0.004, color="#eb6834", label="Locked test", height=0.62)
    for yy, v in zip(y, c.values):
        ax.text(v + c.max() * 0.012, yy, f"{v:,}", va="center", color=INK2)
    ax.set_yticks(y, classes)
    ax.set_xlabel("Breast exams")
    ax.set_title("Breast exams per class")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    F.save(fig, "cohort_class_counts")

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    axs[0].hist(be.age.dropna(), bins=np.arange(25, 95, 3), color="#2a78d6", rwidth=0.9)
    axs[0].set(title="Age at exam", xlabel="Years", ylabel="Breast exams")
    for ax, col, names, title in [(axs[1], "density", {1: "A", 2: "B", 3: "C", 4: "D"}, "Breast density"), (axs[2], "Manufacturer", None, "Scanner manufacturer")]:
        vc = be[col].value_counts()
        if names:
            vc = vc.reindex(sorted(names)).fillna(0)
            lab = [names[k] for k in vc.index]
        else:
            lab = [str(k).split(",")[0].split(" ")[0].title() for k in vc.index]
        ax.bar(lab, vc.values, color="#2a78d6", width=0.65)
        ax.set(title=title, ylabel="Breast exams")
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", rotation=20 if not names else 0)
    fig.tight_layout()
    F.save(fig, "cohort_demographics")

    rows = len(classes)
    rng = np.random.RandomState(7)
    fig, axs = plt.subplots(rows, 6, figsize=(9, 2.2 * rows))
    axs = np.atleast_2d(axs)
    shown = on_disk(d, paths)
    for r in range(rows):
        s = shown[shown[label].astype(int) == r]
        ids = s.image_id.values[rng.permutation(len(s))[:6]]
        for k in range(6):
            ax = axs[r, k]
            ax.axis("off")
            if k < len(ids):
                im = cv2.imread(str(png_path(paths.png, ids[k])), 0)
                ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            if k == 0:
                ax.text(-0.05, 0.5, classes[r], transform=ax.transAxes, ha="right", va="center", fontsize=9, wrap=True)
    fig.suptitle("Sample preprocessed mammograms per class", fontweight="bold")
    F.save(fig, "sample_images")


def training_curves(F, paths, cfg, task):
    folds = cfg["train"][task]["folds"]
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.2), sharex=True)
    metric = "metric"
    for j, arch in enumerate(ARCHS):
        for f in folds:
            p = paths.ckpt / task / arch / f"fold{f}" / "history.csv"
            if not p.exists():
                continue
            h = pd.read_csv(p)
            metric = h.metric_name.iloc[0]
            axs[0, j].plot(h.epoch + 1, h.train_loss, color=SERIES[arch], lw=1.6, alpha=0.75, marker="o", ms=3)
            axs[1, j].plot(h.epoch + 1, h.val_metric, color=SERIES[arch], lw=1.6, alpha=0.75, marker="o", ms=3)
        axs[0, j].set_title(NAMES[arch])
        axs[1, j].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Training loss")
    axs[1, 0].set_ylabel(f"Validation {metric.replace('_', ' ')}")
    fig.suptitle("Training curves (one line per fold)", fontweight="bold", y=1.0)
    fig.tight_layout()
    F.save(fig, "training_curves")


# ---------- task A
def binary_figs(F, paths, rep, kind):
    t = pd.read_parquet(paths.preds / f"a_{kind}_breast.parquet")
    y = t.y.values
    res = rep[kind]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
    rng = np.random.RandomState(1)
    grid = np.linspace(0, 1, 201)
    groups = [np.asarray(v) for v in t.groupby("empi_anon").indices.values()]
    curves = []
    for _ in range(300):
        idx = np.concatenate([groups[i] for i in rng.randint(0, len(groups), len(groups))])
        if len(set(y[idx])) < 2:
            continue
        fp, tp, _t = M.roc_curve(y[idx], t.stack_p1.values[idx])
        curves.append(np.interp(grid, fp, tp))
    if curves:
        axs[0].fill_between(grid, *np.percentile(curves, [2.5, 97.5], axis=0), color=INK, alpha=0.10, lw=0, label="Ensemble 95% CI")
    for name in ARCHS + ["stack"]:
        p = t[f"{name}_p1"].values
        fp, tp, _t = M.roc_curve(y, p)
        pr, rc, _t = M.precision_recall_curve(y, p)
        lw = 2.6 if name == "stack" else 1.6
        axs[0].plot(fp, tp, color=SERIES[name], lw=lw, label=f"{NAMES[name]}  AUC {res[name]['auc_roc']:.3f}")
        axs[1].plot(rc, pr, color=SERIES[name], lw=lw, label=f"{NAMES[name]}  AP {res[name]['auc_pr']:.3f}")
    axs[0].plot([0, 1], [0, 1], color=AXIS, ls="--", lw=1)
    axs[1].axhline(y.mean(), color=AXIS, ls="--", lw=1)
    axs[1].text(0.99, y.mean() + 0.015, f"Prevalence {y.mean():.2f}", ha="right", color=MUTED, fontsize=9)
    axs[0].set(xlabel="1 - specificity", ylabel="Sensitivity", title=f"ROC curve ({kind})", xlim=(0, 1), ylim=(0, 1.01))
    axs[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall curve ({kind})", xlim=(0, 1), ylim=(0, 1.01))
    for ax in axs:
        ax.legend(loc="lower right" if ax is axs[0] else "upper right", fontsize=9)
    fig.tight_layout()
    F.save(fig, f"roc_pr_{kind}")

    ops = res["stack"]["operating_points"]
    names = [("max_accuracy", "Max accuracy threshold"), ("youden", "Youden threshold"), ("sens_90", "90% sensitivity threshold")]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (k, title) in zip(axs, names):
        o = ops[k]
        cm = np.array([[o["tn"], o["fp"]], [o["fn"], o["tp"]]])
        ax.imshow(cm / cm.sum(1, keepdims=True).clip(1), cmap=BLUES, vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(cm):
            frac = v / max(cm[i].sum(), 1)
            ax.text(j, i, f"{v}\n{frac:.0%}", ha="center", va="center", color="white" if frac > 0.55 else INK)
        ax.set_xticks([0, 1], ["Benign", "Malignant"]); ax.set_yticks([0, 1], ["Benign", "Malignant"])
        ax.set(xlabel="Predicted", ylabel="Actual", title=f"{title} ({o['threshold']:.2f})")
        ax.grid(False)
    fig.tight_layout()
    F.save(fig, f"confusion_{kind}")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    edges = np.linspace(0, 1, 11)
    for name in ARCHS + ["stack"]:
        p = t[f"{name}_p1"].values
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
        axs[0].plot(xs, ys, color=SERIES[name], lw=2.6 if name == "stack" else 1.6, marker="o", ms=4, label=f"{NAMES[name]}  ECE {res[name]['ece']:.3f}")
    axs[0].plot([0, 1], [0, 1], color=AXIS, ls="--", lw=1)
    axs[0].set(xlabel="Predicted probability", ylabel="Observed malignant fraction", title="Calibration", xlim=(0, 1), ylim=(0, 1))
    axs[0].legend(fontsize=9)
    b = np.linspace(0, 1, 26)
    axs[1].hist(t.stack_p1[y == 0], bins=b, color=BIN_BEN, alpha=0.7, label="Benign", rwidth=0.9)
    axs[1].hist(t.stack_p1[y == 1], bins=b, color=BIN_MAL, alpha=0.7, label="Malignant", rwidth=0.9)
    axs[1].set(xlabel="Ensemble malignancy probability", ylabel="Breast exams", title="Score distribution")
    axs[1].legend()
    fig.tight_layout()
    F.save(fig, f"calibration_scores_{kind}")

    th = np.linspace(0.01, 0.99, 99)
    p = t.stack_p1.values
    series = {"Sensitivity": ("#2a78d6", []), "Specificity": ("#eb6834", []), "Accuracy": ("#1baf7a", []), "F1": ("#6250d6", [])}
    for x in th:
        yh = (p >= x).astype(int)
        series["Sensitivity"][1].append(((yh == 1) & (y == 1)).sum() / max(y.sum(), 1))
        series["Specificity"][1].append(((yh == 0) & (y == 0)).sum() / max((1 - y).sum(), 1))
        series["Accuracy"][1].append((yh == y).mean())
        series["F1"][1].append(M.f1_score(y, yh, zero_division=0))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k, (col, vals) in series.items():
        ax.plot(th, vals, color=col, lw=2, label=k)
    ax.axhline(0.9, color=AXIS, ls="--", lw=1)
    ax.text(0.01, 0.91, "90% target", color=MUTED, fontsize=9)
    ax.set(xlabel="Decision threshold", ylabel="Metric value", title="Metrics across decision thresholds", ylim=(0, 1.02), xlim=(0, 1))
    ax.legend(loc="lower center", ncol=4)
    F.save(fig, f"threshold_sweep_{kind}")

    sub = paths.metrics / f"a_{kind}_subgroups.csv"
    if sub.exists():
        s = pd.read_csv(sub)
        s = s[s.auc_roc.notna()] if "auc_roc" in s else s.iloc[0:0]
        if len(s):
            fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(s) + 1.3))
            yy = np.arange(len(s))[::-1]
            ax.errorbar(s.auc_roc, yy, xerr=[(s.auc_roc - s.auc_low).clip(0), (s.auc_high - s.auc_roc).clip(0)], fmt="o", color="#2a78d6", ms=5, lw=1.4, capsize=0)
            ax.axvline(res["stack"]["auc_roc"], color=AXIS, ls="--", lw=1)
            ax.set_yticks(yy, [f"{a.replace('_', ' ')}: {b}  (n={n})" for a, b, n in zip(s.group, s.value, s.n)], fontsize=8)
            ax.set(xlabel="AUC with 95% CI", title="Subgroup performance")
            F.save(fig, f"subgroups_{kind}")


# ---------- task B
def multiclass_figs(F, paths, rep, kind):
    classes = spec_labels("b")
    n = len(classes)
    t = pd.read_parquet(paths.preds / f"b_{kind}_breast.parquet")
    y = t.y.values
    res = rep[kind]
    P = t[[f"stack_p{i}" for i in range(n)]].values

    cm = np.array(res["stack"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    norm = cm / cm.sum(1, keepdims=True).clip(1)
    im = ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(n), classes, rotation=35, ha="right"); ax.set_yticks(range(n), classes)
    ax.set(xlabel="Predicted", ylabel="Actual", title=f"Stacked ensemble confusion matrix ({kind})")
    ax.grid(False)
    fig.colorbar(im, fraction=0.04, label="Share of actual class")
    F.save(fig, f"confusion_{kind}")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    for i, c in enumerate(classes):
        yb = (y == i).astype(int)
        if 0 < yb.sum() < len(yb):
            fp, tp, _t = M.roc_curve(yb, P[:, i])
            ax.plot(fp, tp, color=colors[i], lw=1.8, label=f"{c}  AUC {M.roc_auc_score(yb, P[:, i]):.3f}")
    ax.plot([0, 1], [0, 1], color=AXIS, ls="--", lw=1)
    ax.set(xlabel="1 - specificity", ylabel="Sensitivity", title=f"One-vs-rest ROC per BI-RADS class ({kind})", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(fontsize=9, loc="lower right")
    F.save(fig, f"roc_per_class_{kind}")

    per = res["stack"]["per_class"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    yy = np.arange(n)[::-1]
    for ax, k, title in zip(axs, ["precision", "recall_sensitivity", "f1"], ["Precision", "Recall", "F1"]):
        v = [per[c][k] for c in classes]
        ax.barh(yy, v, color="#2a78d6", height=0.62)
        for a, b in zip(yy, v):
            ax.text(min(b + 0.02, 0.86), a, f"{b:.2f}", va="center", color=INK2, fontsize=9)
        ax.set(title=title, xlim=(0, 1))
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(yy, [f"{c} (n={per[c]['support']})" for c in classes])
    fig.tight_layout()
    F.save(fig, f"per_class_metrics_{kind}")

    labels = [NAMES[a] for a in ARCHS + ["stack"]]
    keys = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro F1"), ("macro_auc_ovr", "Macro AUC")]
    fig, axs = plt.subplots(1, 4, figsize=(11, 3), sharey=True)
    for ax, (k, title) in zip(axs, keys):
        vals = [res[a][k] for a in ARCHS + ["stack"]]
        yy = np.arange(4)[::-1]
        ax.barh(yy, vals, color=[SERIES[a] for a in ARCHS + ["stack"]], height=0.62)
        for a, b in zip(yy, vals):
            ax.text(min(b + 0.02, 0.8), a, f"{b:.3f}", va="center", color=INK2, fontsize=9)
        ax.set(title=title, xlim=(0, 1))
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(yy, labels)
    fig.tight_layout()
    F.save(fig, f"model_comparison_{kind}")


def model_comparison_a(F, rep, kind):
    res = rep[kind]
    keys = [("auc_roc", "AUC"), ("auc_pr", "Average precision"), ("accuracy", "Accuracy"), ("sensitivity_recall", "Sensitivity"), ("specificity", "Specificity"), ("f1", "F1")]
    fig, axs = plt.subplots(2, 3, figsize=(11, 5), sharey=True)
    order = ARCHS + ["stack"]
    yy = np.arange(4)[::-1]
    for ax, (k, title) in zip(axs.ravel(), keys):
        vals = [(res[a].get(k) if k in ("auc_roc", "auc_pr") else res[a]["operating_points"]["max_accuracy"][k]) for a in order]
        ax.barh(yy, vals, color=[SERIES[a] for a in order], height=0.62)
        for a, b in zip(yy, vals):
            ax.text(min(b + 0.02, 0.84), a, f"{b:.3f}", va="center", color=INK2, fontsize=9)
        ax.axvline(0.9, color=AXIS, ls="--", lw=1)
        ax.set(title=title, xlim=(0, 1))
        ax.grid(axis="y", visible=False)
        ax.set_yticks(yy, [NAMES[a] for a in order])
    fig.suptitle(f"Model comparison at the max-accuracy threshold ({kind}); dashed line marks 90%", fontweight="bold")
    fig.tight_layout()
    F.save(fig, f"model_comparison_{kind}")


# ---------- embeddings and Grad-CAM
def umap_fig(F, paths, cfg, meta, task, kind):
    import umap

    spec = TASKS[task]
    fold = cfg["train"][task]["folds"][0]
    arch = "convnext_tiny"
    f = paths.preds / f"{task}_{arch}_fold{fold}_{'test' if kind == 'test' else 'val'}_features.npz"
    if not f.exists():
        return
    z = np.load(f, allow_pickle=True)
    ids = z["image_id"].astype(str)
    if len(ids) < 30:
        return
    X = z["features"].astype(np.float32)
    lab = meta.drop_duplicates("image_id").set_index("image_id")[spec["label"]].reindex(ids).astype(int).values
    xy = umap.UMAP(n_neighbors=min(30, len(ids) - 1), min_dist=0.2, random_state=42).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    colors = [BIN_BEN, BIN_MAL] if task == "a" else ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    for i, c in enumerate(spec["classes"]):
        m = lab == i
        ax.scatter(xy[m, 0], xy[m, 1], s=9, color=colors[i], alpha=0.65, label=c, linewidths=0)
    ax.set(title=f"UMAP of ConvNeXt-Tiny image features ({kind})", xlabel="UMAP 1", ylabel="UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    ax.legend(markerscale=2, fontsize=9, loc="best")
    F.save(fig, f"umap_{kind}")


def grad_cam(model, layer, x, cls):
    store = {}
    h1 = layer.register_forward_hook(lambda _m, _i, o: store.update(a=o))
    h2 = layer.register_full_backward_hook(lambda _m, _i, o: store.update(g=o[0]))
    model.zero_grad()
    with torch.amp.autocast("cuda", enabled=False):
        out = model(x)
        out[0, cls].backward()
    h1.remove(); h2.remove()
    w = store["g"].mean((2, 3), keepdim=True)
    cam = torch.relu((w * store["a"]).sum(1))[0].detach().cpu().numpy()
    return cam / max(cam.max(), 1e-6)


def gradcam_gallery(F, paths, cfg, meta, task, kind, rep):
    spec = TASKS[task]
    fold = cfg["train"][task]["folds"][0]
    arch = "resnet50"
    ck = paths.ckpt / task / arch / f"fold{fold}" / "best.pt"
    t = pd.read_parquet(paths.preds / f"{task}_{kind}_breast.parquet")
    if kind == "dev":
        allowed = set(meta[(meta.fold == fold)].breast_exam)
        t = t[t.breast_exam.isin(allowed)]
    mm = on_disk(available_images(meta[meta[spec["flag"]]], paths), paths)
    t = t[t.breast_exam.isin(set(mm.breast_exam))]
    if t.empty:
        return
    model, _ = build_model(arch, spec["n_classes"])
    model.load_state_dict(torch.load(ck, map_location="cuda", weights_only=False)["model"])
    model.to("cuda").eval()
    layer = model.layer4[-1]
    size = tuple(cfg["train"][task]["image_size"])
    rng = np.random.RandomState(3)
    if task == "a":
        thr = rep["thresholds_chosen_on_dev"]["stack"]["max_accuracy"]
        yh = (t.stack_p1.values >= thr).astype(int)
        cases = {"True positive": (t.y == 1) & (yh == 1), "False negative": (t.y == 1) & (yh == 0), "True negative": (t.y == 0) & (yh == 0), "False positive": (t.y == 0) & (yh == 1)}
        cls_of = lambda name, row: 1
    else:
        P = t[[f"stack_p{i}" for i in range(spec["n_classes"])]].values
        yh = P.argmax(1)
        cases = {c: (t.y == i) & (yh == i) for i, c in enumerate(spec["classes"])}
        cls_of = lambda name, row: int(row.y)
    fig, axs = plt.subplots(len(cases), 6, figsize=(9, 2.3 * len(cases)))
    axs = np.atleast_2d(axs)
    for r, (name, mask) in enumerate(cases.items()):
        s = t[mask.values]
        picks = s.iloc[rng.permutation(len(s))[:3]]
        for ax in axs[r]:
            ax.axis("off")
        axs[r, 0].text(-0.05, 0.5, name, transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=9)
        for k, row in enumerate(picks.itertuples()):
            im_row = mm[mm.breast_exam == row.breast_exam].iloc[0]
            ds = Mammo(pd.DataFrame([im_row]), paths.png, size, spec["label"], False)
            x, _, _ = ds[0]
            cam = grad_cam(model, layer, x[None].to("cuda"), cls_of(name, row))
            gray = (x[0].numpy() * STD[0] + MEAN[0]).clip(0, 1)
            axs[r, 2 * k].imshow(gray, cmap="gray", vmin=0, vmax=1)
            axs[r, 2 * k + 1].imshow(gray, cmap="gray", vmin=0, vmax=1)
            axs[r, 2 * k + 1].imshow(cv2.resize(cam, gray.shape[::-1]), cmap="inferno", alpha=0.45, vmin=0, vmax=1)
    fig.tight_layout()
    fig.suptitle(f"Grad-CAM, ResNet-50 ({kind}): image, then heatmap", fontweight="bold", y=1.01)
    F.save(fig, f"gradcam_{kind}")


def clinical_compare(F, rep, task, kind):
    """Image-only stack against the same stack given exam type and age, and the radiologist's reading where it was allowed."""
    series = [(rep[kind]["stack"], INK, "Images only"), (rep[f"{kind}_clinical"]["stack_clinical"], "#2a78d6", "Images + exam type and age")]
    if f"{kind}_reading" in rep:
        series.append((rep[f"{kind}_reading"]["stack_reading"], "#eb6834", "Images + exam type, age and radiologist's reading"))
    keys = [("AUC-ROC", "auc_roc")] if task == "a" else [("Balanced accuracy", "balanced_accuracy"), ("Macro AUC", "macro_auc_ovr")]
    fig, ax = plt.subplots(figsize=(6.5, 1.2 + 0.9 * len(keys)))
    for row, (label, k) in enumerate(keys):
        for i, (m, color, name) in enumerate(series):
            off = 0.2 * ((len(series) - 1) / 2 - i)
            ci = m["ci95"][k]
            y = len(keys) - 1 - row + off
            ax.plot([ci["low"], ci["high"]], [y, y], color=color, linewidth=2, solid_capstyle="round")
            ax.plot(m[k], y, "o", color=color, markersize=8, label=name if row == 0 else None)
            ax.text(ci["high"] + 0.006, y, f"{m[k]:.3f}", va="center", color=INK2, fontsize=9)
    ax.set_yticks(range(len(keys)), [lab for lab, _ in keys][::-1])
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.margins(x=0.12)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Score with 95% CI")
    ax.set_title(f"Effect of adding non-image information ({kind})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1 if len(series) > 2 else 2)
    F.save(fig, f"clinical_compare_{kind}")


def metrics_table(F, rep, task, kind):
    res = rep[kind]
    order = ARCHS + ["stack"]
    if task == "a":
        rows = [("AUC-ROC", "auc_roc", None), ("AUC-PR", "auc_pr", None), ("Accuracy", "accuracy", "op"), ("Balanced accuracy", "balanced_accuracy", "op"),
                ("Sensitivity", "sensitivity_recall", "op"), ("Specificity", "specificity", "op"), ("PPV", "ppv_precision", "op"), ("NPV", "npv", "op"), ("F1", "f1", "op"), ("MCC", "mcc", "op")]
        def val(m, k, how):
            v = m["operating_points"]["max_accuracy"][k] if how else m[k]
            ci = m.get("ci95", {}).get(k)
            return f"{v:.3f} ({ci['low']:.3f}-{ci['high']:.3f})" if ci else f"{v:.3f}"
    else:
        rows = [("Accuracy", "accuracy", None), ("Balanced accuracy", "balanced_accuracy", None), ("Macro F1", "macro_f1", None), ("Macro AUC", "macro_auc_ovr", None), ("MCC", "mcc", None)]
        def val(m, k, how):
            ci = m.get("ci95", {}).get(k)
            return f"{m[k]:.3f} ({ci['low']:.3f}-{ci['high']:.3f})" if ci else f"{m[k]:.3f}"
    cells = [[r[0]] + [val(res[a], r[1], r[2]) for a in order] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 0.38 * len(cells) + 0.9))
    ax.axis("off")
    tb = ax.table(cellText=cells, colLabels=["Metric (95% CI)"] + [NAMES[a] for a in order], loc="center", cellLoc="left", edges="horizontal")
    tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.45)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(fontweight="bold")
    ax.set_title(f"Performance summary, breast-exam level ({kind})")
    F.save(fig, f"metrics_table_{kind}")


def main(paths, cfg, meta, task, unlock, log):
    rep = json.loads((paths.metrics / f"{task}_metrics.json").read_text(encoding="utf-8"))
    F = Fig(paths, task, log)
    kind = "test" if unlock else "dev"
    cohort_figs(F, paths, meta, task)
    training_curves(F, paths, cfg, task)
    for k in (["dev", "test"] if unlock else ["dev"]):
        if task == "a":
            binary_figs(F, paths, rep, k)
            model_comparison_a(F, rep, k)
        else:
            multiclass_figs(F, paths, rep, k)
        metrics_table(F, rep, task, k)
        clinical_compare(F, rep, task, k)
    umap_fig(F, paths, cfg, meta[meta[TASKS[task]["flag"]]], task, kind)
    gradcam_gallery(F, paths, cfg, meta, task, kind, rep)
    write_json(F.dir / "figure_index.json", {"at": now(), "figures": F.made})
    log(f"STAGE_DONE figures {task}: {len(F.made)} figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unlock-test", action="store_true")
    a = ap.parse_args()
    p = Paths(a.dry_run)
    main(p, load_config(a.dry_run), load_metadata(p, a.dry_run), a.task, a.unlock_test, Log(p, "figures"))
