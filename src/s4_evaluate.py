"""Stack the 3 base models and compute every metric, at breast-exam level.

Dev (out-of-fold) results guide all decisions. Test results are produced only with --unlock-test.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax
from sklearn import metrics as M
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from statsmodels.stats.contingency_tables import mcnemar

from embed_common import SEED, Log, Paths, env_info, now, write_json
from pipeline_config import ARCHS, TASKS, load_config, load_metadata

warnings.filterwarnings("ignore", category=UserWarning)


# ---------- metric definitions
def ece(y_correct, conf, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y_correct[idx == b].mean() - conf[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def calib_slope_intercept(y, p):
    x = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    if len(set(y)) < 2:
        return None, None
    lr = LogisticRegression(C=1e6).fit(x, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def at_operating_point(y, p, target_sens=None, target_spec=None):
    fpr, tpr, _ = M.roc_curve(y, p)
    if target_sens is not None:
        return float(1 - fpr[np.argmax(tpr >= target_sens)])
    return float(tpr[np.searchsorted(fpr, 1 - target_spec, side="right") - 1])


def binary_threshold_metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = M.confusion_matrix(y, yh, labels=[0, 1]).ravel()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    ppv, npv = tp / max(tp + fp, 1), tn / max(tn + fn, 1)
    return {"threshold": float(thr), "accuracy": float((yh == y).mean()), "balanced_accuracy": float((sens + spec) / 2),
            "sensitivity_recall": float(sens), "specificity": float(spec), "ppv_precision": float(ppv), "npv": float(npv),
            "f1": float(M.f1_score(y, yh, zero_division=0)), "f2": float(M.fbeta_score(y, yh, beta=2, zero_division=0)),
            "mcc": float(M.matthews_corrcoef(y, yh)), "cohen_kappa": float(M.cohen_kappa_score(y, yh, labels=[0, 1])),
            "youden_j": float(sens + spec - 1), "lr_positive": float(sens / max(1 - spec, 1e-9)), "lr_negative": float((1 - sens) / max(spec, 1e-9)),
            "diagnostic_odds_ratio": float((tp + 0.5) * (tn + 0.5) / ((fp + 0.5) * (fn + 0.5))),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def confident_metrics(y, p):
    """How often the model is right when it is sure, and on how many cases it is sure at all."""
    out = {}
    for c in (0.7, 0.8, 0.9):
        hi, lo = p >= c, p <= 1 - c
        k = hi | lo
        out[f"conf_{c}"] = {"coverage": float(k.mean()), "n": int(k.sum()), "right": float(((p >= 0.5).astype(int) == y)[k].mean()) if k.any() else None,
                            "said_benign": int(lo.sum()), "said_benign_right": float((y[lo] == 0).mean()) if lo.any() else None, "cancers_said_benign": int((y[lo] == 1).sum()),
                            "said_malignant": int(hi.sum()), "said_malignant_right": float((y[hi] == 1).mean()) if hi.any() else None}
    return out


def binary_metrics(y, p, thresholds):
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()), "confident": confident_metrics(y, p)}
    if len(set(y)) == 2:
        slope, intercept = calib_slope_intercept(y, p)
        out.update(auc_roc=float(M.roc_auc_score(y, p)), auc_pr=float(M.average_precision_score(y, p)),
                   partial_auc_fpr10=float(M.roc_auc_score(y, p, max_fpr=0.1)), brier=float(M.brier_score_loss(y, p)),
                   log_loss=float(M.log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
                   ece=ece(y, p), calibration_slope=slope, calibration_intercept=intercept,
                   spec_at_sens90=at_operating_point(y, p, target_sens=0.9), spec_at_sens95=at_operating_point(y, p, target_sens=0.95),
                   sens_at_spec90=at_operating_point(y, p, target_spec=0.9), sens_at_spec95=at_operating_point(y, p, target_spec=0.95))
    out["operating_points"] = {k: binary_threshold_metrics(y, p, t) for k, t in thresholds.items()}
    return out


def pick_thresholds(y, p):
    fpr, tpr, th = M.roc_curve(y, p)
    th = np.clip(th, 0, 1)
    grid = np.unique(np.concatenate([th, [0.5]]))
    acc = [((p >= t).astype(int) == y).mean() for t in grid]
    return {"default_0.5": 0.5, "youden": float(th[np.argmax(tpr - fpr)]), "max_accuracy": float(grid[int(np.argmax(acc))]),
            "sens_90": float(th[np.argmax(tpr >= 0.9)])}


def multiclass_metrics(y, P, classes):
    n = len(classes)
    yh = P.argmax(1)
    labels = list(range(n))
    pr, rc, f1, sup = M.precision_recall_fscore_support(y, yh, labels=labels, zero_division=0)
    per = {}
    aucs, aps = [], []
    for i, c in enumerate(classes):
        yb = (y == i).astype(int)
        tn, fp, fn, tp = M.confusion_matrix(yb, (yh == i).astype(int), labels=[0, 1]).ravel()
        d = {"precision": float(pr[i]), "recall_sensitivity": float(rc[i]), "specificity": float(tn / max(tn + fp, 1)), "f1": float(f1[i]), "support": int(sup[i])}
        if 0 < yb.sum() < len(yb):
            d["auc_ovr"] = float(M.roc_auc_score(yb, P[:, i])); d["auc_pr"] = float(M.average_precision_score(yb, P[:, i]))
            aucs.append(d["auc_ovr"]); aps.append(d["auc_pr"])
        per[c] = d
    conf = P.max(1)
    onehot = np.eye(n)[y]
    w = sup / max(sup.sum(), 1)
    out = {"n": int(len(y)), "accuracy": float((yh == y).mean()), "balanced_accuracy": float(M.recall_score(y, yh, labels=[l for l in labels if sup[l] > 0], average="macro", zero_division=0)),
           "macro_f1": float(f1.mean()), "weighted_f1": float((f1 * w).sum()), "macro_precision": float(pr.mean()), "macro_recall": float(rc.mean()),
           "mcc": float(M.matthews_corrcoef(y, yh)), "cohen_kappa": float(M.cohen_kappa_score(y, yh)),
           "top2_accuracy": float(M.top_k_accuracy_score(y, P, k=2, labels=labels)), "macro_auc_ovr": float(np.mean(aucs)) if aucs else None,
           "weighted_auc_ovr": float(sum(per[classes[i]].get("auc_ovr", 0) * w[i] for i in labels)), "macro_auc_pr": float(np.mean(aps)) if aps else None,
           "log_loss": float(M.log_loss(y, np.clip(P, 1e-7, 1), labels=labels)), "multiclass_brier": float(((P - onehot) ** 2).sum(1).mean()),
           "ece_top_label": ece((yh == y).astype(int), conf), "per_class": per,
           "confusion_matrix": M.confusion_matrix(y, yh, labels=labels).tolist()}
    # BI-RADS 1 < 2 < 3 < 4 < 5 is ordered; category 0 (incomplete) is not on that scale.
    mask = y != 0
    if mask.sum() > 1:
        P15 = P[mask][:, 1:]
        out["qwk_birads_1_to_5"] = float(M.cohen_kappa_score(y[mask], P15.argmax(1) + 1, weights="quadratic"))
        out["mae_birads_1_to_5"] = float(np.abs(y[mask] - (P15.argmax(1) + 1)).mean())
        hi = (y[mask] >= 4).astype(int)
        if 0 < hi.sum() < len(hi):
            s = P15[:, 3:].sum(1) / P15.sum(1)
            out["suspicious_4_5_vs_1_2_3_auc"] = float(M.roc_auc_score(hi, s))
            out["suspicious_4_5_vs_1_2_3_auc_pr"] = float(M.average_precision_score(hi, s))
    return out


def bootstrap(fn, df, n, seed=SEED):
    """Resample patients, so the interval respects multiple exams per patient."""
    rng = np.random.RandomState(seed)
    groups = {k: v.values for k, v in df.groupby("empi_anon").indices.items()} if False else {k: np.asarray(v) for k, v in df.groupby("empi_anon").indices.items()}
    keys = list(groups)
    res = {}
    for _ in range(n):
        pick = rng.randint(0, len(keys), len(keys))
        idx = np.concatenate([groups[keys[i]] for i in pick])
        try:
            for k, v in fn(idx).items():
                if v is not None and np.isfinite(v):
                    res.setdefault(k, []).append(v)
        except ValueError:
            continue
    return {k: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5)), "n_boot": len(v)} for k, v in res.items()}


# ---------- loading predictions
def load_breast(paths, task, cfg, kind):
    """View logits -> mean per breast exam per model. Test logits are averaged over fold models."""
    spec = TASKS[task]
    n = spec["n_classes"]
    cols = [f"logit_{i}" for i in range(n)]
    frames = {}
    for arch in ARCHS:
        parts = []
        for f in cfg["train"][task]["folds"]:
            parts.append(pd.read_parquet(paths.preds / f"{task}_{arch}_fold{f}_{'val' if kind == 'dev' else 'test'}.parquet"))
        v = pd.concat(parts)
        v = v.groupby(["image_id", "empi_anon", "breast_exam", spec["label"]], as_index=False)[cols].mean()
        b = v.groupby(["empi_anon", "breast_exam", spec["label"]], as_index=False)[cols].mean()
        frames[arch] = b.rename(columns={spec["label"]: "y"}).astype({"y": int}).sort_values("breast_exam").reset_index(drop=True)
    base = frames[ARCHS[0]]
    for a in ARCHS[1:]:
        assert (frames[a].breast_exam.values == base.breast_exam.values).all(), "base models disagree on the exam list"
    X = np.hstack([frames[a][cols].values for a in ARCHS])
    return base[["empi_anon", "breast_exam", "y"]].copy(), frames, X


def to_features(X, n):
    """Binary: one log-odds per model. Multiclass: log-probabilities per class per model."""
    blocks = np.split(X, len(ARCHS), axis=1)
    if n == 2:
        return np.hstack([b[:, [1]] - b[:, [0]] for b in blocks])
    return np.hstack([np.log(np.clip(softmax(b, 1), 1e-6, 1)) for b in blocks])


def fit_meta(name, n_classes, X, y):
    if name == "logistic":
        return LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=120, learning_rate=0.03, num_leaves=7, min_child_samples=20, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, random_state=SEED, verbose=-1, n_jobs=3).fit(X, y)
    raise ValueError(name)


def meta_proba(model, X, n):
    P = np.zeros((len(X), n))
    P[:, model.classes_.astype(int)] = model.predict_proba(X)
    return P


def mean_proba(X, n):
    blocks = np.split(X, len(ARCHS), axis=1)
    return softmax(np.mean(blocks, axis=0), axis=1)


def model_proba(frames, a, n):
    return softmax(frames[a][[f"logit_{i}" for i in range(n)]].values, axis=1)


def nested_oof(name, n, Xf, y, groups):
    min_groups = pd.Series(y).groupby(y).apply(lambda s: groups[s.index].size if False else pd.unique(groups[s.index]).size).min()
    k = int(min(5, min_groups))
    splitter = StratifiedGroupKFold(k, shuffle=True, random_state=SEED) if k >= 2 else GroupKFold(2)
    P = np.zeros((len(y), n))
    for tr, va in splitter.split(Xf, y, groups):
        P[va] = meta_proba(fit_meta(name, n, Xf[tr], y[tr]), Xf[va], n)
    return P


def primary(task, y, P):
    if task == "a":
        return float(M.roc_auc_score(y, P[:, 1])) if len(set(y)) == 2 else 0.5
    return float(M.balanced_accuracy_score(y, P.argmax(1)))


def boot_fn_a(y, p, thr):
    def f(i):
        yy, pp = y[i], p[i]
        if len(set(yy)) < 2:
            return {}
        t = binary_threshold_metrics(yy, pp, thr)
        return {"auc_roc": M.roc_auc_score(yy, pp), "auc_pr": M.average_precision_score(yy, pp), "accuracy": t["accuracy"], "balanced_accuracy": t["balanced_accuracy"],
                "sensitivity_recall": t["sensitivity_recall"], "specificity": t["specificity"], "ppv_precision": t["ppv_precision"], "npv": t["npv"], "f1": t["f1"], "mcc": t["mcc"]}
    return f


def boot_fn_b(y, P):
    yh = P.argmax(1)
    def f(i):
        yy, ph = y[i], yh[i]
        d = {"accuracy": float((yy == ph).mean()), "balanced_accuracy": M.balanced_accuracy_score(yy, ph), "macro_f1": M.f1_score(yy, ph, average="macro", zero_division=0), "mcc": M.matthews_corrcoef(yy, ph)}
        present = np.unique(yy)
        if len(present) > 1:
            d["macro_auc_ovr"] = float(np.mean([M.roc_auc_score((yy == c).astype(int), P[i][:, c]) for c in present]))
        return d
    return f


# ---------- subgroups
def add_covariates(tbl, meta):
    be = meta.drop_duplicates("breast_exam").set_index("breast_exam")
    t = tbl.copy()
    for c in ["Manufacturer", "exam_kind", "density", "FinalImageType", "ethnicity", "age"]:
        t[c] = t.breast_exam.map(be[c])
    t["age_group"] = pd.cut(t.age, [0, 50, 60, 70, 200], labels=["under 50", "50-59", "60-69", "70+"]).astype(object)
    t["density"] = t.density.map({1: "A fatty", 2: "B scattered", 3: "C heterogeneous", 4: "D dense"})
    t["ethnicity"] = t.ethnicity.where(t.ethnicity.isin(["African American  or Black", "Caucasian or White", "Asian"]))
    return t


def subgroups(task, t, P, thr, boots):
    rows = []
    for col in ["Manufacturer", "exam_kind", "density", "age_group", "ethnicity", "FinalImageType"]:
        for val, g in t.groupby(col, dropna=True):
            idx = g.index.values
            y = t.y.values[idx]
            if len(idx) < 20:
                continue
            row = {"group": col, "value": str(val), "n": len(idx), "n_patients": int(g.empi_anon.nunique())}
            if task == "a":
                row["n_pos"] = int(y.sum())
                if len(set(y)) == 2 and min(y.sum(), (1 - y).sum()) >= 5:
                    p = P[idx, 1]
                    tm = binary_threshold_metrics(y, p, thr)
                    row.update(auc_roc=M.roc_auc_score(y, p), accuracy=tm["accuracy"], sensitivity=tm["sensitivity_recall"], specificity=tm["specificity"])
                    ci = bootstrap(lambda i: {"auc_roc": M.roc_auc_score(y[i], p[i]) if len(set(y[i])) == 2 else None}, g.reset_index(drop=True), boots)
                    row.update(auc_low=ci["auc_roc"]["low"], auc_high=ci["auc_roc"]["high"])
            else:
                yh = P[idx].argmax(1)
                row.update(accuracy=float((yh == y).mean()), balanced_accuracy=M.balanced_accuracy_score(y, yh), macro_f1=M.f1_score(y, yh, average="macro", zero_division=0))
            rows.append(row)
    return pd.DataFrame(rows)


def clinical_features(tbl, meta):
    """Known before anyone reads the image: exam type and age. Kept apart from the image-only stack so both are reported."""
    be = meta.drop_duplicates("breast_exam").set_index("breast_exam").reindex(tbl.breast_exam)
    return np.column_stack([(be.exam_kind == "diagnostic").astype(float).values, ((be.age.astype(float).fillna(60) - 60) / 12).values])


READING_FIELDS = ["massshape", "massmargin", "massdens", "calcfind", "calcdistri", "otherfind", "asses"]


def load_readings(paths):
    """What the radiologist recorded before the biopsy. Pathology, grade, stage, receptors and surgery are the answer and never inputs.
    BI-RADS K means a cancer was already known, so only ordinary assessments are kept. Returns None where the table is absent."""
    f = paths.tables / "EMBED_OpenData_clinical.csv"
    if not f.exists():
        return None
    from s1_build_metadata import EXCLUDED_VTYPES
    c = pd.read_csv(f, usecols=["acc_anon", "side", "vtype"] + READING_FIELDS, low_memory=False)
    c = c[~c.vtype.astype(str).str.strip().isin(EXCLUDED_VTYPES)].copy()
    c.loc[~c.asses.isin(["N", "B", "P", "A", "S", "M"]), "asses"] = np.nan
    return c


def reading_features(c, tbl, fields=READING_FIELDS, columns=None, min_count=10):
    """One-hot: this finding was recorded on this breast in this exam. Unsided rows count for both breasts.
    The columns come from dev only and are reused unchanged on test."""
    rows = []
    for side in ("L", "R"):
        s = c[(c.side == side) | c.side.isna() | (c.side == "B")]
        for col in fields:
            v = s[["acc_anon", col]].dropna()
            rows.append(pd.DataFrame({"breast_exam": v.acc_anon.astype(str) + "_" + side, "f": col + "=" + v[col].astype(str).str.strip()}))
    long = pd.concat(rows).drop_duplicates()
    long = long[long.breast_exam.isin(set(tbl.breast_exam))]
    if columns is None:
        keep = long.f.value_counts()
        columns = sorted(keep[keep >= min_count].index)
    return pd.crosstab(long.breast_exam, long.f).clip(upper=1).reindex(index=tbl.breast_exam, columns=columns).fillna(0).astype(float)


def evaluate(paths, cfg, meta, task, unlock_test, log):
    spec = TASKS[task]
    n, classes = spec["n_classes"], spec["classes"]
    boots = int(cfg["eval"]["bootstrap"])
    dev_tbl, dev_frames, Xd = load_breast(paths, task, cfg, "dev")
    y_dev, groups = dev_tbl.y.values, dev_tbl.empi_anon.values
    Fd = to_features(Xd, n)

    # choose the stacker using nested out-of-fold predictions on dev only
    cands = {"mean_of_models": mean_proba(Xd, n)}
    for name in ["logistic", "lightgbm"]:
        cands[name] = nested_oof(name, n, Fd, y_dev, groups)
    scores = {k: primary(task, y_dev, v) for k, v in cands.items()}
    chosen = max(scores, key=scores.get)
    log(f"{task} stacker candidates on dev: {scores}; chosen={chosen}")
    stack_dev = cands[chosen]

    dev_probs = {a: model_proba(dev_frames, a, n) for a in ARCHS}
    dev_probs["stack"] = stack_dev

    report = {**env_info(), "task": spec["name"], "level": "breast exam (mean of CC and MLO logits)", "evaluated_at": now(), "stacker": chosen,
              "stacker_scores_on_dev": scores, "base_models": ARCHS, "folds": cfg["train"][task]["folds"], "bootstrap_resamples": boots, "dev": {}}

    def score_set(tbl, probs, thresholds_by_model):
        res = {}
        for name, P in probs.items():
            y = tbl.y.values
            log.beat()
            if task == "a":
                p = P[:, 1]
                res[name] = binary_metrics(y, p, thresholds_by_model[name])
                res[name]["ci95"] = bootstrap(boot_fn_a(y, p, thresholds_by_model[name]["max_accuracy"]), tbl, boots)
            else:
                res[name] = multiclass_metrics(y, P, classes)
                res[name]["ci95"] = bootstrap(boot_fn_b(y, P), tbl, boots)
        return res

    thr = {name: pick_thresholds(y_dev, P[:, 1]) for name, P in dev_probs.items()} if task == "a" else {}
    report["dev"] = score_set(dev_tbl, dev_probs, thr)
    report["thresholds_chosen_on_dev"] = thr

    meta_cov = meta[meta[spec["flag"]]]
    Cd = clinical_features(dev_tbl, meta_cov)
    clin_dev = nested_oof("logistic", n, np.hstack([Fd, Cd]), y_dev, groups)
    clin_thr = {"stack_clinical": pick_thresholds(y_dev, clin_dev[:, 1])} if task == "a" else {}
    report["dev_clinical"] = score_set(dev_tbl, {"stack_clinical": clin_dev}, clin_thr)
    report["thresholds_clinical_on_dev"] = clin_thr
    log(f"{task} dev image stack {scores[chosen]:.4f}, with exam type and age {primary(task, y_dev, clin_dev):.4f}")
    # Malignant vs benign only: on BI-RADS the reading would hand the model its own label.
    readings = load_readings(paths) if task == "a" else None
    read_extra = {}
    if readings is not None:
        Rd = reading_features(readings, dev_tbl)
        Xr = np.hstack([Fd, Cd, Rd.values])
        read_extra = {"stack_reading": nested_oof("logistic", n, Xr, y_dev, groups)}
        read_thr = {"stack_reading": pick_thresholds(y_dev, read_extra["stack_reading"][:, 1])}
        report["dev_reading"] = score_set(dev_tbl, read_extra, read_thr)
        report["thresholds_reading_on_dev"] = read_thr
        report["reading_feature_columns"] = list(Rd.columns)
        log(f"{task} dev with radiologist reading, exam type and age {primary(task, y_dev, read_extra['stack_reading']):.4f}")
    save_breast(paths, task, "dev", dev_tbl, {**dev_probs, "stack_clinical": clin_dev, **read_extra}, n)
    t = add_covariates(dev_tbl, meta_cov)
    thr0 = thr["stack"]["max_accuracy"] if task == "a" else None
    subgroups(task, t, stack_dev, thr0, min(boots, 300)).to_csv(paths.metrics / f"{task}_dev_subgroups.csv", index=False)

    if unlock_test:
        test_tbl, test_frames, Xt = load_breast(paths, task, cfg, "test")
        Ft = to_features(Xt, n)
        if chosen == "mean_of_models":
            stack_test = mean_proba(Xt, n)
        else:
            mdl = fit_meta(chosen, n, Fd, y_dev)
            stack_test = meta_proba(mdl, Ft, n)
            if chosen == "logistic":
                report["stacker_coefficients"] = np.round(mdl.coef_, 4).tolist()
        test_probs = {a: model_proba(test_frames, a, n) for a in ARCHS}
        test_probs["stack"] = stack_test
        report["test"] = score_set(test_tbl, test_probs, thr)
        clin_test = meta_proba(fit_meta("logistic", n, np.hstack([Fd, Cd]), y_dev), np.hstack([Ft, clinical_features(test_tbl, meta_cov)]), n)
        report["test_clinical"] = score_set(test_tbl, {"stack_clinical": clin_test}, clin_thr)
        read_test = {}
        if readings is not None:
            Rt = reading_features(readings, test_tbl, columns=list(Rd.columns))
            read_test = {"stack_reading": meta_proba(fit_meta("logistic", n, Xr, y_dev), np.hstack([Ft, clinical_features(test_tbl, meta_cov), Rt.values]), n)}
            report["test_reading"] = score_set(test_tbl, read_test, read_thr)
        save_breast(paths, task, "test", test_tbl, {**test_probs, "stack_clinical": clin_test, **read_test}, n)
        tt = add_covariates(test_tbl, meta_cov)
        subgroups(task, tt, stack_test, thr0, min(boots, 300)).to_csv(paths.metrics / f"{task}_test_subgroups.csv", index=False)
        report["stack_vs_best_base_test"] = compare(task, test_tbl, test_probs, report["dev"], thr, boots)
        report["test_patient_level"] = patient_level(task, test_tbl, stack_test)
        report["test_opened_at"] = now()

    write_json(paths.metrics / f"{task}_metrics.json", report)
    flat = flatten(report)
    flat.to_csv(paths.metrics / f"{task}_metrics_long.csv", index=False)
    head = "dev" if not unlock_test else "test"
    r = report[head]["stack"]
    log(f"{task} {head} stack: " + (f"AUC={r.get('auc_roc'):.4f} acc@maxacc={r['operating_points']['max_accuracy']['accuracy']:.4f} sens={r['operating_points']['max_accuracy']['sensitivity_recall']:.3f} spec={r['operating_points']['max_accuracy']['specificity']:.3f}"
                                    if task == "a" else f"acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f} macro_f1={r['macro_f1']:.4f} macro_auc={r['macro_auc_ovr']}"))
    log(f"STAGE_DONE evaluate {task} unlock_test={unlock_test}")
    return report


def save_breast(paths, task, kind, tbl, probs, n):
    t = tbl.copy()
    for name, P in probs.items():
        for i in range(n):
            t[f"{name}_p{i}"] = P[:, i]
    t.to_parquet(paths.preds / f"{task}_{kind}_breast.parquet", index=False)


def compare(task, tbl, probs, dev, thr, boots):
    y = tbl.y.values
    key = "auc_roc" if task == "a" else "balanced_accuracy"
    best = max(ARCHS, key=lambda a: dev[a].get(key) or 0)
    if task == "a":
        ps, pb = probs["stack"][:, 1], probs[best][:, 1]
        met = lambda yy, pp: M.roc_auc_score(yy, pp)
        ya, yb = (ps >= thr["stack"]["max_accuracy"]).astype(int), (pb >= thr[best]["max_accuracy"]).astype(int)
    else:
        ps, pb = probs["stack"], probs[best]
        met = lambda yy, pp: M.balanced_accuracy_score(yy, pp.argmax(1))
        ya, yb = ps.argmax(1), pb.argmax(1)
    rng = np.random.RandomState(SEED)
    idx_by = {k: np.asarray(v) for k, v in tbl.groupby("empi_anon").indices.items()}
    keys = list(idx_by)
    deltas = []
    for _ in range(boots):
        i = np.concatenate([idx_by[keys[j]] for j in rng.randint(0, len(keys), len(keys))])
        try:
            deltas.append(met(y[i], ps[i]) - met(y[i], pb[i]))
        except ValueError:
            pass
    d = np.array(deltas)
    a, b = ya == y, yb == y
    table = [[int((a & b).sum()), int((a & ~b).sum())], [int((~a & b).sum()), int((~a & ~b).sum())]]
    return {"best_base_by_dev": best, "metric": key, "delta_mean": float(d.mean()), "delta_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "bootstrap_two_sided_p": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), "mcnemar_table_stack_vs_base": table, "mcnemar_p": float(mcnemar(table, exact=False, correction=True).pvalue)}


def patient_level(task, tbl, P):
    t = tbl[["empi_anon", "y"]].copy()
    if task == "a":
        t["s"] = P[:, 1]
        g = t.groupby("empi_anon").agg(y=("y", "max"), s=("s", "max"))
        return {"n_patients": int(len(g)), "auc_roc_max_pooled": float(M.roc_auc_score(g.y, g.s)) if g.y.nunique() == 2 else None}
    return {"n_patients": int(t.empi_anon.nunique())}


def flatten(report):
    rows = []
    def walk(prefix, o):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(prefix + [str(k)], v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            rows.append({"path": ".".join(prefix), "value": o})
    for part in ("dev", "dev_clinical", "dev_reading", "test", "test_clinical", "test_reading", "stack_vs_best_base_test", "test_patient_level"):
        if part in report:
            walk([part], report[part])
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unlock-test", action="store_true")
    a = ap.parse_args()
    p = Paths(a.dry_run)
    evaluate(p, load_config(a.dry_run), load_metadata(p, a.dry_run), a.task, a.unlock_test, Log(p, "evaluate"))
