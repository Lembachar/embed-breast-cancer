"""Train one base model on one fold. Resumable every epoch; can warm-start when data is added.

Outputs under checkpoints/<task>/<arch>/fold<k>/: last.pt, best.pt, history.csv, config.json, done.json
Predictions under predictions/: <task>_<arch>_fold<k>_{val,test}.parquet (view-level logits)
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

from embed_common import SEED, Log, Paths, env_info, keep_awake, now, write_json
from pipeline_config import ARCHS, TASKS, available_images, load_config, load_metadata
from s2_fetch_images import png_path, remove_marks

MASK_MARKS = "EMBED_MASK_MARKS" in os.environ

cv2.setNumThreads(0)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


class Mammo(Dataset):
    def __init__(self, df, root, size, label_col, train):
        self.ids = df.image_id.values
        self.y = df[label_col].astype(int).values
        self.root, self.size, self.train = Path(root), size, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = cv2.imread(str(png_path(self.root, self.ids[i])), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"unreadable image {self.ids[i]}")
        if MASK_MARKS:
            img = remove_marks(img)
        H, W = self.size
        s = min(H / img.shape[0], W / img.shape[1])
        img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((H, W), np.uint8)
        canvas[: img.shape[0], : img.shape[1]] = img
        if self.train:
            if random.random() < 0.5:
                canvas = canvas[::-1].copy()
            ang, sc = random.uniform(-10, 10), random.uniform(0.9, 1.1)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc)
            M[:, 2] += [random.uniform(-0.05, 0.05) * W, random.uniform(-0.05, 0.05) * H]
            canvas = cv2.warpAffine(canvas, M, (W, H), flags=cv2.INTER_LINEAR)
            x = canvas.astype(np.float32) / 255
            x = np.clip((x - 0.5) * random.uniform(0.85, 1.15) + 0.5 + random.uniform(-0.08, 0.08), 0, 1)
        else:
            x = canvas.astype(np.float32) / 255
        x = np.repeat(x[None], 3, 0)
        x = (x - MEAN[:, None, None]) / STD[:, None, None]
        return torch.from_numpy(x), int(self.y[i]), i


def build_model(arch: str, n: int) -> tuple[nn.Module, nn.Module]:
    if arch == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Linear(m.fc.in_features, n)
        return m, m.fc
    if arch == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
        return m, m.classifier[-1]
    if arch == "convnext_tiny":
        m = models.convnext_tiny(weights="IMAGENET1K_V1")
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
        return m, m.classifier[-1]
    raise ValueError(arch)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, head, loader, device, want_features=False):
    model.eval()
    logits, feats = [], []
    hook = head.register_forward_pre_hook(lambda _m, inp: feats.append(inp[0].flatten(1).float().cpu())) if want_features else None
    for x, _, _ in loader:
        with torch.amp.autocast("cuda"):
            logits.append(model(x.to(device, non_blocking=True)).float().cpu())
    if hook:
        hook.remove()
    return torch.cat(logits).numpy(), (torch.cat(feats).numpy() if want_features else None)


def breast_score(df, logits, task):
    t = df[["breast_exam"]].reset_index(drop=True)
    y = df[TASKS[task]["label"]].astype(int).values
    cols = [f"l{i}" for i in range(logits.shape[1])]
    t[cols] = logits
    t["y"] = y
    g = t.groupby("breast_exam").agg({**{c: "mean" for c in cols}, "y": "first"})
    L = g[cols].values
    if task == "a":
        if g.y.nunique() < 2:
            return 0.5, "auc"
        return float(roc_auc_score(g.y, L[:, 1] - L[:, 0])), "auc"
    return float(balanced_accuracy_score(g.y, L.argmax(1))), "balanced_accuracy"


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_one(paths: Paths, cfg: dict, meta: pd.DataFrame, task: str, arch: str, fold: int, log: Log, warm_start: str | None = None):
    keep_awake()
    spec, label = TASKS[task], TASKS[task]["label"]
    out = paths.ckpt / task / arch / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        log(f"{task}/{arch}/fold{fold} already complete, skipping")
        return
    seed_all(SEED + fold)
    d = available_images(meta[meta[spec["flag"]]], paths)
    dev, test = d[d.split == "dev"], d[d.split == "test"]
    tr, va = dev[dev.fold != fold], dev[dev.fold == fold]
    tc = cfg["train"][task]
    bs, size = tc["batch_size"], tuple(tc["image_size"])
    log(f"start {task}/{arch}/fold{fold} train={len(tr)} val={len(va)} test={len(test)} size={size} bs={bs}")

    counts = tr[label].astype(int).value_counts().sort_index().reindex(range(spec["n_classes"]), fill_value=0).values
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    sampler = WeightedRandomSampler(torch.as_tensor(weights[tr[label].astype(int).values], dtype=torch.double), len(tr), replacement=True,
                                    generator=torch.Generator().manual_seed(SEED + fold))
    nw = cfg["loader_workers"]
    kw = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    tl = DataLoader(Mammo(tr, paths.png, size, label, True), batch_size=bs, sampler=sampler, drop_last=True, **kw)
    vl = DataLoader(Mammo(va, paths.png, size, label, False), batch_size=bs * 2, shuffle=False, **kw)

    device = "cuda"
    model, head = build_model(arch, spec["n_classes"])
    model.to(device)
    if warm_start:
        model.load_state_dict(torch.load(warm_start, map_location=device, weights_only=False)["model"])
        log(f"warm start from {warm_start}")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    epochs, accum = tc["epochs"], tc["accum"]
    steps = max(1, len(tl) // accum)
    warm = steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps * epochs) / (steps * epochs))))
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])
    start, best, bad = 0, -1.0, 0

    hist = out / "history.csv"
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best, bad = ck["epoch"] + 1, ck["best"], ck["bad"]
        torch.set_rng_state(ck["rng"].cpu()); np.random.set_state(ck["np_rng"]); random.setstate(ck["py_rng"])
        log(f"resumed {task}/{arch}/fold{fold} at epoch {start}, best={best:.4f}")
    write_json(out / "config.json", {**env_info(), "task": task, "arch": arch, "fold": fold, "train_cfg": tc,
                                     "n_train": len(tr), "n_val": len(va), "class_counts_train": counts.tolist(),
                                     "train_patients": int(tr.empi_anon.nunique()), "warm_start": warm_start})
    if not hist.exists():
        with open(hist, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_metric", "metric_name", "lr", "seconds", "images_per_s", "at"])

    for ep in range(start, epochs):
        t0 = time.time()
        model.train()
        loss_sum, n_seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _) in enumerate(tl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y) / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at {task}/{arch}/fold{fold} epoch {ep} step {i}")
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            loss_sum += loss.item() * accum * len(x)
            n_seen += len(x)
            if i % 20 == 0:
                log.beat()
        secs = time.time() - t0
        logits, _ = predict(model, head, vl, device)
        val, name = breast_score(va, logits, task)
        lr = opt.param_groups[0]["lr"]
        improved = val > best
        best, bad = (val, 0) if improved else (best, bad + 1)
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
              "epoch": ep, "best": best, "bad": bad, "rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
              "py_rng": random.getstate(), "task": task, "arch": arch, "fold": fold, "cfg": tc, "saved_at": now()}
        atomic_save(ck, out / "last.pt")
        if improved:
            atomic_save({"model": model.state_dict(), "epoch": ep, "val": val, "metric": name, "task": task, "arch": arch, "fold": fold, "saved_at": now()}, out / "best.pt")
        with open(hist, "a", newline="") as f:
            csv.writer(f).writerow([ep, round(loss_sum / max(n_seen, 1), 5), round(val, 5), name, f"{lr:.2e}", round(secs, 1), round(n_seen / secs, 1), now()])
        log(f"{task}/{arch}/fold{fold} epoch {ep + 1}/{epochs} loss={loss_sum / max(n_seen, 1):.4f} val_{name}={val:.4f} best={best:.4f} {n_seen / secs:.1f} img/s {secs / 60:.1f}min")
        if bad >= tc["patience"]:
            log(f"{task}/{arch}/fold{fold} early stop at epoch {ep + 1}")
            break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["model"])
    for name_, df_ in (("val", va), ("test", test)):
        loader = DataLoader(Mammo(df_, paths.png, size, label, False), batch_size=bs * 2, shuffle=False, num_workers=nw, pin_memory=True)
        logits, feats = predict(model, head, loader, device, want_features=True)
        p = df_[["image_id", "empi_anon", "breast_exam", label]].reset_index(drop=True)
        p[[f"logit_{i}" for i in range(spec["n_classes"])]] = logits
        p.to_parquet(paths.preds / f"{task}_{arch}_fold{fold}_{name_}.parquet", index=False)
        if name_ == "val" or fold == cfg["train"][task]["folds"][0]:
            np.savez_compressed(paths.preds / f"{task}_{arch}_fold{fold}_{name_}_features.npz", image_id=df_.image_id.values, features=feats.astype(np.float16))
    write_json(out / "done.json", {"at": now(), "best": best})
    log(f"STAGE_DONE train {task}/{arch}/fold{fold} best={best:.4f}")
    del model, opt, tl, vl
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--warm-start", default=None, help="best.pt to continue from when data has been added")
    a = ap.parse_args()
    p = Paths(a.dry_run)
    train_one(p, load_config(a.dry_run), load_metadata(p, a.dry_run), a.task, a.arch, a.fold, Log(p, "train"), a.warm_start)
