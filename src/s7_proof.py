"""Proof of the runs: hashes of every checkpoint, prediction and metric file, plus panels rendered from the saved logs.
The panels are drawn from log files, not screen captures. Writes outputs_v2/proof/."""
from __future__ import annotations

import csv
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from embed_common import ROOT, now, sha256

P = ROOT / "outputs_v2" / "proof"
RUNS = ["outputs", "outputs_v2", "outputs_masked"]
PARTS = ["checkpoints", "predictions", "metrics", "cohort"]


def manifest():
    n = 0
    with open(P / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "bytes", "modified", "sha256"])
        for run in RUNS:
            for part in PARTS:
                for x in sorted((ROOT / run / part).rglob("*")):
                    if x.is_file():
                        w.writerow([x.relative_to(ROOT).as_posix(), x.stat().st_size, x.stat().st_mtime_ns // 10 ** 9, sha256(x)])
                        n += 1
    print("manifest files", n)


def panel(name, title, text):
    lines = text.splitlines()
    fig = plt.figure(figsize=(12, 0.2 * len(lines) + 0.9))
    fig.patch.set_facecolor("#111111")
    fig.text(0.01, 0.99, f"{title}\nRendered from saved logs on {now()}\n\n" + "\n".join(lines), family="monospace", fontsize=8, color="#e6e6e6", va="top")
    fig.savefig(P / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print("panel", name)


def keep(path, pattern, last=40):
    lines = [l[:150] for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if re.search(pattern, l)]
    return "\n".join(lines[-last:])


def main():
    logs = ROOT / "outputs_v2" / "kaggle_logs"
    pat = r"KAGGLE_|STAGE_DONE (fetch|[ab]_)|RUN_FAILED"
    panel("log_kaggle_malignant", "Kaggle job embed-malignant: stages", keep(logs / "embed-malignant" / "pipeline.log", pat))
    panel("log_kaggle_birads", "Kaggle job embed-birads: stages", keep(logs / "embed-birads" / "pipeline.log", pat))
    panel("log_pc_baseline", "PC baseline: stages", keep(ROOT / "outputs" / "logs" / "pipeline.log", r"STAGE_DONE (train_a|evaluate_a|figures_a)|a test stack"))
    panel("log_pc_evaluation", "Second-pass evaluation on the PC", keep(ROOT / "outputs_v2" / "logs" / "pipeline.log", r"\[evaluate\]|STAGE_DONE figures", 24))
    panel("kaggle_gpu", "nvidia-smi on the Kaggle machine", (P / "kaggle_nvidia_smi.txt").read_text(encoding="utf-8"))
    manifest()


if __name__ == "__main__":
    main()
