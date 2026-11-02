"""Bring a finished Kaggle job into outputs_v2/ (fixed crop, higher resolution).
The PC run in outputs/ is the first-pass baseline and is not touched."""
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
slug = sys.argv[1]
src = next((ROOT / "kaggle" / f"{slug}_out").rglob("run/outputs")).resolve()
dest = ROOT / "outputs_v2"
shutil.copytree(src, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("pipeline.log", "fetched.csv", "fetch_failed.csv", "heartbeat_*", "cohort"))
for rel in ("logs/pipeline.log", "state/fetched.csv", "state/fetch_failed.csv"):
    (dest / "kaggle_logs" / slug).mkdir(parents=True, exist_ok=True)
    if (src / rel).exists():
        shutil.copy2(src / rel, dest / "kaggle_logs" / slug / Path(rel).name)
# Images fetched by any job, for cohort counts. Same cohort, so a plain union is right.
state = dest / "state" / "fetched.csv"
parts = [pd.read_csv(f, usecols=["image_id"], dtype=str) for f in (dest / "kaggle_logs").glob("*/fetched.csv")]
pd.concat(parts).drop_duplicates().to_csv(state, index=False)
print("imported", slug, "runs", sorted(p.parent.relative_to(src / "checkpoints").as_posix() for p in (src / "checkpoints").glob("*/*/fold*/done.json")))
