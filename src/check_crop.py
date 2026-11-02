"""Check the breast crop on original DICOMs: does any breast tissue fall outside the crop box?
Writes outputs/proof/crop_check.png (box drawn in white) and crop_check.csv, plus size statistics."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import boto3
import cv2
import numpy as np
import pandas as pd

from embed_common import BUCKET, REGION, Paths, load_aws_env, write_json
from pipeline_config import load_metadata
from s2_fetch_images import crop_box, read_pixels, to_u8

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
paths = Paths(False)
f = pd.read_csv(paths.state / "fetched.csv").head(10796)  # same rows as the first check, so the same 40 images
f["image_id"] = f.image_id.astype(str)
crop_h, crop_w = f.y1 - f.y0, f.x1 - f.x0
stats = {
    "n": len(f),
    "orig_h_median": float(f.orig_h.median()), "orig_w_median": float(f.orig_w.median()),
    "crop_h_median": float(crop_h.median()), "crop_w_median": float(crop_w.median()),
    "kept_h_median": float((crop_h / f.orig_h).median()), "kept_w_median": float((crop_w / f.orig_w).median()),
    "shrink_to_saved_median": float((crop_h.clip(lower=1).combine(crop_w, max) / 1024).median()),
    "shrink_768x512_median": float((1 / np.minimum(768 / crop_h, 512 / crop_w)).median()),
    "shrink_512x384_median": float((1 / np.minimum(512 / crop_h, 384 / crop_w)).median()),
    "shrink_1536x1024_median": float((1 / np.minimum(1536 / crop_h, 1024 / crop_w)).median()),
}
print(stats)

load_aws_env()
m = load_metadata(paths, False)[["image_id", "anon_dicom_path"]]
s = f.merge(m, on="image_id").sample(N, random_state=7)
s3 = boto3.client("s3", region_name=REGION)
rows, tiles = [], []
with tempfile.TemporaryDirectory() as td:
    for r in s.itertuples():
        p = Path(td) / "x.dcm"
        s3.download_file(BUCKET, r.anon_dicom_path, str(p))
        u8, _ = to_u8(*read_pixels(str(p)))
        fg = u8 > 20
        rec = {"image_id": r.image_id}
        x0, y0, x1, y1, _f = crop_box(u8)
        for tag, box in (("old", (r.x0, r.y0, r.x1, r.y1)), ("new", (x0, y0, x1, y1))):
            inside = np.zeros_like(fg)
            inside[box[1]:box[3], box[0]:box[2]] = True
            rec[f"{tag}_outside_pct"] = 100 * float((fg & ~inside).sum()) / max(int(fg.sum()), 1)
        rec["new_h"], rec["new_w"] = y1 - y0, x1 - x0
        rows.append(rec)
        t = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(t, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), max(4, u8.shape[0] // 300))
        s_ = min(420 / t.shape[0], 330 / t.shape[1])
        t = cv2.resize(t, (round(t.shape[1] * s_), round(t.shape[0] * s_)), interpolation=cv2.INTER_AREA)
        tiles.append(cv2.copyMakeBorder(t, 2, 422 - t.shape[0], 2, 332 - t.shape[1], cv2.BORDER_CONSTANT, value=(60, 60, 60)))
d = pd.DataFrame(rows)
d.to_csv(paths.proof / "crop_check.csv", index=False)
write_json(paths.proof / "crop_check.json", {**stats, "sampled": N, "old_outside_pct_max": float(d.old_outside_pct.max()),
                                             "new_outside_pct_median": float(d.new_outside_pct.median()), "new_outside_pct_max": float(d.new_outside_pct.max())})
print(d.describe().round(3).to_string())
print(d.sort_values("new_outside_pct").tail(4).drop(columns="image_id").round(3).to_string())
cols = 10
grid = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles) - len(tiles) % cols, cols)]
cv2.imwrite(str(paths.proof / "crop_check.png"), np.vstack(grid))
