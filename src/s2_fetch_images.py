"""Download DICOMs from S3, convert to cropped 8-bit PNG, delete the DICOM. Resumable.

Order: every Task A image first, then Task B by rarest class, so Task A can train
while Task B is still downloading.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixels import apply_voi_lut

from embed_common import BUCKET, REGION, Log, Paths, keep_awake, load_aws_env, now, write_json

LONG_SIDE = int(os.environ.get("EMBED_LONG_SIDE", 1024))
FG_LEVEL = 10
MAX_FAIL_RATE = 0.02
MIN_FREE_GB = 15
NETWORK_PATIENCE_S = 3600


def png_path(root: Path, image_id: str) -> Path:
    return root / image_id[-2:] / f"{image_id}.png"


def read_pixels(dcm_path: str):
    """Return (header, stored pixels). dicomsdl gave identical pixels at 2.7x the speed of GDCM; pydicom is the fallback."""
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    try:
        import dicomsdl
        return ds, dicomsdl.open(dcm_path).pixelData(storedvalue=True)
    except Exception:
        return ds, pydicom.dcmread(dcm_path).pixel_array


def to_u8(ds, pixels) -> tuple[np.ndarray, bool]:
    """Apply the DICOM window and return an 8-bit image with a dark background. Every step works on one
    pixel value at a time, so one table over the value range gives the same result as doing it per pixel, far faster."""
    v0, v1 = int(pixels.min()), int(pixels.max())
    values = np.arange(v0, v1 + 1, dtype=pixels.dtype)
    table = values.astype(np.float32)
    windowed = "WindowCenter" in ds or "VOILUTSequence" in ds
    try:
        if windowed:
            table = apply_voi_lut(values, ds, index=0).astype(np.float32)
    except Exception:
        windowed = False
    if float(table.max()) <= float(table.min()):
        raise ValueError("constant image")
    lo, hi = (table.min(), table.max()) if windowed else np.percentile(pixels.astype(np.float32), [0.5, 99.9])
    table = np.clip((table - lo) / max(hi - lo, 1e-6), 0, 1)
    if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
        table = 1 - table
    return (table * 255).astype(np.uint8)[pixels.astype(np.int64) - v0], windowed


def crop_box(u8: np.ndarray) -> tuple[int, int, int, int, float]:
    """Box around the largest non-background region. A fixed low level keeps the whole breast;
    Otsu cut tissue when an implant or dense area was far brighter than the rest."""
    small = cv2.resize(u8, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    mask = (cv2.GaussianBlur(small, (5, 5), 0) > FG_LEVEL).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        raise ValueError("no breast region found")
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[best]
    frac = area / mask.size
    if frac < 0.02:
        raise ValueError(f"breast region too small ({frac:.3f})")
    sx, sy = u8.shape[1] / small.shape[1], u8.shape[0] / small.shape[0]
    x0, y0 = int(max(x * sx - 10, 0)), int(max(y * sy - 10, 0))
    x1, y1 = int(min((x + bw) * sx + 10, u8.shape[1])), int(min((y + bh) * sy + 10, u8.shape[0]))
    return x0, y0, x1, y1, float(frac)


def breast_only(u8: np.ndarray) -> np.ndarray:
    """Black out everything not joined to the breast: burned-in text and view labels. Markers lying on the breast stay."""
    small = cv2.resize(u8, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    mask = (cv2.GaussianBlur(small, (5, 5), 0) > FG_LEVEL).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return u8
    keep = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))
    return u8 * cv2.resize(keep, u8.shape[::-1], interpolation=cv2.INTER_NEAREST)


def marker_mask(u8: np.ndarray) -> np.ndarray:
    """Hollow bright outlines: triangle and circle skin markers. Calcifications are tiny and solid, dense tissue is large and solid."""
    tophat = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, np.ones((7, 7), np.uint8))
    bright = (tophat >= 45).astype(np.uint8)
    out = np.zeros_like(bright)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if not (9 <= w <= 60 and 9 <= h <= 60 and 0.5 <= w / h <= 2.0):
            continue
        comp = (lab[y:y + h, x:x + w] == k).astype(np.uint8)
        cs, tree = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = [cv2.contourArea(c) for c, t in zip(cs, tree[0]) if t[3] >= 0]
        filled = max(cv2.contourArea(c) for c, t in zip(cs, tree[0]) if t[3] < 0)
        if holes and filled > 0 and max(holes) / filled >= 0.25:
            out[lab == k] = 1
    return out


def remove_marks(u8: np.ndarray) -> np.ndarray:
    u8 = breast_only(u8)
    m = marker_mask(u8)
    if m.any():
        u8 = cv2.inpaint(u8, cv2.dilate(m, np.ones((5, 5), np.uint8)), 3, cv2.INPAINT_TELEA)
    return u8


def preprocess(dcm_path: str, out_path: str) -> dict:
    """Runs in a worker process: window, normalise orientation, crop breast, resize, save."""
    ds, pixels = read_pixels(dcm_path)
    u8, windowed = to_u8(ds, pixels)
    h0, w0 = u8.shape[:2]
    x0, y0, x1, y1, frac = crop_box(u8)
    crop = u8[y0:y1, x0:x1]
    flipped = crop[:, : crop.shape[1] // 2].sum() < crop[:, crop.shape[1] // 2:].sum()
    if flipped:
        crop = crop[:, ::-1]
    scale = min(LONG_SIDE / max(crop.shape), 1.0)
    out = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path + ".tmp.png"
    if not cv2.imwrite(tmp, out, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise IOError("png write failed")
    os.replace(tmp, out_path)
    return {"orig_h": h0, "orig_w": w0, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "flipped": int(flipped),
            "out_h": out.shape[0], "out_w": out.shape[1], "windowed": int(windowed),
            "photometric": str(ds.get("PhotometricInterpretation", "")), "breast_frac": round(frac, 4)}


def fetch_order(d: pd.DataFrame) -> pd.DataFrame:
    rare = {"M": 1, "S": 2, "P": 3, "A": 4, "B": 5, "N": 6}
    d = d.copy()
    d["prio"] = np.where(d.in_task_a, 0, d.birads_code.map(rare).fillna(9))
    return d.sort_values(["prio", "image_id"]).reset_index(drop=True)


def run(paths: Paths, subset: pd.DataFrame, cfg: dict, log: Log):
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    keep_awake()
    load_aws_env()
    workers = cfg["fetch_workers"]
    s3 = boto3.client("s3", region_name=REGION, config=Config(
        max_pool_connections=workers + 8, connect_timeout=30, read_timeout=90, retries={"max_attempts": 3, "mode": "standard"}))
    tc = TransferConfig(use_threads=False, multipart_threshold=1 << 40)

    done_csv = paths.state / "fetched.csv"
    fail_csv = paths.state / "fetch_failed.csv"
    tmp_dir = paths.run / "tmp_dcm"
    tmp_dir.mkdir(exist_ok=True)
    fields = ["image_id", "bytes", "orig_h", "orig_w", "x0", "y0", "x1", "y1", "flipped", "out_h", "out_w",
              "windowed", "photometric", "breast_frac", "at"]
    have = set()
    if done_csv.exists():
        have = set(pd.read_csv(done_csv, usecols=["image_id"]).image_id.astype(str))
    failed = set()
    if fail_csv.exists():
        failed = set(pd.read_csv(fail_csv, usecols=["image_id"]).image_id.astype(str))
    d = fetch_order(subset)
    d_a_ids = set(d[d.in_task_a].image_id)
    todo = d[~d.image_id.isin(have | failed)]
    log(f"images total={len(d)} done={len(have)} failed={len(failed)} todo={len(todo)} workers={workers}")

    lock = threading.Lock()
    done_f = open(done_csv, "a", newline="", encoding="utf-8")
    dw = csv.DictWriter(done_f, fieldnames=fields)
    if not have:
        dw.writeheader()
    fail_f = open(fail_csv, "a", newline="", encoding="utf-8")
    fw = csv.writer(fail_f)
    if not failed:
        fw.writerow(["image_id", "key", "error", "at"])
    stop = threading.Event()
    fatal: list[str] = []
    stats = {"ok": 0, "fail": len(failed), "bytes": 0, "t0": time.time()}

    pool = ProcessPoolExecutor(cfg["fetch_decoders"])

    def work(row):
        if stop.is_set():
            return
        image_id, key = row.image_id, row.anon_dicom_path
        tmp = tmp_dir / f"{image_id}.dcm"
        t_first = None
        attempt = 0
        while True:
            if stop.is_set():
                return
            try:
                s3.download_file(BUCKET, key, str(tmp), Config=tc)
                break
            except Exception as e:
                code = (getattr(e, "response", None) or {}).get("Error", {}).get("Code")
                if code in ("404", "NoSuchKey"):
                    with lock:
                        fw.writerow([image_id, key, "missing key", now()]); fail_f.flush()
                        stats["fail"] += 1
                    return
                if code in ("403", "AccessDenied", "ExpiredToken", "InvalidAccessKeyId"):
                    fatal.append(f"S3 access refused: {code}")
                    stop.set()
                    return
                t_first = t_first or time.time()
                attempt += 1
                if time.time() - t_first > NETWORK_PATIENCE_S:
                    fatal.append(f"network down for {NETWORK_PATIENCE_S}s: {type(e).__name__}: {str(e)[:200]}")
                    stop.set()
                    return
                if attempt % 5 == 1:
                    log(f"NETWORK_RETRY attempt {attempt} on {image_id[-12:]}: {type(e).__name__}")
                time.sleep(min(2 ** attempt, 120))
        try:
            size = tmp.stat().st_size
            meta = pool.submit(preprocess, str(tmp), str(png_path(paths.png, image_id))).result()
            with lock:
                dw.writerow({"image_id": image_id, "bytes": size, **meta, "at": now()}); done_f.flush()
                stats["ok"] += 1
                stats["bytes"] += size
        except Exception as e:
            with lock:
                fw.writerow([image_id, key, f"{type(e).__name__}: {str(e)[:200]}", now()]); fail_f.flush()
                stats["fail"] += 1
        finally:
            tmp.unlink(missing_ok=True)

    def check_a_ready():
        marker = paths.state / "task_a_images.done"
        if marker.exists():
            return
        with lock:
            done_f.flush(); fail_f.flush()
        got = set(pd.read_csv(done_csv, usecols=["image_id"]).image_id.astype(str)) if done_csv.exists() else set()
        bad = set(pd.read_csv(fail_csv, usecols=["image_id"]).image_id.astype(str)) if fail_csv.exists() else set()
        if d_a_ids <= (got | bad):
            write_json(marker, {"at": now(), "ok": len(d_a_ids & got), "failed": len(d_a_ids & bad)})
            log(f"STAGE_DONE task_a_images ok={len(d_a_ids & got)} failed={len(d_a_ids & bad)}")

    last_report, last_done, warned = 0.0, time.time(), False
    try:
        check_a_ready()
        with ThreadPoolExecutor(workers) as ex:
            pending = set()
            rows = iter(todo.itertuples())
            exhausted = False
            while True:
                while not exhausted and len(pending) < workers * 2 and not stop.is_set():
                    r = next(rows, None)
                    if r is None:
                        exhausted = True
                        break
                    pending.add(ex.submit(work, r))
                if not pending:
                    break
                finished, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for f in finished:
                    f.result()
                log.beat()
                if finished:
                    last_done, warned = time.time(), False
                elif time.time() - last_done > 900 and not warned:
                    log("FETCH_NO_PROGRESS no image finished for 15 minutes; VPN or network likely down")
                    warned = True
                elif time.time() - last_done > NETWORK_PATIENCE_S + 300:
                    fatal.append("no image finished for over an hour")
                    stop.set()
                attempted = stats["ok"] + stats["fail"]
                if attempted >= 200 and stats["fail"] / max(attempted, 1) > MAX_FAIL_RATE:
                    fatal.append(f"image failure rate {stats['fail']}/{attempted} is above {MAX_FAIL_RATE:.0%}; see {fail_csv}")
                    stop.set()
                if time.time() - last_report > 300:
                    last_report = time.time()
                    el = time.time() - stats["t0"]
                    rate = stats["ok"] / max(el, 1)
                    remain = len(todo) - stats["ok"]
                    free = shutil.disk_usage(paths.png).free / 1e9
                    log(f"progress {stats['ok'] + len(have)}/{len(d)} failed={stats['fail']} {rate:.2f} img/s "
                        f"{stats['bytes'] / max(el, 1) / 1e6:.1f} MB/s eta={remain / max(rate, 1e-6) / 3600:.1f}h free={free:.0f}GB")
                    write_json(paths.state / "fetch_progress.json", {"at": now(), "done": stats["ok"] + len(have), "total": len(d),
                                                                      "failed": stats["fail"], "img_per_s": rate, "free_gb": free})
                    check_a_ready()
                    if free < MIN_FREE_GB:
                        fatal.append(f"disk nearly full: {free:.0f} GB free")
                        stop.set()
                if stop.is_set() and exhausted is False:
                    exhausted = True
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        done_f.close()
        fail_f.close()
    if fatal:
        raise RuntimeError("fetch stopped: " + fatal[0])
    check_a_ready()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    write_json(paths.state / "fetch.done", {"at": now(), "ok": stats["ok"] + len(have), "failed": stats["fail"]})
    log(f"STAGE_DONE fetch ok={stats['ok'] + len(have)} failed={stats['fail']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from pipeline_config import load_config, load_metadata
    p = Paths(a.dry_run)
    c = load_config(a.dry_run)
    run(p, load_metadata(p, a.dry_run), c, Log(p, "fetch"))
