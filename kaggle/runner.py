"""Runs on a Kaggle GPU machine: fetch images through presigned links, then train the listed runs
on both GPUs. Same preprocessing, splits, seeds and training code as the local run.

Results go to /kaggle/working/run; images stay in /tmp so they never reach the notebook output.
Any failure kills everything and exits non-zero, so the notebook shows as failed and I get woken.
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

WORK = Path("/kaggle/working/run")
CODE = Path("/tmp/embed_code")
DCM = Path("/tmp/dcm")
STALL_S = 20 * 60
MAX_FAIL_RATE = 0.02
GPUS = 2


def stage_code(src: Path):
    """Copy the scripts next to a patched embed_common that points at the Kaggle folders."""
    CODE.mkdir(parents=True, exist_ok=True)
    for f in ("embed_common.py", "pipeline_config.py", "s0_weights.py", "s2_fetch_images.py", "s3_train.py", "s4_evaluate.py", "s5_figures.py"):
        shutil.copy(src / f, CODE / f)
    p = CODE / "embed_common.py"
    t = p.read_text(encoding="utf-8")
    for old, new in (('ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path("{WORK}")'),
                     ('(ROOT / "data" / "embed" / "png")', 'Path("/tmp/png")')):
        assert old in t, old
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8")
    (WORK / "configs").mkdir(parents=True, exist_ok=True)
    (WORK / "outputs" / "cohort").mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "full_run.json").read_text(encoding="utf-8"))
    cfg["loader_workers"] = 2
    cfg["fetch_decoders"] = os.cpu_count()
    (WORK / "configs" / "full_run.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    shutil.copy(src / "metadata_images.parquet", WORK / "outputs" / "cohort" / "metadata_images.parquet")
    sys.path.insert(0, str(CODE))


def init_worker():
    import cv2
    cv2.setNumThreads(0)


def fetch(flag: str, src: Path, log, paths):
    import pandas as pd
    from s2_fetch_images import png_path, preprocess
    from embed_common import now

    meta = pd.read_parquet(WORK / "outputs" / "cohort" / "metadata_images.parquet")
    meta["image_id"] = meta.image_id.astype(str)
    urls = pd.read_csv(src / "links.csv", dtype=str).set_index("image_id").url
    ids = meta[meta[flag]].image_id.tolist()
    DCM.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "bytes", "orig_h", "orig_w", "x0", "y0", "x1", "y1", "flipped", "out_h", "out_w",
              "windowed", "photometric", "breast_frac", "at"]
    done_f = open(paths.state / "fetched.csv", "w", newline="", encoding="utf-8")
    dw = csv.DictWriter(done_f, fieldnames=fields)
    dw.writeheader()
    fail_f = open(paths.state / "fetch_failed.csv", "w", newline="", encoding="utf-8")
    fw = csv.writer(fail_f)
    fw.writerow(["image_id", "error", "at"])
    lock = threading.Lock()
    stop = threading.Event()
    fatal = []
    st = {"ok": 0, "fail": 0, "bytes": 0, "t0": time.time(), "last": time.time()}
    pool = ProcessPoolExecutor(os.cpu_count(), mp_context=mp.get_context("spawn"), initializer=init_worker)

    def work(image_id):
        if stop.is_set():
            return
        tmp = DCM / f"{image_id}.dcm"
        try:
            for attempt in range(6):
                try:
                    with urllib.request.urlopen(urls[image_id], timeout=120) as r, open(tmp, "wb") as f:
                        shutil.copyfileobj(r, f, 1 << 20)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 403:
                        fatal.append(f"link refused with 403 for {image_id[-12:]}; links may have expired")
                        stop.set()
                        return
                    if attempt == 5 or e.code == 404:
                        raise
                except OSError:
                    if attempt == 5:
                        raise
                time.sleep(2 ** attempt)
            size = tmp.stat().st_size
            m = pool.submit(preprocess, str(tmp), str(png_path(paths.png, image_id))).result()
            with lock:
                dw.writerow({"image_id": image_id, "bytes": size, **m, "at": now()})
                done_f.flush()
                st["ok"] += 1
                st["bytes"] += size
        except Exception as e:
            with lock:
                fw.writerow([image_id, f"{type(e).__name__}: {str(e)[:200]}", now()])
                fail_f.flush()
                st["fail"] += 1
        finally:
            tmp.unlink(missing_ok=True)
        with lock:
            n = st["ok"] + st["fail"]
            if n >= 200 and st["fail"] / n > MAX_FAIL_RATE:
                fatal.append(f"image failure rate {st['fail']}/{n}")
                stop.set()
            if time.time() - st["last"] > 120:
                st["last"] = time.time()
                el = time.time() - st["t0"]
                log(f"progress {st['ok']}/{len(ids)} failed={st['fail']} {st['ok'] / el:.2f} img/s {st['bytes'] / el / 1e6:.0f} MB/s")

    with ThreadPoolExecutor(os.cpu_count() * 3) as ex:
        list(ex.map(work, ids))
    pool.shutdown()
    done_f.close()
    fail_f.close()
    if fatal:
        raise RuntimeError("fetch stopped: " + fatal[0])
    log(f"STAGE_DONE fetch ok={st['ok']} failed={st['fail']} in {(time.time() - st['t0']) / 60:.1f} min")


def train(runs, log, paths):
    queue = list(runs)
    lock = threading.Lock()
    fatal = []
    active = {}

    def worker(gpu):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(CODE)}
        while not fatal:
            with lock:
                if not queue:
                    return
                task, arch, fold = queue.pop(0)
            name = f"{task}_{arch}_fold{fold}"
            cmd = [sys.executable, "-u", str(CODE / "s3_train.py"), "--task", task, "--arch", arch, "--fold", str(fold)]
            for attempt in (1, 2):
                t0 = time.time()
                with open(paths.logs / f"{name}.err", "a", encoding="utf-8") as err:
                    proc = subprocess.Popen(cmd, env=env, cwd=CODE, stdout=subprocess.DEVNULL, stderr=err)
                    active[gpu] = proc
                    code = proc.wait()
                if code == 0:
                    log(f"STAGE_DONE {name} on gpu{gpu} in {(time.time() - t0) / 60:.1f} min")
                    break
                log(f"RUN_FAILED {name} attempt {attempt} code {code}")
                if attempt == 2:
                    fatal.append(f"{name} failed twice with code {code}, see {name}.err")

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in range(GPUS)]
    for t in threads:
        t.start()
    hb = paths.state / "heartbeat_train.txt"
    t0 = time.time()
    while any(t.is_alive() for t in threads):
        if not fatal and time.time() - max(hb.stat().st_mtime if hb.exists() else 0, t0) > STALL_S:
            fatal.append(f"no training progress for {STALL_S // 60} minutes")
        if fatal:
            for p in active.values():
                if p.poll() is None:
                    p.kill()
            with lock:
                queue.clear()
        time.sleep(10)
    if fatal:
        raise RuntimeError(fatal[0])


def report(task, log, paths):
    """Dev metrics and figures, made here because the images are here. Training results are safe if this fails."""
    env = {**os.environ, "PYTHONPATH": str(CODE)}
    for script in ("s4_evaluate.py", "s5_figures.py"):
        with open(paths.logs / f"{script}.err", "a", encoding="utf-8") as err:
            code = subprocess.call([sys.executable, "-u", str(CODE / script), "--task", task], cwd=CODE, env=env, stdout=subprocess.DEVNULL, stderr=err)
        log(f"REPORT {script} task {task} exit {code}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--flag", required=True, choices=["in_task_a", "in_task_b"])
    ap.add_argument("--runs", required=True, help="comma list of task:arch:fold")
    ap.add_argument("--long-side", required=True, help="saved image long side")
    a = ap.parse_args()
    os.environ["EMBED_LONG_SIDE"] = a.long_side
    src = Path(a.src)
    stage_code(src)
    from embed_common import Log, Paths, env_info, write_json
    paths = Paths(False)
    log = Log(paths, "kaggle")
    write_json(paths.proof / "kaggle_env.json", {**env_info(), "argv": sys.argv, "gpus": GPUS, "cpus": os.cpu_count()})
    (paths.proof / "kaggle_nvidia_smi.txt").write_text(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, encoding="utf-8")
    (paths.proof / "kaggle_pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout, encoding="utf-8")
    import pyarrow  # noqa: F401  checked now because predictions are written only at the end of a run
    runs = [tuple(r.split(":")) for r in a.runs.split(",")]
    log(f"KAGGLE_START flag={a.flag} runs={len(runs)}")
    try:
        subprocess.run([sys.executable, "-u", str(CODE / "s0_weights.py")], check=True, cwd=CODE, env={**os.environ, "PYTHONPATH": str(CODE)})
        fetch(a.flag, src, log, paths)
        train(runs, log, paths)
        report(runs[0][0], log, paths)
    except Exception as e:
        log(f"KAGGLE_ERROR {type(e).__name__}: {e}")
        raise
    log("KAGGLE_DONE")


if __name__ == "__main__":
    main()
