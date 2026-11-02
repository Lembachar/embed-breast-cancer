"""Stage orchestrator. Finished stages are skipped on restart, so a fix resumes where it stopped.

The first failure in any stage, or a stage whose heartbeat goes silent, kills everything and prints
PIPELINE_ERROR to the log the watcher reads. Exit code is non-zero, which also notifies the session.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from embed_common import Log, Paths, env_info, keep_awake, now, stamp, write_json
from pipeline_config import ARCHS, load_config

HERE = Path(__file__).resolve().parent
PY = sys.executable
STALL_S = 15 * 60


class PipelineError(RuntimeError):
    pass


class Runner:
    def __init__(self, paths: Paths, dry: bool, log: Log):
        self.paths, self.dry, self.log = paths, dry, log
        self.fetch = None
        self.fetch_err = None

    def _spawn(self, script, *args, name):
        err = self.paths.logs / f"{name}_{stamp()}.err"
        cmd = [PY, "-u", str(HERE / script), *args] + (["--dry-run"] if self.dry else [])
        fh = open(err, "w", encoding="utf-8")
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=fh, cwd=HERE), err

    def _fail(self, what, err):
        tail = ""
        if err and Path(err).exists():
            tail = "\n".join(Path(err).read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
        for p in (self.fetch,):
            if p and p.poll() is None:
                p.kill()
        write_json(self.paths.state / "error.json", {"at": now(), "what": what, "stderr_file": str(err), "tail": tail})
        raise PipelineError(f"{what}\n{tail}")

    def start_fetch(self):
        if (self.paths.state / "fetch.done").exists():
            return
        self.fetch, self.fetch_err = self._spawn("s2_fetch_images.py", name="fetch")
        self.log("fetch started in background")

    def check_fetch(self):
        if self.fetch and self.fetch.poll() not in (None, 0):
            self._fail(f"fetch exited with code {self.fetch.returncode}", self.fetch_err)

    def wait_for(self, marker: str):
        path = self.paths.state / marker
        while not path.exists():
            self.check_fetch()
            if self.fetch and self.fetch.poll() == 0 and not path.exists():
                self._fail(f"fetch finished without producing {marker}", self.fetch_err)
            self.log.beat()
            time.sleep(5)

    def stage(self, name, script, *args, beat="train"):
        marker = self.paths.state / f"{name}.done"
        if marker.exists():
            self.log(f"skip finished stage {name}")
            return
        self.log(f"STAGE_START {name}")
        proc, err = self._spawn(script, *args, name=name)
        t0 = time.time()
        hb = self.paths.state / f"heartbeat_{beat}.txt"
        while proc.poll() is None:
            self.check_fetch()
            self.log.beat()
            last = max(hb.stat().st_mtime if hb.exists() else 0, t0)
            if time.time() - last > STALL_S:
                proc.kill()
                self._fail(f"STALL: stage {name} had no progress for {STALL_S // 60} minutes", err)
            time.sleep(5)
        if proc.returncode != 0:
            self._fail(f"stage {name} exited with code {proc.returncode}", err)
        write_json(marker, {"at": now(), "seconds": round(time.time() - t0)})
        self.log(f"STAGE_DONE {name} in {(time.time() - t0) / 60:.1f} min")
        with open(self.paths.run / "STATUS.txt", "a", encoding="utf-8") as f:
            f.write(f"{now()}  finished {name} in {(time.time() - t0) / 60:.1f} min\n")

    def train_task(self, cfg, task):
        for arch in ARCHS:
            for fold in cfg["train"][task]["folds"]:
                self.stage(f"train_{task}_{arch}_fold{fold}", "s3_train.py", "--task", task, "--arch", arch, "--fold", str(fold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--final", action="store_true", help="open the locked test set and build final figures")
    ap.add_argument("--no-fetch", action="store_true", help="no local download; BI-RADS training comes from Kaggle (VPN data limit)")
    a = ap.parse_args()
    paths = Paths(a.dry_run)
    log = Log(paths, "pipeline")
    keep_awake()
    cfg = load_config(a.dry_run)
    write_json(paths.state / f"run_{stamp()}.json", {**env_info(), "argv": sys.argv, "config": cfg})
    r = Runner(paths, a.dry_run, log)
    try:
        log(f"PIPELINE_START dry_run={a.dry_run} final={a.final}")
        unlock = ["--unlock-test"] if (a.dry_run or a.final) else []
        tag = "final" if unlock else "dev"
        if not a.final or a.dry_run:
            r.stage("weights", "s0_weights.py", beat="weights")
            if not a.no_fetch:
                r.start_fetch()
            r.wait_for("task_a_images.done")
            r.train_task(cfg, "a")
        r.stage(f"evaluate_a_{tag}", "s4_evaluate.py", "--task", "a", *unlock, beat="evaluate")
        r.stage(f"figures_a_{tag}", "s5_figures.py", "--task", "a", *unlock, beat="figures")
        if not a.final or a.dry_run:
            if a.no_fetch:
                for arch in ARCHS:
                    for fold in cfg["train"]["b"]["folds"]:
                        r.wait_for(f"train_b_{arch}_fold{fold}.done")
            else:
                r.wait_for("fetch.done")
                r.train_task(cfg, "b")
        r.stage(f"evaluate_b_{tag}", "s4_evaluate.py", "--task", "b", *unlock, beat="evaluate")
        r.stage(f"figures_b_{tag}", "s5_figures.py", "--task", "b", *unlock, beat="figures")
        log("PIPELINE_DONE" if unlock else "PIPELINE_DEV_READY")
        return 0
    except PipelineError as e:
        log(f"PIPELINE_ERROR {e}")
        return 1
    except Exception as e:
        log(f"PIPELINE_ERROR orchestrator {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
