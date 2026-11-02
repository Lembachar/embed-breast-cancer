"""Download a finished Kaggle job's output, one file at a time, resuming after a dropped connection.
Skips last.pt by default: best.pt can warm-start more training, and last.pt stays in the notebook output on Kaggle.
Usage: python kaggle/fetch_output.py <slug> [--all]. Prints percent done. Exits non-zero if a file could not be fetched."""
import argparse
import os
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from dotenv import dotenv_values

os.environ["KAGGLE_API_TOKEN"] = dotenv_values(ROOT.parent / ".env")["KAGGLE_API_TOKEN"].strip()
os.environ["KAGGLE_USERNAME"] = "imadlem"
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

USER = "imadlem"
CHUNK = 1 << 20
TRIES = 40


def list_files(api, slug):
    out, token = {}, ""
    with api.build_kaggle_client() as client:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug = USER, slug
            req.page_size, req.page_token = 200, token
            res = client.kernels.kernels_api_client.list_kernel_session_output(req)
            for f in res.files or []:
                out[f.file_name] = f.url
            token = res.next_page_token
            if not token:
                return out


def fetch(api, slug, name, url, dest):
    """Writes to name.part and renames when the server says the file is complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(TRIES):
        try:
            have = part.stat().st_size if part.exists() else 0
            with requests.get(url, stream=True, timeout=120, headers={"Range": f"bytes={have}-"} if have else {}) as r:
                if r.status_code == 416:
                    os.replace(part, dest)
                    return 0
                if r.status_code in (401, 403):
                    url = list_files(api, slug)[name]  # signed link expired
                    continue
                r.raise_for_status()
                if have and r.status_code != 206:
                    have = 0  # server ignored the range: start this file over
                total = int(r.headers.get("Content-Length", 0)) + have
                got = 0
                with open(part, "ab" if have else "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        got += len(chunk)
            if total and part.stat().st_size != total:
                raise IOError(f"short read {part.stat().st_size}/{total}")
            os.replace(part, dest)
            return got
        except (requests.RequestException, OSError) as e:
            print(f"retry {attempt + 1} {name[-40:]}: {type(e).__name__}", flush=True)
            time.sleep(min(10 * (attempt + 1), 120))
    raise RuntimeError(f"could not fetch {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--all", action="store_true", help="include last.pt")
    a = ap.parse_args()
    out = ROOT / "kaggle" / f"{a.slug}_out"
    api = KaggleApi()
    api.authenticate()
    files = list_files(api, a.slug)
    want = {n: u for n, u in files.items() if a.all or not n.endswith("last.pt")}
    todo = {n: u for n, u in want.items() if not (out / n).exists()}
    print(f"{len(files)} files on Kaggle, {len(want)} wanted, {len(todo)} to fetch", flush=True)
    got, t0 = 0, time.time()
    for i, (name, url) in enumerate(todo.items(), 1):
        got += fetch(api, a.slug, name, url, out / name)
        print(f"{100 * i // len(todo)}% files, {got / 1e6:.0f} MB, {got / 1e6 / max(time.time() - t0, 1):.1f} MB/s, {name[-40:]}", flush=True)
    print("FETCH_DONE", got)


if __name__ == "__main__":
    main()
