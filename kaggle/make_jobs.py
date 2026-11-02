"""Build the private Kaggle dataset (code, trimmed metadata, expiring download links) and the two job notebooks."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import boto3
import pandas as pd
from embed_common import BUCKET, REGION, Paths, load_aws_env
from pipeline_config import load_metadata

USER = "imadlem"
DATASET = "embed-run-files"
LINK_HOURS = 24
ARCH_ORDER = ("efficientnet_v2_s", "convnext_tiny", "resnet50")
# slug: (image flag, saved long side, runs). Saved long side equals the training height, so nothing is stored larger than it is used.
JOBS = {
    "embed-birads": ("in_task_b", 1024, [("b", a, 0) for a in ARCH_ORDER]),
    "embed-malignant": ("in_task_a", 1536, [("a", a, k) for a in ARCH_ORDER for k in range(5)]),
}
# Twice the height and width of the first pass. T4 has 16 GB; batch size drops and accumulation rises to keep the same effective batch of 24.
HIRES = {"a": {"image_size": [1536, 1024], "batch_size": 4, "accum": 6},
         "b": {"image_size": [1024, 768], "batch_size": 8, "accum": 3}}
KEEP = ["image_id", "empi_anon", "breast_exam", "split", "fold", "in_task_a", "in_task_b", "label_malignant", "label_birads", "birads_code"]

K = ROOT / "kaggle"
ds = K / "dataset"
shutil.rmtree(ds, ignore_errors=True)
ds.mkdir(parents=True)
for f in ("embed_common.py", "pipeline_config.py", "s0_weights.py", "s2_fetch_images.py", "s3_train.py", "s4_evaluate.py", "s5_figures.py"):
    shutil.copy(ROOT / "scripts" / f, ds / f)
cfg = json.loads((ROOT / "configs" / "full_run.json").read_text(encoding="utf-8"))
for task, change in HIRES.items():
    cfg["train"][task].update(change)
(ds / "full_run.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
shutil.copy(K / "runner.py", ds / "runner.py")

load_aws_env()
d = load_metadata(Paths(False), False)
d[KEEP].to_parquet(ds / "metadata_images.parquet", index=False)
s3 = boto3.client("s3", region_name=REGION)
links = pd.DataFrame({"image_id": d.image_id, "url": [
    s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": k}, ExpiresIn=LINK_HOURS * 3600) for k in d.anon_dicom_path]})
links.to_csv(ds / "links.csv", index=False)
(ds / "dataset-metadata.json").write_text(json.dumps({"title": DATASET, "id": f"{USER}/{DATASET}", "licenses": [{"name": "other"}]}), encoding="utf-8")

for slug, (flag, long_side, runs) in JOBS.items():
    jd = K / slug
    jd.mkdir(exist_ok=True)
    arg = ",".join(f"{t}:{a}:{k}" for t, a, k in runs)
    code = f"""import glob,os,subprocess,sys
subprocess.run([sys.executable,"-m","pip","install","-q","pylibjpeg","pylibjpeg-libjpeg","pylibjpeg-openjpeg","python-gdcm","pydicom","dicomsdl","statsmodels","lightgbm","umap-learn"],capture_output=True)
r=glob.glob("/kaggle/input/**/runner.py",recursive=True)[0]
code=subprocess.call([sys.executable,"-u",r,"--src",os.path.dirname(r),"--flag","{flag}","--long-side","{long_side}","--runs","{arg}"])
print("RUNNER_EXIT",code)
assert code==0
"""
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}
    (jd / "job.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    (jd / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}", "title": slug, "code_file": "job.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [f"{USER}/{DATASET}"], "competition_sources": [], "kernel_sources": [], "model_sources": []}), encoding="utf-8")
print("dataset files", sorted(p.name for p in ds.iterdir()), "images", len(links))
