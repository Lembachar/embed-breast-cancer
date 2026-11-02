"""Build image-level metadata, labels and patient-level splits for both EMBED tasks.

Task A (malignant vs benign) label per breast within an exam, from pathology:
  path_severity codes, decoded from the path1 diagnoses in the data:
    0 invasive, 1 in situ, 2 high-risk, 3 borderline, 4 benign, 5 non-breast malignancy
  malignant if any 0/1; else excluded if any 2/3/5; else benign.
  Only exams 1-180 days before the biopsy, so biopsy clips cannot reveal the label.

Task B (BI-RADS) label per breast within an exam, from `asses`:
  worst finding on that breast, order N < B < P < A < S < M.
  K (known malignancy) and X (no assessment) excluded. Classes capped by BI_RADS_CAP.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from embed_common import Log, Paths, SEED, env_info, now, sha256, write_json

BIRADS_ORDER = ["N", "B", "P", "A", "S", "M"]
BIRADS_NUM = {"A": 0, "N": 1, "B": 2, "P": 3, "S": 4, "M": 5}
BI_RADS_CAP = 2500  # about 1 image/s through the VPN; raise later and fetch only the additions
EXCLUDED_VTYPES = {"1", "2", "B", "N", "L", "M", "T", "Q"}  # procedure, loc, post-biopsy, specimen, post-surgery, pre-RT, reflector


def build(paths: Paths, log: Log) -> pd.DataFrame:
    clin_p = paths.tables / "EMBED_OpenData_clinical.csv"
    meta_p = paths.tables / "EMBED_OpenData_metadata_reduced.csv"
    c = pd.read_csv(clin_p, low_memory=False)
    m = pd.read_csv(meta_p, low_memory=False)
    flow = {"clinical_rows": len(c), "metadata_images": len(m), "metadata_patients": int(m.empi_anon.nunique()),
            "metadata_exams": int(m.acc_anon.nunique())}
    log(f"loaded clinical {c.shape}, metadata {m.shape}")

    c["study_dt"] = pd.to_datetime(c.study_date_anon, errors="coerce")
    c["proc_dt"] = pd.to_datetime(c.procdate_anon, errors="coerce")
    c["vtype_s"] = c.vtype.astype(str).str.strip()
    c = c[~c.vtype_s.isin(EXCLUDED_VTYPES)].copy()
    flow["clinical_rows_after_visit_type_filter"] = len(c)

    img = m[m.ViewPosition.isin(["CC", "MLO"]) & m.spot_mag.isna() & (m.PatientSex == "F")].copy()
    img = img.rename(columns={"ImageLateralityFinal": "side"})
    flow["images_standard_view_female"] = len(img)
    # One image per exam + breast + view: prefer true 2D over synthetic C-View.
    img["type_rank"] = (img.FinalImageType != "2D").astype(int)
    img = img.sort_values(["acc_anon", "side", "ViewPosition", "type_rank", "anon_dicom_path"])
    img = img.drop_duplicates(["acc_anon", "side", "ViewPosition"], keep="first")
    flow["images_one_per_exam_breast_view"] = len(img)

    # ---- Task A
    p = c[c.path_severity.notna()].copy()
    p["days_to_biopsy"] = (p.proc_dt - p.study_dt).dt.days
    flow["pathology_rows"] = len(p)
    p = p[p.days_to_biopsy.between(1, 180)]
    flow["pathology_rows_1_to_180_days_before_biopsy"] = len(p)
    p["bs"] = p.bside.where(p.bside.isin(["L", "R", "B"]), p.side)
    both = p[p.bs == "B"]
    p = pd.concat([p[p.bs.isin(["L", "R"])], both.assign(bs="L"), both.assign(bs="R")])

    def label_a(s):
        vals = set(int(v) for v in s)
        if vals & {0, 1}:
            return 1
        if vals & {2, 3, 5}:
            return -1
        return 0

    a = p.groupby(["acc_anon", "bs"]).path_severity.apply(label_a).reset_index()
    a.columns = ["acc_anon", "side", "label_malignant"]
    flow["task_a_breast_exams_excluded_highrisk_borderline_nonbreast"] = int((a.label_malignant == -1).sum())
    a = a[a.label_malignant >= 0]
    flow["task_a_breast_exams_labelled"] = len(a)

    # ---- Task B
    b = c[c.asses.isin(BIRADS_ORDER + ["K", "X"])].copy()
    un = b[b.side.isna() | (b.side == "B")]
    b = pd.concat([b[b.side.isin(["L", "R"])], un.assign(side="L"), un.assign(side="R")])

    def label_b(s):
        vals = set(s)
        if vals & {"K", "X"}:
            return None
        return max(vals, key=BIRADS_ORDER.index)

    bl = b.groupby(["acc_anon", "side"]).asses.apply(label_b).dropna().reset_index()
    bl.columns = ["acc_anon", "side", "birads_code"]
    flow["task_b_breast_exams_labelled_before_image_join"] = len(bl)

    d = img.merge(a, on=["acc_anon", "side"], how="left").merge(bl, on=["acc_anon", "side"], how="left")
    d = d[d.label_malignant.notna() | d.birads_code.notna()].copy()
    d["breast_exam"] = d.acc_anon.astype(str) + "_" + d.side

    # cap BI-RADS by breast-exam; Task A images are never dropped
    rng = np.random.RandomState(SEED)
    be = d[d.birads_code.notna()].drop_duplicates("breast_exam")[["breast_exam", "birads_code"]]
    keep = []
    flow["task_b_breast_exams_with_images_by_class_before_cap"] = be.birads_code.value_counts().to_dict()
    for code, g in be.groupby("birads_code"):
        ids = g.breast_exam.values
        keep.extend(rng.choice(ids, BI_RADS_CAP, replace=False) if len(ids) > BI_RADS_CAP else ids)
    keep = set(keep)
    d["in_task_b"] = d.birads_code.notna() & d.breast_exam.isin(keep)
    d["in_task_a"] = d.label_malignant.notna()
    d = d[d.in_task_a | d.in_task_b].copy()
    d["label_birads"] = d.birads_code.map(BIRADS_NUM)

    # covariates for subgroup analysis
    exam = c.groupby("acc_anon").agg(age=("age_at_study", "first"), density=("tissueden", "first"),
                                     ethnicity=("ETHNICITY_DESC", "first"), desc=("desc", "first")).reset_index()
    d = d.merge(exam, on="acc_anon", how="left")
    d["exam_kind"] = np.where(d.desc.astype(str).str.contains("screen", case=False), "screening", "diagnostic")
    d["density"] = d.density.where(d.density.isin([1, 2, 3, 4]))

    # ---- patient-level split: same patient never crosses test/dev or folds
    pt = d.groupby("empi_anon").agg(mal=("label_malignant", "max"), bi=("label_birads", "max")).reset_index()
    pt["key"] = pt.mal.fillna(-1).astype(int).astype(str) + "_" + pt.bi.fillna(-1).astype(int).astype(str)
    counts = pt.key.value_counts()
    pt.loc[pt.key.map(counts) < 12, "key"] = "rare"
    pt["split"] = "dev"
    pt["fold"] = -1
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    dev_idx, test_idx = next(iter(skf.split(pt, pt.key)))
    pt.iloc[test_idx, pt.columns.get_loc("split")] = "test"
    dev = pt.iloc[dev_idx]
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED + 1).split(dev, dev.key)):
        pt.loc[dev.index[va], "fold"] = k
    d = d.merge(pt[["empi_anon", "split", "fold"]], on="empi_anon")

    assert d.groupby("empi_anon").split.nunique().max() == 1
    assert d[d.split == "dev"].groupby("empi_anon").fold.nunique().max() == 1
    assert d.groupby(["breast_exam", "ViewPosition"]).size().max() == 1

    d["image_id"] = d.anon_dicom_path.str.rsplit("/", n=1).str[-1].str.replace(".dcm", "", regex=False)
    cols = ["image_id", "anon_dicom_path", "empi_anon", "acc_anon", "breast_exam", "side", "ViewPosition", "FinalImageType",
            "Manufacturer", "ManufacturerModelName", "WindowCenter", "WindowWidth", "VOILUTFunction", "num_ROI", "ROI_coords",
            "cohort_num", "study_date_anon", "age", "density", "ethnicity", "exam_kind", "desc",
            "in_task_a", "label_malignant", "in_task_b", "birads_code", "label_birads", "split", "fold"]
    d = d[cols].sort_values("image_id").reset_index(drop=True)

    for name, flag, lab in [("task_a", "in_task_a", "label_malignant"), ("task_b", "in_task_b", "birads_code")]:
        s = d[d[flag]]
        be = s.drop_duplicates("breast_exam")
        flow[f"{name}_images"] = len(s)
        flow[f"{name}_breast_exams"] = len(be)
        flow[f"{name}_patients"] = int(s.empi_anon.nunique())
        flow[f"{name}_breast_exams_by_class"] = {str(k): int(v) for k, v in be[lab].value_counts().items()}
        flow[f"{name}_test_breast_exams_by_class"] = {str(k): int(v) for k, v in be[be.split == "test"][lab].value_counts().items()}
    flow["total_images_to_fetch"] = len(d)
    flow["total_patients"] = int(d.empi_anon.nunique())
    flow["test_patients"] = int(d[d.split == "test"].empi_anon.nunique())

    write_json(paths.cohort / "cohort_flow.json", flow)
    write_json(paths.cohort / "metadata_manifest.json", {
        **env_info(), "built_at": now(),
        "inputs": {clin_p.name: sha256(clin_p), meta_p.name: sha256(meta_p)},
        "rules": __doc__, "bi_rads_cap": BI_RADS_CAP, "excluded_vtypes": sorted(EXCLUDED_VTYPES)})
    d.to_parquet(paths.cohort / "metadata_images.parquet", index=False)
    d.to_csv(paths.cohort / "metadata_images.csv", index=False)
    log("cohort flow:\n" + json.dumps(flow, indent=2))
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    pths = Paths(False)
    build(pths, Log(pths, "metadata"))
