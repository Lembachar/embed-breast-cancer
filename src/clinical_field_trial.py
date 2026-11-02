"""What the EMBED clinical table adds to malignant vs benign, on dev only, and how many confirmed cancers go unused.

Allowed inputs are what the radiologist recorded before the biopsy. Pathology, grade, stage, receptors and surgery are the
answer itself and are never inputs. BI-RADS code K means a cancer was already known, so it is counted and left out.
Writes metrics/a_clinical_field_trial.json."""
from __future__ import annotations

import numpy as np
import pandas as pd

from embed_common import Log, Paths, write_json
from pipeline_config import load_config, load_metadata
from s1_build_metadata import EXCLUDED_VTYPES
from s4_evaluate import clinical_features, load_breast, nested_oof, primary, reading_features, to_features

MARKINGS = ["massshape", "massmargin", "massdens", "calcfind", "calcdistri", "otherfind"]
ASSESSED = ["N", "B", "P", "A", "S", "M"]


def main():
    paths, cfg = Paths(False), load_config(False)
    log = Log(paths, "clinical_trial")
    c = pd.read_csv(paths.tables / "EMBED_OpenData_clinical.csv", low_memory=False)
    meta = load_metadata(paths, False)
    res = {}

    # ---- cancers in the table against cancers used
    v = c[~c.vtype.astype(str).str.strip().isin(EXCLUDED_VTYPES)]
    p = v[v.path_severity.isin([0, 1])].copy()
    days = (pd.to_datetime(p.procdate_anon, errors="coerce") - pd.to_datetime(p.study_date_anon, errors="coerce")).dt.days
    res["cancer_rows_any_exam"] = int(len(p))
    res["cancer_patients_any_exam"] = int(p.empi_anon.nunique())
    for lo, hi in ((1, 180), (1, 365), (0, 365), (-30, 365)):
        q = p[days.between(lo, hi)]
        res[f"cancer_patients_days_{lo}_to_{hi}"] = int(q.empi_anon.nunique())
        res[f"cancer_exams_days_{lo}_to_{hi}"] = int(q.acc_anon.nunique())
    a = meta[meta.in_task_a].drop_duplicates("breast_exam")
    res["task_a_cancer_breast_exams_used"] = int((a.label_malignant == 1).sum())
    res["task_a_cancer_patients_used"] = int(a[a.label_malignant == 1].empi_anon.nunique())

    # ---- radiologist's recorded reading, dev only
    tbl, _, X = load_breast(paths, "a", cfg, "dev")
    y, g = tbl.y.values, tbl.empi_anon.values
    Fimg = to_features(X, 2)
    be = meta[meta.in_task_a]
    Cbasic = clinical_features(tbl, be)
    v = v[v.acc_anon.astype(str).isin(set(tbl.breast_exam.str.rsplit("_", n=1).str[0]))].copy()
    asses = v.groupby(["acc_anon"]).asses.apply(lambda s: set(s.dropna()))
    known = tbl.breast_exam.str.rsplit("_", n=1).str[0].map(lambda e: "K" in asses.get(e, set()) if e in asses.index else False)
    res["dev_exams_with_K_known_cancer"] = int(known.sum())
    res["dev_K_share_malignant"] = float(y[known.values].mean()) if known.any() else None
    v.loc[~v.asses.isin(ASSESSED), "asses"] = np.nan
    R_mark = reading_features(v, tbl, MARKINGS).values
    R_all = reading_features(v, tbl, MARKINGS + ["asses"]).values
    res["n_dev"], res["reading_markings_features"], res["reading_all_features"] = int(len(y)), int(R_mark.shape[1]), int(R_all.shape[1])
    trials = {"images_only": Fimg, "exam_type_age": Cbasic, "images_exam_type_age": np.hstack([Fimg, Cbasic]),
              "markings_only": R_mark, "markings_age": np.hstack([R_mark, Cbasic]), "images_markings_age": np.hstack([Fimg, R_mark, Cbasic]),
              "reading_with_birads_age": np.hstack([R_all, Cbasic]), "images_reading_with_birads_age": np.hstack([Fimg, R_all, Cbasic])}
    res["auc_dev"] = {k: round(primary("a", y, nested_oof("logistic", 2, f, y, g)), 4) for k, f in trials.items()}
    write_json(paths.metrics / "a_clinical_field_trial.json", res)
    log(f"clinical field trial on dev: {res}")


if __name__ == "__main__":
    main()
