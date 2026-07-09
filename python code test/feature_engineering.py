###############################################################################
# feature_engineering.py
# Purpose: Build baseline and time-varying features (plus anomaly features)
#          for predicting early discontinuation.
#
# Python version used to build/validate this pipeline: 3.12.3
# Required packages (see requirements.txt for exact pinned versions):
#   pandas==3.0.2, numpy==2.4.4, scikit-learn==1.8.0, scipy==1.17.1
#
# Reproducibility:
#   - RANDOM_SEED is imported from config.py and reused for IsolationForest.
#   - IsolationForest is run with n_jobs=1 (avoids thread-scheduling
#     nondeterminism) and a fixed random_state.
#   - All pivot/groupby outputs have their columns sorted explicitly before
#     being merged, so column order never depends on hash/set iteration
#     order and results are identical across machines/runs.
#
# Cutoff-window definition (time-varying features):
#   cutoff_weeks = MIN over discontinued subjects only (cohort.discont == 1),
#                  EXCLUDING subjects whose DS dsterm == 'PROTOCOL ENTRY
#                  CRITERIA NOT MET', of weeks-to-discontinuation, where
#                  weeks-to-discontinuation = (rfendtc - rfstdtc) / 7 days.
#   cutoff_weeks is floored at 0.
#   This single cutoff_weeks value is then applied uniformly to every
#   randomized subject: cutoff_date = rfstdtc + cutoff_weeks (in weeks).
#   All time-varying feature windows (AE, EX, LB, VS) are restricted to
#   records on or before each subject's own cutoff_date.
###############################################################################
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED, set_global_seed


###############################################################################
# 0. Helpers
###############################################################################
def _sort_cols(df: pd.DataFrame, keep_first=("usubjid",)) -> pd.DataFrame:
    """Return df with columns sorted alphabetically, except any columns
    listed in `keep_first`, which are placed first in the order given.
    Ensures column order is deterministic and independent of groupby/pivot
    internals."""
    keep_first = [c for c in keep_first if c in df.columns]
    rest = sorted(c for c in df.columns if c not in keep_first)
    return df[keep_first + rest]


def _pivot_long_to_wide(long_df, index, columns, values, prefix=None) -> pd.DataFrame:
    wide = long_df.pivot_table(index=index, columns=columns, values=values, aggfunc="first")
    if isinstance(wide.columns, pd.MultiIndex):
        wide.columns = ["_".join([str(x) for x in tup]) for tup in wide.columns]
    if prefix:
        wide.columns = [f"{prefix}{c}" for c in wide.columns]
    wide = wide.reset_index()
    return _sort_cols(wide)


###############################################################################
# 1. Cutoff-window derivation
###############################################################################
def compute_cutoff_table(sdtm: dict) -> tuple[pd.DataFrame, float]:
    cohort = sdtm["cohort"]
    ds = sdtm["ds"]

    excl_subjects = set(
        ds.loc[ds["dsterm"] == "PROTOCOL ENTRY CRITERIA NOT MET", "usubjid"]
    )

    disc_only = cohort[
        (cohort["discont"] == 1) & (~cohort["usubjid"].isin(excl_subjects))
    ].copy()
    disc_only["weeks_to_disc"] = (
        disc_only["rfendtc"] - disc_only["rfstdtc"]
    ).dt.total_seconds() / (3600 * 24 * 7)

    cutoff_weeks = disc_only["weeks_to_disc"].min(skipna=True)
    cutoff_weeks = max(cutoff_weeks, 0)

    cutoff_tbl = cohort[["usubjid", "rfstdtc"]].copy()
    cutoff_tbl["cutoff"] = cutoff_tbl["rfstdtc"] + pd.to_timedelta(
        cutoff_weeks * 7, unit="D"
    )
    cutoff_tbl = cutoff_tbl[["usubjid", "cutoff"]]
    return cutoff_tbl, cutoff_weeks


###############################################################################
# 2. Baseline features
###############################################################################
def build_baseline_features(sdtm: dict) -> dict:
    dm, lb, vs, mh, qs = sdtm["dm"], sdtm["lb"], sdtm["vs"], sdtm["mh"], sdtm["qs"]

    # ---- Demographics ----
    dm_cols = [c for c in ["usubjid", "age", "sex", "race", "country", "siteid"] if c in dm.columns]
    dm_base = _sort_cols(dm[dm_cols].copy())

    # ---- Baseline labs: LBBLFL == 'Y' if available, else earliest LBDTC with LBDY <= 1 ----
    grp_has_bl = lb.groupby(["usubjid", "lbtest"])["lbblfl"].transform(lambda s: (s == "Y").any())
    lb_min_dtc = lb.groupby(["usubjid", "lbtest"])["lbdtc"].transform("min")
    keep_mask = np.where(
        grp_has_bl,
        lb["lbblfl"] == "Y",
        (lb["lbdtc"] == lb_min_dtc) & (lb["lbdy"] <= 1),
    )
    lb_base_long = lb[keep_mask][["usubjid", "lbtest", "lbstresn"]]
    lb_base = _pivot_long_to_wide(lb_base_long, "usubjid", "lbtest", "lbstresn")
    if "color" in [c.lower() for c in lb_base.columns]:
        drop_col = [c for c in lb_base.columns if c.lower() == "color"]
        lb_base = lb_base.drop(columns=drop_col)

    # ---- Baseline vitals: VSBLFL == 'Y' ----
    vs_b = vs[vs["vsblfl"] == "Y"][["usubjid", "vstest", "vstpt", "vsstresn"]].copy()
    vs_b["vs_col"] = vs_b["vstest"].astype(str) + "_" + vs_b["vstpt"].fillna("").astype(str)
    vs_base = _pivot_long_to_wide(vs_b, "usubjid", "vs_col", "vsstresn")

    # ---- Medical history ----
    mh_base = (
        mh.groupby("usubjid")
        .agg(
            mh_count=("usubjid", "size"),
            mh_serious=("mhsev", lambda s: s.astype(str).str.lower().str.contains("severe").sum()),
        )
        .reset_index()
    )
    mh_base = _sort_cols(mh_base)

    # ---- Baseline QS: QSBLFL == 'Y' if available, else earliest QSDTC with QSDY <= 1 ----
    grp_has_bl_qs = qs.groupby(["usubjid", "qstestcd"])["qsblfl"].transform(lambda s: (s == "Y").any())
    qs_min_dtc = qs.groupby(["usubjid", "qstestcd"])["qsdtc"].transform("min")
    keep_mask_qs = np.where(
        grp_has_bl_qs,
        qs["qsblfl"] == "Y",
        (qs["qsdtc"] == qs_min_dtc) & (qs["qsdy"] <= 1),
    )
    qs_base_long = qs[keep_mask_qs][["usubjid", "qstestcd", "qsstresn"]]
    qs_base = _pivot_long_to_wide(qs_base_long, "usubjid", "qstestcd", "qsstresn", prefix="qs_")

    return dict(dm_base=dm_base, lb_base=lb_base, vs_base=vs_base, mh_base=mh_base, qs_base=qs_base)


###############################################################################
# 3. Time-varying features (up to each subject's cutoff date)
###############################################################################
def build_time_varying_features(sdtm: dict, cutoff_tbl: pd.DataFrame) -> dict:
    ae, ex, lb, vs, cohort = sdtm["ae"], sdtm["ex"], sdtm["lb"], sdtm["vs"], sdtm["cohort"]

    # ---- Adverse events: counts / seriousness / max severity ----
    ae_c = ae.merge(cutoff_tbl, on="usubjid", how="left")
    ae_c = ae_c[ae_c["aedtc"] <= ae_c["cutoff"]]
    sev_map = {"MILD": 1, "MODERATE": 2, "SEVERE": 3}
    ae_c = ae_c.copy()
    ae_c["sev_num"] = ae_c["aesev"].map(sev_map)
    ae_feat = (
        ae_c.groupby("usubjid")
        .agg(
            ae_count=("usubjid", "size"),
            ae_serious=("aeser", lambda s: (s == "Y").sum()),
            ae_sev_max=("sev_num", "max"),
        )
        .reset_index()
    )
    ae_feat = cohort[["usubjid"]].merge(ae_feat, on="usubjid", how="left")
    ae_feat = ae_feat.fillna(0)
    ae_feat = _sort_cols(ae_feat)

    # ---- Exposure totals ----
    ex_c = ex.merge(cutoff_tbl, on="usubjid", how="left")
    ex_c = ex_c[ex_c["exstdtc"] <= ex_c["cutoff"]]
    ex_feat = (
        ex_c.groupby("usubjid")
        .agg(dose_total=("exdose", "sum"), dose_days=("usubjid", "size"))
        .reset_index()
    )
    ex_feat = _sort_cols(ex_feat)

    # ---- Lab trends ----
    lb_c = lb.merge(cutoff_tbl, on="usubjid", how="left")
    lb_c = lb_c[lb_c["lbdtc"] <= lb_c["cutoff"]]
    lb_c = lb_c[lb_c["lbtest"].str.lower() != "color"]
    lb_trend = (
        lb_c.groupby(["usubjid", "lbtest"])["lbstresn"]
        .agg(lb_mean="mean", lb_max="max", lb_min="min")
        .reset_index()
    )
    lb_feat = lb_trend.pivot_table(index="usubjid", columns="lbtest", values=["lb_mean", "lb_max", "lb_min"])
    lb_feat.columns = [f"{a}_{b}" for a, b in lb_feat.columns]
    lb_feat = _sort_cols(lb_feat.reset_index())

    # ---- Vital-sign trends ----
    vs_c = vs.merge(cutoff_tbl, on="usubjid", how="left")
    vs_c = vs_c[vs_c["vsdtc"] <= vs_c["cutoff"]]
    vs_trend = (
        vs_c.groupby(["usubjid", "vstest"])["vsstresn"]
        .agg(vs_mean="mean", vs_max="max", vs_min="min")
        .reset_index()
    )
    vs_feat = vs_trend.pivot_table(index="usubjid", columns="vstest", values=["vs_mean", "vs_max", "vs_min"])
    vs_feat.columns = [f"{a}_{b}" for a, b in vs_feat.columns]
    vs_feat = _sort_cols(vs_feat.reset_index())

    return dict(ae_features=ae_feat, ex_features=ex_feat, lb_features=lb_feat, vs_features=vs_feat)


###############################################################################
# 4. Anomaly features (Isolation Forest)
###############################################################################
def add_anomaly_features(features: pd.DataFrame, ae_feat: pd.DataFrame) -> pd.DataFrame:
    set_global_seed(RANDOM_SEED)

    features_num = features.select_dtypes(include=[np.number]).drop(columns=["discont"], errors="ignore")
    features_num = features_num[sorted(features_num.columns)]  # deterministic column order

    var = features_num.var(skipna=True)
    keep_cols = sorted(var[(~var.isna()) & (var > 0)].index)
    features_num2 = features_num[keep_cols]

    features_num3 = features_num2.apply(lambda x: x.fillna(x.median()))

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_num3)

    iso_model = IsolationForest(
        n_estimators=500,
        max_samples=min(200, len(features_scaled)),
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    iso_model.fit(features_scaled)
    raw = -iso_model.score_samples(features_scaled)
    iso_score = (raw - raw.min()) / (raw.max() - raw.min())
    iso_flag = (iso_score > 0.5).astype(int)

    ae_num_cols = sorted(c for c in ae_feat.columns if c != "usubjid")
    ae_num = ae_feat[ae_num_cols]
    isoae_model = IsolationForest(
        n_estimators=10,
        max_samples=min(3, len(ae_num)),
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    isoae_model.fit(ae_num)
    raw_ae = -isoae_model.score_samples(ae_num)
    isoae_score = (raw_ae - raw_ae.min()) / (raw_ae.max() - raw_ae.min())
    isoae_flag = (isoae_score > 0.5).astype(int)

    out = features.copy()
    out["iso_score"] = iso_score
    out["iso_flag"] = iso_flag
    out["isoae_score"] = isoae_score
    out["isoae_flag"] = isoae_flag
    return out


###############################################################################
# 5. Key-insight summary
###############################################################################
def print_key_insight(features: pd.DataFrame) -> None:
    n_flagged = int(features["iso_flag"].sum())
    n_total = len(features)

    numeric_cols = [
        c
        for c in features.select_dtypes(include=[np.number]).columns
        if c not in ("iso_score", "iso_flag", "isoae_score", "isoae_flag", "discont")
    ]
    flagged = features[features["iso_flag"] == 1][numeric_cols]
    not_flagged = features[features["iso_flag"] == 0][numeric_cols]

    # Standardized mean difference per feature (flagged vs. not flagged),
    # a simple, deterministic way to rank which features separate the groups.
    diffs = []
    for col in numeric_cols:
        f_mean, nf_mean = flagged[col].mean(), not_flagged[col].mean()
        pooled_std = features[col].std(skipna=True)
        if pooled_std and not np.isnan(pooled_std) and pooled_std > 0:
            diffs.append((col, abs(f_mean - nf_mean) / pooled_std))
    diffs.sort(key=lambda x: (-x[1], x[0]))  # sort by |effect size| desc, ties broken alphabetically
    top_features = diffs[:10]

    print("\n=== KEY INSIGHT: feature_engineering.py ===")
    print(f"{n_flagged} of {n_total} subjects ({n_flagged / n_total:.1%}) flagged as anomalous (iso_flag == 1).")
    print("Top features distinguishing flagged vs. non-flagged subjects (by standardized mean difference):")
    for col, effect in top_features:
        print(f"  {col:40s}  |effect size| = {effect:.3f}")
    print("=============================================\n")


###############################################################################
# 6. Orchestration
###############################################################################
def build_features(sdtm: dict) -> dict:
    set_global_seed(RANDOM_SEED)

    cutoff_tbl, cutoff_weeks = compute_cutoff_table(sdtm)
    print(f"Cutoff window: {cutoff_weeks:.2f} weeks (applied uniformly to all randomized subjects).")

    baseline = build_baseline_features(sdtm)
    time_varying = build_time_varying_features(sdtm, cutoff_tbl)

    features = sdtm["cohort"].copy()
    for tbl in [
        baseline["dm_base"],
        baseline["lb_base"],
        baseline["vs_base"],
        baseline["mh_base"],
        baseline["qs_base"],
        time_varying["ae_features"],
        time_varying["ex_features"],
        time_varying["lb_features"],
        time_varying["vs_features"],
    ]:
        features = features.merge(tbl, on="usubjid", how="left")

    features = add_anomaly_features(features, time_varying["ae_features"])
    print_key_insight(features)

    feature_list = dict(
        features=features,
        baseline_dm=baseline["dm_base"],
        baseline_lb=baseline["lb_base"],
        baseline_vs=baseline["vs_base"],
        baseline_mh=baseline["mh_base"],
        baseline_qs=baseline["qs_base"],
        ae_features=time_varying["ae_features"],
        ex_features=time_varying["ex_features"],
        lb_features=time_varying["lb_features"],
        vs_features=time_varying["vs_features"],
        cutoff_weeks=cutoff_weeks,
    )
    return feature_list


if __name__ == "__main__":
    from load_data import load_all

    sdtm = load_all("sdtm")
    feature_list = build_features(sdtm)
    print(feature_list["features"].shape)
