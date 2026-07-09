###############################################################################
# load_data.py
# Purpose: Load CDISC SDTM domain data (cdiscpilot01) and prepare core tables
#          for downstream feature engineering and modeling.
#
# Python version used to build/validate this pipeline: 3.12.3
# Required packages (see requirements.txt for exact pinned versions):
#   pandas==3.0.2, numpy==2.4.4, pyreadstat==1.3.5
#
# Reproducibility:
#   - RANDOM_SEED is imported from config.py and fixed for the whole pipeline.
#   - This script performs no random sampling itself, but the seed is set on
#     import (via config.set_global_seed) so process-level state is
#     deterministic before feature_engineering.py / modeling.py run.
#   - File discovery and column ordering below are done with explicit sorted()
#     calls so glob/OS directory-listing order never affects results.
#
# Input:
#   A folder (default "sdtm/") containing the following domains, each present
#   as EITHER a .xpt file OR a .csv file (case-insensitive extension/name):
#     dm, ds, ae, ex, lb, vs, mh, qs
#   e.g. sdtm/DM.xpt, sdtm/ds.csv, sdtm/AE.xpt, ...
###############################################################################
import glob
import os

import pandas as pd

from config import RANDOM_SEED, set_global_seed  # noqa: F401  (seed set on import)

DOMAINS = ["dm", "ds", "ae", "ex", "lb", "vs", "mh", "qs"]

DISC_TERMS = [
    "ADVERSE EVENT",
    "DEATH",
    "LACK OF EFFICACY",
    "LOST TO FOLLOW-UP",
    "PHYSICIAN DECISION",
    "PROTOCOL VIOLATION",
    "STUDY TERMINATED BY SPONSOR",
    "WITHDRAWAL BY SUBJECT",
]


###############################################################################
# 0. Helpers
###############################################################################
def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, snake_case column names (equivalent to janitor::clean_names())."""
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace(".", "").replace("-", "_")
        for c in df.columns
    ]
    return df


def find_domain_file(folder: str, domain: str) -> str:
    """Locate a domain's data file in `folder`, accepting .xpt or .csv,
    case-insensitively. Deterministic: if both extensions somehow exist,
    .xpt is preferred; candidate matches are sorted before selection so the
    result never depends on filesystem listing order.
    """
    candidates = sorted(glob.glob(os.path.join(folder, "*")))
    matches = {
        ext: [
            f
            for f in candidates
            if os.path.splitext(f)[0].lower().endswith(domain.lower())
            and os.path.splitext(f)[1].lower() == ext
        ]
        for ext in [".xpt", ".csv"]
    }
    if matches[".xpt"]:
        return sorted(matches[".xpt"])[0]
    if matches[".csv"]:
        return sorted(matches[".csv"])[0]
    raise FileNotFoundError(
        f"Could not find a .xpt or .csv file for domain '{domain}' in '{folder}'."
    )


def read_domain(path: str) -> pd.DataFrame:
    """Read a single domain file, dispatching on extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xpt":
        df = pd.read_sas(path, format="xport", encoding="utf-8")
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension for '{path}': {ext}")
    return clean_names(df)


def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every column ending in 'dtc' to a proper datetime64, parsing
    the first 10 characters as ISO8601 YYYY-MM-DD (equivalent to
    lubridate::ymd(substr(x, 1, 10)))."""
    df = df.copy()
    for col in df.columns:
        if col.endswith("dtc"):
            df[col] = pd.to_datetime(
                df[col].astype(str).str.slice(0, 10), errors="coerce"
            )
    return df


def impute_ae_start_dates(ae: pd.DataFrame) -> pd.DataFrame:
    """Conservative ISO8601 imputation for AESTDTC when AESTDY is missing:
    YYYY -> YYYY-01-01, YYYY-MM -> YYYY-MM-01. Leaves complete dates and
    dates accompanying a non-missing AESTDY untouched."""
    ae = ae.copy()

    def _impute(row):
        if pd.isna(row.get("aestdy")):
            v = row.get("aestdtc")
            if isinstance(v, str):
                if len(v) == 4:
                    return v + "-01-01"
                if len(v) == 7:
                    return v + "-01"
            return v
        return row.get("aestdtc")

    if "aestdtc" in ae.columns:
        ae["aestdtc"] = ae.apply(_impute, axis=1)
    return ae


###############################################################################
# 1. Main loader
###############################################################################
def load_all(folder: str = "sdtm") -> dict:
    """Load all SDTM domains from `folder`, clean/convert them, derive the
    discontinuation cohort, and return a dict:
        {dm, ds, ae, ex, lb, vs, mh, qs, cohort}
    """
    set_global_seed(RANDOM_SEED)

    tabs = {}
    for domain in DOMAINS:
        path = find_domain_file(folder, domain)
        tabs[domain] = read_domain(path)

    ae = impute_ae_start_dates(tabs["ae"])
    tabs["ae"] = ae

    # Convert all *dtc columns to datetime for every domain.
    for domain in DOMAINS:
        tabs[domain] = convert_dates(tabs[domain])

    dm = tabs["dm"]
    ds = tabs["ds"]

    # ---- Remove screen-failure subjects (based on ARM) ----
    if "arm" in dm.columns:
        scrn_fail = sorted(dm.loc[dm["arm"] == "Screen Failure", "usubjid"].unique().tolist())
    else:
        scrn_fail = []
    dm = dm[~dm["usubjid"].isin(scrn_fail)].copy()
    tabs["dm"] = dm

    # ---- Define discontinuation from DS ----
    discont = (
        ds[ds["dsdecod"].isin(DISC_TERMS) | ds["dsterm"].isin(DISC_TERMS)]
        [["usubjid", "dsdecod", "dsterm", "dsdtc"]]
        .copy()
    )
    discont["discont"] = 1

    # ---- Identify randomized subjects (ARMCD populated) ----
    rand_subj = dm.loc[
        dm["armcd"].notna(), ["usubjid", "armcd", "arm", "rfstdtc", "rfendtc"]
    ].copy()

    # ---- Merge discontinuation status into cohort ----
    cohort = rand_subj.merge(discont, on="usubjid", how="left")
    cohort["discont"] = cohort["discont"].fillna(0).astype(int)

    # Deterministic row order for reproducible downstream joins/prints.
    cohort = cohort.sort_values("usubjid").reset_index(drop=True)

    sdtm = dict(
        dm=tabs["dm"],
        ds=tabs["ds"],
        ae=tabs["ae"],
        ex=tabs["ex"],
        lb=tabs["lb"],
        vs=tabs["vs"],
        mh=tabs["mh"],
        qs=tabs["qs"],
        cohort=cohort,
    )
    return sdtm


if __name__ == "__main__":
    sdtm = load_all("sdtm")
    for name, tbl in sdtm.items():
        print(f"{name}: {tbl.shape[0]} rows x {tbl.shape[1]} cols")
    print("\nDiscontinuation counts:")
    print(sdtm["cohort"]["discont"].value_counts().sort_index())
