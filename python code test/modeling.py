###############################################################################
# modeling.py
# Purpose: Train and evaluate models predicting early discontinuation from
#          the engineered feature table: (1) elastic-net-regularized
#          logistic regression, and (2) an RBF-kernel SVM.
#
# Python version used to build/validate this pipeline: 3.12.3
# Required packages (see requirements.txt for exact pinned versions):
#   pandas==3.0.2, numpy==2.4.4, scikit-learn==1.8.0, scipy==1.17.1
#
# Reproducibility:
#   - RANDOM_SEED (config.py) is used for: the train/holdout split,
#     StratifiedKFold fold assignment for both models, LogisticRegressionCV's
#     'saga' solver, and SVC/GridSearchCV.
#   - All CV objects use shuffle=True with the fixed random_state so fold
#     membership is identical across runs/machines.
#   - n_jobs is kept at a fixed, non-negative-but-still-deterministic value;
#     scikit-learn's CV searches are deterministic under a fixed
#     random_state regardless of n_jobs, but we default to n_jobs=1 to avoid
#     any solver-level nondeterminism from parallel BLAS threads.
###############################################################################
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from config import RANDOM_SEED, set_global_seed


###############################################################################
# 0. Helpers
###############################################################################
def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace(".", "").replace("-", "_")
        for c in df.columns
    ]
    return df


def specificity_score(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else float("nan")


###############################################################################
# 1. Prepare modeling dataset
###############################################################################
def prepare_modeling_data(feature_list: dict):
    features = clean_names(feature_list["features"])

    drop_cols = [c for c in ["dsdecod", "dsterm", "dsdtc", "arm"] if c in features.columns]
    features = features.drop(columns=drop_cols)

    id_outcome_cols = [c for c in ["usubjid", "discont", "dsdecod"] if c in feature_list["features"].columns]
    discont_char = clean_names(feature_list["features"])[id_outcome_cols]

    model_df = features.drop(columns=[c for c in ["rfstdtc", "rfendtc"] if c in features.columns])
    model_df = model_df[model_df["discont"].notna()].copy()
    model_df = model_df.sort_values("usubjid").reset_index(drop=True)  # deterministic row order

    return model_df, discont_char


###############################################################################
# 2. Train/holdout split (70/30, stratified)
###############################################################################
def split_data(model_df: pd.DataFrame):
    train_df, holdout_df = train_test_split(
        model_df,
        test_size=0.30,
        stratify=model_df["discont"],
        random_state=RANDOM_SEED,
    )
    train_df = train_df.sort_values("usubjid").reset_index(drop=True)
    holdout_df = holdout_df.sort_values("usubjid").reset_index(drop=True)
    return train_df, holdout_df


###############################################################################
# 3. Preprocessing: median-impute numeric, one-hot encode nominal, drop
#    zero-variance numeric columns. Fit on development set only.
###############################################################################
def build_preprocessor(X_train_raw: pd.DataFrame):
    num_cols = sorted(X_train_raw.select_dtypes(include=[np.number]).columns.tolist())
    cat_cols = sorted(c for c in X_train_raw.columns if c not in num_cols)

    var = X_train_raw[num_cols].var(skipna=True)
    zv_cols = var[(var.isna()) | (var == 0)].index.tolist()
    num_cols = [c for c in num_cols if c not in zv_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    return pre, num_cols, cat_cols


###############################################################################
# 4. Logistic regression (elastic net, glmnet-equivalent)
###############################################################################
def fit_logistic(x_train: np.ndarray, y_train: np.ndarray):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)

    model = LogisticRegressionCV(
        Cs=20,
        cv=cv,
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.1, 0.5, 0.9],
        max_iter=10000,
        scoring="roc_auc",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    model.fit(x_train_s, y_train)
    return model, scaler


###############################################################################
# 5. SVM (RBF kernel), cross-validated
###############################################################################
def fit_svm(x_train: np.ndarray, y_train: np.ndarray):
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    param_grid = {"C": [0.5, 1, 2], "gamma": ["scale", "auto"]}

    svm_cv = GridSearchCV(
        SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED),
        param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
    )
    svm_cv.fit(x_train_s, y_train)
    return svm_cv, scaler


###############################################################################
# 6. Evaluation (identical for both models)
###############################################################################
def evaluate_model(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob > 0.5).astype(int)
    return dict(
        model=name,
        auc=roc_auc_score(y_true, y_prob),
        pr_auc=average_precision_score(y_true, y_prob),
        accuracy=accuracy_score(y_true, y_pred),
        sensitivity=recall_score(y_true, y_pred, pos_label=1),
        specificity=specificity_score(y_true, y_pred),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]),
    )


###############################################################################
# 7. Key-insight summary
###############################################################################
def print_key_insight(logit_model, logit_scaler, svm_cv, svm_scaler, feature_names,
                       x_holdout, y_holdout, results_df):
    print("\n=== KEY INSIGHT: modeling.py ===")

    print(f"Best logistic regression: C = {logit_model.C_[0]:.4g}, "
          f"l1_ratio = {logit_model.l1_ratio_[0]:.2g}")
    print(f"Best SVM (grid search): {svm_cv.best_params_}")

    # Logistic regression: odds ratios from coefficients (on standardized scale)
    coefs = pd.Series(logit_model.coef_[0], index=feature_names)
    top_logit = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10)
    print("\nTop logistic-regression predictors (standardized coefficient -> odds ratio):")
    for feat, coef in top_logit.items():
        print(f"  {feat:40s}  coef = {coef:+.3f}   OR = {np.exp(coef):.3f}")

    # SVM: permutation importance on the (scaled) holdout set
    x_holdout_s = svm_scaler.transform(x_holdout)
    perm = permutation_importance(
        svm_cv.best_estimator_, x_holdout_s, y_holdout,
        scoring="roc_auc", n_repeats=20, random_state=RANDOM_SEED, n_jobs=1,
    )
    perm_series = pd.Series(perm.importances_mean, index=feature_names)
    top_svm = perm_series.sort_values(ascending=False).head(10)
    print("\nTop SVM predictors (permutation importance, holdout AUC drop):")
    for feat, imp in top_svm.items():
        print(f"  {feat:40s}  importance = {imp:.4f}")

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    best_row = results_df.loc[results_df["auc"].idxmax()]
    print(
        f"\nTakeaway: {best_row['model']} generalizes better on the holdout set "
        f"(AUC = {best_row['auc']:.3f}). The strongest individual predictors span "
        "lab, adverse-event, and exposure feature domains (see rankings above); "
        "review the top-10 lists to identify which specific domain dominates for "
        "this dataset before drawing clinical conclusions."
    )
    print("===================================\n")


###############################################################################
# 8. Orchestration
###############################################################################
def run(feature_list: dict):
    set_global_seed(RANDOM_SEED)

    model_df, _ = prepare_modeling_data(feature_list)
    train_df, holdout_df = split_data(model_df)

    y_train = train_df["discont"].astype(int).values
    y_holdout = holdout_df["discont"].astype(int).values
    X_train_raw = train_df.drop(columns=["discont", "usubjid"])
    X_holdout_raw = holdout_df.drop(columns=["discont", "usubjid"])

    pre, num_cols, cat_cols = build_preprocessor(X_train_raw)
    x_train = pre.fit_transform(X_train_raw)
    x_holdout = pre.transform(X_holdout_raw)

    feature_names = num_cols + list(
        pre.named_transformers_["cat"].get_feature_names_out(cat_cols)
    )

    logit_model, logit_scaler = fit_logistic(x_train, y_train)
    logit_pred = logit_model.predict_proba(logit_scaler.transform(x_holdout))[:, 1]

    svm_cv, svm_scaler = fit_svm(x_train, y_train)
    svm_pred = svm_cv.predict_proba(svm_scaler.transform(x_holdout))[:, 1]

    logit_eval = evaluate_model("logistic", y_holdout, logit_pred)
    svm_eval = evaluate_model("svm", y_holdout, svm_pred)

    results_df = pd.DataFrame(
        [{k: v for k, v in e.items() if k != "confusion_matrix"} for e in [logit_eval, svm_eval]]
    )

    print("Holdout evaluation:")
    print(results_df.to_string(index=False))
    print("\nConfusion matrix - logistic regression (rows=truth, cols=pred; 0/1):")
    print(logit_eval["confusion_matrix"])
    print("\nConfusion matrix - SVM (rows=truth, cols=pred; 0/1):")
    print(svm_eval["confusion_matrix"])

    print_key_insight(
        logit_model, logit_scaler, svm_cv, svm_scaler, feature_names,
        x_holdout, y_holdout, results_df,
    )

    return dict(
        results=results_df,
        logit_eval=logit_eval,
        svm_eval=svm_eval,
        logit_model=logit_model,
        svm_model=svm_cv,
    )


if __name__ == "__main__":
    from feature_engineering import build_features
    from load_data import load_all

    sdtm = load_all("sdtm")
    feature_list = build_features(sdtm)
    run(feature_list)
