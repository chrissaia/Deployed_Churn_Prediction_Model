import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import recall_score
from litellm import completion

print("=== Phase 2: Modeling with XGBoost (Optuna + StratifiedKFold) ===")

# --- Load processed data robustly (relative to repo root) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # root folder of the project
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "telco_churn_processed.csv"  # path to processed dataset
assert DATA_PATH.exists(), f"Missing processed dataset at: {DATA_PATH}"

df = pd.read_csv(DATA_PATH)  # full processed dataframe

# --- Target must be numeric 0/1 ---
if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].astype(str).str.strip().map({"No": 0, "Yes": 1})  # map labels to binary

assert df["Churn"].isna().sum() == 0, "Churn has NaNs after mapping"
assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1"

X = df.drop(columns=["Churn"])  # feature matrix
y = df["Churn"].astype(int)     # target vector

# No object columns should remain
obj_cols = X.select_dtypes(include=["object"]).columns.tolist()  # list of non-numeric columns
assert not obj_cols, f"Found object columns in features: {obj_cols[:10]}"

# No NaNs/Infs
X = X.replace([np.inf, -np.inf], np.nan)  # replace infinite values
assert X.isna().sum().sum() == 0, "Features contain NaNs/Infs"

# --- Keep a true held-out test set (no leakage) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 20% of data held out for final testing
    stratify=y,             # preserve churn ratio
    random_state=42         # reproducible split
)

THRESHOLD = 0.40  # probability cutoff for predicting churn

# CV object (StratifiedKFold is correct for imbalanced churn)
CV_SPLITS = int(os.getenv("CV_SPLITS", "3"))  # number of cross-validation folds
cv = StratifiedKFold(
    n_splits=CV_SPLITS,     # number of folds
    shuffle=True,           # shuffle before splitting
    random_state=42         # reproducible folds
)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),         # number of trees
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),    # step size
        "max_depth": trial.suggest_int("max_depth", 3, 10),                  # tree depth
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),             # row sampling
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),# column sampling
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),    # min leaf weight
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),                     # split regularization
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),             # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),           # L2 regularization
        "random_state": 42,        # reproducibility
        "n_jobs": -1,              # use all CPU cores
        "eval_metric": "logloss",  # evaluation metric
        "tree_method": "hist",     # faster CPU training
    }

    recalls = []  # store recall from each fold

    for tr_idx, val_idx in cv.split(X_train, y_train):
        X_tr = X_train.iloc[tr_idx]  # training features for this fold
        X_val = X_train.iloc[val_idx]  # validation features for this fold
        y_tr = y_train.iloc[tr_idx]  # training labels for this fold
        y_val = y_train.iloc[val_idx]  # validation labels for this fold

        spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)  # class imbalance weight

        params_fold = dict(params)  # copy params
        params_fold["scale_pos_weight"] = float(spw)  # apply imbalance weight

        model = XGBClassifier(**params_fold)  # create model
        model.fit(X_tr, y_tr)  # train on fold

        proba = model.predict_proba(X_val)[:, 1]  # churn probability
        y_pred = (proba >= THRESHOLD).astype(int)  # thresholded predictions

        recalls.append(recall_score(y_val, y_pred, pos_label=1))  # store recall

    return float(np.mean(recalls))  # average recall across folds

# Make it fast in CI
N_TRIALS = int(os.getenv("OPTUNA_TRIALS", "10"))  # number of Optuna trials
sampler = optuna.samplers.TPESampler(seed=42)    # reproducible hyperparameter search

study = optuna.create_study(direction="maximize", sampler=sampler)  # create Optuna study
study.optimize(objective, n_trials=N_TRIALS)  # run tuning

print("Best Params:", study.best_params)
print("Best CV Recall:", study.best_value)

# --- Train final model on full training data ---
best_params = study.best_params.copy()  # copy best hyperparameters
best_params.update({
    "random_state": 42,        # reproducibility
    "n_jobs": -1,              # use all CPU cores
    "eval_metric": "logloss",  # evaluation metric
    "tree_method": "hist",     # fast CPU training
})

spw_full = (y_train == 0).sum() / max((y_train == 1).sum(), 1)  # full training imbalance ratio
best_params["scale_pos_weight"] = float(spw_full)  # apply class weight

final_model = XGBClassifier(**best_params)  # final tuned model
final_model.fit(X_train, y_train)           # train on full training data

proba_test = final_model.predict_proba(X_test)[:, 1]  # test churn probabilities
y_pred_test = (proba_test >= THRESHOLD).astype(int)   # thresholded predictions
test_recall = recall_score(y_test, y_pred_test, pos_label=1)  # final recall on test set

print("Test Recall:", test_recall)

# Set a reasonable floor
assert test_recall >= 0.50, f"Test recall too low: {test_recall:.3f}"

