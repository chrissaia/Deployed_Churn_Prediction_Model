#!/usr/bin/env python3
"""
scripts/run_pipeline.py

End-to-end training pipeline:
load -> validate -> preprocess -> feature engineering -> (optional tune) -> train -> log (MLflow) -> save artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import get_mlrrun_metrics

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.utils.validate_data import validate_telco_data

try:
    from src.features.build_feature import build_features
except ImportError:
    from src.features.build_features import build_features

from src.models.train import train_model
from src.models.tune import tune_model
from src.models.explanation import explain


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_params_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--params_json not found: {p}")
    with p.open("r") as f:
        return json.load(f)


def mlflow_log_model_safe(model: Any, name_or_path: str = "model") -> None:
    try:
        mlflow.sklearn.log_model(sk_model=model, name=name_or_path)
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path=name_or_path)


def run(args: argparse.Namespace) -> None:
    log = logging.getLogger("pipeline")

    # --- MLflow setup ---
    tracking_uri = args.mlflow_uri or f"file://{PROJECT_ROOT / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    # --- Baseline model params + overrides ---
    params: Dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
    }
    params.update(load_params_json(args.params_json))

    with mlflow.start_run():
        # --- Log run config ---
        mlflow.log_param("input", args.input)
        mlflow.log_param("target", args.target)
        mlflow.log_param("threshold_cli", args.threshold)
        mlflow.log_param("test_size_cli", args.test_size)
        mlflow.log_param("model", "XGBClassifier")
        mlflow.log_param("tune_enabled", int(args.tune))
        mlflow.log_param("tune_trials", args.tune_trials)
        mlflow.log_param("tune_cv_splits", args.tune_cv_splits)
        mlflow.log_params({f"xgb__base__{k}": v for k, v in params.items()})

        # --- Load ---
        log.info("Loading: %s", args.input)
        df = load_data(args.input)
        log.info("Loaded: %d rows x %d cols", df.shape[0], df.shape[1])

        # --- Validate ---
        if args.skip_validation:
            log.warning("Skipping validation (--skip_validation).")
            is_valid, failed = True, []
        else:
            log.info("Validating data quality...")
            is_valid, failed = validate_telco_data(df)
            mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed: {failed}")


        # --- Preprocess ---
        log.info("Preprocessing...")
        df_processed = preprocess_data(df)

        if args.save_processed:
            processed_dir = PROJECT_ROOT / "data" / "processed"
            ensure_dir(processed_dir)
            processed_path = processed_dir / "telco_churn_processed.csv"
            df_processed.to_csv(processed_path, index=False)
            mlflow.log_artifact(str(processed_path), artifact_path="data")
            log.info("Saved processed dataset: %s", processed_path)

        # --- Feature engineering ---
        if args.target not in df_processed.columns:
            raise ValueError(f"Target column '{args.target}' not found.")

        log.info("Building features...")
        df_fe = build_features(df_processed, target_col=args.target)

        # booleans -> int
        bool_cols = df_fe.select_dtypes(include=["bool"]).columns.tolist()
        if bool_cols:
            df_fe[bool_cols] = df_fe[bool_cols].astype(int)

        # numeric safety
        df_fe = df_fe.replace([np.inf, -np.inf], np.nan)

        # target safety
        if df_fe[args.target].dtype == "object":
            df_fe[args.target] = df_fe[args.target].astype(str).str.strip().map({"No": 0, "Yes": 1})

        if df_fe[args.target].isna().any():
            raise ValueError("Target contains NaN after mapping. Fix in preprocess/build_features.")

        # fill NaNs in features
        feature_df = df_fe.drop(columns=[args.target]).fillna(0)
        df_fe = pd.concat([feature_df, df_fe[[args.target]].astype(int)], axis=1)

        # --- Save feature schema ---
        artifacts_dir = PROJECT_ROOT / "artifacts"
        ensure_dir(artifacts_dir)

        feature_cols = df_fe.drop(columns=[args.target]).columns.tolist()
        feature_path = artifacts_dir / "feature_columns.json"
        with feature_path.open("w") as f:
            json.dump(feature_cols, f, indent=2)

        mlflow.log_artifact(str(feature_path), artifact_path="artifacts")
        mlflow.log_param("n_features", len(feature_cols))
        log.info("Saved feature schema: %s (%d features)", feature_path, len(feature_cols))

        # --- Add scale_pos_weight (imbalance) ---
        y_all = df_fe[args.target].astype(int)
        pos = int((y_all == 1).sum())
        neg = int((y_all == 0).sum())
        if pos > 0:
            params["scale_pos_weight"] = float(neg / pos)
            mlflow.log_param("scale_pos_weight", params["scale_pos_weight"])
            mlflow.log_param("pos_count", pos)
            mlflow.log_param("neg_count", neg)





        # --- OPTIONAL: Hyperparameter tuning (Optuna) ---
        if args.tune:
            log.info("Tuning enabled. Creating train-only set for Optuna...")
            X_all = df_fe.drop(columns=[args.target])
            y_all = df_fe[args.target].astype(int)

            # Holdout test set so tuning doesn't touch it
            X_train, _, y_train, _ = train_test_split(
                X_all, y_all,
                test_size=args.test_size,
                stratify=y_all,
                random_state=42,
            )

            log.info("Running Optuna: trials=%d cv_splits=%d", args.tune_trials, args.tune_cv_splits)
            best_params = tune_model(X_train, y_train, n_trials=args.tune_trials, cv_splits=args.tune_cv_splits, seed=args.seed)

            # Merge + log best params
            params.update(best_params)
            mlflow.log_params({f"xgb__best__{k}": v for k, v in best_params.items()})
            log.info("Optuna complete. Best params merged into training params.")

        # Warn about hardcoded behavior inside your train_model
        if abs(args.threshold - 0.35) > 1e-12:
            log.warning("train_model() hardcodes threshold=0.35; --threshold won't change preds unless you update train_model.")
        if abs(args.test_size - 0.2) > 1e-12:
            log.warning("train_model() hardcodes test_size=0.2; --test_size won't change split unless you update train_model.")

        # --- Train + evaluate (your function) ---
        log.info("Training model...")
        model, metrics, proba, preds = train_model(df_fe, args.target, params)

        # --- Log metrics ---
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # log date and time

        # Summary artifact
        summary = {
            "metrics": metrics,
            "threshold_used_in_train_model": 0.35,
            "test_size_used_in_train_model": 0.2,
            "tuning_enabled": bool(args.tune),
        }
        mlflow.log_text(json.dumps(summary, indent=2), artifact_file="eval_summary.json")

        # --- Log model ---
        log.info("Logging model to MLflow...")
        mlflow_log_model_safe(model, "model")

        # Print metrics
        print("\nFinal metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        log.info("Done.")

        # --- OPTIONAL: llm explanation ---
        if args.explain:
            # pick a single example (first row for now)
            row = df.iloc[0].to_dict()
            feature_importance = dict(
                sorted(
                    zip(df_fe.drop(columns=[args.target]).columns, model.feature_importances_),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            )

            explanation = explain(row, feature_importance, proba, preds)


            print(explanation)
            mlflow.log_text(json.dumps(explanation, indent=2), artifact_file="llm_explanation.json")


        if args.past_models:
            get_mlrrun_metrics.get_mlrrun_metrics()





def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Telco churn pipeline (XGBoost + MLflow)")
    p.add_argument("--input", type=str, default="data/raw/Telco-Customer-Churn.csv")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn - XGBoost")
    p.add_argument("--mlflow_uri", type=str, default=None)
    p.add_argument("--params_json", type=str, default=None)
    p.add_argument("--save_processed", action="store_true")
    p.add_argument("--skip_validation", action="store_true", default=True)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--past_models", action="store_true", default=False)


    # tuning flags
    p.add_argument("--tune", action="store_true", help="Run Optuna tuning on train split (no leakage)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for tuning model")
    p.add_argument("--tune_trials", type=int, default=30, help="Number of Optuna trials")
    p.add_argument("--tune_cv_splits", type=int, default=3, help="CV folds for Optuna tuning")
    p.add_argument("--explain", action="store_true", default=False, help="Whether to call ChatGPT to explain")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    run(args)


"""
Examples:

python scripts/run_pipeline.py \
  --input data/raw/Telco-Customer-Churn.csv \
  --target Churn \
  --save_processed

Override model params:
python scripts/run_pipeline.py \
  --input data/raw/Telco-Customer-Churn.csv \
  --params_json configs/xgb_params.json
"""
