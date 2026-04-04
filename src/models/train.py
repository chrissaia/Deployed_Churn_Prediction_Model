import mlflow.sklearn
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score
import time


def train_model(df: pd.DataFrame, target_col: str, params: dict):
    '''
    Trains XGBoost model and tracks performance metrics using MLFlow

    :param df: dataframe containing training data
    :param target_col:
    :param params: dictionary containing parameters for XGBoost model
    :return: model, metrics, proba, preds
    '''

    X = df.drop(columns=target_col)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(**params)


    # Train model
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    proba = model.predict_proba(X_test)[:, 1]  # Outputs two probabilities for each customer
    preds = (proba >= 0.35).astype(int)  # Makes the prediction based on those probabilities

    metrics = {
        "train_time": train_time,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, pos_label=1),
        "recall": recall_score(y_test, preds, pos_label=1),
        "f1": f1_score(y_test, preds, pos_label=1),
        "roc_auc": roc_auc_score(y_test, proba),
    }

    return model, metrics, proba, preds

