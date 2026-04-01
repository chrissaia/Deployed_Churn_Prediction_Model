from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_test, y_test, threshold: float = 0.35):
    '''
    Evaluate models performance on the test set

    :param model:
    :param X_test:
    :param y_test:
    '''

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    print("Classification Report:\n", classification_report(y_test, preds, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, proba))

    return proba, preds