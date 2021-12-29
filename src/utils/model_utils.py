import numpy as np
from src.utils.parameter_loader import (
    load_gridsearch_parameters,
    load_gridsearch_model_parameters,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import json
import pickle

def find_threshold(clf, y_test, X_test):
    # A function that finds the highest (up to descritization) probability threshold (or decision boundary) that has a recall
    # of req_recall.
    req_recall = 0.8
    threshold = 0.5
    pred_proba_test = clf.predict_proba(X_test)
    pred_test = (pred_proba_test[:, 1] >= threshold).astype("int")
    search_step = 1e-2  # the amount to decrease the probabilty threshold if the recall is not > .8
    current_recall = recall_calc(y_test, pred_test)
    while current_recall < req_recall:
        threshold = threshold - search_step
        pred_proba_test = clf.predict_proba(X_test)
        pred_test = (pred_proba_test[:, 1] >= threshold).astype("int")
        current_recall = recall_calc(y_test, pred_test)
    return threshold


def recall_calc(y_true, y_pred):
    # A calculator for the recall of diabetes. There is a built-in function for this, but I wanted to verify the built-in.
    y_true = y_true.values
    # y_pred = y_pred.values
    true_positives = np.array(
        [
            1 if (y_true[i] == 1 and y_pred[i] == 1) else 0
            for i in np.arange(0, len(y_true))
        ]
    )
    false_negatives = np.array(
        [
            1 if (y_true[i] == 1 and y_pred[i] == 0) else 0
            for i in np.arange(0, len(y_true))
        ]
    )
    recall = true_positives.sum() / (true_positives.sum() + false_negatives.sum())
    return recall

def find_optimal_model(X_train, y_train):
    """
    Uses a grid search to find the best gradient boosting classifier (optimizing f1 score)
    then finds the highest threshold thresh for which when the model predicts
    1 when P(input > thresh) and 0 otherwise, we achieve 80% recall
    on the training set.
    """
    CV, OPT_ON, N_JOBS = load_gridsearch_parameters()
    grid_values_log = load_gridsearch_model_parameters()
    clf_log = LogisticRegression()
    grid_clf_log = GridSearchCV(
        clf_log, param_grid=grid_values_log, cv=CV, scoring=OPT_ON, n_jobs=N_JOBS
    )

    grid_clf_log.fit(X_train, y_train)
    thresh = find_threshold(grid_clf_log, y_train, X_train)

    threshold_data = {}
    threshold_data["thresh"] = thresh

    with open("src/conf/threshold_data.json", "w", encoding="utf-8") as outfile:
        json.dump(threshold_data, outfile)

    pickle.dump(grid_clf_log, open("src/models/model.pkl", "wb"))
    return grid_clf_log
