import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.data_utils import load_data
import pickle
from random import randrange
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

TEST_FRACTION = 0.25  # percentage of the test size
CV = 7  # cross validations to do in grid search
OPT_ON = "f1"  # what to optimize in gridsearches
N_JOBS = 5  # number of cores
RAND_STATE = randrange(
    1000
)  # will be doing multiple train-test-splits, and want to keep everything random, but do the same train-test-splits


def summary_of_model(clf, X_train, X_test, y_train, y_test, threshold):
    # from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score, recall_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

    # This provides a summary of the model given a certain decision threshold of the predicted probability.
    # It includes a summary on recall/accuracy on the training and test sets, a visual display of the confusion matrix
    # and a plot of the precision-recall curve for a given classifier.
    pred_proba_test = clf.predict_proba(X_test)
    pred_test = (pred_proba_test[:, 1] >= threshold).astype("int")
    pred_proba_train = clf.predict_proba(X_train)
    pred_train = (pred_proba_train[:, 1] >= threshold).astype("int")
    print(classification_report(y_test, pred_test))
    print(
        "Recall of diabetes on the training set: {:.2f}".format(
            recall_score(y_train, pred_train)
        )
    )
    print(
        "Accuracy on the training set: {:.2f}".format(
            accuracy_score(y_train, pred_train)
        )
    )
    print(
        "Recall of diabetes class on the test set: {:.2f}".format(
            recall_score(y_test, pred_test)
        )
    )
    print("Accuracy on the test set: {:.2f}".format(accuracy_score(y_test, pred_test)))
    print(confusion_matrix(y_test, pred_test))
    _, ax = plt.subplots(figsize=(9, 9))
    ax = sns.heatmap(
        confusion_matrix(y_test, pred_test),
        annot=True,
        fmt="d",
        cmap="vlag",
        annot_kws={"size": 40, "weight": "bold"},
    )
    labels = ["False", "True"]
    ax.set_xticklabels(labels, fontsize=25)
    ax.set_yticklabels(labels, fontsize=25)
    ax.set_ylabel("Actual", fontsize=30)
    ax.set_xlabel("Prediction", fontsize=30)
    lr_probs = clf.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_probs)
    plt.figure()
    plt.plot(lr_recall, lr_precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


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


from sklearn.metrics import make_scorer


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


df_scaled = load_data()
#print(df_scaled.head())

y = df_scaled.pop("outcome")
X_train, X_test, y_train, y_test = train_test_split(
    df_scaled, y, test_size=TEST_FRACTION, random_state=RAND_STATE
)

grid_values_boost = {
    "n_estimators": [100, 200, 300, 400, 500],
    "learning_rate": [5e-3, 6e-3, 7e-3, 8e-3],
    "max_depth": [2, 3],
}

clf_boost = GradientBoostingClassifier()
grid_clf_boost = GridSearchCV(
    clf_boost, param_grid=grid_values_boost, cv=CV, scoring=OPT_ON, n_jobs=N_JOBS
)

grid_clf_boost.fit(X_train, y_train)
thresh = find_threshold(grid_clf_boost, y_test, X_test)
print(thresh)
summary_of_model(grid_clf_boost, X_train, X_test, y_train, y_test, thresh)

pickle.dump(grid_clf_boost, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0.639947, 1.425995, 0.166619, 0.468492, 0.031990, 0.181541, 0.866045]]))
