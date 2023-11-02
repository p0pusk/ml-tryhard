import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.feature_selection import RFE


def model_build(model, kf, title="Default"):
    model_pkl_file = f"{title}_classifier_model.pkl"
    if os.path.isfile(model_pkl_file):
        with open(model_pkl_file, "rb") as file:
            model = pickle.load(file)
            print("model already build, reading from file")
            return

    data = pd.read_csv("./Data/features_3_sec.csv")
    df = data.iloc[0:, 2:]
    y = df.label.values
    X = df.drop("label", axis=1)
    X_columns = X.columns
    scale = MinMaxScaler()
    scaled_data = scale.fit_transform(X)
    X = pd.DataFrame(scaled_data, columns=X_columns).values

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))

    with open(model_pkl_file, "wb") as file:
        pickle.dump(model, file)
        print("Model saved to: ", model_pkl_file)

    print("Accuracy score of", title, "is:", round(np.mean(accuracy_scores), 2))
