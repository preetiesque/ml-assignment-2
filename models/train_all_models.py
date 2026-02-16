import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

os.makedirs("saved_models", exist_ok=True)

df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

models = {
    "knn": KNeighborsClassifier(),
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": xgb.XGBClassifier(eval_metric="logloss")
}

for name, model in models.items():

    model.fit(X_train, y_train)

    joblib.dump(model, f"saved_models/{name}.pkl")

    print(name, "saved")
