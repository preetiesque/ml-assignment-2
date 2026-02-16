import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# create folder
os.makedirs("saved_models", exist_ok=True)

# load dataset
df = pd.read_csv("data/heart.csv")

# split input and output
X = df.drop("target", axis=1)
y = df["target"]

# split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y)

# train model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# save model
joblib.dump(model, "saved_models/knn.pkl")

print("KNN model saved")
