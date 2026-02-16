import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("Heart Disease Prediction App")

st.write("Upload CSV file and select model")

# model options
model_option = st.selectbox(
    "Select Model",
    ["knn", "logistic", "decision_tree", "naive_bayes", "random_forest", "xgboost"]
)

# file upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# show something even before upload
st.info("Please upload heart.csv file to test")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview:")
    st.write(df.head())

    if "target" not in df.columns:
        st.error("CSV must contain 'target' column")
    else:

        X = df.drop("target", axis=1)
        y = df["target"]

        model_path = os.path.join("saved_models", f"{model_option}.pkl")

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
        else:
            model = joblib.load(model_path)

            y_pred = model.predict(X)

            acc = accuracy_score(y, y_pred)

            st.success(f"Accuracy: {acc:.4f}")
