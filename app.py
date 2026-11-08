from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TARGET = "deposit"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}


def evaluate(pipe, X, y):
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


st.set_page_config(page_title="Bank Deposit Classifier", layout="wide")
st.title("Bank Term Deposit Prediction")
st.write(
    "Upload a test CSV with the same features as the bank marketing data and the "
    f"`{TARGET}` column. Saved models are evaluated on the uploaded data; nothing is retrained."
)

models = load_models()

selected = st.selectbox("Select a model", list(models.keys()))

uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
if uploaded is None:
    st.info("Upload a CSV file to see the results.")
    st.stop()

df = pd.read_csv(uploaded)
if TARGET not in df.columns:
    st.error(f"The uploaded file must contain the `{TARGET}` column.")
    st.stop()

required = list(next(iter(models.values())).feature_names_in_)
missing = [col for col in required if col not in df.columns]
if missing:
    st.error(f"Missing required feature columns: {', '.join(missing)}")
    st.stop()

X = df[required]
y = (df[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)

with st.expander("Preview uploaded data"):
    st.dataframe(df.head())

st.subheader("Comparison of all models on uploaded data")
all_results = {name: evaluate(pipe, X, y) for name, pipe in models.items()}
st.dataframe(pd.DataFrame(all_results).T.round(4), use_container_width=True)

st.subheader("Model details")
pipe = models[selected]
metrics = all_results[selected]

cols = st.columns(len(metrics))
for col, (metric, value) in zip(cols, metrics.items()):
    col.metric(metric, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(pipe, X, y, ax=ax, colorbar=False)
    st.pyplot(fig)

with right:
    st.markdown("**Classification report**")
    report = classification_report(y, pipe.predict(X), output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4))
