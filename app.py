import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="Bank Marketing Classification", layout="centered")

st.title("📊 Bank Marketing Classification App")
st.write("Machine Learning Assignment – BITS Pilani")

# ---------------------------------
# Load saved scaler
# ---------------------------------
scaler = joblib.load("models/scaler.pkl")

# ---------------------------------
# Load trained models
# ---------------------------------
models = {
    "Logistic Regression": joblib.load("models/logistic_regression.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "KNN": joblib.load("models/knn.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl")
}

# ---------------------------------
# Model selection dropdown
# ---------------------------------
model_name = st.selectbox("Select a Model", list(models.keys()))
model = models[model_name]

# ---------------------------------
# Upload CSV file
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file (bank-additional-full.csv format)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, sep=';')

    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    if 'y' not in df.columns:
        st.error("The uploaded file must contain target column 'y'")
    else:
        X = df.drop('y', axis=1)
        y = df['y']

        X_scaled = scaler.transform(X)

        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1]

        st.write("## 📈 Evaluation Metrics")

        st.write("Accuracy:", round(accuracy_score(y, y_pred), 4))
        st.write("AUC Score:", round(roc_auc_score(y, y_prob), 4))
        st.write("Precision:", round(precision_score(y, y_pred), 4))
        st.write("Recall:", round(recall_score(y, y_pred), 4))
        st.write("F1 Score:", round(f1_score(y, y_pred), 4))
        st.write("MCC:", round(matthews_corrcoef(y, y_pred), 4))

        st.write("## 🔍 Confusion Matrix")
        st.write(confusion_matrix(y, y_pred))

        st.write("## 📄 Classification Report")
        st.text(classification_report(y, y_pred))
