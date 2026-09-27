import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page Configuration
st.set_page_config(page_title="Student Score Predictor", layout="wide")

st.title("🎓 Student Performance Predictor")
st.write("Train a Linear Regression model on study hours vs. exam scores and make live predictions.")

# ============================================================
# 1. LOAD & CLEAN DATASET
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "student_performance_dataset.csv")

if not os.path.exists(csv_path):
    st.error(f"Dataset not found at `{csv_path}`. Make sure `student_performance_dataset.csv` is in this folder.")
    st.stop()

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df[["study_time_hours", "final_exam_score"]].dropna()
    return df

df = load_data(csv_path)

# ============================================================
# 2. MODEL TRAINING
# ============================================================
X = df[["study_time_hours"]]
y = df["final_exam_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

@st.cache_resource
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)
y_pred = model.predict(X_test)

# Save trained model
model_path = os.path.join(BASE_DIR, "student_score_model.pkl")
joblib.dump(model, model_path)

# Metrics calculation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

slope = model.coef_[0]
intercept = model.intercept_

# ============================================================
# STREAMLIT UI TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔮 Interactive Prediction", "📊 Dataset & Performance", "📈 Visualizations"])

# TAB 1: Live Interactive Prediction
with tab1:
    st.subheader("Predict Final Exam Score")
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        study_hours = st.number_input(
            "Enter Study Time (Hours):",
            min_value=0.0,
            max_value=24.0,
            value=5.0,
            step=0.5
        )
        
    with col_output:
        new_student = pd.DataFrame({"study_time_hours": [study_hours]})
        predicted_score = model.predict(new_student)[0]
        
        st.metric(
            label="Predicted Score",
            value=f"{predicted_score:.2f} points"
        )
        st.info(f"**Equation:** Score = {slope:.2f} × ({study_hours}) + {intercept:.2f}")

# TAB 2: Dataset Summary & Performance Metrics
with tab2:
    col_data, col_metrics = st.columns([1, 1])
    
    with col_data:
        st.subheader("Dataset Overview")
        st.write(f"**Total Samples:** {len(df)}")
        st.write(f"**Train Samples:** {len(X_train)} | **Test Samples:** {len(X_test)}")
        st.dataframe(df.head(10), use_container_width=True)
        
    with col_metrics:
        st.subheader("Model Evaluation Metrics")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("MAE", f"{mae:.2f}")
        m_col2.metric("MSE", f"{mse:.2f}")
        m_col1.metric("RMSE", f"{rmse:.2f}")
        m_col2.metric("R² Score", f"{r2:.2f}")

# TAB 3: Visualizations
with tab3:
    col_fig1, col_fig2 = st.columns(2)
    
    with col_fig1:
        st.subheader("Study Time vs Final Score")
        sort_idx = np.argsort(X_test["study_time_hours"].values)
        X_sorted = X_test["study_time_hours"].values[sort_idx]
        y_pred_sorted = y_pred[sort_idx]

        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.scatter(X_test["study_time_hours"], y_test, label="Actual Scores", alpha=0.7)
        ax1.plot(X_sorted, y_pred_sorted, color="red", label="Best Fit Line")
        ax1.set_xlabel("Study Time (hours)")
        ax1.set_ylabel("Final Exam Score")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig1)

    with col_fig2:
        st.subheader("Actual vs Predicted Scores")
        minimum = min(y_test.min(), y_pred.min())
        maximum = max(y_test.max(), y_pred.max())

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(y_test, y_pred, alpha=0.7)
        ax2.plot([minimum, maximum], [minimum, maximum], color="red", linestyle="--")
        ax2.set_xlabel("Actual Score")
        ax2.set_ylabel("Predicted Score")  # Fixed set_ylabel syntax
        ax2.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig2)