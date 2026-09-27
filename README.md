# 🎓 Student Performance Predictor

A supervised machine learning project that predicts a student's **final exam score** from their **study time (hours)** using **Linear Regression** — with an interactive Streamlit web app for live predictions.

## 📌 Overview

This project explores whether a student's final exam score can be predicted from how many hours they study. It walks through a complete, beginner-friendly ML regression workflow: loading and cleaning data, training a model, evaluating it, and deploying it as an interactive app.

- **Type:** Supervised Learning — Regression
- **Algorithm:** Linear Regression
- **Input Feature:** `study_time_hours`
- **Target Variable:** `final_exam_score`

## 🗂️ Dataset

The dataset (`student_performance_dataset.csv`) contains 1,000 student records with 12 columns, including:

| Column | Description |
|---|---|
| `student_id` | Unique student identifier |
| `gender` | Student gender |
| `study_time_hours` | Hours spent studying (**model input**) |
| `attendance_percent` | Class attendance percentage |
| `sleep_hours` | Average sleep hours |
| `parental_education` | Parental education level |
| `internet_access` | Internet access at home (Yes/No) |
| `extracurricular_activities` | Participation in extracurriculars |
| `part_time_job` | Whether the student has a part-time job |
| `previous_grade` | Previous academic grade |
| `final_exam_score` | Final exam score (**model target**) |
| `final_grade` | Letter grade (A–D) |

> **Note:** While the dataset has 12 columns, this model currently uses only one input feature — `study_time_hours` — to keep the workflow simple and interpretable.

## ⚙️ Project Workflow

1. **Data Loading** — Dataset loaded with Pandas
2. **Data Inspection** — Checked structure, shape, and missing values
3. **Feature Selection** — Selected `study_time_hours` (X) and `final_exam_score` (y)
4. **Train/Test Split** — 80% training (800 samples) / 20% testing (200 samples), `random_state=42`
5. **Model Training** — Fit a `LinearRegression` model from scikit-learn
6. **Evaluation** — Assessed with MAE, MSE, RMSE, and R²
7. **Model Persistence** — Saved the trained model with `joblib` (`student_score_model.pkl`)
8. **Deployment** — Wrapped everything in an interactive **Streamlit** app (`app.py`)

## 📈 Model Performance

| Metric | Value | Meaning |
|---|---|---|
| **MAE** | 6.91 | Predictions differ from actual scores by ~6.91 points on average |
| **MSE** | 74.58 | Mean squared error across the test set |
| **RMSE** | 8.64 | Typical error, with larger errors penalized more |
| **R²** | 0.31 | Study time alone explains ~31% of score variation |

**Regression Equation:**

```
Final Exam Score = 3.99 × Study Time (hours) + 69.45
```

An additional hour of study time is associated with an estimated **3.99-point increase** in predicted exam score. This reflects an *association*, not a proven causal effect — R² = 0.31 shows study time alone leaves most of the variation in scores unexplained.

## 🖥️ Streamlit App Features

The app (`app.py`) provides three interactive tabs:

- **🔮 Interactive Prediction** — Enter study hours and get a live predicted exam score
- **📊 Dataset & Performance** — Dataset overview and model evaluation metrics
- **📈 Visualizations** — Study Time vs. Exam Score scatter plot with best-fit line, and Actual vs. Predicted score comparison

## 📸 Screenshots

**Interactive Prediction**
![Interactive Prediction tab](screenshots/interactive-prediction.png)

**Dataset & Performance**
![Dataset and Performance tab](screenshots/dataset-performance.png)

**Visualizations**
![Visualizations tab](screenshots/visualizations.png)

## 📁 Repository Structure

```
.
├── ML Model/
│   ├── app.py                                      # Streamlit web application
│   ├── __Import_libraries.py                       # Data loading, training & evaluation script
│   ├── student_performance_dataset.csv             # Dataset
│   ├── student_score_model.pkl                     # Trained Linear Regression model
│   └── student_performance_multiple_regression.pkl # Additional trained model (multi-feature)
├── screenshots/
│   ├── interactive-prediction.png
│   ├── dataset-performance.png
│   └── visualizations.png
├── Presentation.pptx                               # Project presentation slides
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit pandas numpy scikit-learn matplotlib joblib
```

### Run the app

```bash
cd "ML Model"
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> **Note:** `app.py` expects `student_performance_dataset.csv` to be in the same folder (`ML Model/`), since it loads the dataset relative to its own file location.

## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy** — data handling
- **scikit-learn** — model training & evaluation
- **Matplotlib** — visualizations
- **Streamlit** — interactive web app
- **Joblib** — model serialization

## 🔮 Future Improvements

- Incorporate additional features (attendance, sleep, previous grade, etc.) for a multiple regression model
- Experiment with more advanced algorithms: Random Forest, Decision Tree, Gradient Boosting
- Add cross-validation for more robust performance estimates
- Handle missing values (e.g. in `parental_education`) more thoroughly

## 👤 Author

**Rohit Pandey**
Computer Science & AIML

🔗 [Repository](https://github.com/Rohit0847/score-prediction---linear-regression-model-)
