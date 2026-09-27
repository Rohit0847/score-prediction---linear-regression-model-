# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import joblib


# Load dataset
df = pd.read_csv("C:\Users\REEYA\OneDrive\Desktop\Rohit Project\ML Model\student_performance_dataset.csv"

)


# Check dataset
print(df.head())
print("Dataset shape:", df.shape)


# Select input and target
X = df[["study_time_hours"]]
y = df["final_exam_score"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create Linear Regression model
model = LinearRegression()


# Train the model
model.fit(X_train, y_train)


# Save the trained model
joblib.dump(model, "student_score_model.pkl")

print("Model saved successfully.")


# Check current folder and model file
import os

print("Current folder:", os.getcwd())
print("Model exists:", os.path.exists("student_score_model.pkl"))


# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R²:", r2)


# Print regression equation
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print(
    f"Equation: Final Exam Score = "
    f"{model.coef_[0]:.2f} × Study Time + {model.intercept_:.2f}"
)


# Sort values for drawing the regression line
sort_idx = np.argsort(X_test["study_time_hours"].values)

X_sorted = X_test["study_time_hours"].values[sort_idx]
y_pred_sorted = y_pred[sort_idx]


# Plot actual data
plt.scatter(
    X_test["study_time_hours"],
    y_test,
    label="Actual Scores"
)


# Plot regression line
plt.plot(
    X_sorted,
    y_pred_sorted,
    label="Best Fit Line"
)


# Labels and title
plt.xlabel("Study Time (hours)")
plt.ylabel("Final Exam Score")
plt.title("Study Time vs Final Exam Score")

plt.legend()
plt.show()