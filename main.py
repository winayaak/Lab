import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read the CSV file
data = pd.read_csv("Student.csv")

# Input features and target output
X = data[["Study_Hours", "Attendance", "Assignments", "Previous_Marks"]]
y = data["Pass"]

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the test data and display accuracy
y_pred = model.predict(X_test)
print("Accuracy =", accuracy_score(y_test, y_pred))

# Predict a new student's result
new_student = pd.DataFrame(
    [[6, 80, 6, 65]],
    columns=["Study_Hours", "Attendance", "Assignments", "Previous_Marks"]
)
prediction = model.predict(new_student)
probability = model.predict_proba(new_student)
print("Prediction =", prediction)
print("Probability =", probability)

# Plot probability of passing against study hours
hours = np.linspace(1, 9, 200)
graph_data = pd.DataFrame({
    "Study_Hours": hours,
    "Attendance": np.repeat(data["Attendance"].mean(), 200),
    "Assignments": np.repeat(data["Assignments"].mean(), 200),
    "Previous_Marks": np.repeat(data["Previous_Marks"].mean(), 200)
})

prob = model.predict_proba(graph_data)[:, 1]
plt.figure(figsize=(8, 5))
plt.scatter(data["Study_Hours"], data["Pass"], color="blue", s=80,
            label="Actual Data")
plt.plot(hours, prob, color="red", linewidth=3, label="Logistic Curve")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Pass")
plt.title("Logistic Regression on Non-Separable Data")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
