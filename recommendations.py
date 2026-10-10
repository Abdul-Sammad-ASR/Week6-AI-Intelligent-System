import numpy as np
import pandas as pd

from data import LIMITS


def predict(model, student):
    """Predict a student's final marks and keep the result within 0–100."""
    prediction = model.predict(pd.DataFrame([student]))[0]
    return float(np.clip(prediction, 0, 100))


def recommend(model, student):
    """Simulate individual habit changes and rank them by predicted improvement."""
    base_prediction = predict(model, student)
    actions = []

    if student["Study_Hours"] < LIMITS["Study_Hours"][1]:
        actions.append((
            "Study 1 more hour per day",
            {"Study_Hours": min(student["Study_Hours"] + 1, 10)},
        ))

    if student["Attendance"] < LIMITS["Attendance"][1]:
        actions.append((
            "Raise attendance by 10 points",
            {"Attendance": min(student["Attendance"] + 10, 100)},
        ))

    if student["Sleep_Hours"] < 8:
        actions.append(("Sleep 8 hours daily", {"Sleep_Hours": 8.0}))

    results = []
    for label, changes in actions:
        improved_student = {**student, **changes}
        gain = predict(model, improved_student) - base_prediction
        results.append((label, changes, gain))

    results.sort(key=lambda item: item[2], reverse=True)
    return base_prediction, results
