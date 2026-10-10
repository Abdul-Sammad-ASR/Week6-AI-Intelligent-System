import numpy as np
import pandas as pd
import streamlit as st

FEATURES = ["Study_Hours", "Attendance", "Previous_Marks", "Sleep_Hours"]

# Slider limits match the range used to generate the training data.
LIMITS = {
    "Study_Hours": (1.0, 10.0),
    "Attendance": (50, 100),
    "Previous_Marks": (30, 95),
    "Sleep_Hours": (4.0, 10.0),
}


@st.cache_data
def create_dataset():
    rng = np.random.default_rng(42)
    n = 300

    df = pd.DataFrame({
        "Study_Hours": rng.uniform(1, 10, n),
        "Attendance": rng.uniform(50, 100, n),
        "Previous_Marks": rng.uniform(30, 95, n),
        "Sleep_Hours": rng.uniform(4, 10, n),
    })

    noise = rng.normal(0, 5, n)
    df["Final_Marks"] = np.clip(
        df["Study_Hours"] * 4
        + df["Attendance"] * 0.2
        + df["Previous_Marks"] * 0.4
        + df["Sleep_Hours"] * 1.5
        + noise,
        0,
        100,
    )

    # Add a small amount of missing data for preprocessing demonstration.
    for i, col in enumerate(["Study_Hours", "Attendance", "Sleep_Hours"]):
        missing_rows = df.sample(frac=0.03, random_state=100 + i).index
        df.loc[missing_rows, col] = np.nan

    return df
