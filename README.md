<<<<<<< HEAD
# AI Intelligent Student Performance System

A Streamlit web application that predicts student final marks using Linear Regression and a Neural Network, evaluates model performance, visualizes results, and suggests possible study-habit improvements.

## Features

- Generates a reproducible synthetic dataset of 300 student records.
- Demonstrates missing-value handling with median imputation.
- Uses standard scaling in machine-learning pipelines.
- Trains and evaluates Linear Regression and an MLP Neural Network.
- Displays MAE, RMSE, and R² metrics.
- Shows model charts and permutation feature importance.
- Predicts marks from user-provided student details.
- Provides what-if study, attendance, and sleep recommendations.

## Project structure

```text
Week6_AI_Intelligent_System/
├── app.py
├── data.py
├── models.py
├── recommendations.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

Open a terminal in this project folder and install dependencies:

```powershell
& "D:\PYTHON SOFTWARE\python.exe" -m pip install -r requirements.txt
```

## Run the application

Run Streamlit (do not run `app.py` with plain `python`):

```powershell
& "D:\PYTHON SOFTWARE\python.exe" -m streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`.

## Dataset note

The application generates synthetic sample data in code; its predictions demonstrate a machine-learning workflow and are not guaranteed real-world academic outcomes.
=======
# Week6-AI-Intelligent-System
An intelligent student performance prediction system using Machine Learning, Neural Networks, and Streamlit.
>>>>>>> bfde80267a28f94dfcdc610cc1b7bc0c61e74c58
