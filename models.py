import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import FEATURES, create_dataset


def permutation_importance_r2(model, X, y, n_repeats=20, seed=42):
    """Measure the R² drop when each feature is randomly shuffled."""
    rng = np.random.default_rng(seed)
    base_score = r2_score(y, model.predict(X))
    scores = {}

    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled_X = X.copy()
            shuffled_X[col] = rng.permutation(shuffled_X[col].values)
            shuffled_score = r2_score(y, model.predict(shuffled_X))
            drops.append(base_score - shuffled_score)

        scores[col] = float(np.mean(drops))

    return pd.Series(scores)


@st.cache_resource
def train_models():
    df = create_dataset()
    X, y = df[FEATURES], df["Final_Marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    linear = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    neural = TransformedTargetRegressor(
        regressor=Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                max_iter=2000,
                early_stopping=True,
                random_state=42,
            )),
        ]),
        transformer=StandardScaler(),
    )

    linear.fit(X_train, y_train)
    neural.fit(X_train, y_train)

    def calculate_metrics(model):
        predictions = model.predict(X_test)
        return (
            mean_absolute_error(y_test, predictions),
            float(np.sqrt(mean_squared_error(y_test, predictions))),
            r2_score(y_test, predictions),
        )

    importance = permutation_importance_r2(
        neural, X_test, y_test
    ).sort_values()

    return {
        "linear": linear,
        "neural": neural,
        "X_test": X_test,
        "y_test": y_test,
        "metrics": {
            "Linear Regression": calculate_metrics(linear),
            "Neural Network": calculate_metrics(neural),
        },
        "importance": importance,
    }
