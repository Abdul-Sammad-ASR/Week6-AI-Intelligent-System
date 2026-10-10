import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data import LIMITS, create_dataset
from models import train_models
from recommendations import predict, recommend

st.set_page_config(
    page_title="AI Student Performance System",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 AI Intelligent Student Performance System")
st.write(
    "Predict final marks with Machine Learning and a Neural Network, "
    "then get personalised study recommendations."
)

df = create_dataset()
bundle = train_models()

tab_data, tab_models, tab_predict = st.tabs([
    "📊 Data & Preprocessing",
    "🤖 Models & Results",
    "🎯 Predict & Recommend",
])

# Tab 1: Dataset and preprocessing
with tab_data:
    st.subheader("Student dataset")
    st.dataframe(df.head(10))

    col1, col2 = st.columns(2)
    col1.metric("Total student records", len(df))
    col2.metric("Missing values", int(df.isna().sum().sum()))

    st.write(
        "**Preprocessing pipeline:** median imputation → standard scaling → model. "
        "The imputer and scaler learn from training data only (no data leakage)."
    )

    missing = df.isna().sum()
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(missing.index, missing.values, color="#e76f51")
    ax.set_xlabel("Missing values")
    ax.set_title("Missing values per column (handled by the imputer)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Tab 2: Model results and charts
with tab_models:
    st.subheader("Model evaluation (unseen test data)")
    results_table = pd.DataFrame(
        bundle["metrics"], index=["MAE", "RMSE", "R² Score"]
    ).T.round(3)
    st.dataframe(results_table)

    st.caption(
        "Lower MAE/RMSE and higher R² are better. The synthetic marks follow a "
        "mostly linear formula, so Linear Regression may perform very well."
    )

    nn_model = bundle["neural"].regressor_.named_steps["model"]
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(nn_model.loss_curve_, label="Training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (squared error)")
        ax.set_title("Neural network training curve")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        validation_scores = getattr(nn_model, "validation_scores_", None)
        if validation_scores:
            ax.plot(validation_scores, color="#2a9d8f", label="Validation R²")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Validation scores unavailable", ha="center", va="center")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("R²")
        ax.set_title("Validation score (early stopping)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        "Architecture: 4 inputs → 16 neurons (ReLU) → 8 neurons (ReLU) → 1 output, "
        "trained with the Adam optimiser and early stopping."
    )

    col3, col4 = st.columns(2)
    with col3:
        names = list(bundle["metrics"])
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(names, [bundle["metrics"][name][2] for name in names],
               color=["#264653", "#2a9d8f"])
        ax.set_ylabel("R² Score")
        ax.set_title("Comparison of AI models")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.barh(bundle["importance"].index, bundle["importance"].values,
                color="#2a9d8f")
        ax.set_xlabel("Drop in R² when feature is shuffled")
        ax.set_title("What matters most? (Neural network)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    actual = bundle["y_test"]
    ax.scatter(actual, bundle["linear"].predict(bundle["X_test"]),
               label="Linear Regression", alpha=0.7)
    ax.scatter(actual, bundle["neural"].predict(bundle["X_test"]),
               label="Neural Network", alpha=0.7)
    ax.plot([0, 100], [0, 100], "--", color="gray", label="Perfect prediction")
    ax.set_xlabel("Actual marks")
    ax.set_ylabel("Predicted marks")
    ax.set_title("Actual vs predicted marks")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Tab 3: Prediction and recommendations
with tab_predict:
    st.subheader("Enter student details")

    with st.form("student_form"):
        col1, col2 = st.columns(2)

        with col1:
            study_hours = st.slider(
                "Daily study hours", *LIMITS["Study_Hours"], 5.0, 0.5
            )
            attendance = st.slider(
                "Attendance (%)", *LIMITS["Attendance"], 80
            )

        with col2:
            previous_marks = st.slider(
                "Previous marks", *LIMITS["Previous_Marks"], 65
            )
            sleep_hours = st.slider(
                "Daily sleep hours", *LIMITS["Sleep_Hours"], 7.0, 0.5
            )

        choice = st.selectbox(
            "Choose AI model", ["Neural Network", "Linear Regression"]
        )
        submitted = st.form_submit_button("Predict my marks")

    if submitted:
        student = {
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "Previous_Marks": previous_marks,
            "Sleep_Hours": sleep_hours,
        }
        model = bundle["neural"] if choice == "Neural Network" else bundle["linear"]
        base, recommendation_results = recommend(model, student)

        st.metric("Predicted final marks", f"{base:.1f} / 100")

        if base >= 80:
            st.success("Excellent! Keep maintaining your study routine.")
        elif base >= 60:
            st.info("Good progress! Regular revision can help you improve.")
        else:
            st.warning(
                "Consider a consistent study schedule, extra practice "
                "and reviewing difficult topics."
            )

        st.subheader("💡 Personalised recommendations")
        useful = [item for item in recommendation_results if item[2] > 0.05]

        if not useful:
            st.write(
                "Great job - none of the available changes is predicted "
                "to raise your marks further."
            )
        else:
            for label, _, gain in useful:
                st.write(f"- **{label}** → about **+{gain:.1f} marks**")

            plan = {**student}
            for _, changes, _ in useful:
                plan.update(changes)

            combined_prediction = predict(model, plan)
            st.write(
                f"**Combined plan:** {base:.1f} → "
                f"**{combined_prediction:.1f} / 100**"
            )

            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.barh(
                [item[0] for item in useful][::-1],
                [item[2] for item in useful][::-1],
                color="#2a9d8f",
            )
            ax.set_xlabel("Expected gain in marks")
            ax.set_title("Recommended actions ranked by impact")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.caption(
            "Estimates come from synthetic training data and show patterns the "
            "model learned, not guaranteed results."
        )
