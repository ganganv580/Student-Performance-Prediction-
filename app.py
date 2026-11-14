
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction")
st.write("Predict student academic performance using Machine Learning.")

st.info("This application uses a simulated dataset for educational demonstration.")

@st.cache_resource
def train_models():

    rng = np.random.default_rng(42)
    n = 1000

    hours = rng.integers(1, 10, n)
    previous = rng.integers(40, 101, n)

    extracurricular = rng.choice(
        ["Yes", "No"], n, p=[0.45, 0.55]
    )

    sleep = rng.integers(4, 10, n)
    papers = rng.integers(0, 10, n)

    # Convert Yes/No into numerical values
    extra_numeric = (extracurricular == "Yes").astype(int)

    noise = rng.normal(0, 5, n)

    performance = (
        8 + 2.6 * hours
        + 0.55 * previous
        + 1.1 * sleep
        + 0.8 * papers
        + 2 * extra_numeric
        + noise
    )

    performance = np.clip(performance, 0, 100)

    df = pd.DataFrame({
        "Hours Studied": hours,
        "Previous Scores": previous,
        "Extracurricular Activities": extra_numeric,
        "Sleep Hours": sleep,
        "Sample Question Papers Practiced": papers
    })

    X_train, X_test, y_train, y_test = train_test_split(
        df, performance,
        test_size=0.20,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=8,
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42
        )
    }

    trained = {}
    scores = {}
    errors = {}

    for name, model in models.items():

        pipeline = Pipeline([
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        trained[name] = pipeline
        scores[name] = r2_score(y_test, predictions)
        errors[name] = mean_absolute_error(
            y_test, predictions
        )

    best_name = max(scores, key=scores.get)

    return trained, scores, errors, best_name


trained_models, model_scores, model_errors, best_model_name = train_models()

best_model = trained_models[best_model_name]

st.success("Machine Learning models trained successfully!")

st.subheader("Enter Student Details")

hours = st.slider("Hours Studied per Day", 1, 9, 6)

previous = st.slider("Previous Scores", 40, 100, 75)

extracurricular = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

sleep = st.slider("Sleep Hours", 4, 9, 7)

papers = st.slider(
    "Sample Question Papers Practiced",
    0, 9, 5
)

if st.button("Predict Student Performance"):

    student = pd.DataFrame([{
        "Hours Studied": hours,
        "Previous Scores": previous,
        "Extracurricular Activities": (
            1 if extracurricular == "Yes" else 0
        ),
        "Sleep Hours": sleep,
        "Sample Question Papers Practiced": papers
    }])

    prediction = float(
        np.clip(best_model.predict(student)[0], 0, 100)
    )

    st.success("Prediction Completed!")

    st.metric(
        "Predicted Performance Index",
        f"{prediction:.2f} / 100"
    )

    if prediction >= 75:
        st.success("High Performance")
    elif prediction >= 50:
        st.warning("Moderate Performance")
    else:
        st.error("Needs Support")

    st.progress(int(prediction))

st.divider()

st.subheader("Model Comparison")

results = pd.DataFrame({
    "Algorithm": list(model_scores.keys()),
    "R2 Score": list(model_scores.values()),
    "MAE": list(model_errors.values())
})

st.dataframe(results, use_container_width=True)

st.write("Best Model:", best_model_name)
