import joblib
import os

MODEL_FILE = "classsense_model.pkl"


def load_model():
    """Load the trained Random Forest model."""

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "classsense_model.pkl not found. "
            "Run train_model.py first."
        )

    return joblib.load(MODEL_FILE)


def predict_behavior(
    attendance,
    phone_events,
    interactions,
    activity_level
):
    """
    Predict student behaviour category
    using the trained Random Forest model.
    """

    model = load_model()

    data = [[
        attendance,
        phone_events,
        interactions,
        activity_level
    ]]

    prediction = model.predict(data)

    return prediction[0]