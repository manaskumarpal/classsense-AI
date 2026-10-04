import pandas as pd
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# =========================================================
# 1. CREATE TRAINING DATA
# =========================================================

data = []

for _ in range(1000):

    attendance = random.randint(40, 100)

    phone_events = random.randint(0, 10)

    interactions = random.randint(0, 10)

    activity_level = random.randint(20, 100)


    # Create training label
    # This is synthetic training data for the college project.

    score = 0

    if attendance >= 75:
        score += 2

    if phone_events <= 2:
        score += 2

    if interactions <= 4:
        score += 2

    if activity_level >= 65:
        score += 2


    if score >= 7:

        category = "Excellent"

    elif score >= 5:

        category = "Good"

    else:

        category = "Needs Attention"


    data.append(
        {
            "attendance": attendance,
            "phone_events": phone_events,
            "interactions": interactions,
            "activity_level": activity_level,
            "category": category
        }
    )


df = pd.DataFrame(data)


# =========================================================
# 2. SAVE DATASET
# =========================================================

df.to_csv(
    "training_data.csv",
    index=False
)


print("Training dataset created.")
print(
    f"Total training records: {len(df)}"
)


# =========================================================
# 3. PREPARE DATA
# =========================================================

X = df[
    [
        "attendance",
        "phone_events",
        "interactions",
        "activity_level"
    ]
]

y = df["category"]


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


# =========================================================
# 4. TRAIN RANDOM FOREST
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(
    X_train,
    y_train
)


# =========================================================
# 5. TEST MODEL
# =========================================================

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print(
    f"Model accuracy: {accuracy * 100:.2f}%"
)


# =========================================================
# 6. SAVE TRAINED MODEL
# =========================================================

joblib.dump(
    model,
    "classsense_model.pkl"
)


print(
    "Trained ML model saved as classsense_model.pkl"
)

print(
    "ML training completed successfully."
)