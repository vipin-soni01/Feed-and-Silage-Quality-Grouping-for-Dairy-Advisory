import os
import joblib
import pandas as pd


# Project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# Load trained models
imputer = joblib.load(
    os.path.join(BASE_DIR, "models", "imputer.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "models", "scaler.pkl")
)

kmeans = joblib.load(
    os.path.join(BASE_DIR, "models", "kmeans.pkl")
)

quality_mapping = joblib.load(
    os.path.join(BASE_DIR, "models", "quality_mapping.pkl")
)


# Features used by the final K-Means model
FEATURES = [
    "dm.s",
    "ash.s",
    "cp.s",
    "ee.s",
    "ndf.s",
    "adf.s",
    "starch.s",
    "pH",
    "ammonia.s",
    "glucose.s",
    "fructose.s",
    "ethanol.s",
    "lactic.ac.s",
    "acetic.ac.s",
    "propionic.ac.s",
    "butyric.ac.s",
    "dm.loss"
]


def predict_quality(input_data):

    # Create DataFrame
    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    # Handle missing values
    input_imputed = imputer.transform(input_df)

    input_imputed_df = pd.DataFrame(
        input_imputed,
        columns=FEATURES
    )

    input_scaled = scaler.transform(input_imputed_df)

    input_scaled_df = pd.DataFrame(
        input_scaled,
        columns=FEATURES
    )

    cluster = int(
        kmeans.predict(input_scaled_df)[0]
    )

    # Convert cluster to quality
    quality = quality_mapping[cluster]

    # Calculate moisture
    moisture = 100 - input_data["dm.s"]

    # Moisture baseline
    if moisture >= 68.2801304:
        moisture_baseline = "Excellent"
    else:
        moisture_baseline = "Not Excellent"

    return {
        "cluster": cluster,
        "quality": quality,
        "moisture": moisture,
        "moisture_baseline": moisture_baseline
    }