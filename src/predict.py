import json
import sys

import joblib
import pandas as pd


MODEL_PATH = "mini_project_01/models/model.pkl"
SCALER_PATH = "mini_project_01/models/scaler.pkl"

THRESHOLD = 0.5

FEATURES = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]


def load_components():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


def prepare_transaction(transaction):
    missing_features = [feature for feature in FEATURES if feature not in transaction]

    if missing_features:
        raise ValueError(f"Missing features: {', '.join(missing_features)}")

    extra_features = [feature for feature in transaction if feature not in FEATURES]

    if extra_features:
        raise ValueError(f"Unexpected features: {', '.join(extra_features)}")

    data = pd.DataFrame(
        [[transaction[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    return data


def predict(transaction, model, scaler):
    data = prepare_transaction(transaction)

    data_scaled = scaler.transform(data)

    fraud_probability = model.predict_proba(data_scaled)[0, 1]

    prediction = int(fraud_probability >= THRESHOLD)

    return {
        "prediction": prediction,
        "fraud_probability": float(fraud_probability),
        "is_fraud": bool(prediction),
        "threshold": THRESHOLD,
    }


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python predict.py '<transaction_json>'",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        transaction = json.loads(sys.argv[1])

        if not isinstance(transaction, dict):
            raise ValueError("Input must be a JSON object.")

        model, scaler = load_components()

        result = predict(
            transaction,
            model,
            scaler,
        )

        print(json.dumps(result, indent=2))

    except json.JSONDecodeError:
        print(
            "Error: Invalid JSON input.",
            file=sys.stderr,
        )
        sys.exit(1)

    except ValueError as error:
        print(
            f"Error: {error}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
