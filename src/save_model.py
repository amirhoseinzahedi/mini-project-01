import joblib
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


DATA_PATH = "mini_project_01/data/creditcard.csv"
MODEL_PATH = "mini_project_01/models/model.pkl"
SCALER_PATH = "mini_project_01/models/scaler.pkl"


def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_data(df):
    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Fit scaler on the complete dataset
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def train_model(X, y):
    # Final model selected in Phase 8
    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X, y)

    return model


def save_components(model, scaler):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")


def main():
    print("=" * 60)
    print("SAVING FINAL MODEL")
    print("=" * 60)

    df = load_data()

    print(f"\nDataset shape: {df.shape}")

    X, y, scaler = prepare_data(df)

    print(f"Features: {X.shape[1]}")
    print(f"Samples : {X.shape[0]}")

    model = train_model(X, y)

    print("\nFinal model:")
    print("K-Nearest Neighbors")
    print("K = 5")

    save_components(model, scaler)

    print("\nFinal model and scaler saved successfully.")


if __name__ == "__main__":
    main()
