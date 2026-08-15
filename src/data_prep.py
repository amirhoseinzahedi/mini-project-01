import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "mini_project_01/data/creditcard.csv"
SCALER_PATH = "mini_project_01/models/scaler.pkl"


def load_data(path):
    return pd.read_csv(path)


def check_data_quality(df):
    print("===== Data Quality =====")

    print("\nShape:")
    print(df.shape)

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nTotal Missing Values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())


def analyze_class_distribution(y):
    print("\n===== Class Distribution =====")

    class_counts = y.value_counts()
    class_percentages = y.value_counts(normalize=True) * 100

    distribution = pd.DataFrame(
        {"Count": class_counts, "Percentage": class_percentages}
    )

    print(distribution)


def preprocess_data(df):
    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n===== Train/Test Split =====")

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    print("\nTraining Class Distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTesting Class Distribution:")
    print(y_test.value_counts(normalize=True))

    # Create scaler
    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data using training statistics
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)

    print("\nScaler saved to:", SCALER_PATH)

    return X_train_scaled, X_test_scaled, y_train, y_test


def main():
    df = load_data(DATA_PATH)

    check_data_quality(df)

    analyze_class_distribution(df["Class"])

    X_train, X_test, y_train, y_test = preprocess_data(df)


if __name__ == "__main__":
    main()
