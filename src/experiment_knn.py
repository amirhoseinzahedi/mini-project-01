import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score


DATA_PATH = "mini_project_01/data/creditcard.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    return precision, recall, f1


def main():
    df = load_data()

    X_train, X_test, y_train, y_test = prepare_data(df)

    k_values = [1, 5, 20]

    results = []

    print("=" * 60)
    print("KNN HYPERPARAMETER EXPERIMENT")
    print("=" * 60)

    for k in k_values:
        print(f"\nTraining KNN with K={k}...")

        model = KNeighborsClassifier(n_neighbors=k)

        model.fit(X_train, y_train)

        precision, recall, f1 = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        results.append(
            {
                "K": k,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 60)
    print("KNN HYPERPARAMETER RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
