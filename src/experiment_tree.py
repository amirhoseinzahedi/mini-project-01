import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
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


def evaluate_model(model, X, y):
    predictions = model.predict(X)

    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    return precision, recall, f1


def main():
    df = load_data()

    X_train, X_test, y_train, y_test = prepare_data(df)

    depths = [2, 5, 10, None]

    results = []

    print("=" * 60)
    print("DECISION TREE HYPERPARAMETER EXPERIMENT")
    print("=" * 60)

    for depth in depths:
        print(f"\nTraining Decision Tree with max_depth={depth}...")

        model = DecisionTreeClassifier(max_depth=depth, random_state=42)

        model.fit(X_train, y_train)

        train_precision, train_recall, train_f1 = evaluate_model(
            model,
            X_train,
            y_train,
        )

        test_precision, test_recall, test_f1 = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print("\nTraining:")
        print(f"Precision: {train_precision:.4f}")
        print(f"Recall   : {train_recall:.4f}")
        print(f"F1-score : {train_f1:.4f}")

        print("\nTest:")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall   : {test_recall:.4f}")
        print(f"F1-score : {test_f1:.4f}")

        results.append(
            {
                "max_depth": depth,
                "Train Precision": train_precision,
                "Train Recall": train_recall,
                "Train F1": train_f1,
                "Test Precision": test_precision,
                "Test Recall": test_recall,
                "Test F1": test_f1,
            }
        )

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 80)
    print("DECISION TREE HYPERPARAMETER RESULTS")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Train Precision": "{:.4f}".format,
                "Train Recall": "{:.4f}".format,
                "Train F1": "{:.4f}".format,
                "Test Precision": "{:.4f}".format,
                "Test Recall": "{:.4f}".format,
                "Test F1": "{:.4f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
