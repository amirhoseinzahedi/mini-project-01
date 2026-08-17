import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = "mini_project_01/data/creditcard.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y


def create_models():
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    DecisionTreeClassifier(
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    return models


def run_cross_validation(X, y, models):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        precision_scores = cv_results["test_precision"]
        recall_scores = cv_results["test_recall"]
        f1_scores = cv_results["test_f1"]

        print("\nFold Results:")

        for fold in range(5):
            print(
                f"Fold {fold + 1}: "
                f"Precision={precision_scores[fold]:.4f}, "
                f"Recall={recall_scores[fold]:.4f}, "
                f"F1={f1_scores[fold]:.4f}"
            )

        mean_precision = precision_scores.mean()
        mean_recall = recall_scores.mean()
        mean_f1 = f1_scores.mean()

        std_precision = precision_scores.std()
        std_recall = recall_scores.std()
        std_f1 = f1_scores.std()

        print("\nMean:")
        print(f"Precision: {mean_precision:.4f}")
        print(f"Recall   : {mean_recall:.4f}")
        print(f"F1-score : {mean_f1:.4f}")

        print("\nStandard Deviation:")
        print(f"Precision: {std_precision:.4f}")
        print(f"Recall   : {std_recall:.4f}")
        print(f"F1-score : {std_f1:.4f}")

        results.append(
            {
                "Model": name,
                "Mean Precision": mean_precision,
                "Mean Recall": mean_recall,
                "Mean F1": mean_f1,
                "Std Precision": std_precision,
                "Std Recall": std_recall,
                "Std F1": std_f1,
            }
        )

    return pd.DataFrame(results)


def print_summary(results):
    print("\n")
    print("=" * 80)
    print("5-FOLD STRATIFIED CROSS VALIDATION RESULTS")
    print("=" * 80)

    print(
        results[
            [
                "Model",
                "Mean Precision",
                "Mean Recall",
                "Mean F1",
            ]
        ].to_string(
            index=False,
            formatters={
                "Mean Precision": "{:.4f}".format,
                "Mean Recall": "{:.4f}".format,
                "Mean F1": "{:.4f}".format,
            },
        )
    )

    print("\n")
    print("=" * 80)
    print("CROSS-VALIDATION RESULTS WITH STANDARD DEVIATION")
    print("=" * 80)

    print(
        results.to_string(
            index=False,
            formatters={
                "Mean Precision": "{:.4f}".format,
                "Mean Recall": "{:.4f}".format,
                "Mean F1": "{:.4f}".format,
                "Std Precision": "{:.4f}".format,
                "Std Recall": "{:.4f}".format,
                "Std F1": "{:.4f}".format,
            },
        )
    )


def main():
    print("Loading dataset...")

    df = load_data()

    X, y = prepare_data(df)

    print("Dataset shape:", X.shape)
    print("Fraud transactions:", y.sum())
    print("Legitimate transactions:", (y == 0).sum())

    models = create_models()

    results = run_cross_validation(
        X,
        y,
        models,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
