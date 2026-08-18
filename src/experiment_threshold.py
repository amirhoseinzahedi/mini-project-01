import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


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


def evaluate_threshold(
    probabilities,
    y_test,
    threshold,
):
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_test,
        predictions,
    )

    recall = recall_score(
        y_test,
        predictions,
    )

    f1 = f1_score(
        y_test,
        predictions,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
    ).ravel()

    return precision, recall, f1, fp, fn, tp, tn


def main():
    df = load_data()

    X_train, X_test, y_train, y_test = prepare_data(df)

    model = KNeighborsClassifier(
        n_neighbors=5,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = []

    print("=" * 70)
    print("KNN CLASSIFICATION THRESHOLD EXPERIMENT")
    print("=" * 70)

    for threshold in thresholds:
        (
            precision,
            recall,
            f1,
            fp,
            fn,
            tp,
            tn,
        ) = evaluate_threshold(
            probabilities,
            y_test,
            threshold,
        )

        print(f"\nThreshold = {threshold}")

        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        print("\nConfusion Matrix:")
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        results.append(
            {
                "Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "False Positives": fp,
                "False Negatives": fn,
            }
        )

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Threshold": "{:.1f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1": "{:.4f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()
