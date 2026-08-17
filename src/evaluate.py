import joblib
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


DATA_PATH = "mini_project_01/data/creditcard.csv"
SCALER_PATH = "mini_project_01/models/scaler.pkl"

LOGISTIC_PATH = "mini_project_01/models/logistic_regression.pkl"
KNN_PATH = "mini_project_01/models/k-nearest_neighbors.pkl"
DECISION_TREE_PATH = "mini_project_01/models/decision_tree.pkl"
MLP_PATH = "mini_project_01/models/mlp.pth"


# ==============================================================================


def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_data(df):
    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Recreate the same train/test split used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Load the scaler fitted during preprocessing
    scaler = joblib.load(SCALER_PATH)

    # Transform train and test data using the training statistics
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )


# ==============================================================================


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp,
        "True Negatives": tn,
        "Confusion Matrix": cm,
    }


def print_metrics(model_name, metrics):
    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    print(f"\nAccuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1-score : {metrics['F1-score']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["Confusion Matrix"])

    print("\nConfusion Matrix Details:")
    print(f"True Negatives : {metrics['True Negatives']}")
    print(f"False Positives: {metrics['False Positives']}")
    print(f"False Negatives: {metrics['False Negatives']}")
    print(f"True Positives : {metrics['True Positives']}")


# ==============================================================================
# BASELINE MODEL EVALUATION
# ==============================================================================


def evaluate_sklearn_models(X_test_scaled, y_test):
    models = {
        "Logistic Regression": LOGISTIC_PATH,
        "K-Nearest Neighbors": KNN_PATH,
        "Decision Tree": DECISION_TREE_PATH,
    }

    results = []

    for model_name, model_path in models.items():
        print(f"\nEvaluating {model_name}...")

        model = joblib.load(model_path)

        y_pred = model.predict(X_test_scaled)

        metrics = calculate_metrics(y_test, y_pred)

        print_metrics(model_name, metrics)

        results.append(
            {
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-score": metrics["F1-score"],
                "False Positives": metrics["False Positives"],
                "False Negatives": metrics["False Negatives"],
            }
        )

    return results


# ==============================================================================
# MLP
# ==============================================================================


class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


def load_mlp():
    model = SimpleMLP(input_size=30)

    model.load_state_dict(
        torch.load(
            MLP_PATH,
            map_location=torch.device("cpu"),
        )
    )

    return model


def predict_mlp(model, X):
    X_tensor = torch.tensor(X, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits)

        predictions = (probabilities >= 0.5).int()

    return predictions.numpy().ravel()


def evaluate_mlp(X_test_scaled, y_test):
    print("\nEvaluating MLP...")

    model = load_mlp()

    y_pred = predict_mlp(model, X_test_scaled)

    metrics = calculate_metrics(y_test, y_pred)

    print_metrics("MLP", metrics)

    return {
        "Model": "MLP",
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1-score": metrics["F1-score"],
        "False Positives": metrics["False Positives"],
        "False Negatives": metrics["False Negatives"],
    }


# ==============================================================================
# KNN SCALING EXPERIMENT
# ==============================================================================


def evaluate_knn_scaling(
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
):
    print("\n\n" + "=" * 80)
    print("KNN SCALING EXPERIMENT")
    print("=" * 80)

    print("\nTraining KNN on unscaled data...")

    knn_unscaled = KNeighborsClassifier()
    knn_unscaled.fit(X_train, y_train)

    y_pred_unscaled = knn_unscaled.predict(X_test)

    unscaled_metrics = calculate_metrics(
        y_test,
        y_pred_unscaled,
    )

    print_metrics(
        "KNN - Unscaled Features",
        unscaled_metrics,
    )

    print("\nTraining KNN on scaled data...")

    knn_scaled = KNeighborsClassifier()
    knn_scaled.fit(X_train_scaled, y_train)

    y_pred_scaled = knn_scaled.predict(X_test_scaled)

    scaled_metrics = calculate_metrics(
        y_test,
        y_pred_scaled,
    )

    print_metrics(
        "KNN - Scaled Features",
        scaled_metrics,
    )

    results = pd.DataFrame(
        [
            {
                "Version": "Unscaled",
                "Accuracy": unscaled_metrics["Accuracy"],
                "Precision": unscaled_metrics["Precision"],
                "Recall": unscaled_metrics["Recall"],
                "F1-score": unscaled_metrics["F1-score"],
                "False Positives": unscaled_metrics["False Positives"],
                "False Negatives": unscaled_metrics["False Negatives"],
            },
            {
                "Version": "Scaled",
                "Accuracy": scaled_metrics["Accuracy"],
                "Precision": scaled_metrics["Precision"],
                "Recall": scaled_metrics["Recall"],
                "F1-score": scaled_metrics["F1-score"],
                "False Positives": scaled_metrics["False Positives"],
                "False Negatives": scaled_metrics["False Negatives"],
            },
        ]
    )

    print("\nKNN Scaling Comparison:")
    print(
        results.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1-score": "{:.4f}".format,
            },
        )
    )


# ==============================================================================
# DECISION TREE OVERFITTING EXPERIMENT
# ==============================================================================


def evaluate_decision_tree_overfitting(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
):
    print("\n\n" + "=" * 80)
    print("DECISION TREE OVERFITTING EXPERIMENT")
    print("=" * 80)

    model = joblib.load(DECISION_TREE_PATH)

    # Training predictions
    y_train_pred = model.predict(X_train_scaled)

    train_metrics = calculate_metrics(
        y_train,
        y_train_pred,
    )

    # Test predictions
    y_test_pred = model.predict(X_test_scaled)

    test_metrics = calculate_metrics(
        y_test,
        y_test_pred,
    )

    print("\nTraining Performance:")
    print(f"Accuracy : {train_metrics['Accuracy']:.4f}")
    print(f"Precision: {train_metrics['Precision']:.4f}")
    print(f"Recall   : {train_metrics['Recall']:.4f}")
    print(f"F1-score : {train_metrics['F1-score']:.4f}")

    print("\nTest Performance:")
    print(f"Accuracy : {test_metrics['Accuracy']:.4f}")
    print(f"Precision: {test_metrics['Precision']:.4f}")
    print(f"Recall   : {test_metrics['Recall']:.4f}")
    print(f"F1-score : {test_metrics['F1-score']:.4f}")

    comparison = pd.DataFrame(
        [
            {
                "Dataset": "Training",
                "Accuracy": train_metrics["Accuracy"],
                "Precision": train_metrics["Precision"],
                "Recall": train_metrics["Recall"],
                "F1-score": train_metrics["F1-score"],
            },
            {
                "Dataset": "Test",
                "Accuracy": test_metrics["Accuracy"],
                "Precision": test_metrics["Precision"],
                "Recall": test_metrics["Recall"],
                "F1-score": test_metrics["F1-score"],
            },
        ]
    )

    print("\nDecision Tree Train/Test Comparison:")
    print(
        comparison.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1-score": "{:.4f}".format,
            },
        )
    )


# ==============================================================================
# MODEL COMPARISON
# ==============================================================================


def print_model_comparison(results):
    results_df = pd.DataFrame(results)

    print("\n\n" + "=" * 80)
    print("BASELINE MODEL COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1-score": "{:.4f}".format,
            },
        )
    )


# ==============================================================================


def main():
    # Load data
    df = load_data()

    # Prepare train and test data
    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    ) = prepare_data(df)

    print("Test set shape:", X_test_scaled.shape)
    print("Fraud transactions in test set:", y_test.sum())
    print(
        "Legitimate transactions in test set:",
        (y_test == 0).sum(),
    )

    # --------------------------------------------------------------------------
    # 1. Baseline evaluation
    # --------------------------------------------------------------------------

    results = evaluate_sklearn_models(
        X_test_scaled,
        y_test,
    )

    # MLP
    mlp_result = evaluate_mlp(
        X_test_scaled,
        y_test,
    )

    results.append(mlp_result)

    # Compare all four baseline models
    print_model_comparison(results)

    # --------------------------------------------------------------------------
    # 2. KNN scaling experiment
    # --------------------------------------------------------------------------

    evaluate_knn_scaling(
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )

    # --------------------------------------------------------------------------
    # 3. Decision Tree overfitting experiment
    # --------------------------------------------------------------------------

    evaluate_decision_tree_overfitting(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()
