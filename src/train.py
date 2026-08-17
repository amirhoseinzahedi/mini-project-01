import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


from tqdm.auto import tqdm
from time import perf_counter


DATA_PATH = "mini_project_01/data/creditcard.csv"
SCALER_PATH = "mini_project_01/models/scaler.pkl"


def load_data():
    return pd.read_csv(DATA_PATH)


def scaler():
    return joblib.load(SCALER_PATH)


def prepare_data(df):
    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = joblib.load(SCALER_PATH)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, X_test_scaled, y_train, y_test)


def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }

    trained_models = {}
    training_times = {}

    progress = tqdm(models.items(), total=len(models), desc="Training")

    for name, model in progress:
        start_time = perf_counter()

        model.fit(X_train, y_train)

        elapsed_time = perf_counter() - start_time

        trained_models[name] = model
        training_times[name] = elapsed_time

        progress.set_postfix(model=name, time=f"{elapsed_time:.4f}s")

        print(f"\n{name} trained in {elapsed_time:.4f}s")

    # OPTIONAL MODEL
    model_path = "mini_project_01/models/mlp.pth"
    mlp_model = train_mlp(X_train, y_train, epochs=20, batch_size=256)
    torch.save(mlp_model.state_dict(), model_path)
    print(f"Simple MLP (NN) saved to {model_path}")

    return trained_models  # , training_times


################################# NEURAL NETWORK #################################
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


def train_mlp(X_train, y_train, epochs=20, batch_size=256):
    # convert numpy arrays to PyTorch tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)

    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleMLP(input_size=X_train.shape[1])

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(logits, y_batch)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)

        print(f"Epoch [{epoch + 1}/{epochs}]Loss: {average_loss}")
    return model


def predict_mlp(model, X):
    X_tensor = torch.tensor(X, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).int()

        return predictions.numpy().ravel()


################################# NEURAL NETWORK #################################


def main():
    # Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Train models
    trained_models = train_models(X_train, y_train)

    # Save trained models
    for name, model in trained_models.items():
        model_path = f"mini_project_01/models/{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, model_path)
        print(f"{name} saved to {model_path}")


if __name__ == "__main__":
    main()
