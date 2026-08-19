import pandas as pd

from predict import predict, load_components


DATA_PATH = "mini_project_01/data/creditcard.csv"


df = pd.read_csv(DATA_PATH)

model, scaler = load_components()


# Legitimate transaction
legitimate = df[df["Class"] == 0].drop("Class", axis=1).iloc[0].to_dict()

# Fraudulent transaction
fraudulent = df[df["Class"] == 1].drop("Class", axis=1).iloc[0].to_dict()


print("Legitimate transaction:")
print(predict(legitimate, model, scaler))

print("\nFraudulent transaction:")
print(predict(fraudulent, model, scaler))
