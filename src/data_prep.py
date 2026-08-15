import pandas as pd


DATA_PATH = "mini_project_01/data/creditcard.csv"


def load_data(path):
    return pd.read_csv(path)


def analyze_dataset(df):
    num_samples = df.shape[0]
    num_features = df.shape[1] - 1

    print("===== Dataset Structure =====")
    print(f"Number of Samples: {num_samples}")
    print(f"Number of Features: {num_features}")
    print(f"Number of Columns: {df.shape[1]}")

    print("\n===== Columns =====")
    print(df.columns.tolist())

    print("\n===== Data Types =====")
    print(df.dtypes)


def descriptive_statistics(df):
    print("\n===== Descriptive Statistics =====")
    print(df.describe().T)  # .T to see features as RowName and Statistics as ColumnName


def check_missing_values(df):
    missing_values = df.isnull().sum()

    print("\n===== Missing Values =====")
    print(missing_values)

    print("\nTotal Missing Values:", missing_values.sum())


def analyze_class_distribution(df):
    class_counts = df["Class"].value_counts()
    class_percentages = df["Class"].value_counts(normalize=True) * 100

    distribution = pd.DataFrame(
        {"Count": class_counts, "Percentage": class_percentages}
    )

    print("\n===== Class Distribution =====")
    print(distribution)


def main():
    df = load_data(DATA_PATH)

    analyze_dataset(df)
    descriptive_statistics(df)
    check_missing_values(df)
    analyze_class_distribution(df)


if __name__ == "__main__":
    main()
