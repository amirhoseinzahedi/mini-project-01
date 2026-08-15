# Experiments

## Phase 1 — Data Preparation

### Dataset

Credit Card Fraud Detection dataset from Kaggle.

### Dataset Structure

- Samples: 284,807
- Features: 30
- Target: `Class`
- Target type: Binary classification

### Features

The dataset contains:

- `Time`
- `V1` through `V28`
- `Amount`

The `V1`–`V28` features are anonymized PCA-transformed features.

### Missing Values

No missing values were found.

### Class Distribution

| Class | Meaning | Count |
|---|---|---:|
| 0 | Legitimate transaction | 284,315 |
| 1 | Fraudulent transaction | 492 |

The dataset is extremely imbalanced. Fraudulent transactions represent approximately 0.17% of all transactions.

### Initial Observation

Accuracy alone will not be an appropriate primary evaluation metric because a model could achieve very high accuracy while failing to detect fraudulent transactions.

Future experiments should therefore consider metrics such as:

- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC