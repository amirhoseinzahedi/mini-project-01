````markdown
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

Future experiments should therefore consider:

- Precision
- Recall
- F1-score
- ROC-AUC
- Precision-Recall AUC


---

## Phase 2 — Data Preprocessing

### Data Quality

- Dataset shape: 284,807 rows × 31 columns
- All features are numerical.
- Target: `Class`

### Missing Values

No missing values were detected.

### Duplicate Analysis

Exact duplicate rows were checked using:

```python
df.duplicated()
````

Duplicate observations were identified during the data-quality analysis.

The duplicates were not used to alter the original dataset at this stage. The analysis was kept separate from the baseline modeling experiment so that the effect of any data-cleaning decision could be investigated explicitly.

### Train/Test Split

The dataset was split into:

* 80% training
* 20% testing

Parameters:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

Stratification was used to preserve approximately the same proportion of legitimate and fraudulent transactions in both subsets.

### Feature Scaling

`StandardScaler` was used for feature scaling.

The scaler was fitted **only on the training data**:

```python
scaler.fit(X_train)
```

The training and testing data were then transformed using the fitted scaler:

```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This prevents information from the test set from influencing the preprocessing process.

### Data Leakage Prevention

The preprocessing pipeline follows:

```text
Train/Test Split
       ↓
Fit Scaler on Training Data
       ↓
Transform Training Data
       ↓
Transform Test Data
```

The scaler was never fitted on the complete dataset before the split.

---

## Phase 3 — Hypothesis Before Modeling

Before training the models, several hypotheses were established.

### Model Performance

Logistic Regression was expected to provide a strong baseline because it is a simple and interpretable linear classifier.

The Decision Tree was expected to potentially outperform Logistic Regression if fraud patterns contained nonlinear relationships or feature interactions.

KNN was expected to be sensitive to feature scaling because it relies on distance calculations.

The MLP was expected to be capable of learning nonlinear relationships, but potentially sensitive to the severe class imbalance.

### Evaluation Metrics

Recall was expected to be particularly important because False Negatives represent fraudulent transactions that were not detected.

However, Precision and F1-score were also expected to be important because maximizing Recall alone can result in a large number of False Positives.

PR-AUC was expected to be particularly informative because of the severe class imbalance.

### All-Legitimate Prediction

It was hypothesized that a model predicting every transaction as legitimate could achieve extremely high accuracy while having:

```text
Recall = 0
```

This demonstrates why accuracy alone is insufficient for fraud detection.

### KNN and Scaling

It was hypothesized that feature scaling would significantly affect KNN performance because the algorithm depends on distances between observations.

### Decision Tree Overfitting

It was hypothesized that an unrestricted Decision Tree would have a higher risk of overfitting than a constrained tree.

### MLP Overfitting

It was hypothesized that the MLP could also overfit because of its greater model capacity and number of trainable parameters.

---

## Phase 4 — Model Training

Four baseline classification models were trained.

No oversampling, undersampling, SMOTE, or class weighting was applied during the baseline training stage.

The purpose of this phase was to establish baseline model behavior before introducing class-imbalance techniques.

### Model 1 — Logistic Regression

#### Purpose

* Establish a simple classification baseline.
* Analyze the behavior of a linear classifier on an imbalanced dataset.

#### Configuration

```python
LogisticRegression(
    random_state=42,
    max_iter=1000
)
```

---

### Model 2 — K-Nearest Neighbors

#### Purpose

* Analyze distance-based learning.
* Study the impact of feature scaling.

#### Configuration

```python
KNeighborsClassifier(
    n_neighbors=5
)
```

The model was trained using the standardized features.

---

### Model 3 — Decision Tree Classifier

#### Purpose

* Explore nonlinear decision boundaries.
* Analyze overfitting behavior.
* Investigate the effect of class imbalance.

#### Configuration

```python
DecisionTreeClassifier(
    random_state=42
)
```

No `max_depth` constraint was applied to the baseline tree so that potential overfitting could be investigated.

---

### Model 4 — Simple MLP / Neural Network

#### Purpose

* Explore nonlinear decision boundaries.
* Investigate neural-network behavior on tabular data.
* Analyze potential overfitting.
* Compare a neural network with traditional machine learning models.

#### Framework

```text
PyTorch
```

#### Architecture

```text
Input: 30 features
        ↓
Linear(30 → 64)
        ↓
ReLU
        ↓
Linear(64 → 32)
        ↓
ReLU
        ↓
Linear(32 → 1)
```

#### Training Configuration

| Parameter       | Value               |
| --------------- | ------------------- |
| Optimizer       | Adam                |
| Learning Rate   | 0.001               |
| Epochs          | 20                  |
| Batch Size      | 256                 |
| Loss Function   | `BCEWithLogitsLoss` |
| Class Weighting | Not applied         |

Class weighting was intentionally not used in the baseline experiment.

The effect of class imbalance will be investigated in later experiments.

---

## Phase 4 — Training Summary

| Model               | Algorithm Type | Scaling Used | Class Weighting |
| ------------------- | -------------- | ------------ | --------------- |
| Logistic Regression | Linear         | Yes          | No              |
| KNN                 | Distance-based | Yes          | No              |
| Decision Tree       | Tree-based     | Yes          | No              |
| MLP                 | Neural Network | Yes          | No              |

All four models were trained using the same training dataset and preprocessing procedure.

At this stage, no conclusions are made about which model performs best.

---

## Phase 5 — Model Evaluation

*To be completed.*

The trained models will be evaluated on the held-out test set.

The following metrics will be considered:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Precision-Recall AUC
* Confusion Matrix

Particular attention will be given to:

* False Positives
* False Negatives
* Precision/Recall trade-off
* Class imbalance
* Training vs. testing performance
* Evidence of overfitting

### Results

| Model               | Accuracy | Precision | Recall |  F1 | ROC-AUC | PR-AUC |
| ------------------- | -------: | --------: | -----: | --: | ------: | -----: |
| Logistic Regression |      TBD |       TBD |    TBD | TBD |     TBD |    TBD |
| KNN                 |      TBD |       TBD |    TBD | TBD |     TBD |    TBD |
| Decision Tree       |      TBD |       TBD |    TBD | TBD |     TBD |    TBD |
| MLP                 |      TBD |       TBD |    TBD | TBD |     TBD |    TBD |

### Confusion Matrices

*To be added after evaluation.*

### Model Comparison

*To be completed after evaluation.*

---

## After Training Analysis

### Was the Initial Hypothesis Correct?

*To be completed after Phase 5.*

### Which Model Performed Best?

*To be completed after Phase 5.*

### Which Metric Was Most Informative?

*To be completed after Phase 5.*

### How Did Class Imbalance Affect the Results?

*To be completed after Phase 5.*

### What Was the Trade-off Between False Positives and False Negatives?

*To be completed after Phase 5.*

### Did the Decision Tree Overfit?

*To be completed after Phase 5.*

### Did the MLP Overfit?

*To be completed after Phase 5.*

````

