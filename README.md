# Mini Project 01 — Credit Card Fraud Detection

## Project Overview

This project explores credit card fraud detection as a **binary classification problem** using the Credit Card Fraud Detection dataset.

The main objective is to investigate how different machine learning algorithms behave when detecting a highly imbalanced minority class.

The project compares:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Simple MLP / Neural Network using PyTorch

The project is being developed incrementally, with each major phase committed to Git.

---

# Phase 3 — Hypothesis Before Modeling

The following hypotheses were established **before evaluating the trained models**.

## 1. Which model do you expect to perform best?

I expect **Logistic Regression** to provide a strong baseline because this is a binary classification problem and Logistic Regression is simple, interpretable, and computationally efficient.

However, I also expect the **Decision Tree** to potentially outperform Logistic Regression if fraud patterns contain nonlinear relationships or interactions between features.

For **KNN**, I expect performance to depend heavily on feature scaling because KNN relies on distances between observations.

I also expect the **MLP** to be capable of learning nonlinear relationships, but its performance may be strongly affected by the severe class imbalance.

My initial expectation is therefore:

```text
Logistic Regression → strong baseline
Decision Tree       → potentially strongest classical model
KNN                 → potentially competitive, but sensitive to scaling
MLP                 → capable of nonlinear learning, but sensitive to imbalance
```

These hypotheses will only be considered confirmed or rejected after model evaluation.

---

## 2. Which metric is more important: Precision, Recall, or F1-score?

I expect **Recall** to be particularly important for this problem.

Recall answers:

> Of all the actual fraudulent transactions, how many did the model detect?

[
Recall = \frac{TP}{TP + FN}
]

A **False Negative** occurs when a fraudulent transaction is classified as legitimate. In a fraud-detection system, failing to detect fraud can have a significant cost.

However, maximizing Recall alone is not sufficient.

A model could classify a very large number of transactions as fraudulent and achieve high Recall while generating many False Positives.

Therefore, I will consider:

* **Recall** — important for minimizing missed fraud
* **Precision** — important for controlling false alarms
* **F1-score** — useful for balancing Precision and Recall
* **PR-AUC** — particularly useful because of the severe class imbalance

My initial hypothesis is:

> **Recall will be the most important individual metric, but the final model should balance Recall and Precision rather than maximizing Recall alone.**

---

## 3. What happens if the model predicts all transactions as legitimate?

If the model predicts:

```text
Every transaction → Class 0
```

it will correctly classify approximately **99.83%** of the transactions because the dataset is extremely imbalanced.

Therefore, the accuracy would be approximately:

```text
99.83%
```

However:

```text
True Positives  = 0
False Negatives = 492
```

Consequently:

```text
Recall = 0
```

The model would detect **zero fraudulent transactions** despite having extremely high accuracy.

This demonstrates why accuracy alone is misleading for this problem.

My hypothesis is therefore:

> **A model with extremely high accuracy can still be completely ineffective for fraud detection.**

---

## 4. Do I expect feature scaling to significantly affect KNN?

**Yes.**

KNN determines which observations are close to one another using a distance metric.

For example:

```text
Distance =
√((x₁-a₁)² + (x₂-a₂)² + ...)
```

Features with larger numerical scales can therefore have a disproportionate influence on the distance calculation.

In this dataset, `Time` and `Amount` have different scales from the PCA-transformed `V1`–`V28` features.

I therefore expect:

```text
Unscaled KNN
     ↓
Larger-scale features have greater influence
     ↓
Potentially worse distance-based classification
```

while:

```text
Scaled KNN
     ↓
Features are placed on comparable scales
     ↓
Potentially better KNN performance
```

My hypothesis is:

> **Feature scaling will significantly affect KNN performance.**

To avoid data leakage, scaling is performed only after the train/test split, with the scaler fitted exclusively on the training data.

---

## 5. Do I expect the Decision Tree to overfit?

**Yes, particularly when the tree is allowed to grow without constraints.**

A Decision Tree can repeatedly split the training data until it creates highly specific rules for individual observations.

A sufficiently deep tree could therefore produce:

```text
Training performance → extremely high
Testing performance  → significantly lower
```

which would indicate overfitting.

The baseline Decision Tree is therefore intentionally unrestricted:

```python
DecisionTreeClassifier(
    random_state=42
)
```

My hypothesis is:

> **An unrestricted Decision Tree will have a higher risk of overfitting than a constrained tree.**

This behavior will be investigated during model evaluation.

---

## 6. MLP / Neural Network Hypothesis

The project also includes a simple feed-forward neural network implemented using PyTorch.

The baseline architecture is:

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

I expect the MLP to be capable of learning nonlinear relationships between the features and the fraud class.

However, because fraudulent transactions represent only a very small fraction of the dataset, I expect the baseline MLP to be affected substantially by class imbalance.

I also expect the MLP to have some risk of overfitting because it has considerably more trainable parameters than the simpler baseline models.

---

# Initial Hypotheses Summary

| Question                     | Initial Hypothesis                                                                  |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| Expected strongest model     | Decision Tree or Logistic Regression among classical models                         |
| MLP                          | Potentially strong nonlinear model, but sensitive to class imbalance                |
| Most important metric        | Recall, while monitoring Precision, F1-score and PR-AUC                             |
| All predictions = legitimate | Very high accuracy but zero fraud detection                                         |
| Scaling and KNN              | Scaling should significantly affect performance                                     |
| Decision Tree                | Unrestricted tree is likely to overfit                                              |
| MLP                          | Capable of nonlinear learning but potentially affected by imbalance and overfitting |

---

# Phase 4 — Model Training

Four baseline models were implemented and trained.

## Model 1 — Logistic Regression

Purpose:

* Establish a simple classification baseline.
* Analyze the behavior of a linear classifier on an imbalanced dataset.

Configuration:

```python
LogisticRegression(
    random_state=42,
    max_iter=1000
)
```

---

## Model 2 — K-Nearest Neighbors

Purpose:

* Analyze distance-based learning.
* Study the impact of feature scaling.

Configuration:

```python
KNeighborsClassifier(
    n_neighbors=5
)
```

The model was trained using the standardized feature values.

---

## Model 3 — Decision Tree Classifier

Purpose:

* Explore nonlinear decision boundaries.
* Analyze overfitting behavior.
* Investigate the effect of class imbalance.

Configuration:

```python
DecisionTreeClassifier(
    random_state=42
)
```

No `max_depth` constraint was applied in the baseline experiment so that potential overfitting could be investigated.

---

## Model 4 — Simple MLP / Neural Network

Purpose:

* Explore nonlinear decision boundaries.
* Investigate neural-network behavior on tabular data.
* Analyze potential overfitting.
* Compare a neural-network approach with traditional machine learning models.

Framework:

```text
PyTorch
```

Architecture:

```text
Input (30 features)
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

Training configuration:

```text
Optimizer: Adam
Learning Rate: 0.001
Epochs: 20
Batch Size: 256
Loss: BCEWithLogitsLoss
Class Weighting: Not applied
```

Class weighting and other imbalance-handling techniques were intentionally not applied during the baseline experiment.

---

## Baseline Training Strategy

All baseline models use the same fundamental preprocessing procedure:

```text
Dataset
   ↓
Stratified Train/Test Split
   ↓
Fit StandardScaler on Training Data
   ↓
Transform Training Data
   ↓
Transform Test Data
   ↓
Train Model
```

No oversampling, undersampling, SMOTE, or class weighting is used in the baseline training stage.

This allows the effect of class imbalance to be investigated before introducing additional techniques.

---

# Phase 5 — Model Evaluation

*Model evaluation has not yet been completed.*

The models will be compared using metrics appropriate for the imbalanced classification problem, including:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC
* PR-AUC

The evaluation will focus particularly on:

* False Positives
* False Negatives
* The Precision/Recall trade-off
* The effect of class imbalance
* Training vs. testing performance
* Evidence of overfitting

---

## After Training Analysis

This section will be completed after the models have been evaluated.

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
