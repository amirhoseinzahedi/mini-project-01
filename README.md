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

### Was the initial hypothesis correct?

The initial hypotheses were mostly supported by the experiments.

Logistic Regression provided a useful baseline, achieving 99.91% Accuracy, 82.67% Precision, 63.27% Recall, and 71.68% F1-score. This confirmed that a simple linear classifier can detect a significant portion of fraudulent transactions, but its fraud detection performance was weaker than the other models.

The hypothesis that KNN would be affected by feature scaling was strongly supported. When KNN was trained on unscaled features, it achieved 100% Precision but only 3.06% Recall and a 5.94% F1-score. It detected only 3 of the 98 fraudulent transactions. After standardization, Recall increased to 80.61% and F1-score increased to 85.87%. This demonstrates that feature scaling is particularly important for distance-based algorithms such as KNN.

The hypothesis that the Decision Tree could overfit was also supported. The Decision Tree achieved perfect performance on the training set, with 100% Accuracy, Precision, Recall, and F1-score. However, on the test set its F1-score dropped to 74.87%, with Precision of 75.26% and Recall of 74.49%. This large train-test performance gap provides clear evidence of overfitting.

The optional MLP also performed well. It achieved 99.95% Accuracy, 84.54% Precision, 83.67% Recall, and 84.10% F1-score. Its performance was comparable to KNN, although its Recall was slightly higher.

### Which model performed best?

There is no single best model for every metric.

**KNN achieved the highest Precision and F1-score**, with:

* Precision: 91.86%
* Recall: 80.61%
* F1-score: 85.87%
* False Positives: 7
* False Negatives: 19

Therefore, KNN provided the best overall balance between Fraud Precision and Fraud Recall according to F1-score.

However, **the MLP achieved the highest Fraud Recall**, detecting 82 of the 98 fraudulent transactions:

* Precision: 84.54%
* Recall: 83.67%
* F1-score: 84.10%
* False Positives: 15
* False Negatives: 16

Therefore, if minimizing missed fraud is the primary objective, the MLP would be preferable. If a balance between detecting fraud and limiting false alarms is preferred, KNN performed best according to F1-score.

### Which metric was most informative?

**Fraud Recall was one of the most informative metrics for this problem** because it measures the proportion of actual fraudulent transactions that were successfully detected.

This is especially important because a False Negative represents a fraudulent transaction that the model incorrectly classified as legitimate.

For example, Logistic Regression achieved 99.91% Accuracy but detected only 62 of the 98 fraudulent transactions, resulting in 36 False Negatives and a Fraud Recall of only 63.27%.

The KNN scaling experiment provides an even stronger example. Unscaled KNN achieved 99.83% Accuracy and 100% Precision, but its Recall was only 3.06%. It detected only 3 fraudulent transactions while missing 95.

F1-score was also highly informative because it combines Precision and Recall. KNN achieved the highest F1-score of 85.87%, indicating the best balance between detecting fraud and limiting false fraud alerts.

Therefore, Accuracy alone was not an appropriate primary metric for this highly imbalanced dataset.

### How did class imbalance affect the results?

The test set contained 56,864 legitimate transactions but only 98 fraudulent transactions. Fraud therefore represented only a very small portion of the test set.

Because legitimate transactions dominated the dataset, all four baseline models achieved approximately 99.9% Accuracy:

| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   99.91% |    82.67% | 63.27% |   71.68% |
| KNN                 |   99.95% |    91.86% | 80.61% |   85.87% |
| Decision Tree       |   99.91% |    75.26% | 74.49% |   74.87% |
| MLP                 |   99.95% |    84.54% | 83.67% |   84.10% |

Despite the very similar Accuracy values, the models differed substantially in their ability to detect the minority fraud class.

The KNN scaling experiment further demonstrates this problem. Unscaled KNN achieved 99.83% Accuracy while detecting only 3 of 98 fraudulent transactions. Therefore, a high Accuracy score can hide very poor fraud detection performance.

### What was the trade-off between False Positives and False Negatives?

The results demonstrate a trade-off between generating false fraud alerts and missing actual fraudulent transactions.

KNN produced the fewest False Positives among the baseline models, with only 7, and also achieved the highest Precision at 91.86%. However, it still missed 19 fraudulent transactions.

The MLP produced 15 False Positives but had the fewest False Negatives among the baseline models, with only 16. It therefore achieved the highest Fraud Recall at 83.67%.

Logistic Regression produced 13 False Positives but missed 36 fraudulent transactions, resulting in the lowest Fraud Recall among the four models.

The Decision Tree produced the highest number of False Positives, with 24, and also missed 25 fraudulent transactions.

The KNN scaling experiment demonstrates the trade-off particularly clearly. Unscaled KNN produced zero False Positives, but this came at the cost of 95 False Negatives. After scaling, it produced 7 False Positives but reduced False Negatives from 95 to 19.

This shows that reducing False Positives alone is not necessarily desirable in fraud detection. A model that rarely raises false alarms but misses most fraudulent transactions may be less useful than a model that generates some false alarms while successfully detecting a larger proportion of fraud.

Overall, the appropriate balance depends on the relative cost of False Positives and False Negatives. In a fraud detection system, missing an actual fraudulent transaction can be particularly costly, so Fraud Recall should receive significant attention alongside Precision and F1-score.


