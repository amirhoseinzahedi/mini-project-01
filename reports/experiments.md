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

## Phase 4 — Model Training Summary

| Model               | Algorithm Type | Scaling Used | Class Weighting |
| ------------------- | -------------- | ------------ | --------------- |
| Logistic Regression | Linear         | Yes          | No              |
| KNN                 | Distance-based | Yes          | No              |
| Decision Tree       | Tree-based     | Yes          | No              |
| MLP                 | Neural Network | Yes          | No              |

All four models were trained using the same training dataset and preprocessing procedure.

At this stage, no conclusions are made about which model performs best.

---

Your current `experiments.md` is already well structured. I would **not rewrite Phases 1–4**. We only need to complete Phase 5 and make one small correction to the earlier sections.

One important point: you **did not actually run a dedicated MLP train-vs-test experiment**, so the current placeholder:

> Did the MLP overfit?

should remain either **"Not investigated in this experiment"** or be removed. Don't claim MLP overfitting based on test results alone.

Also, you haven't calculated ROC-AUC or PR-AUC, so those should **not remain as TBD requirements** unless you intend to implement them. Your assignment's required metrics are Accuracy, Precision, Recall, F1, and Confusion Matrix. ROC-AUC/PR-AUC were your earlier proposed metrics, not required by the Phase 5 instruction you showed me.

````markdown
## Phase 5 — Model Evaluation

The trained models were evaluated on the held-out test set.

The test set contained:

- Total transactions: 56,962
- Legitimate transactions: 56,864
- Fraudulent transactions: 98

Because the dataset is highly imbalanced, Accuracy was not used as the primary metric.

The following metrics were evaluated:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- False Positives
- False Negatives

Particular attention was given to Fraud Precision and Fraud Recall.

---

### Baseline Model Results

| Model | Accuracy | Precision | Recall | F1-score | False Positives | False Negatives |
| ------------------- | -------: | --------: | -----: | -------: | --------------: | ---------------: |
| Logistic Regression | 99.91% | 82.67% | 63.27% | 71.68% | 13 | 36 |
| KNN | 99.95% | 91.86% | 80.61% | 85.87% | 7 | 19 |
| Decision Tree | 99.91% | 75.26% | 74.49% | 74.87% | 24 | 25 |
| MLP | 99.95% | 84.54% | 83.67% | 84.10% | 15 | 16 |

---

### Confusion Matrices

#### Logistic Regression

```text
[[56851    13]
 [   36    62]]
````

|                   | Predicted Legitimate | Predicted Fraud |
| ----------------- | -------------------: | --------------: |
| Actual Legitimate |               56,851 |              13 |
| Actual Fraud      |                   36 |              62 |

* True Negatives: 56,851
* False Positives: 13
* False Negatives: 36
* True Positives: 62

---

#### K-Nearest Neighbors

```text
[[56857     7]
 [   19    79]]
```

|                   | Predicted Legitimate | Predicted Fraud |
| ----------------- | -------------------: | --------------: |
| Actual Legitimate |               56,857 |               7 |
| Actual Fraud      |                   19 |              79 |

* True Negatives: 56,857
* False Positives: 7
* False Negatives: 19
* True Positives: 79

---

#### Decision Tree

```text
[[56840    24]
 [   25    73]]
```

|                   | Predicted Legitimate | Predicted Fraud |
| ----------------- | -------------------: | --------------: |
| Actual Legitimate |               56,840 |              24 |
| Actual Fraud      |                   25 |              73 |

* True Negatives: 56,840
* False Positives: 24
* False Negatives: 25
* True Positives: 73

---

#### MLP

```text
[[56849    15]
 [   16    82]]
```

|                   | Predicted Legitimate | Predicted Fraud |
| ----------------- | -------------------: | --------------: |
| Actual Legitimate |               56,849 |              15 |
| Actual Fraud      |                   16 |              82 |

* True Negatives: 56,849
* False Positives: 15
* False Negatives: 16
* True Positives: 82

---

## KNN Scaling Experiment

Because KNN is a distance-based algorithm, an additional experiment was performed to investigate the effect of feature scaling.

Two KNN models were trained using the same training and test split:

1. KNN using the original, unscaled features.
2. KNN using standardized features.

### Results

| Version  | Accuracy | Precision | Recall | F1-score | False Positives | False Negatives |
| -------- | -------: | --------: | -----: | -------: | --------------: | --------------: |
| Unscaled |   99.83% |   100.00% |  3.06% |    5.94% |               0 |              95 |
| Scaled   |   99.95% |    91.86% | 80.61% |   85.87% |               7 |              19 |

### Observation

The effect of scaling was substantial.

The unscaled KNN produced zero False Positives and therefore achieved 100% Precision. However, it detected only 3 of the 98 fraudulent transactions and missed 95 fraud cases.

After scaling, Fraud Recall increased from 3.06% to 80.61%, while F1-score increased from 5.94% to 85.87%.

This demonstrates that feature scaling is critical for KNN in this dataset because distance calculations are affected by the scale of the features.

The result also demonstrates why Precision and Accuracy should not be considered in isolation. The unscaled model appeared strong according to these metrics but was extremely poor at detecting fraud.

---

## Decision Tree Overfitting Experiment

To investigate potential overfitting, the Decision Tree was evaluated on both the training set and the held-out test set.

### Results

| Dataset  | Accuracy | Precision |  Recall | F1-score |
| -------- | -------: | --------: | ------: | -------: |
| Training |  100.00% |   100.00% | 100.00% |  100.00% |
| Test     |   99.91% |    75.26% |  74.49% |   74.87% |

### Observation

The Decision Tree achieved perfect performance on the training data but substantially lower Precision, Recall, and F1-score on the test data.

The F1-score decreased from 100.00% on the training set to 74.87% on the test set.

This train-test performance gap provides clear evidence that the unrestricted Decision Tree overfit the training data.

The high test Accuracy alone does not reveal this problem because the test set is dominated by legitimate transactions.

---

## Model Comparison

Based on the baseline test-set results:

* **KNN achieved the highest Precision:** 91.86%.
* **MLP achieved the highest Fraud Recall:** 83.67%.
* **KNN achieved the highest F1-score:** 85.87%.
* **KNN produced the fewest False Positives:** 7.
* **MLP produced the fewest False Negatives:** 16.

Therefore, KNN provided the best balance between Precision and Recall according to F1-score, while MLP was slightly better when the primary objective was maximizing Fraud Recall.

Logistic Regression provided a useful baseline but had the lowest Fraud Recall.

The Decision Tree achieved the lowest Precision and showed clear evidence of overfitting.

---

## Class Imbalance

The test set contained 56,864 legitimate transactions and only 98 fraudulent transactions.

Fraudulent transactions therefore represented a very small minority of the test data.

This caused all four baseline models to achieve approximately 99.9% Accuracy despite having substantially different fraud detection performance.

For example, Logistic Regression achieved 99.91% Accuracy but detected only 62 of the 98 fraudulent transactions.

The KNN scaling experiment provided an even stronger example. Unscaled KNN achieved 99.83% Accuracy but detected only 3 fraudulent transactions.

Therefore, Accuracy is not sufficient for evaluating this fraud detection problem. Fraud Recall, Precision, F1-score, and the Confusion Matrix provide much more useful information about the minority fraud class.

---

## False Positive vs. False Negative Trade-off

A False Positive occurs when a legitimate transaction is incorrectly classified as fraudulent.

A False Negative occurs when a fraudulent transaction is incorrectly classified as legitimate.

The models showed different trade-offs:

| Model               | False Positives | False Negatives |
| ------------------- | --------------: | --------------: |
| Logistic Regression |              13 |              36 |
| KNN                 |               7 |              19 |
| Decision Tree       |              24 |              25 |
| MLP                 |              15 |              16 |

KNN produced the fewest False Positives, while MLP produced the fewest False Negatives.

The KNN scaling experiment demonstrated the trade-off particularly clearly. Unscaled KNN produced zero False Positives but 95 False Negatives. After scaling, it produced 7 False Positives but reduced False Negatives to 19.

This demonstrates that minimizing False Positives alone is not sufficient for fraud detection. A model that rarely generates false alerts but misses most fraudulent transactions may be less useful than a model that generates some false alerts while detecting substantially more fraud.

---

## After Training Analysis

### Was the Initial Hypothesis Correct?

The initial hypotheses were mostly supported by the experiments.

The Logistic Regression baseline provided reasonable fraud detection performance but was weaker than KNN and MLP on the fraud-specific metrics.

The hypothesis that KNN would be sensitive to feature scaling was strongly supported. Scaling increased KNN Fraud Recall from 3.06% to 80.61% and F1-score from 5.94% to 85.87%.

The hypothesis that an unrestricted Decision Tree could overfit was also supported. The Decision Tree achieved perfect training performance but substantially lower test performance.

The MLP performed well on the test set, but MLP overfitting was not specifically investigated through a training-versus-test comparison.

---

### Which Model Performed Best?

There was no single best model across every metric.

KNN achieved the highest Precision (91.86%) and F1-score (85.87%), making it the strongest model according to the balance between Precision and Recall.

MLP achieved the highest Fraud Recall (83.67%), meaning it detected the largest proportion of actual fraudulent transactions.

Therefore:

* **KNN:** best overall balance according to F1-score.
* **MLP:** best Fraud Recall.

---

### Which Metric Was Most Informative?

Fraud Recall was particularly informative because it measures how many actual fraudulent transactions were successfully detected.

F1-score was also highly informative because it combines Precision and Recall.

Accuracy was much less informative because the severe class imbalance allowed all models to achieve approximately 99.9% Accuracy despite significant differences in fraud detection performance.

---

### How Did Class Imbalance Affect the Results?

The severe class imbalance caused Accuracy to remain very high even when a model missed many fraudulent transactions.

The unscaled KNN experiment demonstrated this particularly well: it achieved 99.83% Accuracy while detecting only 3 of 98 fraudulent transactions.

Therefore, the minority fraud class must be evaluated separately using fraud-specific metrics such as Recall, Precision, and F1-score.

---

### What Was the Trade-off Between False Positives and False Negatives?

The experiments showed that reducing False Positives can come at the cost of increasing False Negatives.

Unscaled KNN produced zero False Positives but 95 False Negatives. After scaling, it produced 7 False Positives but only 19 False Negatives.

Among the baseline models, KNN had the fewest False Positives, while MLP had the fewest False Negatives.

The appropriate balance depends on the relative cost of each type of error. In fraud detection, False Negatives are particularly important because they represent actual fraudulent transactions that were not detected.

---

### Did the Decision Tree Overfit?

Yes.

The Decision Tree achieved 100% Accuracy, Precision, Recall, and F1-score on the training set.

On the test set, these values decreased to:

* Accuracy: 99.91%
* Precision: 75.26%
* Recall: 74.49%
* F1-score: 74.87%

The large gap between training and test performance indicates overfitting.

---

### Did the MLP Overfit?

Overfitting was not directly investigated for the MLP in this experiment.

The MLP was evaluated on the held-out test set, but a training-versus-test performance comparison was not performed.

Therefore, the current results are not sufficient to conclude whether the MLP overfit.

````
