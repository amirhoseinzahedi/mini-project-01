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


> Did the MLP overfit?

**"Not investigated in this experiment"**

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


## Phase 6 — 5-Fold Stratified Cross Validation

To evaluate the stability of the baseline models across different
training and validation splits, 5-Fold Stratified Cross Validation
was performed.

`StratifiedKFold` was used because the dataset is highly imbalanced.
Stratification ensures that each fold maintains approximately the
same proportion of legitimate and fraudulent transactions.

The following models were evaluated:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree

The evaluation metrics were:

- Precision
- Recall
- F1-score

Feature scaling was performed inside each cross-validation fold
using a `Pipeline`. Therefore, the scaler was fitted only on the
training portion of each fold and was not influenced by the
validation portion.

### Cross-Validation Results

| Model | Mean Precision | Mean Recall | Mean F1 |
|---|---:|---:|---:|
| Logistic Regression | 87.02% | 62.00% | 72.32% |
| K-Nearest Neighbors | **93.74%** | 77.44% | **84.79%** |
| Decision Tree | 74.49% | **77.24%** | 75.81% |

### Cross-Validation Standard Deviation

| Model | Std Precision | Std Recall | Std F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.0325 | 0.0379 | 0.0285 |
| K-Nearest Neighbors | 0.0455 | 0.0244 | 0.0302 |
| Decision Tree | 0.0332 | 0.0169 | 0.0224 |

### Fold-Level Results

#### Logistic Regression

| Fold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 1 | 87.50% | 56.57% | 68.71% |
| 2 | 82.67% | 62.63% | 71.26% |
| 3 | 86.57% | 59.18% | 70.30% |
| 4 | 92.65% | 64.29% | 75.90% |
| 5 | 85.71% | 67.35% | 75.43% |
| **Mean** | **87.02%** | **62.00%** | **72.32%** |

#### K-Nearest Neighbors

| Fold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 1 | 87.21% | 75.76% | 81.08% |
| 2 | 100.00% | 78.79% | 88.14% |
| 3 | 97.50% | 79.59% | 87.64% |
| 4 | 92.86% | 79.59% | 85.71% |
| 5 | 91.14% | 73.47% | 81.36% |
| **Mean** | **93.74%** | **77.44%** | **84.79%** |

#### Decision Tree

| Fold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 1 | 74.75% | 74.75% | 74.75% |
| 2 | 75.25% | 76.77% | 76.00% |
| 3 | 76.47% | 79.59% | 78.00% |
| 4 | 77.78% | 78.57% | 78.17% |
| 5 | 68.18% | 76.53% | 72.12% |
| **Mean** | **74.49%** | **77.24%** | **75.81%** |

### Interpretation

KNN achieved the highest mean Precision (93.74%) and the highest
mean F1-score (84.79%) across the five folds.

Its mean Recall was 77.44%, which was only slightly higher than the
Decision Tree's 77.24% and substantially higher than Logistic
Regression's 62.00%.

The relatively small standard deviation of KNN's Recall (0.0244)
indicates that its Recall was reasonably consistent across the five
folds.

The cross-validation results are also consistent with the held-out
test-set results from Phase 5. KNN achieved an F1-score of 85.87% on
the test set and a mean F1-score of 84.79% during cross-validation.

This consistency provides additional evidence that KNN is a strong
candidate for the final model.

### Model Selection Implication

Based on the 5-Fold Stratified Cross Validation results, KNN is the
strongest traditional machine learning model among the three
evaluated models.

KNN achieved the highest mean F1-score and the highest mean
Precision while maintaining strong Fraud Recall.

These results will be considered together with the held-out test-set
results when selecting the final model.

# Phase 7 — Mandatory Experiments

## Experiment 1 — Effect of Feature Scaling

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

Decision Trees are less sensitive to scaling because they make decisions using threshold comparisons, not distances. For example, sample_feature < 30 can become Scaled_sample_feature < 0.25 after scaling—the tree can simply adjust its threshold and make the same split. Therefore, scaling usually doesn't change a Decision Tree's decisions, unlike KNN, where scaling directly changes distance calculations.

---

## Experiment 2A — KNN Hyperparameter Analysis

### Investigation

This experiment investigates how the `n_neighbors` hyperparameter
affects KNN performance.

The following values were compared:

- K = 1
- K = 5
- K = 20

### Hypothesis

A very small value of K may make KNN sensitive to individual
observations and increase variance. Increasing K should make the
decision boundary smoother, but an excessively large K may cause
the model to become too general and fail to detect minority fraud
patterns effectively.

### Results

| K | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 1 | 86.96% | 81.63% | 84.21% |
| 5 | 91.86% | 80.61% | 85.87% |
| 20 | 84.52% | 72.45% | 78.02% |

### Interpretation

K=1 achieved the highest Recall, but its Precision was lower than
K=5.

K=5 achieved the highest F1-score and the highest Precision among
the tested values, while maintaining strong Recall.

Increasing K to 20 resulted in lower Precision, Recall, and F1-score.

This suggests that K=20 produced a decision boundary that was too
smooth to effectively capture some of the minority fraud patterns.

Among the tested values, K=5 produced the strongest overall balance
between Precision and Recall.

This result will be considered later during model selection.

---

## Experiment 2B — Decision Tree Hyperparameter Analysis

### Investigation

This experiment investigates how the `max_depth` hyperparameter
affects Decision Tree complexity, generalization, and overfitting.

The following values were compared:

- max_depth = 2
- max_depth = 5
- max_depth = 10
- max_depth = None

### Hypothesis

A shallow tree may underfit because it cannot represent sufficiently
complex relationships.

Increasing the maximum depth should initially improve performance,
but excessive depth may cause the model to memorize the training
data and overfit.

### Results

| max_depth | Train Precision | Train Recall | Train F1 | Test Precision | Test Recall | Test F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 83.66% | 75.38% | 79.31% | 76.53% | 76.53% | 76.53% |
| 5 | 93.51% | 80.46% | 86.49% | 89.41% | 77.55% | 83.06% |
| 10 | 100.00% | 85.53% | 92.20% | 89.02% | 74.49% | 81.11% |
| None | 100.00% | 100.00% | 100.00% | 75.26% | 74.49% | 74.87% |

### Interpretation

At max_depth=2, the Decision Tree had relatively low training and
test performance, suggesting that the model was too constrained
to capture all relevant patterns.

Increasing the depth to 5 improved both training and test
performance. The test F1-score increased to 83.06%.

At max_depth=10, training F1 increased to 92.20%, while test F1
decreased to 81.11%. The increasing difference between training
and test performance provides evidence of overfitting.

The unrestricted tree provides the clearest example of overfitting.
It achieved a training F1-score of 100%, but its test F1-score was
only 74.87%.

Among the tested values, max_depth=5 achieved the highest test
F1-score.

This result will be considered later during model selection.

---

## Experiment 3 — Classification Threshold

### Investigation

This experiment investigates how changing the classification
threshold affects fraud detection performance.

Logistic Regression was used because it produces class
probabilities that can be converted into predictions using a
custom threshold.

The following thresholds were compared:

- 0.3
- 0.5
- 0.7

### Hypothesis

Lowering the threshold should increase Fraud Recall because more
transactions will be classified as fraudulent.

However, this should also increase False Positives and reduce
Precision.

Increasing the threshold should have the opposite effect.

### Results

| Threshold | Precision | Recall | F1 | False Positives | False Negatives |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 73.12% | 69.39% | 71.20% | 25 | 30 |
| 0.5 | 82.67% | 63.27% | 71.68% | 13 | 36 |
| 0.7 | 83.10% | 60.20% | 69.82% | 12 | 39 |

### Interpretation

Lowering the threshold from 0.5 to 0.3 increased Recall from 63.27%
to 69.39% and reduced False Negatives from 36 to 30.

However, Precision decreased from 82.67% to 73.12%, while False
Positives increased from 13 to 25.

Increasing the threshold to 0.7 had the opposite effect. Precision
increased slightly to 83.10%, while Recall decreased to 60.20%.
False Positives decreased to 12, but False Negatives increased to
39.

This demonstrates the Precision-Recall trade-off in fraud detection.

A lower threshold favors fraud detection and reduces missed fraud,
but generates more false alarms. A higher threshold reduces false
alarms but increases the number of missed fraud transactions.

No final threshold was selected at this stage. The appropriate
threshold depends on the relative costs of False Positives and
False Negatives and will be considered during the later model
selection phase.

## Phase 8 — Final Model Selection

### Final Model

The final model selected for this project is **K-Nearest Neighbors (KNN) with K=5**.

The decision was based on the combined results from the test-set evaluation, hyperparameter experiments, and 5-fold stratified cross-validation.

### Model Comparison

| Model                         | Test Precision | Test Recall |    Test F1 | CV Mean Precision | CV Mean Recall | CV Mean F1 |
| ----------------------------- | -------------: | ----------: | ---------: | ----------------: | -------------: | ---------: |
| Logistic Regression           |         0.8267 |      0.6327 |     0.7168 |            0.8702 |         0.6200 |     0.7232 |
| Decision Tree (`max_depth=5`) |         0.8941 |      0.7755 |     0.8306 |            0.7449 |         0.7724 |     0.7581 |
| **KNN (`K=5`)**               |     **0.9186** |  **0.8061** | **0.8587** |        **0.9374** |     **0.7744** | **0.8479** |

KNN achieved the highest test Precision, Recall, and F1-score among the evaluated models. It also achieved the highest mean Precision, Recall, and F1-score during 5-fold stratified cross-validation.

This consistency between cross-validation and test-set performance provides stronger evidence for selecting KNN than relying only on the performance of a single test split.

### KNN Hyperparameter Experiment

Different values of K were evaluated:

|     K |  Precision | Recall |   F1-score |
| ----: | ---------: | -----: | ---------: |
|     1 |     0.8696 | 0.8163 |     0.8421 |
| **5** | **0.9186** | 0.8061 | **0.8587** |
|    20 |     0.8452 | 0.7245 |     0.7802 |

K=5 provided the highest Precision and F1-score while maintaining high Recall. Increasing K to 20 resulted in noticeably worse performance.

Therefore, **K=5** was selected as the final KNN configuration.

### Overfitting Considerations

The Decision Tree experiments demonstrated the effect of excessive model complexity.

For example, with `max_depth=None`, the Decision Tree achieved:

* Training F1-score: 1.0000
* Test F1-score: 0.7487

This indicates substantial overfitting. The tree memorized the training data but generalized poorly to unseen data.

A more constrained tree with `max_depth=5` performed better, achieving a test F1-score of 0.8306. However, it still did not outperform KNN.

KNN therefore provided the strongest overall performance without the severe train/test performance gap observed in the deeper Decision Trees.

---

### Class Imbalance

The dataset is highly imbalanced, with fraudulent transactions representing only a very small fraction of all transactions.

Because of this imbalance, Accuracy is not an appropriate metric for selecting the final model. A model could achieve extremely high Accuracy by correctly classifying the large majority of legitimate transactions while still missing many fraudulent transactions.

Therefore, Precision, Recall, F1-score, and the Confusion Matrix were given greater importance during model selection.

Recall is particularly important because a false negative represents a fraudulent transaction that was not detected.

---

### Final Classification Threshold

The final classification threshold selected for KNN is:

**Threshold = 0.5**

The KNN threshold experiment produced the following results:

| Threshold |  Precision |     Recall |   F1-score | False Positives | False Negatives |
| --------: | ---------: | ---------: | ---------: | --------------: | --------------: |
|       0.3 |     0.8646 | **0.8469** |     0.8557 |              13 |          **15** |
|       0.4 |     0.8646 | **0.8469** |     0.8557 |              13 |          **15** |
|   **0.5** | **0.9186** |     0.8061 | **0.8587** |           **7** |              19 |
|       0.6 |     0.9186 |     0.8061 |     0.8587 |               7 |              19 |
|       0.7 | **0.9595** |     0.7245 |     0.8256 |           **3** |              27 |

For KNN with K=5, the predicted probabilities are discrete because they are based on the five nearest neighbors. Therefore, some thresholds produce identical predictions. For example, thresholds 0.3 and 0.4 produce the same results, as do thresholds 0.5 and 0.6.

Threshold 0.5 was selected because it provides the highest F1-score while maintaining high Precision and Recall. It also produces only 7 false positives while detecting 79 of the 98 fraudulent transactions in the test set.

A lower threshold such as 0.3 detects four additional fraudulent transactions, reducing False Negatives from 19 to 15. However, it also increases False Positives from 7 to 13. Therefore, 0.5 provides a better overall balance between the False Positive and False Negative trade-off for this project.

### Final Decision

The final configuration selected for this project is:

* **Model:** K-Nearest Neighbors
* **K:** 5
* **Classification Threshold:** 0.5

The final model was selected based on the combined evidence from test-set performance, cross-validation, hyperparameter experiments, overfitting behavior, class imbalance, and the False Positive/False Negative trade-off.

KNN with K=5 achieved the strongest overall balance among the evaluated models rather than being selected simply because of Accuracy.

## Phase 9 — Model Saving

After selecting the final model and classification threshold, the components required for the final prediction pipeline were saved.

### Final Model

The selected model from Phase 8 was:

* **Model:** K-Nearest Neighbors
* **K:** 5
* **Classification Threshold:** 0.5

The final KNN model was retrained using the complete dataset and saved using `joblib`.

### Saved Components

The `models/` directory contains:

```text
models/
├── model.pkl
└── scaler.pkl
```

* `model.pkl` contains the final KNN model.
* `scaler.pkl` contains the `StandardScaler` used to transform the input features.

For the final deployment model, the scaler was fitted on the complete feature dataset before the final KNN model was trained.

### Reproducibility

The preprocessing used during prediction must be identical to the preprocessing used during training.

The final pipeline therefore applies the saved `StandardScaler` to a new transaction before passing it to the saved KNN model.

This prevents the prediction process from using a different scaling procedure from the one used when training the model.

---

## Phase 10 — Prediction Script

A reusable prediction script was implemented in:

```text
src/predict.py
```

The script receives a transaction as a JSON object and performs the complete prediction pipeline.

### Prediction Pipeline

```text
JSON Transaction
       ↓
Feature Validation
       ↓
Create DataFrame
       ↓
Load scaler.pkl
       ↓
Scale Features
       ↓
Load model.pkl
       ↓
KNN Prediction
       ↓
Fraud Probability
       ↓
Apply Threshold = 0.5
       ↓
JSON Output
```

The script validates that the transaction contains exactly the 30 features used during training:

* `Time`
* `V1` through `V28`
* `Amount`

The transaction is then transformed using the saved `StandardScaler`.

The transformed transaction is passed to the final KNN model, and `predict_proba()` is used to obtain the fraud-class score. The selected classification threshold of `0.5` is then applied to produce the final prediction.

### Prediction Output

The prediction script returns JSON-compatible output containing:

* `prediction`
* `fraud_probability`
* `is_fraud`
* `threshold`

For example:

```json
{
  "prediction": 1,
  "fraud_probability": 1.0,
  "is_fraud": true,
  "threshold": 0.5
}
```

For KNN with `K=5`, the fraud score is determined by the proportion of the five nearest neighbors belonging to the fraud class. Therefore, the possible scores are discrete values such as `0.0`, `0.2`, `0.4`, `0.6`, `0.8`, and `1.0`.

### Prediction Pipeline Test

The final prediction pipeline was tested using transactions from the original dataset.

A known legitimate transaction produced:

```text
prediction: 0
fraud_probability: 0.0
is_fraud: False
```

A known fraudulent transaction produced:

```text
prediction: 1
fraud_probability: 1.0
is_fraud: True
```

These tests verified that the saved model and scaler can be loaded successfully and that new transactions pass through the same preprocessing and prediction pipeline before the final classification threshold is applied.

````