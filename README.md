# Phase 3 — Hypothesis Before Modeling

## 1. Which model do you expect to perform best?

I expect **Logistic Regression** to provide a strong baseline because this is a binary classification problem and Logistic Regression is simple, interpretable, and often performs well when the classes can be separated reasonably well in feature space.

However, I also expect the **Decision Tree** to potentially outperform Logistic Regression if fraud patterns contain nonlinear relationships or feature interactions.

For **KNN**, I expect performance to depend heavily on the feature representation and scaling because KNN relies on distances between observations.

My initial expectation is therefore:

```text
Logistic Regression → strong baseline
Decision Tree       → potentially best
KNN                 → potentially competitive, but sensitive to scaling
```

I will not consider this hypothesis confirmed until the models are evaluated on the test set.

---

## 2. Which metric is more important: Precision, Recall, or F1-score?

I expect **Recall** to be particularly important for this problem.

Recall answers:

> Of all the actual fraudulent transactions, how many did the model detect?

[
Recall = \frac{TP}{TP + FN}
]

A **False Negative** means that a fraudulent transaction was classified as legitimate. In a fraud-detection system, this can be costly.

However, maximizing Recall alone is not sufficient.

A model could classify almost every transaction as fraud and achieve very high Recall, while generating a huge number of False Positives.

Therefore, I will consider:

* **Recall** — important for minimizing missed fraud
* **Precision** — important for controlling false alarms
* **F1-score** — useful for evaluating the balance between Precision and Recall

Because the dataset is extremely imbalanced, I will also consider **Precision-Recall AUC (PR-AUC)** when comparing models.

My initial hypothesis is:

> **Recall will be the most important individual metric, but the final model should balance Recall and Precision rather than maximizing Recall alone.**

---

## 3. What happens if the model predicts all transactions as legitimate?

If the model predicts:

```text
Every transaction → Class 0
```

it will correctly classify almost all transactions because approximately **99.83%** of the dataset consists of legitimate transactions.

Therefore, its accuracy would be approximately:

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

The model would detect **zero fraudulent transactions** despite its apparently excellent accuracy.

This demonstrates why **accuracy is misleading for this highly imbalanced classification problem**.

My hypothesis is therefore:

> A model with extremely high accuracy can still be completely useless for fraud detection.

---

## 4. Do I expect feature scaling to significantly affect KNN?

**Yes.**

KNN determines which observations are close to one another using a distance metric, commonly Euclidean distance.

For example:

```text
Distance =
√((x₁-a₁)² + (x₂-a₂)² + ...)
```

Therefore, features with larger numerical scales can dominate the distance calculation.

In this dataset, `Amount` and `Time` have very different scales from the PCA-transformed `V1`–`V28` features.

I therefore expect:

```text
Unscaled KNN
     ↓
Distance dominated by larger-scale features
     ↓
Potentially worse performance
```

while:

```text
Scaled KNN
     ↓
More balanced feature contribution
     ↓
Potentially better performance
```

My hypothesis is:

> **Feature scaling will have a significant effect on KNN performance.**

This is also one reason why we correctly performed scaling **after the train/test split** in Phase 2.

---

## 5. Do I expect the Decision Tree to overfit?

**Yes, particularly if the tree is allowed to grow without constraints.**

A Decision Tree can repeatedly split the training data until it creates highly specific rules for individual observations.

A sufficiently deep tree could therefore achieve something close to:

```text
Training performance → extremely high
Testing performance  → significantly lower
```

This would indicate overfitting.

I therefore expect an unrestricted:

```python
DecisionTreeClassifier(random_state=42)
```

to have a higher risk of overfitting than a constrained tree such as:

```python
DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)
```

My hypothesis is:

> **A deep or unrestricted Decision Tree will likely overfit the training data, while limiting `max_depth` should improve generalization.**

---

# Initial Hypotheses Summary

| Question                     | Initial Hypothesis                              |
| ---------------------------- | ----------------------------------------------- |
| Expected strongest model     | Logistic Regression or Decision Tree            |
| Most important metric        | Recall, while monitoring Precision and F1       |
| All predictions = legitimate | Very high accuracy but zero fraud detection     |
| Scaling and KNN              | Scaling should significantly affect performance |
| Decision Tree                | Unrestricted tree is likely to overfit          |


## After Training Analysis

This section will be completed after training and evaluating the models.

### Was the Initial Hypothesis Correct?

_To be completed after model evaluation._

### Which Model Performed Best?

_To be completed after model evaluation._

### Which Metric Was Most Informative?

_To be completed after model evaluation._

### How Did Class Imbalance Affect the Results?

_To be completed after model evaluation._

### False Positive vs False Negative Trade-off

_To be completed after model evaluation._