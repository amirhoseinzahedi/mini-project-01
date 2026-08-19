# Mini Project 01 — Credit Card Fraud Detection

This project investigates credit card fraud detection as a binary classification problem using several machine learning algorithms. The main focus is understanding how different models behave on a highly imbalanced dataset and how preprocessing, hyperparameters, and classification thresholds affect fraud detection performance.

The project compares Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and a simple Multi-Layer Perceptron (MLP). Additional experiments investigate feature scaling, KNN hyperparameters, Decision Tree depth, classification thresholds, and model stability using 5-Fold Stratified Cross Validation.

---

## 1. Problem Description

### Business Scenario

A financial institution needs to identify potentially fraudulent credit card transactions automatically. A fraud detection system should detect as many fraudulent transactions as possible while avoiding an excessive number of false alarms on legitimate transactions.

This creates an important trade-off:

* Missing a fraudulent transaction creates a **False Negative**.
* Incorrectly flagging a legitimate transaction creates a **False Positive**.

Because fraudulent transactions are rare, a model can achieve very high overall accuracy while still performing poorly at detecting fraud.

### Objective

The objective of this project is to build and evaluate binary classification models that can distinguish between:

* `0` — Legitimate transaction
* `1` — Fraudulent transaction

The project focuses particularly on **Precision, Recall, F1-score, and Confusion Matrix**, rather than relying on Accuracy alone.

### Dataset

The project uses the **Credit Card Fraud Detection** dataset from Kaggle.

The dataset contains:

* **284,807 transactions**
* **30 input features**
* **1 binary target variable:** `Class`
* **492 fraudulent transactions**
* **284,315 legitimate transactions**

The `V1`–`V28` features are anonymized PCA-transformed features. The remaining input features are `Time` and `Amount`.

Fraudulent transactions represent approximately **0.17%** of the complete dataset, making this a severely imbalanced classification problem.

---

## 2. Data Analysis

### Dataset Statistics

| Property                |                 Value |
| ----------------------- | --------------------: |
| Total samples           |               284,807 |
| Input features          |                    30 |
| Total columns           |                    31 |
| Legitimate transactions |               284,315 |
| Fraudulent transactions |                   492 |
| Fraud ratio             |                ~0.17% |
| Problem type            | Binary classification |

### Feature Information

The input features are:

* `Time`
* `V1` through `V28`
* `Amount`

The target variable is:

* `Class`

where:

* `Class = 0` → Legitimate
* `Class = 1` → Fraudulent

The `V1`–`V28` variables are anonymized PCA-transformed features.

### Missing-Value Analysis

No missing values were found in the dataset.

All features are numerical, which means the dataset does not require categorical encoding.

### Duplicate Analysis

Exact duplicate rows were also checked during the data-quality analysis. Duplicate observations were identified.

The duplicates were not removed from the baseline dataset. This decision was kept separate from the baseline modeling experiment so that the effect of data-cleaning decisions would not be mixed with the main modeling experiments.

### Class Distribution

| Class | Meaning    |   Count | Approx. Percentage |
| ----- | ---------- | ------: | -----------------: |
| 0     | Legitimate | 284,315 |             99.83% |
| 1     | Fraudulent |     492 |              0.17% |

The severe class imbalance is one of the most important characteristics of this dataset. It is also the main reason Accuracy is not an appropriate primary metric.

---

## 3. Data Preprocessing

### Data-Quality Checks

The following checks were performed:

* Dataset shape was inspected.
* Data types were checked.
* Missing values were checked.
* Duplicate rows were checked.
* Target class distribution was examined.
* All features were confirmed to be numerical.

### Train/Test Split

The dataset was divided into:

* **80% training data**
* **20% testing data**

The split used:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

This produced a test set containing:

* 56,962 total transactions
* 56,864 legitimate transactions
* 98 fraudulent transactions

### Stratification

Stratification was used because the dataset is severely imbalanced.

Without stratification, the relatively small fraud class could be distributed unevenly between the training and testing sets.

Using `stratify=y` preserves approximately the same fraud-to-legitimate ratio in both subsets.

### Scaling

`StandardScaler` was used to standardize the features.

The correct sequence was:

```text
Original Dataset
       ↓
Train/Test Split
       ↓
Fit StandardScaler on X_train
       ↓
Transform X_train
       ↓
Transform X_test
```

The scaler was fitted only on the training data.

### Data Leakage Prevention

Data leakage was avoided by never fitting the scaler on the complete dataset before the train/test split.

The test set therefore remained unseen during the fitting of the preprocessing parameters.

For cross-validation, scaling was performed inside each fold using a `Pipeline`. This ensured that the scaler was fitted only on the training portion of each fold and never on the corresponding validation portion.

---

## 4. Initial Hypothesis

Before evaluating the models, several hypotheses were established.

### Model Performance

* **Logistic Regression** was expected to provide a strong baseline because it is simple, interpretable, and computationally efficient.
* **Decision Tree** was expected to potentially perform well if the fraud patterns contained nonlinear relationships or feature interactions.
* **KNN** was expected to be highly sensitive to feature scaling because it relies on distance calculations.
* **MLP** was expected to learn nonlinear relationships but potentially suffer from the severe class imbalance.

### Evaluation Metrics

Recall was expected to be particularly important because a False Negative represents a fraudulent transaction that the system failed to detect.

However, maximizing Recall alone was not considered sufficient. Precision and F1-score were also expected to be important because increasing Recall can generate more False Positives.

### Accuracy Hypothesis

It was expected that a model predicting every transaction as legitimate could achieve approximately 99.83% Accuracy while detecting zero fraudulent transactions.

Therefore, Accuracy alone was expected to be misleading.

### Scaling Hypothesis

KNN was expected to be strongly affected by feature scaling because it determines neighboring observations using distances.

### Decision Tree Hypothesis

An unrestricted Decision Tree was expected to have a higher risk of overfitting than a constrained tree.

The experiments largely supported these hypotheses.

---

## 5. Model Comparison

### Models

Four baseline models were evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Simple MLP / Neural Network

The MLP used the following architecture:

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

The MLP used Adam optimization, a learning rate of `0.001`, 20 epochs, batch size 256, and `BCEWithLogitsLoss`.

### Evaluation Metrics

The main evaluation metrics were:

* Accuracy
* Precision
* Recall
* F1-score
* False Positives
* False Negatives
* Confusion Matrix

Because of the severe class imbalance, Precision, Recall, and F1-score were given greater importance than Accuracy.

### Test-Set Results

| Model               | Accuracy | Precision |     Recall |   F1-score | False Positives | False Negatives |
| ------------------- | -------: | --------: | ---------: | ---------: | --------------: | --------------: |
| Logistic Regression |   99.91% |    82.67% |     63.27% |     71.68% |              13 |              36 |
| KNN                 |   99.95% |    91.86% |     80.61% | **85.87%** |           **7** |              19 |
| Decision Tree       |   99.91% |    75.26% |     74.49% |     74.87% |              24 |              25 |
| MLP                 |   99.95% |    84.54% | **83.67%** |     84.10% |              15 |          **16** |

The test-set results show that all models achieved approximately 99.9% Accuracy, but their fraud-detection performance was substantially different.

KNN achieved the highest Precision and F1-score, while the MLP achieved the highest Recall.

### Confusion Matrices

#### Logistic Regression

```text
[[56851    13]
 [   36    62]]
```

* True Negatives: 56,851
* False Positives: 13
* False Negatives: 36
* True Positives: 62

#### KNN

```text
[[56857     7]
 [   19    79]]
```

* True Negatives: 56,857
* False Positives: 7
* False Negatives: 19
* True Positives: 79

#### Decision Tree

```text
[[56840    24]
 [   25    73]]
```

* True Negatives: 56,840
* False Positives: 24
* False Negatives: 25
* True Positives: 73

#### MLP

```text
[[56849    15]
 [   16    82]]
```

* True Negatives: 56,849
* False Positives: 15
* False Negatives: 16
* True Positives: 82

### Cross-Validation Results

5-Fold Stratified Cross Validation was used for Logistic Regression, KNN, and Decision Tree.

| Model               | Mean Precision | Mean Recall |    Mean F1 |
| ------------------- | -------------: | ----------: | ---------: |
| Logistic Regression |         87.02% |      62.00% |     72.32% |
| KNN                 |     **93.74%** |  **77.44%** | **84.79%** |
| Decision Tree       |         74.49% |      77.24% |     75.81% |

KNN achieved the highest mean Precision and F1-score and also slightly exceeded the Decision Tree in mean Recall.

The KNN mean F1-score of 84.79% was also close to its held-out test F1-score of 85.87%, providing consistent evidence that KNN generalized well across different data splits.

---

## 6. Scaling Experiment

The effect of feature scaling was specifically investigated for KNN.

Two KNN models were compared using the same train/test split:

1. KNN using unscaled features
2. KNN using standardized features

### Results

| Version      | Accuracy | Precision | Recall |   F1-score | False Positives | False Negatives |
| ------------ | -------: | --------: | -----: | ---------: | --------------: | --------------: |
| Unscaled KNN |   99.83% |   100.00% |  3.06% |      5.94% |               0 |              95 |
| Scaled KNN   |   99.95% |    91.86% | 80.61% | **85.87%** |               7 |              19 |

The effect of scaling was substantial.

The unscaled KNN achieved 100% Precision and 99.83% Accuracy, but detected only 3 of the 98 fraudulent transactions.

After scaling:

* Recall increased from **3.06% → 80.61%**
* F1-score increased from **5.94% → 85.87%**
* False Negatives decreased from **95 → 19**

This confirms the initial hypothesis that KNN is highly sensitive to feature scaling.

The reason is that KNN uses distances between observations. Features with larger numerical scales can therefore have disproportionate influence on the distance calculation.

---

## 7. Hyperparameter Experiment

Two hyperparameter experiments were performed.

### KNN — `n_neighbors`

The values `K=1`, `K=5`, and `K=20` were compared.

|  K |  Precision |     Recall |   F1-score |
| -: | ---------: | ---------: | ---------: |
|  1 |     86.96% | **81.63%** |     84.21% |
|  5 | **91.86%** |     80.61% | **85.87%** |
| 20 |     84.52% |     72.45% |     78.02% |

`K=1` produced the highest Recall, but `K=5` achieved the highest Precision and F1-score.

Increasing K to 20 caused all three metrics to decrease. This suggests that the larger neighborhood produced a decision boundary that was too smooth to capture some minority fraud patterns.

Therefore, **K=5** provided the strongest overall balance.

### Decision Tree — `max_depth`

The values `2`, `5`, `10`, and `None` were compared.

| max_depth | Train F1 | Test Precision | Test Recall |    Test F1 |
| --------: | -------: | -------------: | ----------: | ---------: |
|         2 |   79.31% |         76.53% |      76.53% |     76.53% |
|         5 |   86.49% |     **89.41%** |  **77.55%** | **83.06%** |
|        10 |   92.20% |         89.02% |      74.49% |     81.11% |
|      None |  100.00% |         75.26% |      74.49% |     74.87% |

A depth of 2 was relatively constrained and showed signs of underfitting.

Increasing the depth to 5 improved the test F1-score to 83.06%.

At depth 10, training performance continued to increase while test performance decreased, indicating increasing overfitting.

The unrestricted tree achieved 100% training F1-score but only 74.87% test F1-score, providing clear evidence of overfitting.

Among the tested configurations, `max_depth=5` produced the best Decision Tree test F1-score.

---

## 8. Classification Threshold Experiment

Classification thresholds were investigated to understand the Precision/Recall trade-off.

For Logistic Regression, thresholds of `0.3`, `0.5`, and `0.7` were evaluated.

| Threshold |  Precision |     Recall |   F1-score | False Positives | False Negatives |
| --------: | ---------: | ---------: | ---------: | --------------: | --------------: |
|       0.3 |     73.12% | **69.39%** |     71.20% |              25 |              30 |
|       0.5 |     82.67% |     63.27% | **71.68%** |              13 |              36 |
|       0.7 | **83.10%** |     60.20% |     69.82% |              12 |              39 |

Lowering the threshold from 0.5 to 0.3 caused more transactions to be classified as fraud.

As a result:

* Recall increased from 63.27% to 69.39%.
* False Negatives decreased from 36 to 30.
* Precision decreased from 82.67% to 73.12%.
* False Positives increased from 13 to 25.

Increasing the threshold had the opposite effect.

This demonstrates the central Precision/Recall trade-off in fraud detection:

```text
Lower threshold
      ↓
More fraud detected
      ↓
Higher Recall
      ↓
More False Positives
      ↓
Lower Precision
```

A lower threshold is useful when missing fraud is considered more costly than generating additional alerts.

---

## 9. Final Model Selection

The final selected configuration is:

* **Model:** K-Nearest Neighbors
* **K:** 5
* **Scaling:** StandardScaler
* **Classification Threshold:** 0.5

### Why KNN?

KNN was selected based on the combined evidence from:

* Held-out test-set performance
* 5-Fold Stratified Cross Validation
* KNN hyperparameter experiment
* Scaling experiment
* False Positive / False Negative trade-off
* Comparison with Decision Tree overfitting

KNN achieved:

* Test Precision: **91.86%**
* Test Recall: **80.61%**
* Test F1-score: **85.87%**
* Cross-validation mean Precision: **93.74%**
* Cross-validation mean Recall: **77.44%**
* Cross-validation mean F1-score: **84.79%**

It therefore provided the strongest overall balance between Precision and Recall among the evaluated traditional models.

The MLP achieved slightly higher Recall at 83.67%, but KNN achieved better Precision and F1-score.

### Why K=5?

Among the tested K values:

* `K=1` had higher Recall but lower Precision and F1-score.
* `K=5` achieved the highest F1-score.
* `K=20` produced substantially worse performance.

Therefore, `K=5` provided the best overall balance.

### Why Threshold = 0.5?

For the final KNN model, the threshold experiment produced:

| Threshold |  Precision | Recall |   F1-score | False Positives | False Negatives |
| --------: | ---------: | -----: | ---------: | --------------: | --------------: |
|       0.3 |     86.46% | 84.69% |     85.57% |              13 |              15 |
|       0.4 |     86.46% | 84.69% |     85.57% |              13 |              15 |
|       0.5 | **91.86%** | 80.61% | **85.87%** |           **7** |              19 |
|       0.6 |     91.86% | 80.61% |     85.87% |               7 |              19 |
|       0.7 |     95.95% | 72.45% |     82.56% |               3 |              27 |

KNN with `K=5` produces discrete probability values because the probability is based on the five nearest neighbors. Therefore, some thresholds produce identical predictions.

Threshold `0.5` was selected because it achieved the highest F1-score while also producing the fewest False Positives among the stronger configurations.

A threshold of 0.3 detected four additional fraudulent transactions, reducing False Negatives from 19 to 15, but increased False Positives from 7 to 13.

Therefore, `0.5` was considered the better overall balance for this educational project.

### Final Configuration

```text
K-Nearest Neighbors
        ↓
K = 5
        ↓
StandardScaler
        ↓
Classification Threshold = 0.5
```

The final model and scaler were saved using `joblib` as:

```text
models/
├── model.pkl
└── scaler.pkl
```

---

## 10. Running Instructions

### Installation

Clone the repository:

```bash
git clone https://github.com/amirhoseinzahedi/mini-project-01.git
cd mini-project-01
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The project uses:

* pandas
* NumPy
* matplotlib
* scikit-learn
* joblib
* PyTorch

### Training

The project contains separate scripts for data preparation, model training, evaluation, experiments, model saving, and prediction.

The main training script is:

```bash
python src/train.py
```

Model evaluation can be performed using:

```bash
python src/evaluate.py
```

Cross-validation can be run using:

```bash
python src/cross_validate.py
```

The experiment scripts are:

```bash
python src/experiment_knn.py
python src/experiment_tree.py
python src/experiment_threshold.py
```

### Prediction

The final prediction pipeline is implemented in:

```text
src/predict.py
```

It loads:

```text
models/scaler.pkl
models/model.pkl
```

and applies the same preprocessing used during training before making a prediction.

The prediction pipeline is:

```text
Input Transaction
       ↓
Feature Validation
       ↓
Load Scaler
       ↓
Scale Features
       ↓
Load KNN Model
       ↓
Predict Fraud Probability
       ↓
Apply Threshold = 0.5
       ↓
Prediction
```

The model expects the following 30 input features:

```text
Time
V1 ... V28
Amount
```

---

## 11. Reflection

### Question 1

**Why is Accuracy a misleading metric for this dataset?**

Accuracy is misleading because the dataset is extremely imbalanced. Only 492 of the 284,807 transactions are fraudulent, representing approximately 0.17% of the data. A model that predicts every transaction as legitimate would achieve approximately 99.83% Accuracy while detecting zero fraudulent transactions. The experiments demonstrated the same problem in practice: the unscaled KNN achieved 99.83% Accuracy and 100% Precision but detected only 3 of the 98 fraudulent transactions in the test set. Therefore, Accuracy does not adequately measure the model's ability to detect the minority fraud class.

### Question 2

**What is the trade-off between detecting more fraudulent transactions and generating more false alarms?**

Detecting more fraudulent transactions generally requires accepting more False Positives. Lowering the classification threshold makes the model more likely to classify transactions as fraudulent, which increases Recall and reduces False Negatives, but it also increases the number of legitimate transactions incorrectly flagged as fraud. For example, lowering the Logistic Regression threshold from 0.5 to 0.3 increased Recall from 63.27% to 69.39%, but False Positives increased from 13 to 25. In a real fraud detection system, the appropriate threshold depends on the relative business costs of missed fraud and false alarms.

### Question 3

**If you had one additional week, what would you improve in your fraud detection system?**

I would focus on improving the handling of the severe class imbalance and on evaluating the system using more realistic fraud-detection criteria. I would investigate techniques such as class weighting, SMOTE, and alternative sampling strategies, while carefully avoiding data leakage. I would also evaluate additional models such as Random Forest or gradient-boosting methods and compare them using Precision-Recall AUC. Finally, I would perform more systematic threshold optimization based on an explicit business cost for False Positives and False Negatives instead of selecting the threshold only from a small set of candidate values.

---

## 12. Conclusion

This project demonstrated that credit card fraud detection is fundamentally different from ordinary balanced classification because fraudulent transactions are extremely rare.

The experiments showed that Accuracy alone can be misleading. All baseline models achieved approximately 99.9% Accuracy, yet their ability to detect fraud differed considerably.

Feature scaling had a particularly strong effect on KNN. Without scaling, KNN detected only 3 of 98 fraudulent transactions. After standardization, Recall increased from 3.06% to 80.61% and F1-score increased from 5.94% to 85.87%.

The hyperparameter experiments showed that `K=5` provided the best overall KNN performance among the tested values. The Decision Tree experiments also demonstrated how excessive model complexity can lead to overfitting.

Based on the combined test-set results, cross-validation, hyperparameter experiments, and error trade-offs, **KNN with K=5 and a classification threshold of 0.5** was selected as the final model.

The final result emphasizes an important practical lesson: a fraud detection system should not be optimized simply for overall Accuracy. The ability to detect fraudulent transactions while controlling false alarms must be considered together, with particular attention to Precision, Recall, F1-score, and the business cost of False Positives and False Negatives.
