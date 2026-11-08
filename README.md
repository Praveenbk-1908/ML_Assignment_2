# ML Assignment 2: Bank Marketing Classification

## Problem Statement

Predict whether a bank client will subscribe to a term deposit (`deposit` = yes/no). This is a binary classification problem. Five classification models are trained, compared on the same held-out test set, and deployed in a Streamlit app.

## Dataset

- **Source:** Bank Marketing dataset (UCI), file `bank.csv`
- **Instances:** 11,162
- **Features:** 16 (mix of numeric and categorical)
- **Target:** `deposit` (yes/no)
- **Split:** 80% train / 20% test (stratified, `random_state=42`). The test split is saved as `test_data.csv`.

## Models Used

Each model is a scikit-learn pipeline: `ColumnTransformer` (StandardScaler for numeric features, OneHotEncoder for categorical features) followed by the classifier.

- Logistic Regression
- Decision Tree (max_depth=8)
- K-Nearest Neighbors (k=7)
- Naive Bayes (Gaussian)
- Support Vector Machine (RBF kernel, probability=True)

## Model Comparison (Test Set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8262 | 0.9071 | 0.8278 | 0.7996 | 0.8135 | 0.6513 |
| Decision Tree | 0.8213 | 0.8802 | 0.7838 | 0.8601 | 0.8202 | 0.6461 |
| KNN | 0.8222 | 0.8881 | 0.8282 | 0.7883 | 0.8077 | 0.6433 |
| Naive Bayes | 0.7201 | 0.8042 | 0.7837 | 0.5652 | 0.6568 | 0.4472 |
| Support Vector Machine | 0.8554 | 0.9194 | 0.8249 | 0.8819 | 0.8524 | 0.7124 |

## Observations

- **Logistic Regression:** Balanced performance. A linear decision boundary is a solid baseline for this data.
- **Decision Tree:** Highest recall among the single models (0.860) but lower precision (0.784). Max depth 8 limits overfitting.
- **KNN:** Good precision (0.828) but lower recall (0.788). Depends on feature scaling.
- **Naive Bayes:** Weakest model. Recall is only 0.565, so it misses many clients who subscribe. The feature independence assumption does not fit this data well.
- **Support Vector Machine:** Best on most metrics (Accuracy, AUC, Recall, F1, MCC). The RBF kernel captures non-linear patterns.

**Best model:** Support Vector Machine.

## Project Structure

```
app.py                  Streamlit app
train_models.py         Training script (same logic as the notebook)
ML_Assignment_2.ipynb   Notebook with training, evaluation, and observations
bank.csv                Dataset
test_data.csv           Held-out test split for the app
requirements.txt        Dependencies
model/                  Saved pipelines (.joblib) and metrics.json
```

## How to Run

1. Create and activate a virtual environment:

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```
   streamlit run app.py
   ```

4. Upload `test_data.csv` in the app to view metrics, the confusion matrix, and the classification report for each model.

To retrain the models, run the notebook or `python train_models.py`.

## Links

- GitHub repository: _add link_
- Streamlit app: _add link_
