# 🔐 Phishing URL Detector

A machine-learning project that classifies a URL as **Phishing** or **Legitimate**.
The model is trained and evaluated in **Google Colab**, then used in a **Streamlit** app for predictions.

## Overview

1. Load and explore the dataset (Colab)
2. Clean the data (remove duplicates, check missing/invalid values)
3. Train and compare models, keep the best one (Random Forest)
4. Evaluate on unseen test data
5. Save the model and load it in Streamlit (no training happens in the app)

## Dataset

- Kaggle "Phishing URL" dataset: `phishing_url_dataset.csv`
- 2,488 rows, 13 numeric features, target `0 = Phishing`, `1 = Legitimate`
- 1,134 duplicate rows removed, leaving **1,354 rows** (about 67% phishing, 33% legitimate)
- Features: `url_length`, `valid_url`, `at_symbol`, `sensitive_words_count`, `path_length`, `isHttps`, `nb_dots`, `nb_hyphens`, `nb_and`, `nb_or`, `nb_www`, `nb_com`, `nb_underscore`

## Model

**Final model: Random Forest Classifier** (scikit-learn)

| Setting | Value |
|---|---|
| Trees (`n_estimators`) | 300 |
| `min_samples_leaf` | 2 |
| `class_weight` | balanced |
| Train / test split | 80% / 20%, stratified, `random_state=42` |
| Scaling | Not needed for Random Forest |

> Random Forest is not a neural network, so there is **no optimizer (Adam) and no epochs**.
> Its main settings are the number of trees and tree depth/leaf size, shown above.

Compared against a majority-class baseline, Logistic Regression and a Decision Tree:

| Model | Test accuracy |
|---|---|
| Baseline (always "phishing") | 66.8% |
| Logistic Regression | 72.7% |
| Decision Tree | 74.9% |
| **Random Forest** | **78.2%** |

## Results (Random Forest, 271 unseen test URLs)

| Metric | Score |
|---|---|
| Accuracy | 78.2% |
| Precision (phishing) | 83.2% |
| Recall (phishing) | 84.5% |
| F1-score (phishing) | 83.8% |
| ROC-AUC | 0.850 |
| 5-fold cross-validation accuracy | 81.1% |

Confusion matrix: 153 phishing caught, 28 missed, 59 legitimate passed, 31 legitimate wrongly flagged.

## Important points

- **Most important features:** `valid_url`, `url_length`, `path_length`
- **Overfitting:** training accuracy is 93.2% vs 78.2% on test data, so the model partly memorised the training data. The test score is the realistic one.
- **Weaker on legitimate URLs:** only 65.6% of legitimate URLs are correctly passed.
- **Small dataset:** only 1,354 unique rows, and the features are simple counts, so accuracy has a limit.
- **URL mode in the app is approximate:** the dataset only has pre-computed counts and does not document how `valid_url` and `nb_or` were made, so the app lets the user set those two.
- This tool is a learning project. Do not rely on it alone to judge whether a website is safe.

## Project structure

```
week5_Phishing_URL_Detector/
├── app.py                  # Streamlit app
├── requirements.txt
├── dataset/                # phishing_url_dataset.csv
├── notebook/               # Phishing_Url_Ml_Model.ipynb (Google Colab)
└── model_files/            # files exported from Colab
    ├── phishing_model.pkl
    ├── feature_names.pkl
    ├── model_config.json
    ├── feature_importance.csv
    └── permutation_importance.csv
```

## Run locally

```bash
cd week5_Phishing_URL_Detector
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. Use Python 3.10 to 3.12 and `scikit-learn==1.6.1` (the version used to train the model).

## Tech stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, Streamlit
