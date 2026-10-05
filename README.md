# An Explainable Machine Learning Framework for Depression Screening Using Psychosocial and Demographic Indicators: A Leakage-Free Benchmark on an Expanded Bangladeshi Dataset

This repository contains the code accompanying the manuscript *"An Explainable Machine Learning Framework for Depression Screening Using Psychosocial and Demographic Indicators: A Leakage-Free Benchmark on an Expanded Bangladeshi Dataset."* The study presents a leakage-free machine learning framework for depression screening on an extended Bangladesh-relevant psychosocial survey dataset, coupled with a dual explainable AI layer (SHAP + LIME) and a deployed screening prototype.

> ⚠️ **Note on data:** The dataset used in this study is **not included** in this repository. It extends a previously published dataset (Zulfiker et al., 2021) with 135 newly collected, anonymous survey responses, and is available from the corresponding author upon reasonable request (see Declarations in the manuscript).

## Overview

- **Leakage-free pipeline:** feature scaling, SMOTE oversampling, and hyperparameter tuning are performed strictly within the training fold of a stratified 5-fold cross-validation, with the champion model selected on cross-validation accuracy alone before the holdout test set is touched.
- **Nine benchmarked classifiers:** Logistic Regression, SVM, KNN, Naive Bayes, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, and XGBoost.
- **Extended robustness search:** wider hyperparameter grids, alternative resamplers, and three additional ensembles (LightGBM, CatBoost, Extra Trees), compared against the champion using McNemar's exact test.
- **Explainability:** SHAP (global feature attribution) and LIME (local, case-level explanations) applied to the champion model.
- **Calibration:** post-hoc sigmoid (Platt) calibration of the champion's probability outputs.
- **Deployment:** a Streamlit-based, anonymous web screening prototype with an integrated emergency-referral pathway, generated directly from the notebook.

## Repository Structure

```
.
├── Depression_Final_Code.ipynb   # End-to-end pipeline: preprocessing, modeling, XAI,
│                                  # calibration, and the Streamlit app (written out via
│                                  # %%writefile app.py within the notebook)
├── sample_data/
│   └── sample_data.csv           # Small synthetic sample (structure only, no real respondent data)
├── requirements.txt               # Python dependencies
└── README.md
```

## Key Results

| Model | CV Accuracy | Test Accuracy | MCC | Test ROC-AUC |
|---|---|---|---|---|
| **Random Forest (champion)** | 83.88% | 82.19% | 0.5775 | 0.8754 |
| Calibrated champion | — | **85.62%** | — | — |

Full results for all nine classifiers are reported in the manuscript (Table 3).

## Requirements

```
pip install -r requirements.txt
```

Main dependencies: `scikit-learn`, `imbalanced-learn`, `xgboost`, `lightgbm`, `catboost`, `shap`, `lime`, `streamlit`, `pandas`, `numpy`, `matplotlib`.

## Running the Pipeline

1. Place your dataset (same schema as described in the manuscript's Table 1 codebook) in `sample_data/` or update the path at the top of the notebook.
2. Run `Depression_Final_Code.ipynb` sequentially in Google Colab or Jupyter — preprocessing, nested cross-validation, holdout evaluation, explainability, and calibration are executed in order.
3. The final cells write out and launch the Streamlit screening prototype (`app.py`) directly from the notebook, including a public tunnel link for live testing. To run the app independently afterward:
   ```
   streamlit run app.py
   ```

## Citation

If you use this code, please cite:

> [Authors]. ([Year]). An Explainable Machine Learning Framework for Depression Screening Using Psychosocial and Demographic Indicators: A Leakage-Free Benchmark on an Expanded Bangladeshi Dataset. *[Journal Name]*. [DOI — to be added upon publication]

## License

This project's source code is licensed under the MIT License — see [`LICENSE`](./LICENSE) for details. The underlying survey dataset is not covered by this license (see Data note above).

## Authors

- Md. Arifur Rahman Rahad
- K. M. Shahriar Islam
- Imran Mahmud

Department of Software Engineering, Daffodil International University
