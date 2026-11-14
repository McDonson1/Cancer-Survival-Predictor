# Cancer Survival Prediction Using Machine Learning

An educational machine-learning project that explores binary cancer-survival prediction and demonstrates how a trained scikit-learn pipeline can be deployed with Streamlit.

> **Important:** This application is for educational and demonstration purposes only. It is not clinically validated and must not be used to diagnose cancer, estimate an individual's prognosis, or guide treatment decisions.

## Live application

- **Streamlit app:** Add your public Streamlit deployment URL here.
- **Source code:** https://github.com/McDonson1/Cancer-Survival-Predictor

## Project overview

The project trains and evaluates classification models to predict the dataset's binary target, `Survival Prediction` (`Yes` or `No`). It explores multiple classifiers and resampling methods for class imbalance, then saves a Gradient Boosting pipeline with ADASYN for use in the Streamlit application.

The saved pipeline includes preprocessing and the classifier. The app sends a single-row pandas DataFrame with the expected 28 input columns to the pipeline and displays the predicted class.

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit prediction interface |
| `requirements.txt` | Runtime dependencies for the deployed app |
| `Cancer Survival Prediction.ipynb` | Model exploration, training and evaluation notebook |
| `GradientBoosting.pkl` | Serialized fitted model pipeline used by the app |
| `GradientBoosting.jlb` | Additional serialized model artifact; not referenced by the current app |
| `Final Report.docx` | Existing project report |
| `Sample Prediction File.csv` | Predictions generated from the notebook's held-out split |

## Model workflow

1. Load and inspect the labelled training data.
2. Explore data quality, feature types and target distribution.
3. Separate the target (`Survival Prediction`) from the input features.
4. Compare classifiers including AdaBoost, Random Forest, Gradient Boosting, XGBoost and Decision Tree.
5. Compare resampling approaches including SMOTE, RandomOverSampler, RandomUnderSampler and ADASYN.
6. Fit a preprocessing-and-classification pipeline.
7. Evaluate the selected Gradient Boosting + ADASYN pipeline on a held-out split.
8. Serialize the fitted pipeline with pickle.
9. Load the pipeline in Streamlit for interactive predictions.

The selected pipeline uses a `ColumnTransformer` with categorical encoding and numerical scaling/imputation, ADASYN oversampling during training, and `GradientBoostingClassifier`.

## Reported evaluation

The notebook records the following results for its final 80/20 split (random state 66):

| Metric | Result |
|---|---:|
| Accuracy | 0.600244 (60.02%) |
| Recall | 0.999099 (99.91%) |
| Precision | 0.600541 (60.05%) |
| F1-score | 0.750169 (75.02%) |

These figures describe performance on the notebook's held-out subset. They are not evidence of clinical validity or performance on an independent external dataset. The very high recall alongside modest precision means many positive predictions may be false positives. Review the confusion matrix and class-specific error counts before making claims about model usefulness.

## Run locally

Use a Python environment compatible with the versions pinned in `requirements.txt`.

```bash
git clone https://github.com/McDonson1/Cancer-Survival-Predictor.git
cd Cancer-Survival-Predictor
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies and start the app:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The model file `GradientBoosting.pkl` must be in the repository root beside `app.py`.

## Reproducibility and validation notes

- The saved model is a pickle artifact. Pickle-based models are version-sensitive; use the compatible dependency versions and Python environment used to create the artifact.
- The notebook's final evaluation uses a random 80/20 split of the labelled data. Use `stratify=y` in a future retraining run and keep a truly independent test set untouched until final evaluation.
- The notebook fills missing `Healthcare Access` values separately in the training and test frames. For a more rigorous retraining workflow, put missing-value imputation inside the preprocessing pipeline and fit it only on training data.
- In the notebook, the categorical pipeline currently applies `OneHotEncoder` before `SimpleImputer`. Prefer imputing categorical values before one-hot encoding, and imputing numerical values before scaling.
- The checked-in `Sample Prediction File.csv` was generated from the notebook's held-out split using `test_df.index` as IDs. It is not a verified submission for a separate 75,000-row external test file. Do not describe it as such.
- The notebook compares models and samplers; reported comparison results are rounded. The final model metrics above come from the later dedicated training/evaluation cells.
- The app currently accepts 28 fields. Ensure any future retrained model uses the same feature names, types and category conventions, or update and test the app at the same time.

## Recommended next improvements

1. Correct the notebook preprocessing order and use a stratified split.
2. Keep training, validation and any external test dataset in separate variables.
3. Evaluate with balanced accuracy, specificity, ROC-AUC, PR-AUC, confusion matrix and class-wise error counts.
4. Validate on an independent dataset and assess potential bias across relevant subgroups.
5. Record the exact Python and package versions used to train and serialize the model.
6. Regenerate a submission CSV only when the original external test data and its true ID column are available.
7. Add automated tests for input schema, model loading and both output classes.

## Disclaimer

This project is an academic demonstration of a machine-learning workflow. It is not a medical device, has not been clinically validated, and must not be used for patient care or medical decision-making.
