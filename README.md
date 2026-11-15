# Cancer Survival Prediction Using Machine Learning

An educational machine-learning project that explores binary cancer-survival classification and demonstrates how a fitted scikit-learn pipeline can be used in a Streamlit interface.

> **Important:** This is an academic demonstration only. It is not clinically validated and must not be used to diagnose cancer, estimate an individual's prognosis, select treatment, or guide patient care.

## Project overview

The project uses a labelled training dataset to predict the binary target `Survival Prediction` (`Yes` / `No`). The notebook compares modelling approaches and the selected approach is a Gradient Boosting classifier with ADASYN oversampling.

The corrected training workflow is in **[Cancer Survival Prediction.ipynb](./Cancer%20Survival%20Prediction.ipynb)**. It keeps the internal validation data separate from the external test data, fits preprocessing only on the training fold, evaluates with a stratified split, and preserves the original external test IDs when creating a submission file.

## Live application

- **Streamlit deployment:** A public deployment URL was not discoverable in this repository. Add the verified public URL here when available.
- **Source repository:** https://github.com/McDonson1/Cancer-Survival-Predictor
- **Application code:** [app.py](./app.py)

The existing app and deployed model artifact were intentionally left unchanged during this audit.

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit prediction interface |
| `requirements.txt` | Runtime package dependencies |
| `Cancer Survival Prediction.ipynb` | Corrected, reproducible training and evaluation workflow |
| `GradientBoosting.pkl` | Existing serialized model used by the current app; not replaced by this audit |
| `GradientBoosting.jlb` | Additional serialized artifact; not referenced by the current app |
| `GradientBoosting_candidate.pkl` | Generated locally by the corrected notebook after the data files are supplied; not committed or deployed by this audit |
| `Final Report.docx` | Existing project report |
| `Sample Prediction File.csv` | Legacy validation output currently in the repository; see the submission warning below |

## Dataset description and provenance

The notebook expects two local CSV files:

- `data/patient_train_data.csv`: labelled training records, including `ID` and `Survival Prediction`.
- `data/patient_test_data.csv`: external test records, including the original `ID` and predictor columns, with exactly **75,000 data rows**.

The source datasets are not included in this public repository. Their source, licence, collection method, population, time period and data dictionary are not documented here. Do not infer or claim dataset provenance without confirming it with the dataset provider.

The model uses 28 predictors:

`Alcohol Consumption`, `Cancer Stage`, `Country`, `Diabetes`, `Diabetes History`, `Diet Risk`, `Early Detection`, `Family History`, `Gender`, `Genetic Mutation`, `Healthcare Access`, `Healthcare Costs`, `Heart Disease History`, `Hypertension`, `Incidence Rate per 100K`, `Inflammatory Bowel Disease`, `Insurance Costs`, `Insurance Status`, `Mortality Rate per 100K`, `Non Smoker`, `Obesity BMI`, `Physical Activity`, `Screening History`, `Smoking History`, `Treatment Type`, `Tumor Size (mm)`, `Urban or Rural`, and `Age`.

The notebook derives age from `Date of Birth` using a fixed reference date of 1 January 2026. Change that date only if the study's intended reference date is known and documented.

### Data-cleaning decisions

- `M` and `F` are normalized to `Male` and `Female`.
- Other gender codes, including `P`, are retained as categories. The earlier notebook removed `P` without documenting its meaning. The app currently offers only Male and Female; the meaning of `P` must be confirmed before changing the interface.
- Observed treatment labels `Radiotherapy` and `Combination` are normalized to `Radiation` and `Combined` to match the current app's input options.
- `?` is treated as missing. Categorical values are imputed before one-hot encoding; numerical values are imputed before scaling.
- Predictor rows are not dropped just because a value is missing. Rows with missing or unrecognized target labels cannot be used for supervised training and cause the notebook to stop with an explicit error.
- The earlier outlier-removal function operated on temporary copies and did not change the model training data. It has been removed from the corrected workflow; no outlier rows are dropped without a justified, training-derived rule.

## Methodology

1. Load and inspect the labelled training and external test CSV files.
2. Validate the external test row count and retain its original IDs.
3. Apply deterministic feature normalization and construct the same 28-column schema for training and external data.
4. Split only the labelled data into an 80% training fold and 20% validation fold using stratification and random state 42.
5. Fit categorical imputation, one-hot encoding, numerical imputation and scaling inside a preprocessing pipeline.
6. Apply ADASYN only within the training pipeline.
7. Evaluate a Gradient Boosting classifier using accuracy, precision, recall, F1-score, specificity, balanced accuracy, ROC-AUC and a confusion matrix.
8. Fit a fresh candidate pipeline on all labelled training rows.
9. Predict the untouched 75,000-row external test file and write `Sample Prediction File.csv` with columns `ID` and `Survival Prediction`.

The validation metrics are internal holdout estimates, not external clinical performance estimates.

## Evaluation status

The previous notebook recorded the following results on a non-stratified 80/20 split (random state 66):

| Metric | Previously recorded value |
|---|---:|
| Accuracy | 0.600244 (60.02%) |
| Precision | 0.600541 (60.05%) |
| Recall | 0.999099 (99.91%) |
| F1-score | 0.750169 (75.02%) |
| Specificity | Not recorded |
| Balanced accuracy | Not recorded |
| ROC-AUC | Not recorded |
| Confusion-matrix counts | Not recorded in the notebook's text output |

These are **legacy results**, not results from the corrected stratified workflow. They must not be presented as the new validation results. Run the corrected notebook with the original training CSV to generate the complete metric set and confusion matrix.

The earlier high recall and modest precision indicate that the old model predicted the positive class very frequently; this can produce many false positives. Accuracy alone is not sufficient to assess the model.

## Important external-submission warning

The currently committed `Sample Prediction File.csv` contains **14,784 data rows**, with IDs `0` through `14783`. The notebook shows that it was created from the internal validation split, not from the original 75,000-row external test file. **Do not submit the current checked-in CSV as the external test prediction file.**

The corrected notebook now validates that the external test file contains exactly 75,000 rows, preserves its original IDs and writes a new CSV with exactly 75,000 predictions when run. The external test CSV was not present in the repository during this audit, so a new 75,000-row prediction file could not be generated or verified in this GitHub change.

## Run locally

1. Clone the repository:

   ```bash
   git clone https://github.com/McDonson1/Cancer-Survival-Predictor.git
   cd Cancer-Survival-Predictor
   ```

2. Create and activate a virtual environment:

   **Windows PowerShell**
   ```powershell
   python -m venv .venv
   .venv\\Scripts\\Activate.ps1
   ```

   **macOS/Linux**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install the app dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create a local `data/` folder and place the two source CSV files in it using the filenames listed above. These files are not committed to the repository.

5. Open `Cancer Survival Prediction.ipynb` in Jupyter and run all cells from top to bottom. The notebook requires the installed packages plus Jupyter. It writes:
   - `Sample Prediction File.csv` — expected to contain exactly 75,000 rows and the required columns.
   - `GradientBoosting_candidate.pkl` — a newly trained candidate artifact, separate from the deployed model.

6. Verify the notebook's final assertions and review the validation metrics before using the output.

To run the existing Streamlit application:

```bash
streamlit run app.py
```

The existing app expects `GradientBoosting.pkl` in the repository root. This audit did not replace that file or alter `app.py`.

## Deployment and model compatibility

The current app's Treatment Type choices include `Radiation` and `Combined`, while the earlier notebook's observed source labels included `Radiotherapy` and `Combination`. The corrected notebook normalizes those labels for a future candidate model. The existing serialized model could not be inspected or reloaded through the GitHub file interface during this audit, so the existing app/model category compatibility remains unverified. Do not replace the deployed model until the candidate model has been tested with every app input and the deployed Streamlit app has been smoke-tested.

The current app also exposes only Male and Female. The training data contained a `P` gender code; the corrected notebook retains it rather than guessing its meaning. Confirm the data dictionary before expanding or changing the app's gender choices.

The existing dependency pins in `requirements.txt` were left unchanged because the user had previously confirmed that the deployed app was working with them. Pickle artifacts are version-sensitive; use the compatible Python and package versions when loading the existing model.

## Limitations and next steps

- The dataset source and licence need to be documented.
- The corrected stratified metrics have not yet been computed because the source CSV files are not in the repository.
- The new 75,000-row external prediction CSV has not yet been generated for the same reason.
- The old sample CSV is a validation output and must not be submitted as the external test result.
- The existing model artifact could not be independently deserialized in this audit. Its full compatibility with the current app's categorical choices remains unresolved.
- No independent labelled test set, clinical validation, calibration assessment, fairness assessment, or regulatory review has been established.
- Add the verified public Streamlit URL when available.

## Disclaimer

This project is an educational machine-learning demonstration. It is not a medical device, has not been clinically validated, and must not be used for diagnosis, prognosis, treatment decisions or patient care.
