# Model Card — Cancer Survival Prediction

## Summary

- **Task:** Binary classification of the dataset field `Survival Prediction` into `Yes` or `No`.
- **Model:** `GradientBoostingClassifier` inside a fitted preprocessing and ADASYN pipeline.
- **Artifact used by app:** `GradientBoosting.pkl`.
- **Interface:** Streamlit.
- **Status:** Educational prototype; not clinically validated.

## Intended use

Demonstrating a supervised machine-learning workflow, model serialization, and interactive inference in a Streamlit application.

## Out-of-scope use

Do not use this model to diagnose cancer, estimate an individual's survival probability, make treatment decisions, prioritize patients, or replace a qualified healthcare professional. The output is a class label, not a calibrated clinical probability.

## Inputs

The application accepts 28 dataset features, including cancer stage, country, demographic and history fields, healthcare access/cost fields, incidence and mortality rates, tumor size, treatment type, and age. The app constructs a pandas DataFrame using the feature names expected by the saved pipeline.

## Training and evaluation

The notebook explores multiple classifiers and resampling methods. It selects a Gradient Boosting classifier with ADASYN and reports a final random 80/20 hold-out evaluation (random state 66).

| Metric | Notebook result |
|---|---:|
| Accuracy | 60.0244% |
| Recall | 99.9099% |
| Precision | 60.0541% |
| F1-score | 75.0169% |

The notebook also contains an earlier model-comparison experiment using a different split (random state 44), with rounded scores. Do not mix those comparison scores with the final dedicated evaluation above.

## Limitations and risks

- The reported evaluation is from a single random hold-out split, not independent external validation.
- Very high recall and modest precision indicate a potentially substantial false-positive burden.
- The available repository does not establish clinical representativeness, calibration, subgroup fairness, or prospective performance.
- The notebook's categorical imputation is performed separately on training and test frames, and its categorical pipeline places one-hot encoding before imputation. These should be corrected in a future retraining workflow.
- The checked-in prediction CSV is based on the notebook's held-out subset and its row index; it is not confirmed to represent a separate 75,000-row external test dataset.
- Serialized scikit-learn pipelines can be incompatible across package versions.

## Ethical use

Treat every output as a software demonstration only. Do not expose real patient information in a public app. Avoid collecting identifiable or sensitive health data. Clearly communicate uncertainty and limitations.

## Maintenance

Before any new release, validate the model artifact in the target runtime, test representative inputs and both predicted classes, verify the exact feature schema, and rerun evaluation on a genuinely independent dataset.
