# Cancer Survival Prediction Using Machine Learning and Streamlit Deployment

**Capstone Project Report**

**Author:** McDonald Chibueze Madufor  
**Project repository:** https://github.com/McDonson1/Cancer-Survival-Predictor  
**Application:** Streamlit deployment (insert the public app URL before submission)

---

## Abstract

This project demonstrates an end-to-end supervised machine-learning workflow for a binary classification task using a cancer-related dataset. The target variable, `Survival Prediction`, is represented as `Yes` or `No`. Several classification algorithms and class-imbalance techniques were explored. The selected approach was a Gradient Boosting classifier combined with ADASYN oversampling, embedded in a preprocessing pipeline and deployed through Streamlit.

The notebook reports 60.02% accuracy, 99.91% recall, 60.05% precision and a 75.02% F1-score on a random 20% hold-out split. The results show that the model identifies nearly all positive-class examples in that split, but its modest precision indicates a considerable false-positive risk. The project therefore demonstrates technical implementation rather than clinical readiness. The deployed interface accepts 28 input features and returns a binary class label. This report documents the workflow, evaluation, deployment, limitations and recommended improvements.

## 1. Introduction

Machine learning can identify patterns in structured datasets and support exploratory analysis in many domains. In healthcare-related applications, however, model outputs require particular caution because errors can have serious consequences. A model that performs well on one dataset may not generalize to other hospitals, populations, countries or periods.

The goal of this project was to develop a learning prototype that predicts the dataset's binary cancer-survival label and make the fitted model accessible through a simple web interface. The work also provided practical experience with data preparation, exploratory analysis, preprocessing, class-imbalance handling, model comparison, evaluation, serialization and deployment.

The project is not intended to provide a medical diagnosis or an individual's clinical prognosis. It is an educational demonstration based on the available dataset and its labels.

## 2. Problem Statement and Objectives

The task is a supervised binary classification problem. Given the input fields in a record, the model predicts one of two target classes: `Yes` or `No`.

The project objectives were to:

1. Inspect and prepare the structured dataset for machine learning.
2. Explore the target distribution and relationships among available variables.
3. Compare classification algorithms and resampling techniques.
4. Build a preprocessing and classification pipeline.
5. Evaluate the selected model with classification metrics and a confusion matrix.
6. Save the fitted pipeline and integrate it into a Streamlit application.
7. Document limitations and improvements needed for more reliable evaluation.

## 3. Dataset and Data Preparation

The notebook uses a labelled training dataset with a mixture of categorical and numerical variables. The 28 predictors supplied to the deployed application include Alcohol Consumption, Cancer Stage, Country, Diabetes, Diabetes History, Diet Risk, Early Detection, Family History, Gender, Genetic Mutation, Healthcare Access, Healthcare Costs, Heart Disease History, Hypertension, Incidence Rate per 100K, Inflammatory Bowel Disease, Insurance Costs, Insurance Status, Mortality Rate per 100K, Non Smoker, Obesity BMI, Physical Activity, Screening History, Smoking History, Treatment Type, Tumor Size (mm), Urban or Rural and Age.

The target field is `Survival Prediction`. The notebook maps `No` to 0 and `Yes` to 1 for model training. Categorical and numerical predictors are identified separately for preprocessing.

The exploratory and preparation stages inspect missing values, category values, distributions and relationships. The notebook also replaces a question-mark marker in Healthcare Access with a missing value and fills it using the mode. One improvement is required here: the current notebook calculates the mode separately in the training and test frames. In a more rigorous workflow, the imputer should be fitted only on the training data and then reused for validation and test data.

The notebook's final preprocessing pipeline uses one-hot encoding for categorical variables and standard scaling for numerical variables. Its categorical pipeline currently places the encoder before the imputer, and its numerical pipeline places the scaler before the imputer. For future retraining, imputation should precede encoding or scaling so missing values are handled consistently.

## 4. Exploratory Data Analysis

Exploratory data analysis was used to understand the structure and quality of the dataset before fitting models. The notebook includes inspection of data types and unique values, target-class distribution and visual exploration.

This stage is important because class imbalance can cause a classifier to favor the majority class. A model may achieve apparently reasonable accuracy while performing poorly on the less frequent class. The project therefore explored resampling techniques as part of model development.

The analysis should be interpreted as descriptive exploration of the supplied dataset. Relationships observed in the data do not establish causation, and the variables should not be interpreted as independently validated clinical risk factors.

## 5. Methodology

### 5.1 Train-validation split

The final model-development section uses an 80/20 random train-test split with `random_state=66`. The 20% subset is used to calculate the reported evaluation metrics. The earlier model-comparison class uses a separate split with `random_state=44`; its rounded comparison results should not be confused with the final dedicated evaluation.

A recommended improvement is to use a stratified split so the target-class proportions are preserved in both subsets. A separate external test dataset should remain untouched until the final evaluation.

### 5.2 Preprocessing

A ColumnTransformer applies separate transformations to categorical and numerical columns. Categorical features are one-hot encoded with unknown categories ignored. Numerical features are standardized. Missing-value imputation is included in both pipelines, although the order should be corrected before future retraining.

Keeping preprocessing inside the saved pipeline is useful because the same fitted transformations can be applied during inference. This reduces the risk of manually transforming application inputs differently from training inputs.

### 5.3 Model comparison and class imbalance

The notebook explores AdaBoost, Random Forest, Gradient Boosting, XGBoost and Decision Tree classifiers. It also compares SMOTE, RandomOverSampler, RandomUnderSampler and ADASYN.

The notebook's model comparison ranks results with emphasis on recall. The selected approach is Gradient Boosting with ADASYN. ADASYN generates synthetic minority-class examples during training to address class imbalance. Resampling must be performed only on training data; it must not be applied to the held-out evaluation data.

### 5.4 Final pipeline

The selected fitted pipeline combines the ColumnTransformer, ADASYN and GradientBoostingClassifier. The resulting pipeline is serialized to `GradientBoosting.pkl` using pickle. The Streamlit app loads this artifact and passes it a DataFrame containing the expected feature names.

## 6. Model Evaluation

The final dedicated evaluation cells in the notebook report:

| Metric | Score |
|---|---:|
| Accuracy | 0.600244 (60.02%) |
| Recall | 0.999099 (99.91%) |
| Precision | 0.600541 (60.05%) |
| F1-score | 0.750169 (75.02%) |

Accuracy measures the proportion of all predictions that are correct. Recall measures the proportion of actual positive cases identified by the model. Precision measures the proportion of positive predictions that are correct. The F1-score is the harmonic mean of precision and recall.

The recall is extremely high, but precision is much lower. This combination indicates that the model predicts the positive class frequently and may label many negative examples as positive. Such false positives are important and should be quantified using the confusion matrix and class-specific counts. Accuracy alone is also insufficient when classes are imbalanced.

The reported values are from one random hold-out split. They should not be presented as evidence that the model will achieve the same performance on new populations or independent datasets. Further evaluation should include specificity, balanced accuracy, ROC-AUC, precision-recall AUC, calibration, confidence intervals and subgroup performance. Cross-validation and external validation are also recommended.

## 7. Streamlit Deployment

The application is implemented in `app.py`. It presents a form with 28 fields, including select boxes for categorical variables and numeric inputs for continuous variables. When the user selects the prediction button, the application constructs a one-row pandas DataFrame with the expected feature names and calls the saved pipeline's `predict` method.

The model artifact is loaded from a path relative to the application file and cached as a Streamlit resource. The app converts the numeric model output into the readable labels `Yes` and `No`. Error handling is included for a missing model file, model-loading failures and prediction failures.

The application and dependency file are hosted in the GitHub repository and deployed through Streamlit Community Cloud. The deployment was tested by the project owner, who confirmed that the application loads and produces predictions. The deployment check demonstrates that the application can run; it does not independently verify clinical accuracy.

## 8. Repository and Reproducibility

The repository includes the training notebook, Streamlit application, requirements file, serialized model artifacts, an existing report and a sample prediction CSV. The README should be used as the entry point for setup and project navigation.

Pickle artifacts depend on compatible Python and package versions. The deployed requirements pin the core numerical and machine-learning packages. Anyone reproducing the project should use a compatible runtime and should not assume that a pickle created with one scikit-learn version will load safely with every other version.

The sample prediction CSV in the repository was generated from the notebook's 20% held-out subset, with the subset index used as the ID. It is therefore not verified as a prediction file for a separate external test dataset. A true external submission file can only be generated when the original test dataset and its actual ID column are available.

## 9. Limitations and Ethical Considerations

Several limitations affect interpretation of the results:

- Evaluation is based on one random hold-out split rather than independent external validation.
- The high recall and modest precision suggest a potentially large number of false positives.
- The notebook's imputation order and separate train/test mode filling should be improved in a future training run.
- The sample prediction file does not establish that all rows in a separate 75,000-row test dataset have been predicted.
- The repository does not establish clinical representativeness, calibration, prospective performance or fairness across demographic groups.
- The model's class label is not a calibrated probability and should not be treated as an individual's survival likelihood.

Health-related predictions can influence consequential decisions. This prototype must not be used for diagnosis, treatment selection, patient triage or personal prognosis. Real-world use would require appropriate clinical validation, governance, privacy safeguards, expert oversight and regulatory review.

## 10. Recommendations for Future Work

Future development should begin by correcting the preprocessing pipeline so imputers are fitted on training data and applied consistently, with imputation before encoding and scaling. The train-test split should be stratified, and validation and external test datasets should be kept separate. The training process should be rerun after these changes, because a corrected pipeline is a new model and must be re-evaluated before replacing the deployed artifact.

The evaluation should report confusion-matrix counts, specificity, balanced accuracy, ROC-AUC and precision-recall AUC. Cross-validation and independent external validation should be added. The project should also assess subgroup performance, calibration and the impact of false-positive and false-negative errors.

For deployment, automated checks should verify that the model file exists, the input schema matches the fitted pipeline, and the application can return valid outputs. The repository should record the training environment and provide clear setup instructions.

## 11. Conclusion

This project demonstrates an end-to-end machine-learning workflow, from exploratory analysis and model comparison to serialization and Streamlit deployment. Gradient Boosting with ADASYN was selected from the approaches explored, and the fitted pipeline was integrated into an interactive application that returns a binary prediction.

The notebook reports 60.02% accuracy, 99.91% recall, 60.05% precision and 75.02% F1-score on its final hold-out split. These results show why multiple evaluation metrics are necessary: a very high recall does not by itself establish strong overall predictive performance. The false-positive burden, generalization and data-processing workflow require further investigation.

The project is therefore best presented as an educational prototype and a foundation for further experimentation, not as a clinically reliable tool. Its main achievement is demonstrating the practical stages of building, evaluating, packaging and deploying a machine-learning application while identifying the validation and ethical work required before any consequential use.
