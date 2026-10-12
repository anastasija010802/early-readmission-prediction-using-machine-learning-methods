# Early Readmission Prediction Using Machine Learning Methods


## Overview

This project focuses on predicting early hospital readmission within 30 days among patients with diabetes-related inpatient encounters. Machine learning algorithms and different techniques for handling class imbalance were applied to identify patients at increased risk of early readmission.

The final model was based on the XGBoost algorithm, with its performance evaluated using the F1-score.


## Dataset

The main dataset used in this project is the publicly available Diabetes 130 US hospitals for years 1999-2008 dataset from Kaggle. It contains 101,766 inpatient encounters involving patients with diabetes and includes demographic information, diagnoses, laboratory tests, medications, and hospitalization characteristics. The dataset also contains information on readmission, which was used to define the target variable.

The data originate from the Health Facts database developed by Cerner Corporation and were collected between 1999 and 2008 from 130 hospitals and integrated healthcare systems in the United States. The original data were de-identified and later made publicly available through the UCI Machine Learning Repository and Kaggle. The original dataset was filtered according to the following criteria:
Diabetes had to be recorded as a diagnosis;
Laboratory tests had to be performed;
Medication had to be administered;
The length of hospital stay had to be between 1 and 14 days.

An auxiliary file, id_mapping.csv, is included in this repository and was used during data preprocessing to interpret coded attributes, including admission source, discharge disposition, and admission type. The file also documents that ? represents a missing value.

ICD-9 reference file — A CSV file converted from a Word document containing ICD-9 diagnostic code information. It was used to support the grouping of diagnoses and the creation of additional diagnosis-related features during preprocessing.


### Data Sources

Main Dataset: https://www.kaggle.com/datasets/brandao/diabetes/data
ICD-9-CM Reference Files: https://ftp.cdc.gov/pub/health_statistics/nchs/publications/icd9-cm/2008/


## Project Workflow

## 1. Exploratory data analysis 
 
 The first step in the project which includes following analyses:
Dropping the hospitalization ID — Removed because it uniquely identifies each hospital encounter.
Retaining the patient ID — Kept at this stage because it was required to split the dataset into training and test sets at the patient level.
Dropping the 'weight' and 'max_glu_serum' attributes due to the high percentage of missing values(>90%).
Imputing missing values in A1Cresult with "not_measured" and in medical_specialty with "Missing", to account for non-random missingness.
Dropping citoglipton, examide, acetohexamide, metformin-pioglitazone, metformin-rosiglitazone, glimepiride-pioglitazone, glipizide-metformin, tolbutamide, and troglitazone due to extremely low variability(<0,0206%).
Dropping hospitalizations with discharge dispositions of "Expired", "Hospice / home", "Hospice / medical facility", "Expired at home. Medicaid only, hospice.", and "Expired in a medical facility. Medicaid only, hospice." because these encounters are not relevant to readmission prediction.
The medical_specialty attribute was grouped into broader categories representing different medical specialties.

Note: All analyses and transformations performed during the EDA phase were based on established facts and clear, well-founded reasoning, ensuring that no data leakage was introduced.

Transforming the target feature: the categories ">30" and "NO" were grouped into a single category, "OK", while the category "<30" was assigned to the positive class, indicating a potential failure of the healthcare system. Readmissions occurring more than 30 days after discharge may, on the other hand, be associated with disease progression rather than a failure in the quality of care.
After transforming the target feature, its distribution is shown below:

![Target variable distribution](images/Screenshot%20(385).png)

## 2. Data engineering

The diagnosis attributes 'diag_1', 'diag_2', and 'diag_3' represent the first three digits of the ICD-9 classification codes, except for diabetes mellitus diagnoses (category 250), which were represented as follows:
250- group od diabetes melitus
250.x – a more specific diabetes mellitus category in which the fourth digit indicates the type of complication: 0 indicates no complication, while digits 1–9 indicate specific types of complications.
250.xx – the fifth digit provides additional information about the type of diabetes mellitus and its control status.
The fifth digit was removed, reducing the codes to four digits. This resulted in less specific codes and fewer distinct categories. 

Based on the diagnosis codes, the following binary attributes were created:

- has_complications
- has_circulatory
- has_endocrine_metabolic
- has_respiratory
- has_digestive
- has_genitourinary
- has_skin
- has_blood
- has_mental
- has_injury_poisoning
- has_neoplasms
- has_infections
- has_musculoskeletal
- has_nervous

Each binary attribute was created as follows: if at least one of the three diagnoses met the corresponding condition, the attribute was assigned a value of 1; otherwise, it was assigned a value of 0.


## 3. Data Preparation and Feature Selection

### 3.1. Train-Test Split

The dataset was split into training and test sets using `GroupShuffleSplit`, with `patient_nbr` as the grouping variable. This ensured that encounters belonging to the same patient were not present in both sets, reducing the risk of data leakage. The split was configured with a test size of 20% and a random state of 42.

### 3.2. Rare Category Grouping

Rare categories were grouped to reduce the number of distinct categorical values.

- **Diagnosis attributes (`diag_1_reduced`, `diag_2_reduced`, and `diag_3_reduced`):** Categories occurring in less than 0.5% of observations were mapped to broader ICD-9 diagnostic groups, such as infectious diseases, neoplasms, endocrine and metabolic diseases, and diseases of the circulatory system. External causes and supplementary factors were handled as separate groups.
- **Other categorical attributes (`payer_code`, `admission_type`, `discharge_disposition`, and `admmission_source`):** Categories with fewer than 50 observations were grouped into the `Other` category.

The category grouping rules were learned from the training data and applied to the data being transformed.

### 3.3. Missing Value Imputation

A custom imputer was implemented to handle missing values in diagnosis attributes and selected categorical attributes.

- **Diagnosis attributes (`diag_1_reduced`, `diag_2_reduced`, and `diag_3_reduced`):** Missing values were first imputed using the most frequent diagnosis associated with the combination of the other two diagnosis attributes. Any remaining missing values were replaced with the most frequent value of the corresponding diagnosis attribute.
- **`payer_code`, `race`, and `gender`:** Missing values were replaced with the most frequent category of each attribute.

The imputation rules were fitted on the training data and applied to the data being transformed.

### 3.4. Filter-Based Feature Selection

Feature selection was implemented within the modelling pipeline. Numerical and categorical attributes were evaluated separately, and the top 50% of attributes in each group were retained according to their respective ranking criteria. Redundancy filtering was then applied to remove highly associated attributes.

#### Numerical Attributes

1. **Mann–Whitney U test:** Numerical attributes were ranked according to the absolute value of the standardized Z-statistic. The top 50% of attributes were retained.
2. **Redundancy filtering:** Absolute Spearman rank correlations were calculated between the retained attributes. When the absolute correlation was at least 0.90, the lower-ranked attribute was removed.

#### Categorical Attributes

1. **Minimum variability filter:** Attributes had to contain at least two observed categories, with observations outside the dominant category accounting for at least 0.05% of the observations.
2. **Feature ranking:** The chi-square test was used to assess the association between each categorical attribute and the target variable. Cramér's V was calculated to measure the strength of the association, and the top 50% of attributes were retained according to their Cramér's V scores.
3. **Redundancy filtering:** Cramér's V was calculated between pairs of retained categorical attributes. When the coefficient was at least 0.90, the lower-ranked attribute was removed.

The final feature set consisted of the numerical and categorical attributes retained after ranking and redundancy filtering.

## 4. Experiments and Model Evaluation

### 4.1. Experimental Setup

All experiments were conducted using a machine learning pipeline that integrated missing value imputation, rare category grouping, feature selection, categorical encoding, class imbalance handling, and model training. Cross-validation was performed using `GroupKFold` with five folds, grouping observations by `patient_nbr` to prevent encounters belonging to the same patient from appearing in both the training and validation subsets of a fold.

This pipeline-based approach ensured that data-dependent preprocessing and feature selection were fitted within each training fold and subsequently applied to its corresponding validation fold, reducing the risk of data leakage.

### 4.2. Preliminary Experiments

Preliminary experiments were conducted to compare alternative categorical encoding and class imbalance handling strategies before selecting the final modelling configuration.

The following categorical encoding methods were evaluated:

- One-Hot Encoding;
- Target Encoding.

The following class imbalance handling strategies were evaluated:

- No resampling;
- SMOTE;
- Random oversampling;
- Class weighting.

These combinations were initially evaluated using Logistic Regression to identify promising encoding and class imbalance handling strategies under a consistent modelling setup.

The preliminary results were relatively similar and showed limited differences between the evaluated configurations. However, One-Hot Encoding combined with class weighting achieved stable results and was therefore selected for subsequent experiments.

### 4.3. Model Comparison

Following the preliminary experiments, three classification algorithms were evaluated:

- Logistic Regression;
- Random Forest;
- XGBoost.

The selected preprocessing and class imbalance handling configuration was incorporated into the pipeline used to train and evaluate the candidate models.

Model performance was assessed using multiple evaluation metrics, including precision, recall, F1-score, and ROC-AUC, to account for the imbalanced target distribution and evaluate different aspects of classification performance.

Random Forest was excluded from further experiments because it achieved consistently low F1-scores on the selected feature set, regardless of the class imbalance handling method used. Its performance was insufficient compared with the other candidate models.

### 4.4. Hyperparameter Tuning and Model Comparison

Following the preliminary experiments, Logistic Regression and XGBoost were selected for further evaluation. Hyperparameter tuning was performed for both models using the previously described preprocessing pipeline and the selected class imbalance handling approach.

The models were compared based on their performance after hyperparameter tuning. XGBoost achieved a higher F1-score than Logistic Regression, indicating a better balance between precision and recall on this dataset.

The results of the two models are presented in the table below.

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| Accuracy | 0.664 | 0.756 |
| Precision | 0.183 | 0.213 |
| Recall | 0.559 | 0.423 |
| F1-score | 0.276 | 0.283 |
| AUC | 0.667 | 0.672 |

### Probability Calibration

Probability calibration was applied to improve the reliability of the predicted probabilities. Since the model was used to estimate the likelihood of early hospital readmission, it was important for the predicted probabilities to better reflect the actual frequency of readmissions. Calibration was performed using training data, without using the held-out test set for calibration.

Calibration Curve Before Calibration

![Calibration Curve Before Calibration](images/Screenshot (378)(1).png)

Calibration Curve After Calibration

![Calibration Curve After Calibration](images/Screenshot (379)(1).png)

After calibration, a classification threshold of 0.13 was selected, as it achieved the highest F1-score among the evaluated thresholds.


## Results and discusion

### 1. Model performance

The final model performance on the held-out test set, using this threshold, is presented in the table below.

| Metric | XGBoost |
|---|---:|
| Accuracy | 0.7091 |
| Precision | 0.1915 |
| Recall | 0.4964 |
| F1-score | 0.2764 |
| AUC | 0.6717 |

### 2. Shap analysis

1. Most Important Features

![Most Important Features](images/Screenshot (380)(1).png)

2. Feature Impact on Predictions

![Feature Impact on Predictions](images/Screenshot (381)(1).png)

### 3. Conclusion

The SHAP analysis identified discharge disposition, particularly transfers to another short-term hospital, and the use of home health services as important features for predicting early hospital readmission. The number of inpatient visits also contributed to the model's predictions. These findings suggest that information related to discharge arrangements and previous healthcare utilization may be relevant for identifying patients at higher risk of early readmission.

The predictive performance remained limited, indicating that the available features may not capture all factors associated with early readmission. Due to computational resource constraints, feature selection was performed using filter-based methods, and the number of selected features was not treated as a hyperparameter during model optimization. An additional experiment retaining 70% of the features produced similar results, without substantial improvements in predictive performance.

Future work could explore additional patient and clinical information, alternative feature selection strategies, and transformations of important features to capture potentially more complex relationships. Further investigation of model calibration and decision thresholds could also help improve the practical usefulness of predicted readmission probabilities.


##  How to Run the Project

### 1. Clone the Repository

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/anastasija010802/early-readmission-prediction-using-machine-learning-methods.git
cd early-readmission-prediction-using-machine-learning-methods
```

### 2. Download the Datasets

Download the datasets from the links provided in the **Data Sources** section of this README.

- **Main dataset:** Open the dataset link and download the dataset file.
- **ICD-9-CM reference tables:** Open the [Dtab09.zip](https://ftp.cdc.gov/pub/health_statistics/nchs/publications/icd9-cm/2008/Dtab09.zip) file, download it, and extract its contents.

Place the downloaded main dataset file and the extracted ICD-9-CM reference files in the `datasets/` directory.

### 3. Prepare the Project Directories

Ensure that the following directories exist in the project root:

```text
ML_project/
├── datasets/
├── preprocessed/
└── README.md
```

The `datasets/` directory contains the downloaded source data, while `preprocessed/` is used to store preprocessed data generated during the workflow. Create the `preprocessed/` directory manually before running the project if it does not already exist.


