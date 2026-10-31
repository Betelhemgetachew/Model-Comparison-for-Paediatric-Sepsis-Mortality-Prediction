# Pediatric Sepsis: In-Hospital Mortality Prediction

Predicting in-hospital mortality among children admitted with suspected sepsis, using admission-time clinical, demographic, and household data. The project covers data cleaning, clinically informed imputation, hypothesis testing, ordinal/label encoding, and a comparison of four gradient-boosting and tree-based classifiers tuned for a highly imbalanced outcome.

## Dataset

| Item | Detail |
|---|---|
| Source file | `SyntheticData_Training.csv` (synthetic pediatric sepsis data) |
| Records | 2,686 patients |
| Raw features | 138 columns (23 float, 3 int, 112 categorical) |
| Target | `inhospital_mortality` (0 = survived, 1 = died) |
| Class balance | 2,567 survived / 119 died (≈ 4.4% mortality) |
| Age variable | `agecalc_adm`, in months (≈ 6–62 months) |

Feature groups include vital signs (heart rate, respiratory rate, blood pressure, temperature, SpO₂), anthropometrics (height, weight, MUAC), labs (hematocrit, lactate, glucose), Blantyre Coma Scale components, vaccination history, symptoms, comorbidities, birth history, caregiver/maternal information, and household environment (water source, cooking fuel, bed-net use).

### Reference files (WHO growth standards)

Used to impute missing anthropometrics by age and sex:

- `girls_length_height_for_age.csv`, `boys_length_height_for_age.csv`
- `girls_weight_for_age.csv`, `boys_weight_for_age.csv`
- `acfa-girls-3-5-zscores.csv`, `acfa-boys-3-5-zscores.csv` (arm circumference-for-age)

## Requirements

```
pandas
numpy
scipy
scikit-learn
imbalanced-learn
matplotlib
seaborn
lightgbm
xgboost
catboost
```

Install with:

```bash
pip install pandas numpy scipy scikit-learn imbalanced-learn matplotlib seaborn lightgbm xgboost catboost
```

Place the dataset and WHO reference CSVs in the same folder as the notebook, then run all cells in order.

## Methodology

### 1. Data cleaning

**Columns removed**
- 21 `admitabx_adm___*` columns: antibiotics given at admission are interventions, not pre-treatment predictors.
- `spo2other_adm` (93.3% missing), `nonexclbreastfed_adm` (96.9% not applicable), `vaccmeaslessource_adm` (35.2% missing, low relevance), and `lactate2_mmolpl_adm` (post-intervention measurement).

**Numeric imputation**

| Variable | Strategy |
|---|---|
| Systolic / diastolic BP | Mean (skewness between −0.5 and 0.5) |
| Height, weight | WHO median for age (rounded months) and sex |
| MUAC | WHO arm-circumference median (SD0) for age and sex |
| Hematocrit | Mean within age-group × sex |
| Lactate | Median within a high-risk group (SpO₂ < 90% or shock index > 1) |
| SpO₂ site 1 / 2 | Cross-filled from the other site, then median by age group |
| SQI 1 / 2 | Cross-filled from the other site, then overall median |
| Remaining numeric columns | Median |

**Categorical imputation**

| Variable | Strategy |
|---|---|
| `bcseye_adm` | "Fails to watch or follow" (single case with no motor response) |
| `bcsverbal_adm` | Mode among patients with the same eye and motor scores |
| `hivstatus_adm` | New category: `NotTested` |
| `hctpretransfusion_adm` | "Yes" if hematocrit < 10 or > 50, otherwise mode |
| Remaining categorical columns | Mode |

Helper columns created during imputation (age groups, WHO medians, shock index, high-risk flag) were dropped afterwards, leaving 113 columns.

### 2. Hypothesis testing

- **Chi-square test of independence** for categorical features vs. mortality.
- **Mann–Whitney U test** for continuous features (non-parametric, chosen because clinical measurements are often non-normal).

At p < 0.01, 20 features were significant, including oxygen availability, SpO₂ on oxygen, respiratory distress, capillary refill, all three Blantyre Coma Scale components, BCG scar, feeding status, breastfeeding duration, bed-net use, weight, MUAC, and lactate.

### 3. Encoding

- **Ordinal encoding** with clinically ordered categories for 19 variables (e.g., Blantyre Coma Scale components, oxygen availability, feeding status, vaccine doses, travel time, maternal education, prior hospitalisation, bed-net use).
- **Label encoding** for the remaining nominal variables.

Ordering decisions are documented in the notebook. For example, "Oxygen available and not being used" is ranked above "being used" because the child did not need oxygen, and "Currently exclusively breastfed" is placed just above "5 months" because the youngest admitted child was about 5 months old.

### 4. Sparse feature review

Symptom and comorbidity columns that were nearly constant were checked against mortality rates. `symptoms_adm___15`, `comorbidity_adm___10`, `comorbidity_adm___11`, and `comorbidity_adm___12` showed class separation and were retained. Constant columns (`symptoms_adm___18`, `comorbidity_adm___4`) carry no information.

## Models

| Model | Features | Imbalance handling | Validation |
|---|---|---|---|
| CatBoost | 41 selected clinical features | SMOTE + StandardScaler in pipeline | 5-fold stratified CV |
| LightGBM | All features | `class_weight='balanced'` | 5-fold stratified CV |
| Random Forest | Top 30 by importance | SMOTE (0.7) + balanced class weights, GridSearch on recall, threshold tuned with F2 | 80/20 split + 5-fold CV (SMOTE inside folds) |
| XGBoost | All features | `scale_pos_weight` | 80/20 split + 5-fold CV |

## Results (mortality class = 1)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| CatBoost (5-fold CV) | 0.31 | 0.24 | 0.27 | 0.759 |
| LightGBM (5-fold CV) | 0.57 | 0.26 | 0.36 | 0.82 |
| Random Forest, test set (threshold 0.10) | 0.09 | 0.67 | 0.15 | 0.676 |
| Random Forest, 5-fold CV (threshold 0.10) | 0.117 ± 0.013 | 0.737 ± 0.088 | 0.201 ± 0.022 | 0.829 ± 0.022 |
| XGBoost, test set | 0.17 | 0.12 | 0.14 | 0.671 |
| XGBoost, 5-fold CV | – | – | – | 0.763 |

**Interpretation**
- Accuracy is uninformative here: predicting "survived" for everyone already gives ≈ 95.6%.
- LightGBM gives the best balance of precision and AUC but misses roughly three in four deaths at the default 0.5 threshold.
- The threshold-tuned Random Forest catches about two-thirds to three-quarters of deaths, at the cost of many false alarms (≈ 1 true death per 9–10 flagged patients). For a screening tool where missing a death is costlier than an extra review, this trade-off may be preferable.
- The most important Random Forest features were oxygen availability, the Blantyre Coma Scale components, vaccination status, MUAC, heart rate, age, and respiratory rate, which is broadly consistent with the hypothesis-testing results.

Output plots saved by the notebook: `precision_recall_curve.png`, `roc_curve.png`.

## Limitations and next steps

- **Possible leakage in the LightGBM and XGBoost models.** `X` is built from all columns except the target, so it includes `lengthadm` (length of admission, only known after the outcome) and `studyid_adm` (an identifier). Both should be removed before reporting these models.
- **Early stopping on the evaluation fold.** In the CatBoost and LightGBM loops, the same fold is used for early stopping and for scoring, which makes the CV estimates optimistic. A separate inner validation split would fix this.
- **Threshold chosen on the test set.** The Random Forest threshold was chosen by evaluating the test set, so the test metrics are not fully held-out. The threshold should be chosen within cross-validation on the training data.
- **GridSearch on SMOTE-resampled data.** Synthetic samples leak across CV folds during tuning. Putting SMOTE inside an `imblearn` pipeline passed to `GridSearchCV` would avoid this.
- **Imputation before splitting.** Means, medians, and modes were computed on the full dataset. Fitting imputers on training folds only would be stricter.
- **Data entry anomalies.** Some MUAC values (e.g., 14–17) appear to be recorded in cm rather than mm and should be corrected.
- **Synthetic data.** Results may not generalise to real clinical populations without external validation.
- **Next steps:** calibration analysis, SHAP-based interpretation, and a single consistent evaluation protocol across all four models.

## Project structure

```
├── sepsis_mortality_prediction.ipynb       # Main notebook
├── SyntheticData_Training.csv              # Dataset
├── girls_length_height_for_age.csv         # WHO reference tables
├── boys_length_height_for_age.csv
├── girls_weight_for_age.csv
├── boys_weight_for_age.csv
├── acfa-girls-3-5-zscores.csv
├── acfa-boys-3-5-zscores.csv
├── precision_recall_curve.png              # Generated output
├── roc_curve.png                           # Generated output
└── README.md
```

## Author

Betelhem Getachew Kebede
Department of Data Science & Analytics, USIU-Africa
