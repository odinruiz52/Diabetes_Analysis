# Diabetes Risk Prediction using Machine Learning

Predicting diabetes risk using machine learning on 250k+ patient health records from the BRFSS survey.

## Problem Statement

Diabetes affects millions globally and early detection can prevent serious complications. This project builds a machine learning model to identify individuals at high risk of diabetes using common health indicators. By analyzing lifestyle factors, demographics, and basic health metrics, we can help healthcare providers focus preventive care on those who need it most.

## Dataset Overview

**Source**: Behavioral Risk Factor Surveillance System (BRFSS) 2015 Survey
**Size**: 253,680 patient records with 22 health features
**Target**: Diabetes status (No Diabetes, Prediabetes, Diabetes)

**Key Features**:
- **BMI**: Body Mass Index (12-98 range)
- **Age**: Age groups from 18-24 to 80+ years
- **Physical Activity**: Whether person exercises regularly
- **General Health**: Self-reported health status (1-5 scale)
- **High Blood Pressure**: History of hypertension
- **Income**: Household income levels (8 brackets)
- **Mental/Physical Health**: Days of poor health in past month

## Methodology

**Step 1: Data Exploration**
- Analyzed 253k health records for patterns and relationships
- Visualized diabetes prevalence across age, income, and lifestyle factors
- Identified key risk factor associations

**Step 2: Data Preparation**
- Cleaned and validated health survey data
- Handled missing values using healthcare-appropriate methods
- Created binary diabetes outcome (diabetes/prediabetes vs. healthy)

**Step 3: Model Training**
- Trained Random Forest classifier on 80% of data (203k records)
- Selected 12 most predictive health features
- Used cross-validation to ensure model reliability

**Step 4: Model Evaluation**
- Tested on 20% holdout set (50k records)
- Measured accuracy, precision, recall, and clinical metrics
- Analyzed feature importance and model performance

## Results Summary

**Model Performance on Test Set (50,736 patients):**

| Metric | Value |
|--------|-------|
| **ROC-AUC** | 0.818 |
| **Precision** | 0.609 |
| **Recall (Sensitivity)** | 0.134 |
| **Specificity** | 0.984 |
| **F1-Score** | 0.219 |

**What This Means**: The model is excellent at correctly identifying healthy patients (98% specificity) but conservative in flagging diabetes cases, catching only 13% of actual diabetes patients. This creates a very reliable screening tool that minimizes false alarms but may miss some at-risk individuals. For healthcare screening, this trade-off reduces unnecessary follow-ups while maintaining high confidence in positive predictions.

## Key Findings

**Top Risk Factors**: General health status, high blood pressure, and BMI emerged as the strongest predictors of diabetes risk, accounting for over 50% of the model's decision-making power. Age and high cholesterol were also important but secondary factors.

**Model Strength**: The 98% specificity means the model rarely flags healthy people as diabetic, making it excellent for reducing false alarms in screening programs. When the model says someone is at risk, there's a 61% chance they actually have diabetes.

**Population Insights**: The data revealed clear patterns linking lifestyle factors (physical activity, diet) and socioeconomic status (income level) to diabetes outcomes, confirming that diabetes risk extends beyond just medical factors.

## Limitations & Next Steps

**Key Limitations**: The model misses 87% of actual diabetes cases due to conservative thresholds, which could delay treatment for many patients. The data comes from 2015 surveys where people self-reported their health status, potentially introducing bias. Survey respondents may also not represent the broader population accurately.

**Next Steps**: Adjust the prediction threshold to catch more diabetes cases, even if it means more false alarms. Test the model separately for different age groups and genders to ensure fair performance across populations. Consider adding more recent data or lab test results to improve accuracy.

## How to Run

Follow these steps to reproduce the project results:

1. **Make sure you have Python 3.8 or higher** installed.
2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
3. **From the project root, run the training pipeline:**
   ```bash
   python src/train.py
   ```

**Expected Results:**
- The script will finish in about 2–3 minutes
- Results will be saved automatically:
  - Trained model in `models/`
  - Performance metrics in `results/model_metrics.csv`
  - Feature importance in `results/feature_importance.csv`
  - Plots in `results/plots/`

This single command handles data preparation, training, and results generation.
