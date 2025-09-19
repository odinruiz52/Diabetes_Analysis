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
