# Diabetes Prediction Model – Summary Report

## Model Performance
- Accuracy: 0.850
- Precision: 0.609
- Recall (Sensitivity): 0.134
- Specificity: 0.984
- F1-Score: 0.220
- AUC: 0.818

## Threshold Analysis
See `results/threshold_analysis.csv` for detailed trade-offs.
Lower thresholds catch more cases (higher recall), while higher thresholds reduce false alarms (higher precision).

## Calibration Check
See `results/calibration_results.csv` and `results/plots/calibration_curve.png`.
If the curve is close to the diagonal, predicted probabilities are reliable.

## Fairness Check
See `results/fairness_check.csv`.
Shows performance across sex and age groups to detect disparities.

## Interpretability (SHAP Analysis)
See `results/shap_summary.csv` and `results/plots/shap_summary.png`.
Highlights the health factors that most influence predictions.
For example: General Health, High Blood Pressure, and BMI are top drivers.

---
**End of Report – All outputs saved in the `results/` folder.**
