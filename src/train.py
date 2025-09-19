"""
Diabetes Risk Prediction Model
Simple, reproducible pipeline for diabetes classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import joblib
import os
import warnings
try:
    from .visual import create_model_performance_plots
except ImportError:
    from visual import create_model_performance_plots
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_prepare_data():
    """Load diabetes data and prepare for modeling"""
    print("Loading diabetes dataset...")
    
    # Load data
    df = pd.read_csv('data/diabetes_data.csv')
    print(f"Dataset loaded: {df.shape[0]:,} records, {df.shape[1]} features")
    
    # Create binary target (0: No Diabetes, 1: Diabetes/Prediabetes)
    df['has_diabetes'] = (df['Diabetes_012'] > 0).astype(int)
    
    # Select key features for modeling
    feature_cols = [
        'BMI', 'Age', 'PhysActivity', 'Income', 'HighBP', 'HighChol',
        'MentHlth', 'PhysHlth', 'Sex', 'Fruits', 'Veggies', 'GenHlth'
    ]
    
    X = df[feature_cols]
    y = df['has_diabetes']
    
    print(f"Target distribution:")
    print(f"  No Diabetes: {(y==0).sum():,} ({(y==0).mean():.1%})")
    print(f"  Has Diabetes: {(y==1).sum():,} ({(y==1).mean():.1%})")
    
    return X, y, feature_cols

def train_model(X_train, y_train):
    """Train Random Forest model"""
    print("Training Random Forest model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate model and return metrics"""
    print("Evaluating model performance...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'predictions': {
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        },
        'feature_importance': feature_importance
    }
    
    print(f"Model Performance:")
    print(f"  ROC-AUC: {roc_auc:.3f}")
    print(f"  PR-AUC: {pr_auc:.3f}")
    print(f"  Sensitivity: {sensitivity:.3f}")
    print(f"  Specificity: {specificity:.3f}")
    
    return metrics

def analyze_thresholds(y_test, y_pred_proba):
    """Analyze model performance at different thresholds"""
    print("Analyzing threshold trade-offs...")

    # Test different thresholds
    thresholds = [0.3, 0.5, 0.6]
    threshold_results = []

    for threshold in thresholds:
        # Make predictions at this threshold
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        threshold_results.append({
            'Threshold': threshold,
            'Precision': round(precision, 3),
            'Recall': round(recall, 3),
            'Specificity': round(specificity, 3),
            'F1_Score': round(f1_score, 3)
        })

        print(f"  Threshold {threshold}: Precision={precision:.3f}, Recall={recall:.3f}, Specificity={specificity:.3f}")

    # Create DataFrame and save results
    threshold_df = pd.DataFrame(threshold_results)

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    threshold_df.to_csv('results/threshold_analysis.csv', index=False)

    print("Threshold analysis complete!")
    print("Key Trade-offs:")
    print("  Lower thresholds catch more diabetes cases but increase false alarms")
    print("  Higher thresholds reduce false alarms but miss more cases")
    print("Threshold analysis saved: results/threshold_analysis.csv")

    return threshold_df

def calibration_check(y_test, y_pred_proba):
    """Check model calibration with a reliability diagram"""
    print("Running calibration check...")

    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    # Calculate calibration curve (10 bins)
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

    # Save calibration results as CSV
    calib_df = pd.DataFrame({
        'Predicted_Prob': prob_pred,
        'True_Prob': prob_true
    })
    os.makedirs('results', exist_ok=True)
    calib_df.to_csv('results/calibration_results.csv', index=False)

    # Plot reliability diagram
    plt.figure(figsize=(6,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model Calibration')
    plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/calibration_curve.png')
    plt.close()

    # Print plain-English interpretation
    print("Calibration analysis complete!")
    print("Key Insight: If points are close to the diagonal, predictions are well-calibrated.")
    print("Results saved: results/calibration_results.csv and results/plots/calibration_curve.png")

    return calib_df

def create_visualizations(metrics):
    """Create and save key visualizations using visual.py"""
    print("Creating visualizations using visual.py...")

    # Ensure plots directory exists
    os.makedirs('results/plots', exist_ok=True)

    # Use the enhanced visualization suite from visual.py
    create_model_performance_plots(metrics)

    print("Visualizations saved to results/plots/")

def save_results(model, metrics, feature_names):
    """Save model and results"""
    print("Saving model and results...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save model
    joblib.dump(model, 'models/diabetes_model.joblib')
    
    # Save metrics
    results_df = pd.DataFrame({
        'metric': ['ROC-AUC', 'PR-AUC', 'Sensitivity', 'Specificity', 'Precision'],
        'value': [metrics['roc_auc'], metrics['pr_auc'], 
                 metrics['sensitivity'], metrics['specificity'], metrics['precision']]
    })
    results_df.to_csv('results/model_metrics.csv', index=False)
    
    # Save feature importance
    metrics['feature_importance'].to_csv('results/feature_importance.csv', index=False)
    
    print("Model saved: models/diabetes_model.joblib")
    print("Results saved: results/model_metrics.csv")
    print("Feature importance saved: results/feature_importance.csv")

def main():
    """Main pipeline"""
    print("=" * 50)
    print("DIABETES PREDICTION MODEL PIPELINE")
    print("=" * 50)
    
    # Load and prepare data
    X, y, feature_names = load_and_prepare_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData split completed:")
    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Test set: {X_test.shape[0]:,} samples")
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, feature_names)

    # Analyze different thresholds
    threshold_df = analyze_thresholds(y_test, metrics['predictions']['y_pred_proba'])

    # Create visualizations
    create_visualizations(metrics)
    
    # Save results
    save_results(model, metrics, feature_names)
    
    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("Generated files:")
    print("  - 5 visualization charts in results/plots/")
    print("  - Trained model: models/diabetes_model.joblib")
    print("  - Performance metrics: results/model_metrics.csv")
    print("  - Feature importance: results/feature_importance.csv")
    print("  - Threshold analysis: results/threshold_analysis.csv")

if __name__ == "__main__":
    main()