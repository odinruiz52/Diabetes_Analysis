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

if __name__ == "__main__":
    main()