"""
Simple test to verify the diabetes prediction pipeline works correctly
"""

import os
import pandas as pd
import joblib

def test_pipeline_outputs():
    """Test that all expected files are generated"""
    print("Testing pipeline outputs...")
    
    # Check required files exist
    required_files = [
        'models/diabetes_model.joblib',
        'results/model_metrics.csv',
        'results/feature_importance.csv',
        # Data exploration plots (5)
        'results/plots/physical_activity_diabetes.png',
        'results/plots/age_diabetes_distribution.png',
        'results/plots/bmi_income_heatmap.png',
        'results/plots/mental_health_analysis.png',
        'results/plots/nutrition_composite_analysis.png',
        # Model performance plots (5)
        'results/plots/roc_curve.png',
        'results/plots/feature_importance.png',
        'results/plots/confusion_matrix.png',
        'results/plots/precision_recall_curve.png',
        'results/plots/performance_summary.png'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"FAIL: Missing files: {missing_files}")
        return False
    
    print("PASS: All required files exist")
    
    # Check metrics are reasonable
    metrics_df = pd.read_csv('results/model_metrics.csv')
    roc_auc = metrics_df[metrics_df['metric'] == 'ROC-AUC']['value'].iloc[0]
    
    if roc_auc < 0.7:
        print(f"FAIL: ROC-AUC too low: {roc_auc:.3f}")
        return False
    
    print(f"PASS: ROC-AUC is good: {roc_auc:.3f}")
    
    # Check model can be loaded
    try:
        model = joblib.load('models/diabetes_model.joblib')
        print("PASS: Model loads successfully")
    except Exception as e:
        print(f"FAIL: Cannot load model: {e}")
        return False
    
    # Check feature importance file
    importance_df = pd.read_csv('results/feature_importance.csv')
    if len(importance_df) < 10:
        print(f"FAIL: Feature importance file too short: {len(importance_df)} rows")
        return False
    
    print(f"PASS: Feature importance file has {len(importance_df)} features")
    
    # Final validation: Check total plot count
    import glob
    plot_files = glob.glob('results/plots/*.png')
    if len(plot_files) < 10:
        print(f"FAIL: Expected 10 plots, found {len(plot_files)}")
        return False

    print(f"PASS: Found {len(plot_files)} visualization files")
    print("ALL TESTS PASSED!")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("DIABETES PIPELINE TEST")
    print("=" * 50)
    
    success = test_pipeline_outputs()
    
    if success:
        print("\nSUCCESS: Pipeline is working correctly!")
        print("This project is ready for recruiters to review.")
    else:
        print("\nERROR: Pipeline has issues that need fixing.")
        print("Run 'python src/train.py' to generate required files.")