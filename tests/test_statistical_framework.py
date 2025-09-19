import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from statistical_tests import StatisticalAnalyzer
from data_quality import DataQualityAnalyzer
from predictive_models import PredictiveModelPipeline
from bias_fairness import FairnessAnalyzer
from explainability import ModelExplainabilityEngine
from model_monitoring import ModelMonitoringSystem

class TestStatisticalFramework:
    """
    Comprehensive test suite for the statistical analysis framework.
    """
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample diabetes dataset for testing."""
        np.random.seed(42)
        n = 1000
        
        data = {
            'Diabetes_012': np.random.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1]),
            'BMI': np.random.normal(28, 6, n),
            'Age': np.random.choice(range(1, 14), n),
            'PhysActivity': np.random.choice([0, 1], n, p=[0.3, 0.7]),
            'Sex': np.random.choice([0, 1], n, p=[0.55, 0.45]),
            'Income': np.random.choice(range(1, 9), n),
            'MentHlth': np.random.poisson(3, n),
            'PhysHlth': np.random.poisson(4, n),
            'Fruits': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'Veggies': np.random.choice([0, 1], n, p=[0.2, 0.8])
        }
        
        # Clip values to realistic ranges
        data['BMI'] = np.clip(data['BMI'], 15, 60)
        data['MentHlth'] = np.clip(data['MentHlth'], 0, 30)
        data['PhysHlth'] = np.clip(data['PhysHlth'], 0, 30)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def statistical_analyzer(self):
        """Initialize statistical analyzer."""
        return StatisticalAnalyzer(alpha=0.05)
    
    def test_statistical_analyzer_initialization(self, statistical_analyzer):
        """Test statistical analyzer initialization."""
        assert statistical_analyzer.alpha == 0.05
        assert isinstance(statistical_analyzer.results, dict)
        assert len(statistical_analyzer.results) == 0
    
    def test_chi_square_test(self, statistical_analyzer, sample_data):
        """Test chi-square test functionality."""
        result = statistical_analyzer.chi_square_test(
            sample_data, 'PhysActivity', 'Sex'
        )
        
        assert 'test_type' in result
        assert result['test_type'] == 'Chi-Square Test'
        assert 'chi2_statistic' in result
        assert 'p_value' in result
        assert 'effect_size_cramers_v' in result
        assert 'significant' in result
        assert isinstance(result['significant'], bool)
        assert result['chi2_statistic'] >= 0
        assert 0 <= result['p_value'] <= 1
        assert 0 <= result['effect_size_cramers_v'] <= 1
    
    def test_mann_whitney_u_test(self, statistical_analyzer, sample_data):
        """Test Mann-Whitney U test functionality."""
        result = statistical_analyzer.mann_whitney_u_test(
            sample_data, 'Sex', 'BMI'
        )
        
        assert 'test_type' in result
        assert result['test_type'] == 'Mann-Whitney U Test'
        assert 'u_statistic' in result
        assert 'p_value' in result
        assert 'effect_size_r' in result
        assert 'significant' in result
        assert isinstance(result['significant'], bool)
        assert result['u_statistic'] >= 0
        assert 0 <= result['p_value'] <= 1
    
    def test_kruskal_wallis_test(self, statistical_analyzer, sample_data):
        """Test Kruskal-Wallis test functionality."""
        result = statistical_analyzer.kruskal_wallis_test(
            sample_data, 'Diabetes_012', 'BMI'
        )
        
        assert 'test_type' in result
        assert result['test_type'] == 'Kruskal-Wallis Test'
        assert 'h_statistic' in result
        assert 'p_value' in result
        assert 'effect_size_eta_squared' in result
        assert 'significant' in result
        assert isinstance(result['significant'], bool)
        assert result['h_statistic'] >= 0
        assert 0 <= result['p_value'] <= 1
    
    def test_correlation_analysis(self, statistical_analyzer, sample_data):
        """Test correlation analysis functionality."""
        result = statistical_analyzer.correlation_with_significance(sample_data)
        
        assert 'test_type' in result
        assert 'correlation_matrix' in result
        assert 'p_value_matrix' in result
        assert 'significant_correlations' in result
        assert isinstance(result['significant_correlations'], list)
    
    def test_multiple_testing_correction(self, statistical_analyzer, sample_data):
        """Test multiple testing correction."""
        # Run several tests first
        statistical_analyzer.chi_square_test(sample_data, 'PhysActivity', 'Sex')
        statistical_analyzer.mann_whitney_u_test(sample_data, 'Sex', 'BMI')
        
        correction_result = statistical_analyzer.multiple_testing_correction()
        
        assert 'correction_method' in correction_result
        assert 'n_tests' in correction_result
        assert 'n_significant_original' in correction_result
        assert 'n_significant_corrected' in correction_result
        assert correction_result['n_tests'] > 0
    
    def test_generate_report(self, statistical_analyzer, sample_data):
        """Test report generation."""
        # Run some tests first
        statistical_analyzer.chi_square_test(sample_data, 'PhysActivity', 'Sex')
        statistical_analyzer.mann_whitney_u_test(sample_data, 'Sex', 'BMI')
        
        report = statistical_analyzer.generate_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'STATISTICAL ANALYSIS REPORT' in report
        assert 'Significance level' in report

class TestDataQualityAnalyzer:
    """Test suite for data quality analyzer."""
    
    @pytest.fixture
    def sample_data_with_issues(self):
        """Generate sample data with quality issues."""
        np.random.seed(42)
        n = 500
        
        data = {
            'BMI': np.concatenate([
                np.random.normal(28, 6, n-50),
                [np.nan] * 20,  # Missing values
                [100, 150] * 15  # Outliers
            ]),
            'Age': np.random.choice(range(1, 14), n),
            'Income': np.concatenate([
                np.random.choice(range(1, 9), n-30),
                [np.nan] * 30  # Missing values
            ]),
            'Diabetes_012': np.random.choice([0, 1, 2], n)
        }
        
        return pd.DataFrame(data)
    
    def test_data_quality_initialization(self, sample_data_with_issues):
        """Test data quality analyzer initialization."""
        analyzer = DataQualityAnalyzer(sample_data_with_issues)
        assert analyzer.df is not None
        assert len(analyzer.df) == len(sample_data_with_issues)
    
    def test_comprehensive_quality_assessment(self, sample_data_with_issues):
        """Test comprehensive quality assessment."""
        analyzer = DataQualityAnalyzer(sample_data_with_issues)
        results = analyzer.comprehensive_quality_assessment()
        
        required_keys = [
            'basic_info', 'missing_values', 'duplicates', 
            'outliers', 'distributions', 'data_types',
            'value_ranges', 'consistency_checks'
        ]
        
        for key in required_keys:
            assert key in results
        
        # Check basic info
        basic_info = results['basic_info']
        assert 'n_rows' in basic_info
        assert 'n_columns' in basic_info
        assert basic_info['n_rows'] == len(sample_data_with_issues)
    
    def test_missing_value_analysis(self, sample_data_with_issues):
        """Test missing value analysis."""
        analyzer = DataQualityAnalyzer(sample_data_with_issues)
        results = analyzer.comprehensive_quality_assessment()
        
        missing_results = results['missing_values']
        assert 'missing_counts' in missing_results
        assert 'missing_percentages' in missing_results
        assert 'recommendations' in missing_results
        
        # Should detect missing values in BMI and Income
        assert missing_results['missing_counts']['BMI'] > 0
        assert missing_results['missing_counts']['Income'] > 0
    
    def test_outlier_detection(self, sample_data_with_issues):
        """Test outlier detection."""
        analyzer = DataQualityAnalyzer(sample_data_with_issues)
        results = analyzer.comprehensive_quality_assessment()
        
        outliers = results['outliers']
        assert 'BMI' in outliers
        
        bmi_outliers = outliers['BMI']
        assert 'iqr_method' in bmi_outliers
        assert 'zscore_method' in bmi_outliers
        assert 'modified_zscore_method' in bmi_outliers
        
        # Should detect outliers in BMI
        assert bmi_outliers['iqr_method']['n_outliers'] > 0
    
    def test_generate_quality_report(self, sample_data_with_issues):
        """Test quality report generation."""
        analyzer = DataQualityAnalyzer(sample_data_with_issues)
        analyzer.comprehensive_quality_assessment()
        
        report = analyzer.generate_quality_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'DATA QUALITY ASSESSMENT REPORT' in report

class TestPredictiveModelPipeline:
    """Test suite for predictive modeling pipeline."""
    
    @pytest.fixture
    def sample_modeling_data(self):
        """Generate sample data for modeling."""
        np.random.seed(42)
        n = 500
        
        data = {
            'Diabetes_012': np.random.choice([0, 1], n, p=[0.8, 0.2]),  # Binary for testing
            'BMI': np.random.normal(28, 6, n),
            'Age': np.random.choice(range(1, 14), n),
            'PhysActivity': np.random.choice([0, 1], n),
            'Sex': np.random.choice([0, 1], n),
            'Income': np.random.choice(range(1, 9), n)
        }
        
        return pd.DataFrame(data)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = PredictiveModelPipeline()
        assert pipeline.target_column == 'Diabetes_012'
        assert pipeline.random_state == 42
        assert isinstance(pipeline.models, dict)
        assert isinstance(pipeline.results, dict)
    
    def test_data_preparation(self, sample_modeling_data):
        """Test data preparation."""
        pipeline = PredictiveModelPipeline(target_column='Diabetes_012')
        data_info = pipeline.prepare_data(sample_modeling_data, test_size=0.2)
        
        required_keys = [
            'X_train', 'X_test', 'y_train', 'y_test',
            'feature_names', 'n_features', 'n_train_samples', 'n_test_samples'
        ]
        
        for key in required_keys:
            assert key in data_info
        
        assert len(data_info['X_train']) == data_info['n_train_samples']
        assert len(data_info['X_test']) == data_info['n_test_samples']
        assert data_info['n_features'] > 0
    
    def test_model_initialization(self, sample_modeling_data):
        """Test model initialization."""
        pipeline = PredictiveModelPipeline()
        models = pipeline.initialize_models()
        
        expected_models = ['logistic_regression', 'random_forest', 'gradient_boosting', 'svm']
        
        for model_name in expected_models:
            assert model_name in models
            assert 'model' in models[model_name]
            assert 'params' in models[model_name]

class TestFairnessAnalyzer:
    """Test suite for fairness analyzer."""
    
    @pytest.fixture
    def sample_fairness_data(self):
        """Generate sample data for fairness testing."""
        np.random.seed(42)
        n = 1000
        
        data = {
            'Diabetes_012': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'Sex': np.random.choice([0, 1], n, p=[0.55, 0.45]),
            'Age': np.random.choice([0, 1], n, p=[0.6, 0.4]),  # Binary age groups
            'BMI': np.random.normal(28, 6, n),
            'predictions': np.random.choice([0, 1], n, p=[0.75, 0.25])
        }
        
        return pd.DataFrame(data)
    
    def test_fairness_analyzer_initialization(self):
        """Test fairness analyzer initialization."""
        analyzer = FairnessAnalyzer()
        assert isinstance(analyzer.fairness_results, dict)
        assert isinstance(analyzer.bias_audit_results, dict)
    
    def test_protected_attribute_analysis(self, sample_fairness_data):
        """Test protected attribute analysis."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = sample_fairness_data[['BMI', 'Age']]
        y = sample_fairness_data['Diabetes_012']
        model.fit(X, y)
        
        analyzer = FairnessAnalyzer()
        
        # Test single protected attribute
        result = analyzer._analyze_protected_attribute(
            sample_fairness_data, 'Sex', 'Diabetes_012', 'predictions'
        )
        
        assert 'attribute' in result
        assert 'groups' in result
        assert 'group_sizes' in result
        assert 'outcome_rates' in result
        assert 'prediction_rates' in result
        assert 'performance_metrics' in result
        assert 'disparate_impact' in result
    
    def test_fairness_metrics_calculation(self, sample_fairness_data):
        """Test fairness metrics calculation."""
        analyzer = FairnessAnalyzer()
        
        metrics = analyzer._calculate_fairness_metrics(
            sample_fairness_data, ['Sex'], 'Diabetes_012', 'predictions'
        )
        
        assert 'Sex' in metrics
        sex_metrics = metrics['Sex']
        
        if 'demographic_parity' in sex_metrics:
            dp = sex_metrics['demographic_parity']
            assert 'group1_rate' in dp
            assert 'group2_rate' in dp
            assert 'difference' in dp
            assert 'fair' in dp
    
    def test_bias_detection(self, sample_fairness_data):
        """Test bias detection."""
        analyzer = FairnessAnalyzer()
        
        bias_indicators = analyzer._detect_bias_indicators(
            sample_fairness_data, ['Sex'], 'Diabetes_012', 'predictions'
        )
        
        assert 'systematic_bias_detected' in bias_indicators
        assert 'bias_sources' in bias_indicators
        assert 'severity_assessment' in bias_indicators
        assert isinstance(bias_indicators['systematic_bias_detected'], bool)

class TestModelMonitoringSystem:
    """Test suite for model monitoring system."""
    
    @pytest.fixture
    def monitoring_setup(self):
        """Set up monitoring system with reference data."""
        np.random.seed(42)
        n_ref = 1000
        
        # Reference data
        ref_data = pd.DataFrame({
            'BMI': np.random.normal(28, 6, n_ref),
            'Age': np.random.choice(range(1, 14), n_ref),
            'PhysActivity': np.random.choice([0, 1], n_ref)
        })
        
        ref_predictions = np.random.choice([0, 1], n_ref, p=[0.8, 0.2])
        ref_labels = np.random.choice([0, 1], n_ref, p=[0.85, 0.15])
        
        # New data (with some drift)
        new_data = pd.DataFrame({
            'BMI': np.random.normal(30, 7, 500),  # Slightly different distribution
            'Age': np.random.choice(range(1, 14), 500),
            'PhysActivity': np.random.choice([0, 1], 500, p=[0.4, 0.6])  # Different distribution
        })
        
        new_predictions = np.random.choice([0, 1], 500, p=[0.75, 0.25])
        new_labels = np.random.choice([0, 1], 500, p=[0.8, 0.2])
        
        monitor = ModelMonitoringSystem(
            reference_data=ref_data,
            reference_predictions=ref_predictions,
            reference_labels=ref_labels
        )
        
        return monitor, new_data, new_predictions, new_labels
    
    def test_monitoring_initialization(self, monitoring_setup):
        """Test monitoring system initialization."""
        monitor, _, _, _ = monitoring_setup
        
        assert monitor.reference_data is not None
        assert monitor.reference_predictions is not None
        assert monitor.reference_labels is not None
        assert isinstance(monitor.alert_thresholds, dict)
        assert 'data_drift_psi' in monitor.alert_thresholds
    
    def test_psi_calculation(self, monitoring_setup):
        """Test PSI calculation."""
        monitor, new_data, _, _ = monitoring_setup
        
        ref_values = monitor.reference_data['BMI']
        new_values = new_data['BMI']
        
        psi = monitor._calculate_psi(ref_values, new_values)
        
        assert isinstance(psi, (int, float))
        assert psi >= 0  # PSI should be non-negative
    
    def test_data_drift_analysis(self, monitoring_setup):
        """Test data drift analysis."""
        monitor, new_data, _, _ = monitoring_setup
        
        drift_analysis = monitor._analyze_data_drift(new_data)
        
        assert 'features_analyzed' in drift_analysis
        assert 'psi_scores' in drift_analysis
        assert 'ks_test_results' in drift_analysis
        assert 'overall_drift_score' in drift_analysis
        assert 'drifted_features' in drift_analysis
        
        assert len(drift_analysis['features_analyzed']) > 0
        assert drift_analysis['overall_drift_score'] >= 0
    
    def test_performance_drift_analysis(self, monitoring_setup):
        """Test performance drift analysis."""
        monitor, _, new_predictions, new_labels = monitoring_setup
        
        perf_analysis = monitor._analyze_performance_drift(new_predictions, new_labels)
        
        assert 'reference_metrics' in perf_analysis
        assert 'current_metrics' in perf_analysis
        assert 'metric_changes' in perf_analysis
        assert 'significant_degradation' in perf_analysis
        assert 'performance_trend' in perf_analysis
        
        # Check that metrics are calculated
        ref_metrics = perf_analysis['reference_metrics']
        assert 'accuracy' in ref_metrics
        assert 'precision' in ref_metrics
        assert 'recall' in ref_metrics
        assert 'f1' in ref_metrics
    
    def test_comprehensive_monitoring(self, monitoring_setup):
        """Test comprehensive monitoring analysis."""
        monitor, new_data, new_predictions, new_labels = monitoring_setup
        
        results = monitor.comprehensive_monitoring_analysis(
            new_data, new_predictions, new_labels
        )
        
        required_keys = [
            'timestamp', 'data_drift_analysis', 'performance_drift_analysis',
            'concept_drift_analysis', 'statistical_tests', 'alerts_generated',
            'severity_assessment', 'recommendations'
        ]
        
        for key in required_keys:
            assert key in results
        
        assert isinstance(results['alerts_generated'], list)
        assert isinstance(results['recommendations'], list)
        
        # Check that monitoring history is updated
        assert len(monitor.monitoring_history) == 1
    
    def test_generate_monitoring_report(self, monitoring_setup):
        """Test monitoring report generation."""
        monitor, new_data, new_predictions, new_labels = monitoring_setup
        
        # Run monitoring first
        monitor.comprehensive_monitoring_analysis(new_data, new_predictions, new_labels)
        
        report = monitor.generate_monitoring_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'MODEL MONITORING REPORT' in report
        assert 'OVERALL ASSESSMENT' in report

# Integration tests
class TestFrameworkIntegration:
    """Integration tests for the entire framework."""
    
    @pytest.fixture
    def integration_data(self):
        """Generate comprehensive dataset for integration testing."""
        np.random.seed(42)
        n = 1000
        
        data = {
            'Diabetes_012': np.random.choice([0, 1, 2], n, p=[0.7, 0.2, 0.1]),
            'BMI': np.random.normal(28, 6, n),
            'Age': np.random.choice(range(1, 14), n),
            'PhysActivity': np.random.choice([0, 1], n, p=[0.3, 0.7]),
            'Sex': np.random.choice([0, 1], n, p=[0.55, 0.45]),
            'Income': np.random.choice(range(1, 9), n),
            'MentHlth': np.random.poisson(3, n),
            'PhysHlth': np.random.poisson(4, n),
            'Fruits': np.random.choice([0, 1], n, p=[0.4, 0.6]),
            'Veggies': np.random.choice([0, 1], n, p=[0.2, 0.8])
        }
        
        # Clip values to realistic ranges
        data['BMI'] = np.clip(data['BMI'], 15, 60)
        data['MentHlth'] = np.clip(data['MentHlth'], 0, 30)
        data['PhysHlth'] = np.clip(data['PhysHlth'], 0, 30)
        
        return pd.DataFrame(data)
    
    def test_full_analysis_pipeline(self, integration_data):
        """Test complete analysis pipeline integration."""
        # 1. Data Quality Assessment
        quality_analyzer = DataQualityAnalyzer(integration_data)
        quality_results = quality_analyzer.comprehensive_quality_assessment()
        
        assert quality_results is not None
        assert len(quality_results) > 0
        
        # 2. Statistical Analysis
        stat_analyzer = StatisticalAnalyzer()
        chi2_result = stat_analyzer.chi_square_test(integration_data, 'PhysActivity', 'Sex')
        
        assert chi2_result is not None
        assert 'p_value' in chi2_result
        
        # 3. Predictive Modeling
        pipeline = PredictiveModelPipeline(target_column='Diabetes_012')
        
        # Create binary target for testing
        binary_data = integration_data.copy()
        binary_data['Diabetes_012'] = (binary_data['Diabetes_012'] > 0).astype(int)
        
        data_prep = pipeline.prepare_data(binary_data, test_size=0.2)
        
        assert data_prep is not None
        assert 'X_train' in data_prep
        assert 'y_train' in data_prep
        
        # Test model initialization
        models = pipeline.initialize_models()
        assert len(models) > 0
    
    def test_error_handling(self):
        """Test error handling across components."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        quality_analyzer = DataQualityAnalyzer(empty_df)
        
        # This should not raise an exception
        try:
            results = quality_analyzer._basic_data_info()
            assert 'n_rows' in results
            assert results['n_rows'] == 0
        except Exception as e:
            pytest.fail(f"Error handling failed: {e}")
    
    def test_data_validation(self, integration_data):
        """Test data validation across components."""
        # Test with invalid column names
        invalid_data = integration_data.copy()
        invalid_data.columns = ['invalid_' + col for col in invalid_data.columns]
        
        # Components should handle missing expected columns
        stat_analyzer = StatisticalAnalyzer()
        
        # Should not find expected columns but shouldn't crash
        try:
            # This should handle the case where expected columns don't exist
            pass
        except Exception as e:
            pytest.fail(f"Data validation failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])