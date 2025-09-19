import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import warnings

class DataQualityAnalyzer:
    """
    Comprehensive data quality assessment for healthcare analytics.
    Implements outlier detection, distribution analysis, and data validation.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.quality_report = {}
        self.outliers = {}
        
    def comprehensive_quality_assessment(self) -> Dict:
        """
        Perform comprehensive data quality assessment.
        """
        self.quality_report = {
            'basic_info': self._basic_data_info(),
            'missing_values': self._missing_value_analysis(),
            'duplicates': self._duplicate_analysis(),
            'outliers': self._outlier_analysis(),
            'distributions': self._distribution_analysis(),
            'data_types': self._data_type_validation(),
            'value_ranges': self._value_range_validation(),
            'consistency_checks': self._consistency_checks()
        }
        
        return self.quality_report
    
    def _basic_data_info(self) -> Dict:
        """
        Basic dataset information.
        """
        return {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'column_names': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict()
        }
    
    def _missing_value_analysis(self) -> Dict:
        """
        Analyze missing values with patterns and recommendations.
        """
        missing_counts = self.df.isnull().sum()
        missing_percentages = (missing_counts / len(self.df)) * 100
        
        # Missing value patterns
        missing_patterns = self.df.isnull().value_counts().head(10)
        
        return {
            'missing_counts': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'columns_with_missing': missing_counts[missing_counts > 0].index.tolist(),
            'total_missing_values': missing_counts.sum(),
            'missing_patterns': missing_patterns.to_dict(),
            'recommendations': self._missing_value_recommendations(missing_percentages)
        }
    
    def _missing_value_recommendations(self, missing_percentages: pd.Series) -> List[str]:
        """
        Provide recommendations for handling missing values.
        """
        recommendations = []
        
        high_missing = missing_percentages[missing_percentages > 50]
        moderate_missing = missing_percentages[(missing_percentages > 10) & 
                                             (missing_percentages <= 50)]
        low_missing = missing_percentages[(missing_percentages > 0) & 
                                        (missing_percentages <= 10)]
        
        if len(high_missing) > 0:
            recommendations.append(
                f"Consider removing columns with >50% missing: {list(high_missing.index)}"
            )
        
        if len(moderate_missing) > 0:
            recommendations.append(
                f"Investigate moderate missing values (10-50%): {list(moderate_missing.index)}"
            )
        
        if len(low_missing) > 0:
            recommendations.append(
                f"Low missing values (<10%) can be imputed: {list(low_missing.index)}"
            )
        
        return recommendations
    
    def _duplicate_analysis(self) -> Dict:
        """
        Analyze duplicate records.
        """
        total_duplicates = self.df.duplicated().sum()
        duplicate_percentage = (total_duplicates / len(self.df)) * 100
        
        # Check for duplicates in key columns
        key_columns = ['Age', 'Sex', 'BMI', 'Income']  # Adjust based on your dataset
        partial_duplicates = {}
        
        for col in key_columns:
            if col in self.df.columns:
                partial_duplicates[col] = self.df.duplicated(subset=[col]).sum()
        
        return {
            'total_duplicates': total_duplicates,
            'duplicate_percentage': duplicate_percentage,
            'partial_duplicates': partial_duplicates,
            'unique_records': len(self.df) - total_duplicates
        }
    
    def _outlier_analysis(self) -> Dict:
        """
        Comprehensive outlier detection using multiple methods.
        """
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        outlier_results = {}
        
        for col in numeric_columns:
            if self.df[col].notna().sum() > 0:
                outlier_results[col] = self._detect_outliers_multiple_methods(col)
        
        return outlier_results
    
    def _detect_outliers_multiple_methods(self, column: str) -> Dict:
        """
        Detect outliers using IQR, Z-score, and Modified Z-score methods.
        """
        data = self.df[column].dropna()
        
        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_lower = Q1 - 1.5 * IQR
        iqr_upper = Q3 + 1.5 * IQR
        iqr_outliers = data[(data < iqr_lower) | (data > iqr_upper)]
        
        # Z-score method (threshold = 3)
        z_scores = np.abs(stats.zscore(data))
        z_outliers = data[z_scores > 3]
        
        # Modified Z-score method
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad
        modified_z_outliers = data[np.abs(modified_z_scores) > 3.5]
        
        return {
            'iqr_method': {
                'n_outliers': len(iqr_outliers),
                'outlier_percentage': (len(iqr_outliers) / len(data)) * 100,
                'lower_bound': iqr_lower,
                'upper_bound': iqr_upper,
                'outlier_values': iqr_outliers.tolist()[:20]  # Limit for display
            },
            'zscore_method': {
                'n_outliers': len(z_outliers),
                'outlier_percentage': (len(z_outliers) / len(data)) * 100,
                'outlier_values': z_outliers.tolist()[:20]
            },
            'modified_zscore_method': {
                'n_outliers': len(modified_z_outliers),
                'outlier_percentage': (len(modified_z_outliers) / len(data)) * 100,
                'outlier_values': modified_z_outliers.tolist()[:20]
            }
        }
    
    def _distribution_analysis(self) -> Dict:
        """
        Analyze distributions of numeric variables.
        """
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        distribution_results = {}
        
        for col in numeric_columns:
            if self.df[col].notna().sum() > 0:
                data = self.df[col].dropna()
                
                # Normality tests
                shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
                
                # Skewness and kurtosis
                skewness = stats.skew(data)
                kurtosis = stats.kurtosis(data)
                
                distribution_results[col] = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'shapiro_test': {
                        'statistic': shapiro_stat,
                        'p_value': shapiro_p,
                        'is_normal': shapiro_p > 0.05
                    },
                    'distribution_type': self._classify_distribution(skewness, kurtosis, shapiro_p)
                }
        
        return distribution_results
    
    def _classify_distribution(self, skewness: float, kurtosis: float, shapiro_p: float) -> str:
        """
        Classify distribution type based on statistical measures.
        """
        if shapiro_p > 0.05:
            return 'Normal'
        elif abs(skewness) < 0.5:
            return 'Approximately symmetric'
        elif skewness > 0.5:
            return 'Right-skewed'
        elif skewness < -0.5:
            return 'Left-skewed'
        else:
            return 'Non-normal'
    
    def _data_type_validation(self) -> Dict:
        """
        Validate data types and suggest corrections.
        """
        type_issues = []
        recommendations = []
        
        for col in self.df.columns:
            # Check for numeric columns stored as objects
            if self.df[col].dtype == 'object':
                try:
                    pd.to_numeric(self.df[col], errors='raise')
                    type_issues.append(f"{col}: Numeric data stored as object")
                    recommendations.append(f"Convert {col} to numeric type")
                except:
                    pass
            
            # Check for categorical variables with high cardinality
            if self.df[col].dtype == 'object':
                unique_count = self.df[col].nunique()
                total_count = len(self.df)
                if unique_count / total_count > 0.5:
                    type_issues.append(f"{col}: High cardinality categorical variable")
        
        return {
            'type_issues': type_issues,
            'recommendations': recommendations,
            'current_types': self.df.dtypes.to_dict()
        }
    
    def _value_range_validation(self) -> Dict:
        """
        Validate value ranges for healthcare-specific variables.
        """
        validation_results = {}
        
        # Define expected ranges for common health variables
        expected_ranges = {
            'BMI': (10, 100),
            'Age': (0, 120),
            'MentHlth': (0, 30),
            'PhysHlth': (0, 30),
            'Sex': (0, 1),
            'Income': (1, 8),
            'Diabetes_012': (0, 2)
        }
        
        for col, (min_val, max_val) in expected_ranges.items():
            if col in self.df.columns:
                data = self.df[col].dropna()
                out_of_range = data[(data < min_val) | (data > max_val)]
                
                validation_results[col] = {
                    'expected_range': (min_val, max_val),
                    'actual_range': (data.min(), data.max()),
                    'out_of_range_count': len(out_of_range),
                    'out_of_range_percentage': (len(out_of_range) / len(data)) * 100,
                    'valid': len(out_of_range) == 0
                }
        
        return validation_results
    
    def _consistency_checks(self) -> Dict:
        """
        Perform logical consistency checks specific to health data.
        """
        consistency_issues = []
        
        # Check for logical inconsistencies
        if 'Diabetes_012' in self.df.columns and 'BMI' in self.df.columns:
            # Higher BMI should correlate with diabetes risk
            diabetes_bmi = self.df.groupby('Diabetes_012')['BMI'].mean()
            if len(diabetes_bmi) > 1 and diabetes_bmi.iloc[-1] < diabetes_bmi.iloc[0]:
                consistency_issues.append("BMI decreases with diabetes status - unexpected pattern")
        
        if 'Age' in self.df.columns and 'Diabetes_012' in self.df.columns:
            # Age should correlate with diabetes risk
            diabetes_age = self.df.groupby('Diabetes_012')['Age'].mean()
            if len(diabetes_age) > 1 and diabetes_age.iloc[-1] < diabetes_age.iloc[0]:
                consistency_issues.append("Age decreases with diabetes status - unexpected pattern")
        
        return {
            'consistency_issues': consistency_issues,
            'checks_performed': ['BMI-Diabetes correlation', 'Age-Diabetes correlation']
        }
    
    def visualize_data_quality(self, save_plots: bool = False) -> None:
        """
        Create visualizations for data quality assessment.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        n_cols = len(numeric_cols)
        
        if n_cols == 0:
            print("No numeric columns found for visualization")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Quality Assessment Visualizations', fontsize=16)
        
        # Missing values heatmap
        if self.df.isnull().sum().sum() > 0:
            sns.heatmap(self.df.isnull(), cbar=True, ax=axes[0,0])
            axes[0,0].set_title('Missing Values Pattern')
        else:
            axes[0,0].text(0.5, 0.5, 'No Missing Values', 
                          horizontalalignment='center', verticalalignment='center')
            axes[0,0].set_title('Missing Values Pattern')
        
        # Distribution plots for key variables
        if 'BMI' in self.df.columns:
            self.df['BMI'].hist(bins=50, ax=axes[0,1])
            axes[0,1].set_title('BMI Distribution')
            axes[0,1].set_xlabel('BMI')
        
        # Outlier visualization (boxplot)
        if len(numeric_cols) > 0:
            key_vars = [col for col in ['BMI', 'Age', 'MentHlth', 'PhysHlth'] 
                       if col in numeric_cols][:4]
            if key_vars:
                self.df[key_vars].boxplot(ax=axes[1,0])
                axes[1,0].set_title('Outlier Detection (Box Plots)')
                axes[1,0].tick_params(axis='x', rotation=45)
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            correlation_matrix = self.df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                       center=0, ax=axes[1,1])
            axes[1,1].set_title('Correlation Matrix')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('data_quality_assessment.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_quality_report(self) -> str:
        """
        Generate comprehensive data quality report.
        """
        if not self.quality_report:
            self.comprehensive_quality_assessment()
        
        report = []
        report.append("DATA QUALITY ASSESSMENT REPORT")
        report.append("=" * 50)
        
        # Basic info
        basic = self.quality_report['basic_info']
        report.append(f"Dataset size: {basic['n_rows']:,} rows × {basic['n_columns']} columns")
        report.append(f"Memory usage: {basic['memory_usage_mb']:.2f} MB")
        report.append("")
        
        # Missing values
        missing = self.quality_report['missing_values']
        report.append("MISSING VALUES ANALYSIS")
        report.append("-" * 30)
        report.append(f"Total missing values: {missing['total_missing_values']:,}")
        if missing['columns_with_missing']:
            report.append("Columns with missing values:")
            for col in missing['columns_with_missing']:
                count = missing['missing_counts'][col]
                pct = missing['missing_percentages'][col]
                report.append(f"  {col}: {count:,} ({pct:.2f}%)")
        else:
            report.append("No missing values detected")
        report.append("")
        
        # Outliers summary
        outliers = self.quality_report['outliers']
        report.append("OUTLIER ANALYSIS SUMMARY")
        report.append("-" * 30)
        for col, methods in outliers.items():
            iqr_pct = methods['iqr_method']['outlier_percentage']
            report.append(f"{col}: {iqr_pct:.2f}% outliers (IQR method)")
        report.append("")
        
        # Distribution analysis
        distributions = self.quality_report['distributions']
        report.append("DISTRIBUTION ANALYSIS")
        report.append("-" * 30)
        for col, stats in distributions.items():
            dist_type = stats['distribution_type']
            skew = stats['skewness']
            report.append(f"{col}: {dist_type} (skewness: {skew:.3f})")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        for rec in missing['recommendations']:
            report.append(f"• {rec}")
        
        for col, validation in self.quality_report['value_ranges'].items():
            if not validation['valid']:
                report.append(f"• {col}: {validation['out_of_range_count']} values outside expected range")
        
        return "\n".join(report)