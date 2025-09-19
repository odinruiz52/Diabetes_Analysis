import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings

class AdvancedPreprocessor:
    """
    Advanced data preprocessing pipeline for healthcare analytics.
    Handles missing values, outliers, feature engineering, and data transformation.
    """
    
    def __init__(self):
        self.imputers = {}
        self.scalers = {}
        self.outlier_bounds = {}
        self.transformations = {}
        self.feature_engineering_params = {}
        self.preprocessing_log = []
        
    def comprehensive_preprocessing(self, df: pd.DataFrame, 
                                  target_column: str = None,
                                  strategy: str = 'healthcare') -> Dict:
        """
        Perform comprehensive preprocessing tailored for healthcare data.
        """
        df_processed = df.copy()
        preprocessing_results = {}
        
        # 1. Data Quality Assessment
        quality_assessment = self._assess_data_quality(df_processed)
        preprocessing_results['quality_assessment'] = quality_assessment
        
        # 2. Handle Missing Values
        df_processed, missing_results = self._handle_missing_values(
            df_processed, strategy=strategy
        )
        preprocessing_results['missing_value_handling'] = missing_results
        
        # 3. Outlier Detection and Treatment
        df_processed, outlier_results = self._handle_outliers(
            df_processed, method='multiple', strategy=strategy
        )
        preprocessing_results['outlier_handling'] = outlier_results
        
        # 4. Feature Engineering
        df_processed, feature_results = self._engineer_features(
            df_processed, target_column=target_column
        )
        preprocessing_results['feature_engineering'] = feature_results
        
        # 5. Data Transformation
        df_processed, transform_results = self._transform_data(df_processed)
        preprocessing_results['data_transformation'] = transform_results
        
        # 6. Data Scaling
        df_processed, scaling_results = self._scale_data(df_processed, target_column)
        preprocessing_results['data_scaling'] = scaling_results
        
        # 7. Final Validation
        final_validation = self._validate_processed_data(df, df_processed)
        preprocessing_results['final_validation'] = final_validation
        
        self.preprocessing_log.append(f"Comprehensive preprocessing completed for dataset with {len(df)} rows")
        
        return {
            'processed_data': df_processed,
            'preprocessing_results': preprocessing_results,
            'preprocessing_log': self.preprocessing_log.copy()
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Assess data quality before preprocessing.
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        quality_issues = {
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'infinite_values': {},
            'zero_variance_features': [],
            'high_cardinality_categorical': []
        }
        
        # Check for infinite values in numeric columns
        for col in numeric_columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                quality_issues['infinite_values'][col] = inf_count
        
        # Check for zero variance features
        for col in numeric_columns:
            if df[col].var() == 0:
                quality_issues['zero_variance_features'].append(col)
        
        # Check for high cardinality categorical variables
        for col in categorical_columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:
                quality_issues['high_cardinality_categorical'].append(col)
        
        return quality_issues
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str = 'healthcare') -> Tuple[pd.DataFrame, Dict]:
        """
        Handle missing values with domain-specific strategies.
        """
        df_imputed = df.copy()
        missing_results = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        # Healthcare-specific missing value handling
        if strategy == 'healthcare':
            # For healthcare data, missing values often have clinical meaning
            missing_strategies = {
                'BMI': 'median',
                'MentHlth': 'zero',  # Assume no poor mental health days if not reported
                'PhysHlth': 'zero',  # Assume no poor physical health days if not reported
                'Age': 'median',
                'Income': 'mode',
                'Sex': 'mode'
            }
            
            for col in numeric_columns:
                if col in df_imputed.columns and df_imputed[col].isnull().any():
                    if col in missing_strategies:
                        if missing_strategies[col] == 'median':
                            imputer = SimpleImputer(strategy='median')
                        elif missing_strategies[col] == 'zero':
                            imputer = SimpleImputer(strategy='constant', fill_value=0)
                        else:
                            imputer = SimpleImputer(strategy='most_frequent')
                    else:
                        # Use KNN imputation for complex patterns
                        imputer = KNNImputer(n_neighbors=5)
                    
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).ravel()
                    self.imputers[col] = imputer
                    
                    missing_results[col] = {
                        'original_missing': df[col].isnull().sum(),
                        'strategy_used': missing_strategies.get(col, 'knn'),
                        'imputed_values': (df_imputed[col] != df[col]).sum()
                    }
            
            # Handle categorical missing values
            for col in categorical_columns:
                if col in df_imputed.columns and df_imputed[col].isnull().any():
                    mode_value = df_imputed[col].mode()
                    if not mode_value.empty:
                        df_imputed[col].fillna(mode_value[0], inplace=True)
                        missing_results[col] = {
                            'original_missing': df[col].isnull().sum(),
                            'strategy_used': 'mode',
                            'imputed_value': mode_value[0]
                        }
        
        else:
            # Standard missing value handling
            for col in numeric_columns:
                if df_imputed[col].isnull().any():
                    imputer = SimpleImputer(strategy='median')
                    df_imputed[col] = imputer.fit_transform(df_imputed[[col]]).ravel()
                    self.imputers[col] = imputer
            
            for col in categorical_columns:
                if df_imputed[col].isnull().any():
                    df_imputed[col].fillna(df_imputed[col].mode()[0], inplace=True)
        
        self.preprocessing_log.append(f"Missing value imputation completed for {len(missing_results)} columns")
        return df_imputed, missing_results
    
    def _handle_outliers(self, df: pd.DataFrame, method: str = 'multiple', 
                        strategy: str = 'healthcare') -> Tuple[pd.DataFrame, Dict]:
        """
        Comprehensive outlier detection and treatment.
        """
        df_cleaned = df.copy()
        outlier_results = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df_cleaned.columns:
                outliers_detected = self._detect_outliers_comprehensive(df_cleaned[col], col, method)
                
                if strategy == 'healthcare':
                    # Healthcare-specific outlier handling
                    df_cleaned, treatment_result = self._treat_healthcare_outliers(
                        df_cleaned, col, outliers_detected
                    )
                else:
                    # Standard outlier treatment
                    df_cleaned, treatment_result = self._treat_outliers_standard(
                        df_cleaned, col, outliers_detected
                    )
                
                outlier_results[col] = {
                    'outliers_detected': outliers_detected,
                    'treatment_applied': treatment_result
                }
        
        self.preprocessing_log.append(f"Outlier detection and treatment completed for {len(numeric_columns)} numeric columns")
        return df_cleaned, outlier_results
    
    def _detect_outliers_comprehensive(self, series: pd.Series, column_name: str, method: str) -> Dict:
        """
        Detect outliers using multiple methods.
        """
        outliers = {}
        data = series.dropna()
        
        # IQR Method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_lower = Q1 - 1.5 * IQR
        iqr_upper = Q3 + 1.5 * IQR
        iqr_outliers = data[(data < iqr_lower) | (data > iqr_upper)]
        
        # Z-score Method
        z_scores = np.abs(stats.zscore(data))
        z_outliers = data[z_scores > 3]
        
        # Modified Z-score Method
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad > 0:
            modified_z_scores = 0.6745 * (data - median) / mad
            modified_z_outliers = data[np.abs(modified_z_scores) > 3.5]
        else:
            modified_z_outliers = pd.Series(dtype=float)
        
        # Isolation Forest (for complex outlier patterns)
        try:
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
            iso_outliers = data[outlier_labels == -1]
        except ImportError:
            iso_outliers = pd.Series(dtype=float)
        
        outliers = {
            'iqr_method': {
                'outliers': iqr_outliers.index.tolist(),
                'count': len(iqr_outliers),
                'bounds': (iqr_lower, iqr_upper)
            },
            'zscore_method': {
                'outliers': z_outliers.index.tolist(),
                'count': len(z_outliers)
            },
            'modified_zscore_method': {
                'outliers': modified_z_outliers.index.tolist(),
                'count': len(modified_z_outliers)
            },
            'isolation_forest': {
                'outliers': iso_outliers.index.tolist() if not iso_outliers.empty else [],
                'count': len(iso_outliers) if not iso_outliers.empty else 0
            }
        }
        
        # Store bounds for later use
        self.outlier_bounds[column_name] = (iqr_lower, iqr_upper)
        
        return outliers
    
    def _treat_healthcare_outliers(self, df: pd.DataFrame, column: str, 
                                 outliers_detected: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Treat outliers with healthcare domain knowledge.
        """
        df_treated = df.copy()
        
        # Healthcare-specific outlier treatment
        healthcare_bounds = {
            'BMI': (10, 100),
            'Age': (0, 120),
            'MentHlth': (0, 30),
            'PhysHlth': (0, 30),
            'Income': (1, 8)
        }
        
        if column in healthcare_bounds:
            min_val, max_val = healthcare_bounds[column]
            
            # Clip values to clinically reasonable ranges
            original_outliers = len(df_treated[(df_treated[column] < min_val) | 
                                             (df_treated[column] > max_val)])
            
            df_treated[column] = df_treated[column].clip(lower=min_val, upper=max_val)
            
            treatment_result = {
                'method': 'clinical_clipping',
                'bounds_applied': (min_val, max_val),
                'values_modified': original_outliers
            }
        else:
            # Use IQR method for other variables
            lower_bound, upper_bound = self.outlier_bounds[column]
            
            # Winsorization (clip to 5th and 95th percentiles)
            p5 = df_treated[column].quantile(0.05)
            p95 = df_treated[column].quantile(0.95)
            
            original_outliers = len(df_treated[(df_treated[column] < p5) | 
                                             (df_treated[column] > p95)])
            
            df_treated[column] = df_treated[column].clip(lower=p5, upper=p95)
            
            treatment_result = {
                'method': 'winsorization',
                'bounds_applied': (p5, p95),
                'values_modified': original_outliers
            }
        
        return df_treated, treatment_result
    
    def _treat_outliers_standard(self, df: pd.DataFrame, column: str, 
                               outliers_detected: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Standard outlier treatment methods.
        """
        df_treated = df.copy()
        
        # Use IQR bounds
        lower_bound, upper_bound = self.outlier_bounds[column]
        
        # Count outliers before treatment
        outliers_count = len(df_treated[(df_treated[column] < lower_bound) | 
                                      (df_treated[column] > upper_bound)])
        
        # Clip outliers to bounds
        df_treated[column] = df_treated[column].clip(lower=lower_bound, upper=upper_bound)
        
        treatment_result = {
            'method': 'iqr_clipping',
            'bounds_applied': (lower_bound, upper_bound),
            'values_modified': outliers_count
        }
        
        return df_treated, treatment_result
    
    def _engineer_features(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Create new features based on domain knowledge.
        """
        df_engineered = df.copy()
        feature_results = {}
        
        # Healthcare-specific feature engineering
        new_features_created = []
        
        # BMI categories
        if 'BMI' in df_engineered.columns:
            df_engineered['BMI_Category'] = pd.cut(
                df_engineered['BMI'],
                bins=[0, 18.5, 25, 30, 35, 100],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II+']
            )
            new_features_created.append('BMI_Category')
        
        # Age groups
        if 'Age' in df_engineered.columns:
            # Assuming Age is encoded (1-13), create meaningful groups
            age_mapping = {
                1: 'Young_Adult', 2: 'Young_Adult', 3: 'Young_Adult', 4: 'Young_Adult',
                5: 'Middle_Age', 6: 'Middle_Age', 7: 'Middle_Age', 8: 'Middle_Age',
                9: 'Older_Adult', 10: 'Older_Adult', 11: 'Senior', 12: 'Senior', 13: 'Senior'
            }
            df_engineered['Age_Group_Detailed'] = df_engineered['Age'].map(age_mapping)
            new_features_created.append('Age_Group_Detailed')
        
        # Health burden score
        health_vars = ['MentHlth', 'PhysHlth']
        if all(var in df_engineered.columns for var in health_vars):
            df_engineered['Total_Health_Burden'] = (
                df_engineered['MentHlth'] + df_engineered['PhysHlth']
            )
            new_features_created.append('Total_Health_Burden')
        
        # Lifestyle score
        lifestyle_vars = ['PhysActivity', 'Fruits', 'Veggies']
        if all(var in df_engineered.columns for var in lifestyle_vars):
            df_engineered['Healthy_Lifestyle_Score'] = (
                df_engineered['PhysActivity'] + df_engineered['Fruits'] + df_engineered['Veggies']
            )
            new_features_created.append('Healthy_Lifestyle_Score')
        
        # Risk factor combinations
        if 'BMI' in df_engineered.columns and 'Age' in df_engineered.columns:
            # High BMI + older age interaction
            df_engineered['High_BMI_Older'] = (
                (df_engineered['BMI'] > 30) & (df_engineered['Age'] > 8)
            ).astype(int)
            new_features_created.append('High_BMI_Older')
        
        # Socioeconomic health interaction
        if 'Income' in df_engineered.columns and 'MentHlth' in df_engineered.columns:
            # Low income + high mental health burden
            df_engineered['Low_Income_High_MentHlth'] = (
                (df_engineered['Income'] <= 3) & (df_engineered['MentHlth'] > 7)
            ).astype(int)
            new_features_created.append('Low_Income_High_MentHlth')
        
        feature_results = {
            'new_features_created': new_features_created,
            'original_feature_count': len(df.columns),
            'final_feature_count': len(df_engineered.columns),
            'features_added': len(new_features_created)
        }
        
        self.feature_engineering_params = feature_results
        self.preprocessing_log.append(f"Feature engineering completed: {len(new_features_created)} new features created")
        
        return df_engineered, feature_results
    
    def _transform_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply data transformations to improve model performance.
        """
        df_transformed = df.copy()
        transform_results = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df_transformed.columns:
                original_data = df_transformed[col].dropna()
                
                # Test for normality
                if len(original_data) > 3:
                    shapiro_stat, shapiro_p = stats.shapiro(original_data.sample(min(5000, len(original_data))))
                    
                    # If data is highly skewed, apply transformation
                    skewness = stats.skew(original_data)
                    
                    transformation_applied = None
                    
                    if abs(skewness) > 1 and shapiro_p < 0.01:
                        if skewness > 0:  # Right-skewed
                            if (original_data > 0).all():
                                # Log transformation
                                df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
                                transformation_applied = 'log'
                            else:
                                # Square root transformation (after shifting if needed)
                                min_val = df_transformed[col].min()
                                if min_val < 0:
                                    df_transformed[f'{col}_sqrt'] = np.sqrt(df_transformed[col] - min_val + 1)
                                else:
                                    df_transformed[f'{col}_sqrt'] = np.sqrt(df_transformed[col])
                                transformation_applied = 'sqrt'
                        else:  # Left-skewed
                            # Square transformation
                            df_transformed[f'{col}_squared'] = np.square(df_transformed[col])
                            transformation_applied = 'square'
                    
                    transform_results[col] = {
                        'original_skewness': skewness,
                        'shapiro_p_value': shapiro_p,
                        'transformation_applied': transformation_applied,
                        'is_normal': shapiro_p > 0.01
                    }
                    
                    if transformation_applied:
                        self.transformations[col] = transformation_applied
        
        self.preprocessing_log.append(f"Data transformation completed for {len(transform_results)} columns")
        return df_transformed, transform_results
    
    def _scale_data(self, df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Scale numerical features appropriately.
        """
        df_scaled = df.copy()
        scaling_results = {}
        
        # Identify numeric columns to scale (exclude target if specified)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_columns:
            numeric_columns.remove(target_column)
        
        # Remove binary columns from scaling
        binary_columns = []
        for col in numeric_columns:
            unique_values = df[col].dropna().unique()
            if len(unique_values) == 2 and set(unique_values).issubset({0, 1}):
                binary_columns.append(col)
        
        columns_to_scale = [col for col in numeric_columns if col not in binary_columns]
        
        if columns_to_scale:
            # Use RobustScaler for healthcare data (less sensitive to outliers)
            scaler = RobustScaler()
            
            df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
            self.scalers['robust'] = scaler
            
            scaling_results = {
                'scaler_used': 'RobustScaler',
                'columns_scaled': columns_to_scale,
                'columns_not_scaled': binary_columns,
                'scaling_parameters': {
                    'center': scaler.center_.tolist() if hasattr(scaler, 'center_') else None,
                    'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
                }
            }
        
        self.preprocessing_log.append(f"Data scaling completed for {len(columns_to_scale)} columns")
        return df_scaled, scaling_results
    
    def _validate_processed_data(self, original_df: pd.DataFrame, 
                               processed_df: pd.DataFrame) -> Dict:
        """
        Validate the preprocessing results.
        """
        validation_results = {
            'data_shape_original': original_df.shape,
            'data_shape_processed': processed_df.shape,
            'missing_values_original': original_df.isnull().sum().sum(),
            'missing_values_processed': processed_df.isnull().sum().sum(),
            'infinite_values_processed': np.isinf(processed_df.select_dtypes(include=[np.number])).sum().sum(),
            'data_types_changed': {},
            'preprocessing_successful': True
        }
        
        # Check for data type changes
        for col in original_df.columns:
            if col in processed_df.columns:
                if original_df[col].dtype != processed_df[col].dtype:
                    validation_results['data_types_changed'][col] = {
                        'original': str(original_df[col].dtype),
                        'processed': str(processed_df[col].dtype)
                    }
        
        # Validate no data loss in critical columns
        critical_columns = ['Diabetes_012', 'BMI', 'Age', 'Sex']
        for col in critical_columns:
            if col in original_df.columns and col in processed_df.columns:
                if len(processed_df[col].dropna()) < len(original_df[col].dropna()) * 0.95:
                    validation_results['preprocessing_successful'] = False
                    validation_results[f'{col}_data_loss'] = True
        
        return validation_results
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned preprocessing transformations to new data.
        """
        if not self.imputers and not self.scalers:
            raise ValueError("No preprocessing has been performed yet. Fit the preprocessor first.")
        
        df_transformed = df.copy()
        
        # Apply imputation
        for col, imputer in self.imputers.items():
            if col in df_transformed.columns:
                df_transformed[col] = imputer.transform(df_transformed[[col]]).ravel()
        
        # Apply outlier bounds
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in df_transformed.columns:
                df_transformed[col] = df_transformed[col].clip(lower=lower, upper=upper)
        
        # Apply transformations
        for col, transformation in self.transformations.items():
            if col in df_transformed.columns:
                if transformation == 'log':
                    df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
                elif transformation == 'sqrt':
                    df_transformed[f'{col}_sqrt'] = np.sqrt(df_transformed[col])
                elif transformation == 'square':
                    df_transformed[f'{col}_squared'] = np.square(df_transformed[col])
        
        # Apply scaling
        if 'robust' in self.scalers:
            scaler = self.scalers['robust']
            numeric_columns = df_transformed.select_dtypes(include=[np.number]).columns
            columns_to_scale = [col for col in numeric_columns 
                              if col in scaler.feature_names_in_]
            
            if columns_to_scale:
                df_transformed[columns_to_scale] = scaler.transform(df_transformed[columns_to_scale])
        
        return df_transformed
    
    def generate_preprocessing_report(self) -> str:
        """
        Generate comprehensive preprocessing report.
        """
        report = []
        report.append("DATA PREPROCESSING REPORT")
        report.append("=" * 40)
        
        for log_entry in self.preprocessing_log:
            report.append(f"• {log_entry}")
        
        report.append("\nPREPROCESSING COMPONENTS FITTED:")
        report.append("-" * 35)
        report.append(f"Imputers: {len(self.imputers)} columns")
        report.append(f"Scalers: {len(self.scalers)} fitted")
        report.append(f"Outlier bounds: {len(self.outlier_bounds)} columns")
        report.append(f"Transformations: {len(self.transformations)} applied")
        
        return "\n".join(report)