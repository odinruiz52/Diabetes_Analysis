"""
data_preprocessing.py

This module cleans and prepares the diabetes dataset
so it is ready for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    """Load the diabetes dataset from CSV file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {len(df):,} rows, {len(df.columns)} columns")
    return df

def handle_missing_values(df):
    """Fill or drop missing values in the dataset."""
    print("Handling missing values...")

    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        print("No missing values found")
        return df

    # Fill numerical columns with mean
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)
            print(f"  Filled missing values in {col} with mean")

    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  Filled missing values in {col} with mode")

    return df

def encode_categorical(df):
    """Convert categorical variables into numeric codes."""
    print("Encoding categorical variables...")

    categorical_cols = df.select_dtypes(include=['object']).columns

    if len(categorical_cols) == 0:
        print("No categorical variables to encode")
        return df

    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        print(f"  Encoded {col}: {len(encoder.classes_)} unique values")

    return df

def remove_outliers(df, columns=None):
    """Remove extreme outliers using the IQR method."""
    print("Removing extreme outliers...")

    if columns is None:
        # Focus on key health metrics
        columns = ['BMI', 'MentHlth', 'PhysHlth']
        columns = [col for col in columns if col in df.columns]

    original_size = len(df)

    for col in columns:
        # Calculate IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds (1.5 * IQR rule)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Remove outliers
        outliers_before = len(df)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        outliers_removed = outliers_before - len(df)

        if outliers_removed > 0:
            print(f"  Removed {outliers_removed} outliers from {col}")

    total_removed = original_size - len(df)
    print(f"Total rows removed: {total_removed} ({total_removed/original_size:.1%})")

    return df

def scale_features(df, target_column=None):
    """Scale continuous features to similar ranges."""
    print("Scaling numerical features...")

    # Get numerical columns (exclude target if specified)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column and target_column in numerical_cols:
        numerical_cols.remove(target_column)

    # Skip binary columns (0/1 values)
    cols_to_scale = []
    for col in numerical_cols:
        unique_vals = df[col].unique()
        if not (len(unique_vals) == 2 and set(unique_vals).issubset({0, 1})):
            cols_to_scale.append(col)

    if len(cols_to_scale) == 0:
        print("No continuous features to scale")
        return df

    # Apply standard scaling
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    print(f"  Scaled {len(cols_to_scale)} features: {', '.join(cols_to_scale)}")
    return df

def create_target_variable(df, target_col='Diabetes_012'):
    """Create binary target variable for diabetes prediction."""
    print("Creating target variable...")

    if target_col not in df.columns:
        print(f"Warning: {target_col} not found in dataset")
        return df

    # Create binary target: 0 = No Diabetes, 1 = Diabetes/Prediabetes
    df['has_diabetes'] = (df[target_col] > 0).astype(int)

    print(f"  Target distribution:")
    print(f"    No Diabetes: {(df['has_diabetes']==0).sum():,} ({(df['has_diabetes']==0).mean():.1%})")
    print(f"    Has Diabetes: {(df['has_diabetes']==1).sum():,} ({(df['has_diabetes']==1).mean():.1%})")

    return df

def preprocess_data(filepath, remove_outliers_flag=False, scale_features_flag=False):
    """
    Main preprocessing function for the diabetes dataset.

    Steps:
    1. Load data
    2. Handle missing values
    3. Encode categorical variables
    4. Remove outliers (optional)
    5. Scale features (optional)
    6. Create target variable
    """
    print("=" * 50)
    print("DIABETES DATA PREPROCESSING")
    print("=" * 50)

    # Step 1: Load data
    df = load_data(filepath)

    # Step 2: Handle missing values
    df = handle_missing_values(df)

    # Step 3: Encode categorical features
    df = encode_categorical(df)

    # Step 4: Remove outliers (optional)
    if remove_outliers_flag:
        df = remove_outliers(df)

    # Step 5: Scale numeric features (optional)
    if scale_features_flag:
        df = scale_features(df, target_column='Diabetes_012')

    # Step 6: Create target variable
    df = create_target_variable(df)

    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETED!")
    print("=" * 50)
    print(f"Final dataset: {len(df):,} rows, {len(df.columns)} columns")
    print("Dataset ready for machine learning!")

    return df

# Example usage
if __name__ == "__main__":
    # Test the preprocessing pipeline
    processed_df = preprocess_data('data/diabetes_data.csv')
    print(f"\nProcessed dataset shape: {processed_df.shape}")