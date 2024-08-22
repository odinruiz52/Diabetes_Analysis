import pandas as pd
from load_data import load_diabetes_data

def perform_eda(df):
    """
    Perform basic exploratory data analysis on the diabetes dataset.

    Parameters:
        df (pd.DataFrame): The diabetes dataset.

    Returns:
        None
    """
    # Display the first few rows
    print("First 5 Rows of Data:")
    print(df.head())

    # Display the structure of the DataFrame
    print("\nDataFrame Info:")
    print(df.info())

    # Display summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

if __name__ == "__main__":
    df = load_diabetes_data()
    perform_eda(df)
