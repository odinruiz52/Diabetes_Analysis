import pandas as pd

def load_diabetes_data(file_path='./data/diabetes_data.csv'):
    """
    Load the diabetes dataset from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

if __name__ == "__main__":
    df = load_diabetes_data()
    print(df.head())
