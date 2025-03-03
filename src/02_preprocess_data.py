import pandas as pd
import os

def preprocess_data(load_path, save_path):
    """Loads the dataset, drops missing values, and saves the cleaned dataset."""
    print("Loading dataset...")
    df = pd.read_csv(load_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Drop rows with missing values
    df = df.dropna()
    print(f"Dataset shape after dropping missing values: {df.shape}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Cleaned dataset saved at {save_path}")
    
    print("Column names:", df.columns.tolist())
    return df

if __name__ == "__main__":
    LOAD_PATH = "data/weatherAUS.csv"
    SAVE_PATH = "data/weatherAUS_clean.csv"
    
    df_clean = preprocess_data(LOAD_PATH, SAVE_PATH)
