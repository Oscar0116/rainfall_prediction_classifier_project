import pandas as pd
import os

def load_and_save_data(url, save_path):
    """Downloads the dataset from the URL and saves it locally."""
    print("Downloading dataset...")
    df = pd.read_csv(url)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Dataset saved at {save_path}")
    return df

if __name__ == "__main__":
    DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
    SAVE_PATH = "data/weatherAUS.csv"
    
    df = load_and_save_data(DATA_URL, SAVE_PATH)
    print("Data loaded successfully!")
    print(df.head())
    df.count()