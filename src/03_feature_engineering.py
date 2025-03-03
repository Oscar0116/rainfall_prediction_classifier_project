import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def date_to_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

def feature_engineering(load_path, save_path):
    """Performs feature engineering on the dataset."""
    print("Loading cleaned dataset...")
    df = pd.read_csv(load_path)
    
    # Rename columns
    df = df.rename(columns={'RainToday': 'RainYesterday', 'RainTomorrow': 'RainToday'})
    
    # Filter specific locations
    df = df[df.Location.isin(['Melbourne', 'MelbourneAirport', 'Watsonia'])]
    print(f"Dataset shape after filtering locations: {df.shape}")
    
    # Convert Date column to datetime and extract season
    df['Date'] = pd.to_datetime(df['Date'])
    df['Season'] = df['Date'].apply(date_to_season)
    df = df.drop(columns=['Date'])
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Final dataset saved at {save_path}")
    
    print("Column names:", df.columns.tolist())
    return df

if __name__ == "__main__":
    LOAD_PATH = "data/weatherAUS_clean.csv"
    SAVE_PATH = "data/weatherAUS_final.csv"
    
    df_final = feature_engineering(LOAD_PATH, SAVE_PATH)
