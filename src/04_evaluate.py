import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def build_pipeline(load_path, save_path):
    """Builds a preprocessing pipeline for numerical and categorical features."""
    print("Loading final dataset...")
    df = pd.read_csv(load_path)
    
    # Define numerical and categorical columns
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Define transformations
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Save preprocessor
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.to_pickle(preprocessor, save_path)
    print(f"Preprocessing pipeline saved at {save_path}")
    
    return preprocessor

if __name__ == "__main__":
    LOAD_PATH = "data/weatherAUS_final.csv"
    SAVE_PATH = "models/preprocessor.pkl"
    
    preprocessor = build_pipeline(LOAD_PATH, SAVE_PATH)
