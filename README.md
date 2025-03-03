# Rainfall Prediction Classifier Project

## Overview
This project aims to build a machine learning classifier to predict whether it will rain the next day based on historical weather data. The dataset used is the "Rain in Australia" dataset from Kaggle, which contains daily weather observations from 2008 to 2017.

## Objectives
- Perform feature engineering on real-world data.
- Build a classifier pipeline and optimize it using Grid Search CV.
- Evaluate the model using various performance metrics and visualizations.
- Implement different classifiers and tune their parameters for optimization.

## Dataset
The dataset includes weather attributes such as:
- Temperature
- Rainfall
- Evaporation
- Wind speed & direction
- Humidity
- Pressure
- Cloud cover
- Sunshine hours

The target variable is `RainTomorrow`, which indicates whether there will be at least 1mm of rain the next day.

## Project Structure
```
📂 rainfall_prediction_classifier_project
├── 📂 data               # Raw and processed data
├── 📂 models             # Saved trained models
├── 📂 results            # Evaluation results and visualizations
├── 📂 src                # Python scripts for different stages
│   ├── 01_get_data.py            # Load and preprocess the dataset
│   ├── 02_preprocess_data.py      # Handle missing values and clean data
│   ├── 03_feature_engineering.py  # Feature extraction and transformation
│   ├── 04_build_pipeline.py       # Define the preprocessing pipeline
│   ├── 05_train_model.py          # Train machine learning models
│   ├── 06_model_comparison.py     # Compare multiple classifiers
│   ├── 07_evaluate_model.py       # Evaluate the final model
└── README.md          # Project summary and instructions
```

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rainfall_prediction_classifier_project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd rainfall_prediction_classifier_project
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project
To execute the full pipeline:
```bash
python src/01_get_data.py
python src/02_preprocess_data.py
python src/03_feature_engineering.py
python src/04_build_pipeline.py
python src/05_train_model.py
python src/06_model_comparison.py
python src/07_evaluate_model.py
```

## Results
The best model achieved an accuracy of **77.51%** on the test set. Key evaluation metrics:
- **Confusion Matrix**:
  ```
  [[907 247]
   [ 93 265]]
  ```
- **Classification Report**:
  ```
              precision    recall  f1-score   support

           0       0.91      0.79      0.84      1154
           1       0.52      0.74      0.61       358

    accuracy                           0.78      1512
   macro avg       0.71      0.76      0.73      1512
   weighted avg       0.81      0.78      0.79      1512
  ```
- **ROC Curve**: Saved in `results/roc_curve.png`
  ---
📌 **Author**: Oscar 
