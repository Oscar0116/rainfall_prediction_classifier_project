import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Define paths
project_dir = "C:/Users/oscar/OneDrive/Escritorio/Python_Projects/rainfall_prediction_classifier_project"
data_path = os.path.join(project_dir, "data", "weatherAUS_final.csv")
model_path = os.path.join(project_dir, "models", "rainfall_model.pkl")
results_path = os.path.join(project_dir, "results")

# Ensure results directory exists
os.makedirs(results_path, exist_ok=True)

# Load data
df = pd.read_csv(data_path)
X = df.drop(columns=["RainToday"])
y = df["RainToday"]

# Load trained Random Forest model
with open(model_path, "rb") as f:
    rf_model = pickle.load(f)

# Extract feature importances
feature_importances = rf_model.named_steps['classifier'].feature_importances_

# Get transformed feature names from the pipeline
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numeric_features = X.select_dtypes(exclude=['object']).columns.tolist()

# Get the OneHotEncoder object
ohe = rf_model.named_steps['preprocessor'].named_transformers_['cat']

# Fix: Get one-hot encoded feature names using stored feature names
ohe_feature_names = ohe.get_feature_names_out(ohe.feature_names_in_)

# Combine numeric and categorical (one-hot encoded) feature names
feature_names = numeric_features + list(ohe_feature_names)

# Ensure lengths match
if len(feature_names) != len(feature_importances):
    raise ValueError(f"Feature name count ({len(feature_names)}) does not match importance count ({len(feature_importances)})")

# Store feature importances in a DataFrame
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Save feature importances
importance_csv = os.path.join(results_path, "feature_importances.csv")
importance_df.to_csv(importance_csv, index=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"].head(20), importance_df["Importance"].head(20), color='skyblue')
plt.gca().invert_yaxis()
plt.title("Top 20 Most Important Features")
plt.xlabel("Importance Score")
plt.savefig(os.path.join(results_path, "feature_importances.png"))
plt.show()

# Apply preprocessing before training Logistic Regression
X_transformed = rf_model.named_steps['preprocessor'].transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

# Save X_test and y_test for evaluation in 07_evaluate_model.py
with open(os.path.join(results_path, "X_test.pkl"), "wb") as f:
    pickle.dump(X_test, f)

with open(os.path.join(results_path, "y_test.pkl"), "wb") as f:
    pickle.dump(y_test, f)

# Train Logistic Regression model
log_model = LogisticRegression(random_state=42, solver='liblinear', penalty='l1', class_weight='balanced')
log_model.fit(X_train, y_train)  # Use transformed training data

# Predict with Logistic Regression
y_pred_log = log_model.predict(X_train)

# Evaluate Logistic Regression model
log_report = classification_report(y_train, y_pred_log, output_dict=True)
log_report_df = pd.DataFrame(log_report).transpose()
log_report_df.to_csv(os.path.join(results_path, "logistic_classification_report.csv"))

# Save the trained Logistic Regression model as the best model
best_model_path = os.path.join(project_dir, "models", "best_model.pkl")
with open(best_model_path, "wb") as f:
    pickle.dump(log_model, f)

print("Best model saved successfully in the models folder.")

# Confusion Matrix for Logistic Regression
log_cm = confusion_matrix(y_train, y_pred_log)
plt.figure(figsize=(6, 4))
sns.heatmap(log_cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig(os.path.join(results_path, "logistic_confusion_matrix.png"))
plt.show()

print("Comparison results saved in results folder.")
