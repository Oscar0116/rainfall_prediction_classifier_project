import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# Define base directory and data path
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(base_dir, "data", "weatherAUS_final.csv")

# Load dataset
print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# Ensure categorical features are properly encoded
df["RainYesterday"] = df["RainYesterday"].astype(str)  # Convert 'RainYesterday' to string (important for encoding)
df["RainToday"] = df["RainToday"].astype(str)          # Convert target variable to string

# Define features and target
X = df.drop(columns=["RainToday"])  # 'RainTomorrow' was renamed to 'RainToday'
y = df["RainToday"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify categorical and numerical features
categorical_features = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "Season", "RainYesterday"]
numerical_features = [col for col in X.columns if col not in categorical_features]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
model_path = os.path.join(base_dir, "models", "rainfall_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to: {model_path}")

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Extract TP (True Positives) and FN (False Negatives) for class "Yes" (Rain)
TP = conf_matrix[1, 1]  # True Positives (Predicted Yes, Actual Yes)
FN = conf_matrix[1, 0]  # False Negatives (Predicted No, Actual Yes)

# Calculate TPR (Recall)
TPR = TP / (TP + FN)
print(f"True Positive Rate (TPR) for 'Yes' (Rain): {TPR:.4f}")
