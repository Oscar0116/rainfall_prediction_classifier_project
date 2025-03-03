import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

# Load test data
with open("results/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("results/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# Load trained model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict probabilities and classes
y_prob = model.predict_proba(X_test)[:, 1]  # Probability for class 1 (Yes)
y_pred = model.predict(X_test)

# Convert 'Yes'/'No' predictions to binary (0,1)
y_pred = (y_pred == "Yes").astype(int)  # Convert 'Yes' -> 1, 'No' -> 0

# Convert 'Yes'/'No' true labels to binary
lb = LabelBinarizer()
y_binary = lb.fit_transform(y_test).ravel()  # Converts 'Yes' -> 1 and 'No' -> 0

# Evaluate model
accuracy = accuracy_score(y_binary, y_pred)
conf_matrix = confusion_matrix(y_binary, y_pred)
class_report = classification_report(y_binary, y_pred)

# Print evaluation results
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# ROC Curve
fpr, tpr, _ = roc_curve(y_binary, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig("results/roc_curve.png")
plt.show()

print("Evaluation results saved in results folder.")
