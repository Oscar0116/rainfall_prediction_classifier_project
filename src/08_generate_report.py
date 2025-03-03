import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from fpdf import FPDF

# Load test data
with open("results/X_test.pkl", "rb") as f:
    X_test = pickle.load(f)

with open("results/y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

# Load trained model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict probabilities and classes
y_prob = model.predict_proba(X_test)[:, 1]  # Probability for class 1 ("Yes")
y_pred = model.predict(X_test)  # Predictions (still in "Yes"/"No" format)

# Convert 'Yes'/'No' labels to binary (0 and 1)
lb = LabelBinarizer()
y_test_binary = lb.fit_transform(y_test).ravel()  # "Yes" → 1, "No" → 0
y_pred_binary = lb.transform(y_pred).ravel()  # Convert predictions to binary as well

# Evaluate model
accuracy = accuracy_score(y_test_binary, y_pred_binary)
conf_matrix = confusion_matrix(y_test_binary, y_pred_binary)
class_report = classification_report(y_test_binary, y_pred_binary)

# Print evaluation results
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test_binary, y_prob)
roc_auc = auc(fpr, tpr)

# Save Confusion Matrix Plot
plt.figure()
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No', 'Yes'])
plt.yticks(tick_marks, ['No', 'Yes'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("results/confusion_matrix.png")
plt.close()

# Save ROC Curve Plot
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig("results/roc_curve.png")
plt.close()

# Generate PDF Report
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", style="B", size=16)
pdf.cell(200, 10, "Rainfall Prediction Classifier - Model Report", ln=True, align="C")
pdf.ln(10)

# Model Performance Summary
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, f"Model Accuracy: {accuracy:.4f}", ln=True)
pdf.ln(5)
pdf.multi_cell(0, 8, "Classification Report:\n" + class_report)
pdf.ln(5)

# Add Confusion Matrix Image
pdf.cell(200, 10, "Confusion Matrix:", ln=True)
pdf.image("results/confusion_matrix.png", x=30, w=150)
pdf.ln(10)

# Add ROC Curve Image
pdf.cell(200, 10, "ROC Curve:", ln=True)
pdf.image("results/roc_curve.png", x=30, w=150)
pdf.ln(10)

# Save the PDF
pdf.output("results/model_report.pdf")
print("PDF report saved in results folder.")
