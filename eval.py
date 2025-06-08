# evaluate_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from tensorflow.keras.models import load_model
import joblib

# Load dataset
df = pd.read_csv("Cleaned_HR_Data_Analysis.csv")

# Convert target
df["Performance Binary"] = df["Performance Score"].apply(lambda x: 1 if x == "Fully Meets" else 0)

# Features and target
features = ["Engagement Score", "Satisfaction Score", "Training Cost", "Work-Life Balance Score"]
X = df[features]
y = df["Performance Binary"]

# Load scaler and scale features
scaler = joblib.load("scaler.pkl")
X_scaled = scaler.transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Load trained model
model = load_model("employee_performance_model.h5")

# ---------- 1. Plot Training History (Accuracy & Loss) ----------
# (Assumes training script saved 'history' as a .npz file)
try:
    history_data = np.load("training_history.npz", allow_pickle=True)
    history = history_data['history'].item()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
except Exception as e:
    print("⚠️ Could not plot training history. Reason:", e)

# ---------- 2. Evaluate on Test Set ----------
y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# ---------- 3. Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Needs Improvement", "Fully Meets"], yticklabels=["Needs Improvement", "Fully Meets"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ---------- 4. ROC Curve & AUC ----------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='green', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# ---------- 5. Try Manual Predictions ----------
print("\n🔍 Sample predictions:")
sample_inputs = np.array([
    [0, 0, 0, 0],
    [3, 3, 100, 3],
    [5, 5, 200, 5]
])
sample_scaled = scaler.transform(sample_inputs)
sample_probs = model.predict(sample_scaled).flatten()

for i, prob in enumerate(sample_probs):
    print(f"Input {i+1}: {sample_inputs[i]} --> Prob: {prob:.4f} --> {'Fully Meets ✅' if prob > 0.5 else 'Needs Improvement ⚠️'}")
