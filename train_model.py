# train_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load dataset
df = pd.read_csv("Cleaned_HR_Data_Analysis.csv")

# Convert Performance Score to binary
df["Performance Binary"] = df["Performance Score"].apply(lambda x: 1 if x == "Fully Meets" else 0)

# Features and target
features = ["Engagement Score", "Satisfaction Score", "Training Cost", "Work-Life Balance Score"]
X = df[features]
y = df["Performance Binary"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Class weights
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = dict(enumerate(weights))

# Build model
model = Sequential([
    Dense(16, input_dim=4, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=8,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# Show sample predictions
print("Sample predictions:")
print(model.predict(X_test[:5]))

# Save model and scaler
model.save("employee_performance_model.h5")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved successfully!")