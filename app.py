# app.py

from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model("employee_performance_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    score = None
    if request.method == "POST":
        try:
            engagement = float(request.form["engagement"])
            satisfaction = float(request.form["satisfaction"])
            training_cost = float(request.form["training_cost"])
            work_life = float(request.form["work_life"])

            input_data = np.array([[engagement, satisfaction, training_cost, work_life]])
            input_scaled = scaler.transform(input_data)

            prob = float(model.predict(input_scaled)[0][0])
            prediction = "✅ Fully Meets Performance" if prob > 0.5 else "⚠️ Needs Improvement"
            score = round(prob * 100, 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"
            score = None
    return render_template("index.html", prediction=prediction, score=score)

if __name__ == "__main__":
    app.run(debug=True)
