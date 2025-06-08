🧠 Employee Performance Predictor

A Flask-based machine learning web application that predicts whether an employee fully meets performance expectations based on engagement, satisfaction, training cost, and work-life balance.

🚀Features

- Predicts employee performance in real time
- User-friendly web interface
- Model trained using a neural network (Keras + TensorFlow)
- Displays probability score along with the prediction

🛠️ Tech Stack

- Python 3.8+
- Flask
- TensorFlow, Keras
- Scikit-learn
- Pandas, NumPy
- HTML, CSS (for frontend)

📁 Project Structure

project/
├── ap.py                   # Flask app
├── train.py                # Model training script
├── eval.py                 # Evaluation & metrics
├── Cleaned_HR_Data_Analysis.csv
├── employee_performance_model.h5
├── scaler.pkl
├── static/
│   └── style.css
└── templates/
    └── index.html


📦 Installation

 1. Clone the repository
"bash
git clone https://github.com/your-username/hr-performance-predictor.git
cd hr-performance-predictor
"

 2. Install dependencies
"bash
pip install -r requirements.txt
"

▶️ Running the App

1. Train the model:
"bash
python train_moodel.py
"

2. (Optional)Evaluate performance:
"bash
python eval.py"


3. Launch the web app:
"bash
python ap.py"


Open your browser at: `http://127.0.0.1:5000`

📊 Sample Inputs

📍 Engagement Score: 0 – 5  
📍 Satisfaction Score: 0 – 5  
📍 Training Cost: Integer   
📍 Work-Life Balance: 0 – 5


🧾 License

This project is licensed for educational and personal use.
