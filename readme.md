# ML Model Drift Detection & Performance Monitoring System

## 🔍 Overview
This project demonstrates how to monitor deployed Machine Learning models by detecting **data drift** and **prediction drift** on new incoming datasets.

The system compares new data distributions against baseline training statistics and flags potential risks to model reliability.

---

## ⚙️ Tech Stack
- Python
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Joblib

---

## 🧠 Key Concepts Implemented
- Feature Drift Detection (mean shift vs baseline standard deviation)
- Prediction Drift Detection (positive-rate distribution change)
- ML Pipeline with ColumnTransformer
- Baseline statistics persistence
- Model health status classification (STABLE / MONITOR / HIGH RISK)

---

## 🚀 How It Works
1. Train the ML model and save:
   - Model pipeline
   - Baseline feature statistics
   - Baseline prediction distribution
2. Upload a new dataset via Streamlit
3. System computes:
   - Feature-level drift
   - Prediction drift
4. Model health status is reported as:
   - **STABLE**
   - **MONITOR**
   - **HIGH RISK**

---

## ▶️ Run the Application
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
├── app.py                 # Streamlit UI and workflow control
├── model_utils.py         # Model & baseline loaders
├── drift_utils.py         # Feature & prediction drift logic
├── artifacts/
│   ├── model_pipeline.pkl
│   ├── baseline_stats.pkl
│   └── baseline_positive_rate.pkl
├── requirements.txt
└── README.md
