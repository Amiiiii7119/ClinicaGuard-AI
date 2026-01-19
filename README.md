# ClinicaGuard AI Pro

ClinicaGuard AI Pro is a clinical risk assessment and decision-support application built to help visualize and understand diabetes risk using patient health data. It combines a trained ML model with an interactive Streamlit dashboard to make risk insights easier to interpret for both technical and non-technical users.

---

 🎯 Overview

- Takes basic patient clinical data (age, BMI, HbA1c, glucose, etc.)
- Predicts diabetes risk probability
- Classifies risk into **Low / Moderate / High**
- Explains why risk is high using visual charts
- Suggests clinical recommendations based on risk factors
- Saves assessments for future reference

> **Note:** The goal is decision support, not diagnosis.



 🧠 Model & Logic

- Uses a trained machine learning classification model (`model.pkl`)
- Categorical inputs (gender, smoking history) are handled via label encoders
- Produces a probability score (not a simple yes/no)
- Risk categories:
    - Low Risk → < 30%
    - Moderate Risk → 30–60%
    - High Risk → > 60%
- Visual explanations are provided through charts instead of raw numbers.



📊 Key Features

- Interactive risk gauge
- Risk factor contribution chart
- 10-year risk projection (with and without intervention)
- Population comparison
- Clean light/dark themed UI
- Patient history saved to CSV
- Export-ready summaries



 🖥️ Tech Stack

- Python
- Streamlit (frontend)
- Scikit-learn / XGBoost (model)
- Plotly (charts & visuals)
- Pandas / NumPy (data handling)

---

 📂 Project Structure

ClinicaGuard-AI/
│
├── backend/
│   └── app.py               # Main Streamlit application
│
├── model/
│   ├── model.pkl            # Trained ML model
│   └── label_encoders.pkl   # Encoders for categorical data
│
├── requirements.txt
├── patient_assessments.csv  # Auto-generated
└── README.md

⚙️ How to Run Locally


1️⃣ Clone the repository
git clone https://github.com/Amiiiii7119/ClinicaGuard-AI

cd ClinicaGuard-AI

2️⃣ Create & activate virtual environment

python -m venv venv

venv\Scripts\activate

3️⃣ Install dependencies

pip install -r requirements.txt

4️⃣ Run the app

streamlit run backend/app.py



⚠️ Clinical Disclaimer

This system is not a diagnostic tool.
ClinicaGuard AI Pro is intended for:
Educational use
Risk awareness
Decision support
All outputs must be interpreted by qualified healthcare professionals alongside proper clinical evaluation.



🚀 Why this project stands out

Focuses on interpretability, not just predictions
Designed like a real clinical dashboard
Emphasizes responsible AI usage
Built with production-style structure and UI discipline
Easy for judges and reviewers to understand
