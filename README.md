🏥 DiabetesGuard AI
Preventive Risk & Clinical Decision Support System for Diabetes


📌 Overview

DiabetesGuard AI is a preventive healthcare decision support system that helps identify early diabetes risk using routine patient data.
It is designed to support clinicians, not replace them, and to communicate risk safely and clearly to both doctors and patients.
The system combines Machine Learning (ML) and human-centered explanations to surface early risk signals, highlight contributing factors, and suggest next-step actions while handling uncertainty responsibly.

🎯 Problem Statement

Chronic diseases like diabetes often develop silently.
By the time symptoms appear, treatment becomes expensive and outcomes worsen.
In real OPD/clinic workflows:
Doctors have limited time
Patient history is often fragmented
Risk indicators are probabilistic, not binary
Patients struggle to understand medical risk numbers
There is a need for a clinical decision support workflow that:
Detects early risk
Explains uncertainty
Supports informed intervention
Avoids misleading or overconfident predictions

💡 Why This Problem?

Diabetes is one of the fastest-growing chronic diseases
Early intervention can delay or prevent disease progression
Most existing tools focus on diagnosis, not prevention
Many ML systems act as black boxes, reducing clinical trust
This project focuses on preventive medicine, interpretability, and human-AI collaboration.

🧠 Solution Summary

DiabetesGuard AI:
Trains a machine learning model on structured diabetes data
Generates a risk probability with confidence level
Identifies key contributing health factors
Provides counterfactual insights (what would reduce risk most)
Separates Clinician View and Patient View
Tracks risk over time for preventive follow-ups
Applies ethical guardrails and bias awareness


⚠️ The system does NOT diagnose diabetes.


🏗️ System Architecture (High Level)Patient Data Input (Streamlit UI)
        ↓
Data Encoding & Preprocessing
        ↓
Random Forest ML Model
        ↓
Risk Probability + Confidence Estimation
        ↓
Explainability & Counterfactual Engine
        ↓
Clinician / Patient Decision Support Views


🛠️ Tech Stack

Layer	Technology
Frontend	Streamlit
Backend Logic	Python
Machine Learning	Scikit-learn (Random Forest)
Visualization	Plotly
Data Handling	Pandas, NumPy

🤖 Machine Learning Integration

Model Used
Random Forest Classifier

Why Random Forest?

Handles non-linear clinical data well
Robust to noisy health data
Provides feature importance for explainability
Input Features
Age
Gender
BMI
HbA1c level
Blood glucose
Hypertension
Heart disease
Smoking history Output
Diabetes risk probability
Risk category (Low / Moderate / High / Critical)
Model confidence (High / Medium / Low)
The model outputs probability, not diagnosis.

🧩 GenAI Integration (Responsible Use)

GenAI concepts are applied in:
Patient-friendly explanations
Action-oriented summaries
Non-numeric, non-alarming language
All explanations:
Avoid deterministic claims
Avoid diagnosis language
Encourage professional consultation
The design allows future integration with full LLM-based explanation engines.

🧑‍⚕️ Clinician vs Patient Views

Clinician View
Exact risk percentage
Model confidence
Feature importance chart
Counterfactual insights
Clinical recommendations
Longitudinal risk trends
Patient View
No percentages
No medical jargon
Simple health summaries
Lifestyle-focused guidance
Calm, supportive explanations
This separation prevents misunderstanding and over-trust.

🔍 Counterfactual Reasoning

The system answers:
“What would reduce risk the most?”
Examples:
Reducing BMI by ~5%
Improving HbA1c levels
Results are shown as ranges, not exact numbers, to respect uncertainty.


📈 Longitudinal Risk Tracking

Each assessment is timestamped
Risk trends are visualized over time
Helps monitor preventive progress
Supports follow-up decision making

⚖️ Ethics, Bias & Safety Considerations

Safety
Human-in-the-loop enforced
Low-confidence warnings shown
Clear disclaimer: Not a diagnostic tool
Bias Awareness
Reduced reliability at age extremes
Reduced reliability at BMI extremes
Dataset demographic limitations acknowledged
Privacy
No personal identifiers stored
Local/session-based processing
Privacy-safe patient hashing for tracking

⚠️ Limitations

Depends on training dataset quality
No real EHR integration yet
No pediatric-specific model
GenAI explanations are template-based in this prototype


🚀 Future Scope

EHR integration
Counterfactual optimization using SHAP
Full GenAI conversational explanation layer
Fairness audits across demographics
Multi-disease preventive expansion


💼 Business Feasibility

Target Users
Clinics
Hospitals
Preventive health programs
Corporate wellness platforms
Value Proposition
Early risk detection
Reduced treatment cost
Faster OPD decision support
Better patient engagement
Revenue Model
SaaS subscription for clinics
Enterprise hospital licensing
Preventive screening partnerships


▶️ How to Run Locally

pip install -r requirements.txt
streamlit run app.py



⚠️ Disclaimer

This tool is intended for clinical decision support only.
It does not provide medical diagnosis and must be used alongside professional medical judgment.


 🌐 Live Demo
🔗 https://diabetesguard-ai.streamlit.app/


