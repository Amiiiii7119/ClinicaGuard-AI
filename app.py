import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from datetime import datetime
import hashlib

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="DiabetesGuard AI - Clinical Risk Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'patient_records' not in st.session_state:
    st.session_state.patient_records = []

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None

if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}

if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []

if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = 0.0

if 'patient_history' not in st.session_state:
    st.session_state.patient_history = {}

if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'clinician'

# ============================================================================
# PROFESSIONAL DARK THEME
# ============================================================================

def apply_theme():
    """Apply dark theme with light blue wave background"""
    
    theme_css = """
    <style>
        /* Dark theme with light blue wave background */
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        }
        
        .stApp {
            background: linear-gradient(135deg, 
                rgba(15, 23, 42, 0.95) 0%, 
                rgba(30, 41, 59, 0.95) 100%),
                url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><path d="M0,50 Q25,40 50,50 T100,50" fill="none" stroke="%230ea5e9" stroke-width="0.5" opacity="0.1"/><path d="M0,60 Q25,50 50,60 T100,60" fill="none" stroke="%230ea5e9" stroke-width="0.5" opacity="0.1"/><path d="M0,70 Q25,60 50,70 T100,70" fill="none" stroke="%230ea5e9" stroke-width="0.5" opacity="0.1"/></svg>');
            background-size: 300px 300px;
        }
        
        /* Custom Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem;
            font-weight: 600;
            color: #e2e8f0;
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            font-weight: 500;
            color: #94a3b8;
            text-transform: none;
            letter-spacing: 0.3px;
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        
        /* Card Styling */
        .metric-card {
            background: rgba(30, 41, 59, 0.7);
            padding: 1.2rem;
            border-radius: 10px;
            border: 1px solid rgba(100, 116, 139, 0.3);
            backdrop-filter: blur(10px);
            transition: all 0.2s ease;
        }
        
        .metric-card:hover {
            border-color: rgba(14, 165, 233, 0.5);
            background: rgba(30, 41, 59, 0.9);
        }
        
        /* Button Styling */
        .stButton>button {
            background: linear-gradient(135deg, #0ea5e9 0%, #0369a1 100%);
            color: white;
            border: none;
            padding: 0.6rem 1.5rem;
            border-radius: 6px;
            font-weight: 500;
            font-size: 0.9rem;
            font-family: 'Segoe UI', system-ui, sans-serif;
            transition: all 0.2s ease;
        }
        
        .stButton>button:hover {
            background: linear-gradient(135deg, #0369a1 0%, #0ea5e9 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background-color: rgba(15, 23, 42, 0.5);
            padding: 0.5rem;
            border-radius: 8px;
            border: 1px solid rgba(100, 116, 139, 0.2);
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 6px;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            color: #94a3b8;
            font-family: 'Segoe UI', system-ui, sans-serif;
            transition: all 0.2s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background: rgba(14, 165, 233, 0.2);
            color: #e2e8f0;
            border: 1px solid rgba(14, 165, 233, 0.3);
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            border-right: 1px solid rgba(100, 116, 139, 0.3);
        }
        
        [data-testid="stSidebar"] .stMarkdown {
            color: #e2e8f0;
        }
        
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stRadio label {
            color: #e2e8f0 !important;
            font-weight: 500;
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        
        [data-testid="stSidebar"] .stNumberInput>div>div>input,
        [data-testid="stSidebar"] .stSelectbox>div>div>div {
            background-color: rgba(30, 41, 59, 0.7) !important;
            color: #f1f5f9 !important;
            border: 1px solid rgba(100, 116, 139, 0.3) !important;
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        
        /* Input Styling */
        .stNumberInput>div>div>input:focus,
        .stSelectbox>div>div>div:focus {
            border-color: #0ea5e9 !important;
            box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.2) !important;
        }
        
        /* Alert Boxes */
        .stAlert {
            border-radius: 8px;
            border-left: 4px solid;
            padding: 1rem;
            font-weight: 500;
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(100, 116, 139, 0.2);
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #e2e8f0;
            font-family: 'Segoe UI', system-ui, sans-serif;
            font-weight: 600;
        }
        
        h1 {
            font-size: 2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(14, 165, 233, 0.3);
        }
        
        h2 {
            font-size: 1.5rem;
            margin-top: 1.5rem;
        }
        
        h3 {
            font-size: 1.2rem;
        }
        
        /* Text colors */
        .stMarkdown {
            color: #cbd5e1;
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        
        /* Divider */
        hr {
            margin: 1.5rem 0;
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(100, 116, 139, 0.3), transparent);
        }
        
        /* Dataframe Styling */
        .dataframe {
            background: rgba(30, 41, 59, 0.7);
            border-radius: 8px;
            border: 1px solid rgba(100, 116, 139, 0.2);
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.5);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(100, 116, 139, 0.5);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(148, 163, 184, 0.7);
        }
        
        /* Safety banner */
        .safety-banner {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-left: 4px solid #ef4444;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        /* Confidence indicators */
        .confidence-high {
            color: #10b981;
            font-weight: 600;
        }
        
        .confidence-medium {
            color: #f59e0b;
            font-weight: 600;
        }
        
        .confidence-low {
            color: #ef4444;
            font-weight: 600;
        }
        
        /* Health meter colors */
        .health-good {
            color: #10b981;
            font-weight: 600;
        }
        
        .health-warning {
            color: #f59e0b;
            font-weight: 600;
        }
        
        .health-critical {
            color: #ef4444;
            font-weight: 600;
        }
        
        /* Toggle styling */
        .stRadio [role="radiogroup"] {
            background: rgba(30, 41, 59, 0.7);
            padding: 0.5rem;
            border-radius: 8px;
            border: 1px solid rgba(100, 116, 139, 0.3);
        }
        
        .stRadio [role="radio"] {
            color: #94a3b8;
        }
        
        .stRadio [role="radio"][aria-checked="true"] {
            color: #0ea5e9;
        }
        
        /* Expandable section */
        .streamlit-expanderHeader {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(100, 116, 139, 0.2);
            border-radius: 6px;
            font-weight: 500;
        }
        
        .streamlit-expanderContent {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid rgba(100, 116, 139, 0.2);
            border-radius: 0 0 6px 6px;
        }
    </style>
    """
    
    st.markdown(theme_css, unsafe_allow_html=True)

# ============================================================================
# MACHINE LEARNING MODEL TRAINING
# ============================================================================

@st.cache_resource
def train_model_from_csv(uploaded_file):
    """Train Random Forest model on uploaded diabetes dataset"""
    
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Prepare encoders
        label_encoders = {}
        
        # Encode categorical variables
        categorical_cols = ['gender', 'smoking_history']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        # Separate features and target
        X = df.drop('diabetes', axis=1)
        y = df['diabetes']
        
        # Store feature names
        feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        
        return model, label_encoders, feature_names, accuracy
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, 0.0

# ============================================================================
# CONFIDENCE & UNCERTAINTY HANDLING
# ============================================================================

def calculate_confidence(probability):
    """Calculate model confidence based on distance from 0.5"""
    distance = abs(probability - 0.5)
    
    if distance >= 0.30:
        return "High", "confidence-high"
    elif distance >= 0.15:
        return "Medium", "confidence-medium"
    else:
        return "Low", "confidence-low"

def get_confidence_explanation(confidence_level):
    """Get explanation text for confidence level"""
    explanations = {
        "High": "Model shows high confidence in this assessment",
        "Medium": "Moderate confidence - consider clinical context",
        "Low": "Low confidence - additional evaluation recommended"
    }
    return explanations.get(confidence_level, "Confidence assessment unavailable")

# ============================================================================
# RISK CALCULATION AND CATEGORIZATION
# ============================================================================

def calculate_risk_score(model, patient_data):
    """Calculate diabetes risk probability using trained model"""
    
    try:
        # Reshape data for prediction
        patient_array = np.array(patient_data).reshape(1, -1)
        
        # Get probability of diabetes (class 1)
        risk_probability = model.predict_proba(patient_array)[0][1]
        
        # Convert to percentage
        risk_score = risk_probability * 100
        
        # Calculate confidence
        confidence_level, confidence_class = calculate_confidence(risk_probability)
        
        return risk_score, risk_probability, confidence_level, confidence_class
        
    except Exception as e:
        st.error(f"Error calculating risk: {str(e)}")
        return 0.0, 0.5, "Low", "confidence-low"

def get_risk_category(risk_score):
    """Categorize risk level with clinical thresholds"""
    
    if risk_score < 25:
        return "Low Risk", "#10b981", "health-good"
    elif risk_score < 50:
        return "Moderate Risk", "#f59e0b", "health-warning"
    elif risk_score < 75:
        return "High Risk", "#f97316", "health-critical"
    else:
        return "Critical Risk", "#ef4444", "health-critical"

# ============================================================================
# COUNTERFACTUAL INSIGHT ENGINE
# ============================================================================

def calculate_counterfactual_reduction(model, patient_data, feature_names, feature_values):
    """Calculate potential risk reduction from modifying key factors"""
    
    modifiable_factors = ['bmi', 'HbA1c_level', 'blood_glucose_level']
    reductions = []
    
    # Get current prediction
    patient_array = np.array(patient_data).reshape(1, -1)
    current_risk = model.predict_proba(patient_array)[0][1]
    
    for factor in modifiable_factors:
        if factor in feature_names:
            idx = list(feature_names).index(factor)
            
            # Create modified patient data
            modified_data = patient_data.copy()
            
            if factor == 'bmi':
                # Reduce BMI by 5%
                modified_value = feature_values[factor] * 0.95
                modified_data[idx] = modified_value
                
                # Calculate new risk
                modified_array = np.array(modified_data).reshape(1, -1)
                new_risk = model.predict_proba(modified_array)[0][1]
                
                risk_reduction = (current_risk - new_risk) / current_risk * 100
                
                reductions.append({
                    'factor': 'BMI',
                    'improvement': '5% reduction',
                    'reduction_range': f"{max(5, int(risk_reduction*0.8))}-{min(25, int(risk_reduction*1.2))}%",
                    'current_value': feature_values[factor],
                    'suggestion': f"Consider reducing BMI from {feature_values[factor]:.1f} to {modified_value:.1f}"
                })
            
            elif factor == 'HbA1c_level':
                # Reduce HbA1c by 0.5%
                if feature_values[factor] > 5.0:
                    modified_value = max(4.0, feature_values[factor] - 0.5)
                    modified_data[idx] = modified_value
                    
                    # Calculate new risk
                    modified_array = np.array(modified_data).reshape(1, -1)
                    new_risk = model.predict_proba(modified_array)[0][1]
                    
                    risk_reduction = (current_risk - new_risk) / current_risk * 100
                    
                    reductions.append({
                        'factor': 'HbA1c',
                        'improvement': '0.5% reduction',
                        'reduction_range': f"{max(3, int(risk_reduction*0.7))}-{min(20, int(risk_reduction*1.3))}%",
                        'current_value': feature_values[factor],
                        'suggestion': f"Consider lowering HbA1c from {feature_values[factor]:.1f}% to {modified_value:.1f}%"
                    })
    
    # Sort by potential impact
    reductions.sort(key=lambda x: int(x['reduction_range'].split('-')[0].replace('%', '')), reverse=True)
    
    return reductions[:2]  # Return top 2 recommendations

# ============================================================================
# LONGITUDINAL RISK TRACKING
# ============================================================================

def get_patient_hash(patient_info):
    """Generate hash for patient identification"""
    patient_str = f"{patient_info['age']}_{patient_info['gender']}_{patient_info['bmi']:.1f}"
    return hashlib.md5(patient_str.encode()).hexdigest()[:8]

def update_patient_history(patient_hash, risk_score, patient_info):
    """Update patient's risk history"""
    timestamp = datetime.now()
    
    if patient_hash not in st.session_state.patient_history:
        st.session_state.patient_history[patient_hash] = {
            'patient_info': patient_info,
            'assessments': []
        }
    
    st.session_state.patient_history[patient_hash]['assessments'].append({
        'timestamp': timestamp,
        'risk_score': risk_score
    })

def create_trend_chart(patient_hash):
    """Create trend chart for patient's risk scores over time"""
    if patient_hash not in st.session_state.patient_history:
        return None
    
    assessments = st.session_state.patient_history[patient_hash]['assessments']
    
    if len(assessments) < 2:
        return None
    
    # Sort by timestamp
    assessments.sort(key=lambda x: x['timestamp'])
    
    dates = [a['timestamp'].strftime('%Y-%m-%d') for a in assessments]
    scores = [a['risk_score'] for a in assessments]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#0ea5e9', width=3),
        marker=dict(size=8, color='#0ea5e9'),
        hovertemplate='Date: %{x}<br>Risk: %{y:.1f}%<extra></extra>'
    ))
    
    # Add trend line
    if len(scores) >= 3:
        x_numeric = np.arange(len(scores))
        z = np.polyfit(x_numeric, scores, 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=p(x_numeric),
            mode='lines',
            name='Trend',
            line=dict(color='#10b981', width=2, dash='dash'),
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title='Risk Trend Over Time',
        xaxis_title='Assessment Date',
        yaxis_title='Risk Score (%)',
        height=300,
        margin=dict(l=20, r=20, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Segoe UI'),
        legend=dict(
            bgcolor='rgba(30, 41, 59, 0.8)',
            bordercolor='rgba(100, 116, 139, 0.3)',
            borderwidth=1
        ),
        xaxis=dict(gridcolor='rgba(100, 116, 139, 0.2)'),
        yaxis=dict(gridcolor='rgba(100, 116, 139, 0.2)')
    )
    
    return fig

# ============================================================================
# PATIENT-FRIENDLY LANGUAGE
# ============================================================================

def get_patient_friendly_risk(risk_category):
    """Convert risk category to patient-friendly language"""
    mapping = {
        "Low Risk": "Your current health profile suggests a lower likelihood of developing diabetes-related issues",
        "Moderate Risk": "Some of your health indicators suggest room for improvement to maintain optimal health",
        "High Risk": "Several health factors indicate that focused lifestyle improvements could be beneficial",
        "Critical Risk": "Your health profile shows significant areas where professional guidance could help improve outcomes"
    }
    return mapping.get(risk_category, "Your health profile has been assessed")

def get_patient_friendly_recommendations(risk_category, bmi):
    """Generate patient-friendly recommendations"""
    
    base_recommendations = {
        "Low Risk": [
            "Continue with your healthy lifestyle habits",
            "Maintain regular physical activity",
            "Keep up with annual health check-ups"
        ],
        "Moderate Risk": [
            "Consider increasing your daily physical activity",
            "Focus on balanced meals with plenty of vegetables",
            "Schedule a follow-up health review in 6 months"
        ],
        "High Risk": [
            "Consult with a healthcare provider for personalized guidance",
            "Consider joining a health improvement program",
            "Focus on sustainable lifestyle changes"
        ],
        "Critical Risk": [
            "Seek professional medical advice for comprehensive evaluation",
            "Consider working with a healthcare team for support",
            "Focus on gradual, sustainable improvements"
        ]
    }
    
    recommendations = base_recommendations.get(risk_category, [])
    
    # Add weight-specific advice
    if bmi >= 25:
        recommendations.append("Gradual weight management through diet and exercise can be beneficial")
    elif bmi < 18.5:
        recommendations.append("Focus on nutrient-rich foods to maintain healthy weight")
    
    return recommendations

def generate_recommendations(risk_category, patient_info):
    """Generate clinician-focused recommendations dict used by the UI"""
    bmi = patient_info.get('bmi', 0)
    actions = []
    timeline = "Routine follow-up"
    primary = "General Advice"

    if risk_category == "Low Risk":
        primary = "Low risk - maintain"
        actions = [
            "Continue current lifestyle and preventive care",
            "Routine monitoring annually",
        ]
        timeline = "Annual follow-up"
    elif risk_category == "Moderate Risk":
        primary = "Moderate risk - lifestyle interventions"
        actions = [
            "Increase physical activity to at least 150 minutes/week",
            "Adopt a balanced diet with reduced refined carbohydrates",
            "Consider weight management if BMI >= 25"
        ]
        timeline = "Follow-up in 3-6 months"
    elif risk_category == "High Risk":
        primary = "High risk - clinical review recommended"
        actions = [
            "Refer for clinical assessment and consider structured programs",
            "Initiate targeted lifestyle and possible pharmacologic review",
            "Close monitoring of HbA1c and glucose"
        ]
        timeline = "Follow-up in 1-3 months"
    else:
        primary = "Critical risk - urgent clinical review"
        actions = [
            "Urgent referral to specialist care",
            "Comprehensive evaluation and management plan",
            "Consider immediate metabolic stabilization if needed"
        ]
        timeline = "Immediate/within 1 month"

    if bmi >= 25:
        actions.insert(0, "Address weight management strategies")

    return {"primary": primary, "actions": actions, "timeline": timeline}

# ============================================================================
# BIAS & FAIRNESS AWARENESS
# ============================================================================

def get_bias_notes(patient_info):
    """Generate bias and fairness awareness notes"""
    
    notes = []
    
    # Age considerations
    if patient_info['age'] < 18:
        notes.append("Model reliability reduced for pediatric populations")
    elif patient_info['age'] > 80:
        notes.append("Limited training data for advanced age groups")
    
    # BMI extremes
    if patient_info['bmi'] < 16 or patient_info['bmi'] > 45:
        notes.append("Reduced model reliability at BMI extremes")
    
    # General notes
    notes.append("Training data primarily from clinical populations")
    notes.append("Performance may vary across demographic subgroups")
    notes.append("Clinical judgment should always override model predictions")
    
    return notes

# ============================================================================
# GENAI EXPLANATION LAYER
# ============================================================================

def generate_patient_explanation(risk_category, bmi, hba1c):
    """Generate plain language explanation for patients"""
    
    # Base explanation
    if risk_category == "Low Risk":
        explanation = "Your current health indicators are within favorable ranges. Maintaining your healthy habits will help continue this positive trend."
    elif risk_category == "Moderate Risk":
        explanation = "Some of your health measurements suggest opportunities for improvement. Small, consistent changes to daily habits can make a meaningful difference."
    else:
        explanation = "Your health profile shows areas where focused attention could be beneficial. Working with healthcare professionals can help develop a personalized plan for improvement."
    
    # Add specific guidance
    specific_guidance = []
    
    if bmi >= 25:
        specific_guidance.append("Managing your weight through balanced nutrition and regular activity")
    elif bmi < 18.5:
        specific_guidance.append("Ensuring adequate nutrition to support your health")
    
    if hba1c > 5.7:
        specific_guidance.append("Focusing on consistent meal patterns and carbohydrate awareness")
    
    if specific_guidance:
        explanation += " Areas to consider include " + ", ".join(specific_guidance[:-1])
        if len(specific_guidance) > 1:
            explanation += ", and " + specific_guidance[-1]
        else:
            explanation += specific_guidance[0]
        explanation += "."
    
    explanation += " Remember that health is a journey, and even small improvements can have significant benefits over time."
    
    return explanation

# ============================================================================
# SAFETY GUARDRAILS
# ============================================================================

def check_safety_guardrails(confidence_level, patient_info):
    """Apply safety guardrails to output"""
    
    warnings = []
    
    # Low confidence warning
    if confidence_level == "Low":
        warnings.append("Risk estimate uncertain — additional clinical evaluation recommended before interpretation")
    
    # Extreme values warning
    if patient_info['bmi'] < 16 or patient_info['bmi'] > 45:
        warnings.append("Caution: BMI outside typical validation range")
    
    if patient_info['age'] < 18 or patient_info['age'] > 85:
        warnings.append("Note: Age outside primary validation range")
    
    return warnings

# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================

def create_risk_gauge(risk_score, risk_category, color, show_percentage=True):
    """Create interactive risk gauge visualization"""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': risk_category, 'font': {'size': 18, 'color': '#e2e8f0'}},
        number={
            'suffix': "%" if show_percentage else "",
            'font': {'size': 36, 'color': color},
            'valueformat': '.1f' if show_percentage else ''
        },
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#64748b'},
            'bar': {'color': color, 'thickness': 0.6},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1,
            'bordercolor': "#475569",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [25, 50], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [50, 75], 'color': 'rgba(249, 115, 22, 0.2)'},
                {'range': [75, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.6,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': "Segoe UI, system-ui, sans-serif"}
    )
    
    return fig

def create_feature_importance_chart(model, feature_names):
    """Visualize which factors most influence risk prediction"""
    
    importances = model.feature_importances_
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    # Map feature names to readable format
    feature_mapping = {
        'gender': 'Gender',
        'age': 'Age',
        'hypertension': 'Hypertension',
        'heart_disease': 'Heart Disease',
        'smoking_history': 'Smoking History',
        'bmi': 'Body Mass Index',
        'HbA1c_level': 'HbA1c Level',
        'blood_glucose_level': 'Blood Glucose'
    }
    
    importance_df['Feature'] = importance_df['Feature'].map(feature_mapping).fillna(importance_df['Feature'])
    
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=importance_df['Importance'],
            colorscale='Blues',
            showscale=False
        ),
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Factors Influencing Risk Assessment",
            'font': {'size': 16, 'color': '#e2e8f0'}
        },
        xaxis_title="Relative Impact",
        yaxis_title="Health Factor",
        height=350,
        margin=dict(l=150, r=20, t=50, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': "Segoe UI, system-ui, sans-serif", 'size': 12, 'color': '#e2e8f0'},
        xaxis=dict(gridcolor='rgba(100, 116, 139, 0.2)'),
        yaxis=dict(gridcolor='rgba(100, 116, 139, 0.2)')
    )
    
    return fig

# ============================================================================
# SIDEBAR COMPONENTS
# ============================================================================

def render_sidebar():
    """Render sidebar with patient input fields and controls"""
    
    with st.sidebar:
        st.markdown("### DiabetesGuard AI")
        st.markdown("Clinical Risk Assessment Platform")
        st.markdown("---")
        
        # View mode selection
        st.markdown("#### View Mode")
        view_mode = st.radio(
            "Select view mode:",
            ["Clinician View", "Patient View"],
            index=0 if st.session_state.view_mode == 'clinician' else 1,
            label_visibility="collapsed"
        )
        
        st.session_state.view_mode = 'clinician' if view_mode == "Clinician View" else 'patient'
        
        st.markdown("---")
        
        # Model Training Section
        st.markdown("#### Model Training")
        
        uploaded_file = st.file_uploader(
            "Upload Training Dataset",
            type=['csv'],
            help="Upload diabetes dataset with required columns"
        )
        
        if uploaded_file is not None:
            if st.button("Train Model", type="primary", use_container_width=True):
                with st.spinner("Training machine learning model..."):
                    model, encoders, features, accuracy = train_model_from_csv(uploaded_file)
                    
                    if model is not None:
                        st.session_state.trained_model = model
                        st.session_state.label_encoders = encoders
                        st.session_state.feature_names = features
                        st.session_state.model_accuracy = accuracy
                        st.session_state.model_trained = True
                        
                        st.success("Model trained successfully")
                        st.metric("Model Accuracy", f"{accuracy*100:.1f}%")
                    else:
                        st.error("Model training failed")
        
        if st.session_state.model_trained:
            st.success("Model Ready")
            st.metric("Accuracy", f"{st.session_state.model_accuracy*100:.1f}%")
        
        st.markdown("---")
        
        # Patient Information Input
        st.markdown("#### Patient Information")
        
        gender = st.selectbox(
            "Gender",
            options=['Female', 'Male', 'Other'],
            help="Patient's biological sex"
        )
        
        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=45,
            step=1,
            help="Patient's age in years"
        )
        
        st.markdown("#### Health Status")
        
        hypertension = st.selectbox(
            "Hypertension",
            options=['No', 'Yes'],
            help="Diagnosed hypertension"
        )
        
        heart_disease = st.selectbox(
            "Heart Disease",
            options=['No', 'Yes'],
            help="Diagnosed heart disease"
        )
        
        smoking_history = st.selectbox(
            "Smoking History",
            options=['never', 'former', 'current', 'not current', 'ever', 'No Info'],
            help="Patient's smoking history"
        )
        
        st.markdown("#### Measurements")
        
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1,
            help="Body Mass Index"
        )
        
        # Show BMI category
        bmi_category = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
        st.caption(f"BMI Category: {bmi_category}")
        
        hba1c = st.number_input(
            "HbA1c Level (%)",
            min_value=3.0,
            max_value=15.0,
            value=5.5,
            step=0.1,
            help="Glycated Hemoglobin percentage"
        )
        
        blood_glucose = st.number_input(
            "Blood Glucose (mg/dL)",
            min_value=50,
            max_value=400,
            value=100,
            step=1,
            help="Fasting blood glucose level"
        )
        
        st.markdown("---")
        
        # Assessment Button
        assess_button = st.button("Assess Risk", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        # Data Management
        if len(st.session_state.patient_records) > 0:
            st.markdown("#### Data Management")
            
            if st.button("Export All Data (CSV)", use_container_width=True):
                df_records = pd.DataFrame(st.session_state.patient_records)
                csv_data = df_records.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"patient_assessments_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        if st.button("Clear All Records", type="secondary", use_container_width=True):
            if st.checkbox("Confirm deletion"):
                st.session_state.patient_records = []
                st.success("All records cleared")
                st.rerun()
        
        # Prepare patient info dictionary
        patient_info = {
            'gender': gender,
            'age': age,
            'hypertension': 1 if hypertension == 'Yes' else 0,
            'heart_disease': 1 if heart_disease == 'Yes' else 0,
            'smoking_history': smoking_history,
            'bmi': bmi,
            'HbA1c_level': hba1c,
            'blood_glucose_level': blood_glucose
        }
        
        return patient_info, assess_button

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application logic and layout"""
    
    # Apply theme
    apply_theme()
    
    # Render sidebar and get inputs
    sidebar_result = render_sidebar()
    
    # Check if sidebar returned values
    if sidebar_result is None:
        patient_info = {}
        assess_button = False
    else:
        patient_info, assess_button = sidebar_result
    
    # Main content area
    st.title("DiabetesGuard AI")
    st.markdown("Clinical Diabetes Risk Assessment Platform")
    
    # Safety banner
    st.markdown("""
    <div class="safety-banner">
    <strong>Important:</strong> This tool supports clinical decisions and does not provide diagnoses. 
    All assessments should be reviewed by qualified healthcare professionals.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Check if model is trained
    if not st.session_state.model_trained:
        st.info("Upload and train a model using the sidebar to begin risk assessments.")
        
        st.markdown("""
        #### Required Dataset Format
        
        Your CSV file should contain these columns:
        - **gender**: Female, Male, Other
        - **age**: Patient age in years
        - **hypertension**: 0 (No) or 1 (Yes)
        - **heart_disease**: 0 (No) or 1 (Yes)
        - **smoking_history**: never, former, current, not current, ever, No Info
        - **bmi**: Body Mass Index
        - **HbA1c_level**: Glycated Hemoglobin percentage
        - **blood_glucose_level**: Blood glucose in mg/dL
        - **diabetes**: 0 (No diabetes) or 1 (Has diabetes) - Target variable
        """)
        
        # Show sample data
        sample_data = pd.DataFrame({
            'gender': ['Female', 'Male'],
            'age': [45, 60],
            'hypertension': [0, 1],
            'heart_disease': [0, 0],
            'smoking_history': ['never', 'former'],
            'bmi': [25.1, 28.5],
            'HbA1c_level': [5.5, 6.2],
            'blood_glucose_level': [100, 140],
            'diabetes': [0, 1]
        })
        
        st.dataframe(sample_data, use_container_width=True)
        return
    
    # Process risk assessment
    if assess_button and patient_info:
        with st.spinner("Analyzing patient data..."):
            try:
                # Check if model is trained
                if not st.session_state.model_trained:
                    st.error("Please train the model first using the sidebar.")
                    return
                
                # Prepare patient data for prediction
                model = st.session_state.trained_model
                encoders = st.session_state.label_encoders
                feature_names = st.session_state.feature_names
                
                # Encode categorical variables
                patient_data_encoded = []
                
                for feature in feature_names:
                    if feature in encoders:
                        try:
                            encoded_value = encoders[feature].transform([patient_info[feature]])[0]
                        except Exception:
                            encoded_value = 0
                        patient_data_encoded.append(encoded_value)
                    else:
                        patient_data_encoded.append(patient_info[feature])
                
                # Calculate risk with confidence
                risk_score, risk_probability, confidence_level, confidence_class = calculate_risk_score(model, patient_data_encoded)
                risk_category, color, health_class = get_risk_category(risk_score)
                
                # Get patient hash for tracking
                patient_hash = get_patient_hash(patient_info)
                
                # Check safety guardrails
                safety_warnings = check_safety_guardrails(confidence_level, patient_info)
                
                # Display safety warnings if any
                for warning in safety_warnings:
                    st.warning(warning)
                
                # Store patient record
                patient_record = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    **patient_info,
                    'risk_score': risk_score,
                    'risk_category': risk_category,
                    'confidence': confidence_level
                }
                
                st.session_state.patient_records.append(patient_record)
                
                # Update patient history for longitudinal tracking
                update_patient_history(patient_hash, risk_score, patient_info)
                
                # Display success message
                st.success("Risk assessment completed")
                
                # ============================================
                # CLINICIAN VIEW
                # ============================================
                if st.session_state.view_mode == 'clinician':
                    
                    # Metric cards
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="Risk Score",
                            value=f"{risk_score:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            label="Risk Category",
                            value=risk_category
                        )
                    
                    with col3:
                        st.metric(
                            label="Model Confidence",
                            value=confidence_level
                        )
                    
                    with col4:
                        assessment_count = len(st.session_state.patient_history.get(patient_hash, {}).get('assessments', []))
                        st.metric(
                            label="Assessments",
                            value=assessment_count
                        )
                    
                    # Confidence notice
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.5); padding: 1rem; border-radius: 8px; border-left: 3px solid {color}; margin: 1rem 0;">
                    <p style="margin: 0; color: #94a3b8;">
                    <strong>Confidence:</strong> <span class="{confidence_class}">{confidence_level}</span> - 
                    {get_confidence_explanation(confidence_level)}
                    </p>
                    <p style="margin: 0.5rem 0 0 0; color: #94a3b8; font-size: 0.9rem;">
                    This risk estimate includes statistical uncertainty and should be interpreted alongside clinical judgment.
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Main visualization
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.plotly_chart(
                            create_risk_gauge(risk_score, risk_category, color, show_percentage=True),
                            use_container_width=True
                        )
                    
                    with col2:
                        # Health indicators
                        st.markdown("#### Health Indicators")
                        
                        indicators = [
                            ("BMI", patient_info['bmi'], "kg/m²", 
                             "Normal" if 18.5 <= patient_info['bmi'] < 25 else 
                             "Underweight" if patient_info['bmi'] < 18.5 else "Overweight"),
                            ("HbA1c", patient_info['HbA1c_level'], "%", 
                             "Normal" if patient_info['HbA1c_level'] < 5.7 else "Elevated"),
                            ("Glucose", patient_info['blood_glucose_level'], "mg/dL", 
                             "Normal" if patient_info['blood_glucose_level'] < 100 else "Elevated")
                        ]
                        
                        for name, value, unit, status in indicators:
                            status_class = "health-good" if "Normal" in status else "health-warning" if "Elevated" in status else "health-critical"
                            st.markdown(f"""
                            <div style="margin-bottom: 0.8rem;">
                            <div style="display: flex; justify-content: space-between;">
                            <span style="color: #cbd5e1;">{name}</span>
                            <span style="color: #e2e8f0; font-weight: 500;">{value:.1f}{unit}</span>
                            </div>
                            <div style="color: #94a3b8; font-size: 0.9rem;">
                            Status: <span class="{status_class}">{status}</span>
                            </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Tabs for detailed information
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "Analysis",
                        "Recommendations",
                        "Trend",
                        "Records",
                        "Notes"
                    ])
                    
                    with tab1:
                        # Feature importance
                        st.plotly_chart(
                            create_feature_importance_chart(model, feature_names),
                            use_container_width=True
                        )
                        
                        # Counterfactual insights
                        st.markdown("#### What Would Reduce Risk Most?")
                        
                        counterfactuals = calculate_counterfactual_reduction(
                            model, patient_data_encoded, feature_names, patient_info
                        )
                        
                        if counterfactuals:
                            for cf in counterfactuals:
                                st.info(f"""
                                **{cf['factor']}**: {cf['improvement']}
                                
                                *Could lower estimated risk by approximately {cf['reduction_range']}*
                                
                                {cf['suggestion']}
                                """)
                        else:
                            st.info("All modifiable factors are within optimal ranges.")
                    
                    with tab2:
                        # Generate recommendations
                        from_recommendations = generate_recommendations(risk_category, patient_info)
                        
                        st.markdown(f"#### {from_recommendations['primary']}")
                        st.markdown("##### Recommended Actions:")
                        
                        for action in from_recommendations['actions']:
                            st.markdown(f"- {action}")
                        
                        st.info(f"**Timeline:** {from_recommendations['timeline']}")
                        
                        # GenAI explanation button
                        if st.button("Explain This Assessment"):
                            explanation = generate_patient_explanation(
                                risk_category, 
                                patient_info['bmi'], 
                                patient_info['HbA1c_level']
                            )
                            st.markdown(f"""
                            <div style="background: rgba(30, 41, 59, 0.7); padding: 1.5rem; border-radius: 8px; border: 1px solid rgba(100, 116, 139, 0.3); margin-top: 1rem;">
                            <p style="color: #cbd5e1; margin: 0;">{explanation}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with tab3:
                        # Longitudinal tracking
                        trend_chart = create_trend_chart(patient_hash)
                        
                        if trend_chart:
                            st.plotly_chart(trend_chart, use_container_width=True)
                            
                            assessments = st.session_state.patient_history[patient_hash]['assessments']
                            if len(assessments) >= 2:
                                first_score = assessments[0]['risk_score']
                                last_score = assessments[-1]['risk_score']
                                change = last_score - first_score
                                
                                if abs(change) > 1:
                                    direction = "increased" if change > 0 else "decreased"
                                    st.info(f"Risk score has {direction} by {abs(change):.1f}% over {len(assessments)} assessments")
                        else:
                            st.info("Complete another assessment to see risk trends over time")
                    
                    with tab4:
                        # Patient records
                        if st.session_state.patient_records:
                            df_records = pd.DataFrame(st.session_state.patient_records)
                            st.dataframe(df_records, use_container_width=True)
                        else:
                            st.info("No assessment records available")
                    
                    with tab5:
                        # Bias and fairness notes
                        with st.expander("Model Bias & Reliability Notes"):
                            bias_notes = get_bias_notes(patient_info)
                            
                            st.markdown("##### Limitations:")
                            for note in bias_notes:
                                st.markdown(f"- {note}")
                            
                            st.markdown("""
                            ##### Clinical Guidance:
                            - Model predictions should complement clinical judgment
                            - Consider patient context and individual factors
                            - Use results as one component of comprehensive assessment
                            """)
                
                # ============================================
                # PATIENT VIEW
                # ============================================
                else:
                    # Patient-friendly display
                    st.markdown(f"""
                    <div style="text-align: center; margin: 2rem 0;">
                    <h2 style="color: {color};">{risk_category}</h2>
                    <p style="color: #cbd5e1; font-size: 1.1rem;">
                    {get_patient_friendly_risk(risk_category)}
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Simple gauge (no percentages)
                    st.plotly_chart(
                        create_risk_gauge(risk_score, risk_category, color, show_percentage=False),
                        use_container_width=True
                    )
                    
                    st.markdown("---")
                    
                    # Health summary
                    st.markdown("#### Your Health Summary")
                    
                    health_items = [
                        ("Weight Status", "Healthy weight range" if 18.5 <= patient_info['bmi'] < 25 else 
                         "Below healthy range" if patient_info['bmi'] < 18.5 else "Above healthy range"),
                        ("Blood Sugar Trend", "Within expected range" if patient_info['HbA1c_level'] < 5.7 else 
                         "Slightly elevated" if patient_info['HbA1c_level'] < 6.5 else "Elevated"),
                        ("Overall Health", "Good foundation" if risk_category == "Low Risk" else 
                         "Opportunities for improvement" if risk_category == "Moderate Risk" else "Benefit from guidance")
                    ]
                    
                    for item, status in health_items:
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; margin-bottom: 1rem; padding: 0.8rem; background: rgba(30, 41, 59, 0.5); border-radius: 6px;">
                        <span style="color: #e2e8f0;">{item}</span>
                        <span style="color: #94a3b8;">{status}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Patient-friendly recommendations
                    st.markdown("#### Suggestions for Your Health Journey")
                    
                    recommendations = get_patient_friendly_recommendations(
                        risk_category, patient_info['bmi']
                    )
                    
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                    
                    # Explain button for patients
                    if st.button("Learn More About This Assessment"):
                        explanation = generate_patient_explanation(
                            risk_category, 
                            patient_info['bmi'], 
                            patient_info['HbA1c_level']
                        )
                        st.markdown(f"""
                        <div style="background: rgba(30, 41, 59, 0.7); padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                        <p style="color: #cbd5e1; margin: 0;">{explanation}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Export options (clinician only)
                if st.session_state.view_mode == 'clinician':
                                        st.markdown("#### Export Options")
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # Generate simple text report
                                            bmi_category_local = "Underweight" if patient_info['bmi'] < 18.5 else "Normal" if patient_info['bmi'] < 25 else "Overweight" if patient_info['bmi'] < 30 else "Obese"
                                            report_text = f"""
                    DiabetesGuard AI Assessment Report
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    
                    Patient Information:
                    - Age: {patient_info['age']} years
                    - Gender: {patient_info['gender']}
                    - BMI: {patient_info['bmi']:.1f} ({bmi_category_local})
                    - HbA1c: {patient_info['HbA1c_level']:.1f}%
                    - Blood Glucose: {patient_info['blood_glucose_level']} mg/dL
                    
                    Assessment Results:
                    - Risk Score: {risk_score:.1f}%
                    - Risk Category: {risk_category}
                    - Model Confidence: {confidence_level}
                    
                    Clinical Notes:
                    {get_confidence_explanation(confidence_level)}
                    
                    Important: This tool supports clinical decisions and does not provide diagnoses.
                    """
                                            
                                            st.download_button(
                                                label="Download Text Report",
                                                data=report_text,
                                                file_name=f"assessment_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                                                mime="text/plain",
                                                use_container_width=True
                                            )
                                        
                                        with col2:
                                            # Export current record
                                            current_record_df = pd.DataFrame([patient_record])
                                            csv_data = current_record_df.to_csv(index=False).encode('utf-8')
                                            
                                            st.download_button(
                                                label="Export Assessment Data",
                                                data=csv_data,
                                                file_name=f"assessment_data_{datetime.now().strftime('%Y%m%d')}.csv",
                                                mime="text/csv",
                                                use_container_width=True
                                            )
            
            except Exception as e:
                st.error(f"An error occurred during assessment: {str(e)}")
    
    elif st.session_state.model_trained:
        # Show instructions when no assessment has been run
        st.info("""
        Ready to assess diabetes risk:
        
        1. Enter patient information in the sidebar
        2. Select view mode (Clinician or Patient)
        3. Click 'Assess Risk' to generate analysis
        
        The system provides risk assessment with confidence indicators and personalized recommendations.
        """)
        
        # Show model info
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"""
            **Model Status**: Trained and ready
            
            **Model Accuracy**: {st.session_state.model_accuracy*100:.1f}%
            
            **Assessments Completed**: {len(st.session_state.patient_records)}
            """)
        
        with col2:
            st.info("""
            **Key Features:**
            - Confidence-aware risk assessment
            - Clinician and patient view modes
            - Counterfactual improvement insights
            - Longitudinal risk tracking
            - Bias awareness notes
            """)

if __name__ == "__main__":
    main()