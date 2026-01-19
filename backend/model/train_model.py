import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

from xgboost import XGBClassifier


# --------------------------------------------------
# Step 1: Load the dataset safely
# --------------------------------------------------
# Using python engine to avoid CSV encoding issues
# commonly seen on Windows or Excel-generated files

DATA_PATH = "../../data/diabetes_clean.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Dataset file not found. Please check the path.")

data = pd.read_csv(
    DATA_PATH,
    engine="python",
    encoding="utf-8"
)

# If the CSV does not have a proper header,
# we manually assign column names
EXPECTED_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "diabetes"
]

if data.shape[1] != len(EXPECTED_COLUMNS):
    data = pd.read_csv(
        DATA_PATH,
        engine="python",
        encoding="utf-8",
        header=None
    )
    data.columns = EXPECTED_COLUMNS

print("Dataset loaded successfully")
print("Dataset shape:", data.shape)
print(data.head())


# --------------------------------------------------
# Step 2: Basic data cleaning
# --------------------------------------------------

# Remove duplicate records
data = data.drop_duplicates()

# Normalize inconsistent smoking history values
data["smoking_history"] = data["smoking_history"].replace(
    ["No Info", "no info", "None", None],
    "unknown"
)


# --------------------------------------------------
# Step 3: Encode categorical variables
# --------------------------------------------------
# Label encoding is used for simplicity and
# interpretability in tree-based models

label_encoders = {}

for column in ["gender", "smoking_history"]:
    encoder = LabelEncoder()
    data[column] = encoder.fit_transform(data[column].astype(str))
    label_encoders[column] = encoder


# --------------------------------------------------
# Step 4: Split features and target
# --------------------------------------------------

X = data.drop("diabetes", axis=1)
y = data["diabetes"].astype(int)


# --------------------------------------------------
# Step 5: Train-test split
# --------------------------------------------------
# Stratification ensures class balance is preserved

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Step 6: Train the risk prediction model
# --------------------------------------------------
# XGBoost is chosen for:
# - strong performance on tabular data
# - compatibility with explainability tools (SHAP)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# --------------------------------------------------
# Step 7: Model evaluation
# --------------------------------------------------
# ROC-AUC is emphasized since missing high-risk
# patients is costly in preventive healthcare

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", round(roc_auc, 4))


# --------------------------------------------------
# Step 8: Save trained model and encoders
# --------------------------------------------------
# These files are used later during inference

with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("label_encoders.pkl", "wb") as encoder_file:
    pickle.dump(label_encoders, encoder_file)

print("\nModel training completed successfully")
print("Saved files: model.pkl, label_encoders.pkl")
