import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# =========================================================
# 1. PATHS
# =========================================================

DATA_PATH = "data/processed/featured_churn.csv"
MODEL_PATH = "models/xgboost.pkl"


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# =========================================================
# 3. FEATURES AND TARGET
# =========================================================

X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]


# =========================================================
# 4. FEATURE LISTS
# =========================================================

numerical_features = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
    "Spend_Per_Tenure",
    "Support_Calls_Per_Tenure",
]

categorical_features = [
    "Gender",
    "Subscription Type",
    "Contract Length",
    "Payment_Delay_Category",
]


# =========================================================
# 5. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# =========================================================
# 6. NUMERICAL PIPELINE
# =========================================================

numerical_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)


# =========================================================
# 7. CATEGORICAL PIPELINE
# =========================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


# =========================================================
# 8. PREPROCESSOR
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# =========================================================
# 9. XGBOOST MODEL
# =========================================================

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="logloss",
)


# =========================================================
# 10. COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


# =========================================================
# 11. TRAIN
# =========================================================

print("\nTraining XGBoost...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# =========================================================
# 12. PREDICTION
# =========================================================

y_pred = pipeline.predict(X_test)


# =========================================================
# 13. EVALUATION
# =========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n========== XGBOOST RESULTS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# =========================================================
# 14. CLASSIFICATION REPORT
# =========================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(y_test, y_pred))


# =========================================================
# 15. CONFUSION MATRIX
# =========================================================

print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(y_test, y_pred))


# =========================================================
# 16. SAVE MODEL
# =========================================================

joblib.dump(pipeline, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
