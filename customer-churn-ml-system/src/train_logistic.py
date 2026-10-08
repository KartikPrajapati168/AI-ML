import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import joblib

DATA_PATH = "data/processed/featured_churn.csv"
MODEL_PATH = "models/logistic_regression.pkl"


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["CustomerID", "Churn"])

y = df["Churn"]


# --------------------------------------------------
# 3. Define feature groups
# --------------------------------------------------

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


# --------------------------------------------------
# 4. Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# --------------------------------------------------
# 5. Numerical preprocessing
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)


# --------------------------------------------------
# 6. Categorical preprocessing
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


# --------------------------------------------------
# 7. Combine preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# --------------------------------------------------
# 8. Create Logistic Regression model
# --------------------------------------------------

model = LogisticRegression(max_iter=1000, random_state=42)


# --------------------------------------------------
# 9. Complete ML Pipeline
# --------------------------------------------------

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


# --------------------------------------------------
# 10. Train
# --------------------------------------------------

print("\nTraining Logistic Regression...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# --------------------------------------------------
# 11. Prediction
# --------------------------------------------------

y_pred = pipeline.predict(X_test)


# --------------------------------------------------
# 12. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)


print("\n========== MODEL RESULTS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(y_test, y_pred))


print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# 13. Save model
# --------------------------------------------------

joblib.dump(pipeline, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
