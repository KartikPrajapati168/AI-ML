import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV


DATA_PATH = "data/processed/featured_churn.csv"
MODEL_PATH = "models/calibrated_final_model.pkl"


print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


numeric_features = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
    "Spend_Per_Tenure",
    "Support_Calls_Per_Tenure"
]

categorical_features = [
    "Gender",
    "Subscription Type",
    "Contract Length",
    "Payment_Delay_Category"
]


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


base_model = DecisionTreeClassifier(
    random_state=42
)


base_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", base_model)
])


calibrated_model = CalibratedClassifierCV(
    estimator=base_pipeline,
    method="sigmoid",
    cv=5
)


print("Training calibrated model...")

calibrated_model.fit(X_train, y_train)

print("Training completed.")


joblib.dump(
    calibrated_model,
    MODEL_PATH
)

print("\nCalibrated model saved:")
print(MODEL_PATH)