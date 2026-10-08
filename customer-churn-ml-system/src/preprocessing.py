import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = "data/processed/featured_churn.csv"

# load data
df = pd.read_csv(DATA_PATH)

# seperate features and target
X = df.drop(columns=["CustomerID", "Churn"])

y = df["Churn"]

# Numerical columns
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

# Categorical columns
categorical_features = [
    "Gender",
    "Subscription Type",
    "Contract Length",
    "Payment_Delay_Category",
]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Numerical preprocesssing
numerical_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Combine both
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)

# Fit ONLY on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using the already fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

print("Original training shape:")
print(X_train.shape)

print("\nProcessed training shape:")
print(X_train_processed.shape)

print("\nOriginal testing shape:")
print(X_test.shape)

print("\nProcessed testing shape:")
print(X_test_processed.shape)
