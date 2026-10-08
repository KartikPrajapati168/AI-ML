import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/processed/featured_churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. FEATURES
# ============================================================

numeric_features = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

categorical_features = ["Gender", "Subscription Type", "Contract Length"]

target = "Churn"


X = df[numeric_features + categorical_features]
y = df[target]


# ============================================================
# 3. SHUFFLE TARGET
# ============================================================

print("\nOriginal target distribution:")
print(y.value_counts())

y_shuffled = y.sample(frac=1, random_state=42).reset_index(drop=True)

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)


print("\nShuffled target distribution:")
print(y_shuffled.value_counts())


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_shuffled, test_size=0.20, random_state=42, stratify=y_shuffled
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)


categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)


preprocessor = ColumnTransformer(
    [
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# ============================================================
# 6. DECISION TREE
# ============================================================

model = DecisionTreeClassifier(random_state=42)


pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])


# ============================================================
# 7. TRAIN
# ============================================================

print("\nTraining model on SHUFFLED target...")

pipeline.fit(X_train, y_train)


# ============================================================
# 8. PREDICTION
# ============================================================

predictions = pipeline.predict(X_test)


# ============================================================
# 9. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions, zero_division=0)

recall = recall_score(y_test, predictions, zero_division=0)

f1 = f1_score(y_test, predictions, zero_division=0)


# ============================================================
# 10. RESULTS
# ============================================================

print("\n" + "=" * 60)
print("SHUFFLED TARGET TEST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# 11. INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

if f1 < 0.60:
    print("PASS: Model performance collapsed after target shuffling.")
    print(
        "This supports that the original high performance "
        "was not caused by a basic pipeline leakage issue."
    )

else:
    print("WARNING: Performance is still unusually high " "after target shuffling.")
    print("Further leakage investigation is required.")
