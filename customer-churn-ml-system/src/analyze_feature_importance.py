import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

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
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# ============================================================
# 4. PREPROCESSING
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
# 5. DECISION TREE
# ============================================================

model = DecisionTreeClassifier(random_state=42)


pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])


# ============================================================
# 6. TRAIN
# ============================================================

print("\nTraining Decision Tree...")

pipeline.fit(X_train, y_train)


# ============================================================
# 7. GET FEATURE NAMES
# ============================================================

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()


# ============================================================
# 8. GET FEATURE IMPORTANCES
# ============================================================

importances = pipeline.named_steps["model"].feature_importances_


importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})


# ============================================================
# 9. SORT
# ============================================================

importance_df = importance_df.sort_values(by="Importance", ascending=False)


# ============================================================
# 10. DISPLAY
# ============================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

print(importance_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# ============================================================
# 11. ORIGINAL FEATURE GROUP IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("TOP 15 FEATURES")
print("=" * 80)

print(importance_df.head(15).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# ============================================================
# 12. SAVE RESULTS
# ============================================================

importance_df.to_csv("reports/feature_importance.csv", index=False)

print("\nSaved: reports/feature_importance.csv")


# ============================================================
# 13. MODEL INFORMATION
# ============================================================

tree_model = pipeline.named_steps["model"]

print("\n" + "=" * 80)
print("TREE INFORMATION")
print("=" * 80)

print("Tree depth :", tree_model.get_depth())
print("Leaves     :", tree_model.get_n_leaves())
