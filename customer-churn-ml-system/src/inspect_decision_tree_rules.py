import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

# ============================================================
# 1. PATH
# ============================================================

DATA_PATH = "data/processed/featured_churn.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. FEATURES
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


# ============================================================
# 4. X AND Y
# ============================================================

X = df[numeric_features + categorical_features]

y = df[target]


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# ============================================================
# 7. TRANSFORM TRAINING DATA
# ============================================================

X_train_transformed = preprocessor.fit_transform(X_train)


# ============================================================
# 8. GET FEATURE NAMES
# ============================================================

feature_names = preprocessor.get_feature_names_out()


# ============================================================
# 9. TRAIN DECISION TREE
# ============================================================

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train_transformed, y_train)


# ============================================================
# 10. TREE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE INFORMATION")
print("=" * 70)

print("Tree depth:", model.get_depth())
print("Number of leaves:", model.get_n_leaves())


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": model.feature_importances_}
)

importance_df = importance_df.sort_values(by="Importance", ascending=False)

print(importance_df.head(20).to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# ============================================================
# 12. DECISION TREE RULES
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE RULES")
print("=" * 70)

rules = export_text(model, feature_names=list(feature_names), max_depth=5)

print(rules)


# ============================================================
# 13. SAVE RULES
# ============================================================

with open("reports/decision_tree_rules.txt", "w", encoding="utf-8") as file:

    file.write(rules)


print("\nSaved: reports/decision_tree_rules.txt")
