import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
# 7. DIFFERENT TREE DEPTHS
# ============================================================

depths = [1, 2, 3, 4, 5, 7, 10, 15, None]


results = []


# ============================================================
# 8. TRAIN DIFFERENT TREES
# ============================================================

for depth in depths:

    print("\n" + "-" * 70)

    print(f"Testing Decision Tree " f"max_depth = {depth}")

    model = DecisionTreeClassifier(max_depth=depth, random_state=42)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = pipeline.predict(X_test)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions, zero_division=0)

    recall = recall_score(y_test, predictions, zero_division=0)

    f1 = f1_score(y_test, predictions, zero_division=0)

    # --------------------------------------------------------
    # Tree information
    # --------------------------------------------------------

    tree_model = pipeline.named_steps["model"]

    tree_depth = tree_model.get_depth()

    leaves = tree_model.get_n_leaves()

    print(f"Actual depth: {tree_depth}")
    print(f"Leaves      : {leaves}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1 Score    : {f1:.4f}")

    results.append(
        {
            "Max Depth": depth,
            "Actual Depth": tree_depth,
            "Leaves": leaves,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    )


# ============================================================
# 9. RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)


print("\n" + "=" * 80)
print("DECISION TREE DEPTH COMPARISON")
print("=" * 80)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results_df.to_csv("reports/tree_depth_comparison.csv", index=False)


print("\nSaved: " "reports/tree_depth_comparison.csv")
