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
# 3. TARGET
# ============================================================

target = "Churn"

y = df[target]


# ============================================================
# 4. FEATURE GROUPS
# ============================================================

original_numeric = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

original_categorical = ["Gender", "Subscription Type", "Contract Length"]

engineered_numeric = ["Spend_Per_Tenure", "Support_Calls_Per_Tenure"]

engineered_categorical = ["Payment_Delay_Category"]


# ============================================================
# 5. FEATURE GROUP DEFINITIONS
# ============================================================

feature_groups = {
    "Original Features": {
        "numeric": original_numeric,
        "categorical": original_categorical,
    },
    "Original + Engineered": {
        "numeric": (original_numeric + engineered_numeric),
        "categorical": (original_categorical + engineered_categorical),
    },
    "Original WITHOUT Payment Delay": {
        "numeric": [
            "Age",
            "Tenure",
            "Usage Frequency",
            "Support Calls",
            "Total Spend",
            "Last Interaction",
        ],
        "categorical": original_categorical,
    },
}


# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

X_train_full, X_test_full, y_train, y_test = train_test_split(
    df, y, test_size=0.20, random_state=42, stratify=y
)


# ============================================================
# 7. RESULTS
# ============================================================

results = []


# ============================================================
# 8. TEST EACH FEATURE GROUP
# ============================================================

for group_name, features in feature_groups.items():

    print("\n" + "=" * 70)
    print(group_name)
    print("=" * 70)

    numeric_features = features["numeric"]
    categorical_features = features["categorical"]

    selected_features = numeric_features + categorical_features

    X_train = X_train_full[selected_features]
    X_test = X_test_full[selected_features]

    # --------------------------------------------------------
    # Numeric preprocessing
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # --------------------------------------------------------
    # Column transformer
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    # --------------------------------------------------------
    # Decision Tree
    # --------------------------------------------------------

    model = DecisionTreeClassifier(random_state=42)

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

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    results.append(
        {
            "Feature Group": group_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    )


# ============================================================
# 9. FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="F1", ascending=False)

print("\n" + "=" * 70)
print("FEATURE GROUP COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# 10. SAVE
# ============================================================

results_df.to_csv("reports/feature_group_comparison.csv", index=False)

print("\nSaved: " "reports/feature_group_comparison.csv")
