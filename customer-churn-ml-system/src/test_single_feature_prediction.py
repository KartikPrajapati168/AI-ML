import pandas as pd

from sklearn.model_selection import train_test_split
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

features = [
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

target = "Churn"


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# ============================================================
# 5. TEST EACH FEATURE INDIVIDUALLY
# ============================================================

results = []

print("\n" + "=" * 70)
print("SINGLE FEATURE PREDICTION TEST")
print("=" * 70)

for feature in features:

    print("\n" + "-" * 70)
    print(f"Testing feature: {feature}")
    print("-" * 70)

    # --------------------------------------------------------
    # Single-feature Decision Tree
    # max_depth=1 means Decision Stump
    # --------------------------------------------------------

    model = DecisionTreeClassifier(max_depth=1, random_state=42)

    model.fit(X_train[[feature]], y_train)

    predictions = model.predict(X_test[[feature]])

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
            "Feature": feature,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
    )


# ============================================================
# 6. SORT RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(by="F1", ascending=False)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SINGLE FEATURE RESULTS")
print("=" * 70)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv("reports/single_feature_results.csv", index=False)

print("\nSaved: reports/single_feature_results.csv")


# ============================================================
# 9. INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

best_feature = results_df.iloc[0]

print(f"\nBest single feature: " f"{best_feature['Feature']}")

print(f"Best single-feature F1: " f"{best_feature['F1']:.4f}")

print("\nThis test uses only ONE feature at a time " "with a depth-1 Decision Tree.")
