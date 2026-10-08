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
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


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

print("\nTraining model...")

pipeline.fit(X_train, y_train)


# ============================================================
# 7. PREDICT TRAIN + TEST
# ============================================================

train_predictions = pipeline.predict(X_train)

test_predictions = pipeline.predict(X_test)


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================


def evaluate_model(y_true, predictions):

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(y_true, predictions, zero_division=0)

    recall = recall_score(y_true, predictions, zero_division=0)

    f1 = f1_score(y_true, predictions, zero_division=0)

    return accuracy, precision, recall, f1


# ============================================================
# 9. TRAIN RESULTS
# ============================================================

train_accuracy, train_precision, train_recall, train_f1 = evaluate_model(
    y_train, train_predictions
)


# ============================================================
# 10. TEST RESULTS
# ============================================================

test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(
    y_test, test_predictions
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TRAIN VS TEST PERFORMANCE")
print("=" * 70)

print(f"{'Metric':<15}" f"{'Train':>15}" f"{'Test':>15}" f"{'Difference':>15}")

print("-" * 70)

print(
    f"{'Accuracy':<15}"
    f"{train_accuracy:>15.4f}"
    f"{test_accuracy:>15.4f}"
    f"{train_accuracy - test_accuracy:>15.4f}"
)

print(
    f"{'Precision':<15}"
    f"{train_precision:>15.4f}"
    f"{test_precision:>15.4f}"
    f"{train_precision - test_precision:>15.4f}"
)

print(
    f"{'Recall':<15}"
    f"{train_recall:>15.4f}"
    f"{test_recall:>15.4f}"
    f"{train_recall - test_recall:>15.4f}"
)

print(
    f"{'F1 Score':<15}"
    f"{train_f1:>15.4f}"
    f"{test_f1:>15.4f}"
    f"{train_f1 - test_f1:>15.4f}"
)


# ============================================================
# 12. MODEL COMPLEXITY
# ============================================================

tree_model = pipeline.named_steps["model"]

print("\n" + "=" * 70)
print("TREE COMPLEXITY")
print("=" * 70)

print("Tree depth :", tree_model.get_depth())
print("Leaves     :", tree_model.get_n_leaves())


# ============================================================
# 13. INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

gap = train_f1 - test_f1

print(f"F1 gap: {gap:.4f}")

if gap < 0.02:
    print("Small train-test gap.")
    print("No strong evidence of overfitting.")

elif gap < 0.05:
    print("Moderate train-test gap.")
    print("Some overfitting may be present.")

else:
    print("Large train-test gap.")
    print("The model may be overfitting.")
