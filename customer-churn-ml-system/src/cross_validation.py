import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

# =========================================================
# 1. LOAD DATA
# =========================================================

DATA_PATH = "data/processed/featured_churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# =========================================================
# 2. FEATURES AND TARGET
# =========================================================

X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]


# =========================================================
# 3. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# =========================================================
# 4. LOAD TRAINED MODELS
# =========================================================

models = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
}


# =========================================================
# 5. STRATIFIED K-FOLD
# =========================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# =========================================================
# 6. EVALUATION METRICS
# =========================================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


# =========================================================
# 7. STORE RESULTS
# =========================================================

results = []


# =========================================================
# 8. CROSS-VALIDATE EACH MODEL
# =========================================================

for model_name, model_path in models.items():

    print(f"\nRunning Cross-Validation for {model_name}...")

    model = joblib.load(model_path)

    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    results.append(
        {
            "Model": model_name,
            "Accuracy Mean": scores["test_accuracy"].mean(),
            "Accuracy Std": scores["test_accuracy"].std(),
            "Precision Mean": scores["test_precision"].mean(),
            "Precision Std": scores["test_precision"].std(),
            "Recall Mean": scores["test_recall"].mean(),
            "Recall Std": scores["test_recall"].std(),
            "F1 Mean": scores["test_f1"].mean(),
            "F1 Std": scores["test_f1"].std(),
        }
    )


# =========================================================
# 9. CREATE RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(results)


# =========================================================
# 10. SORT BY F1
# =========================================================

results_df = results_df.sort_values(by="F1 Mean", ascending=False)


# =========================================================
# 11. DISPLAY RESULTS
# =========================================================

print("\n========== 5-FOLD CROSS-VALIDATION ==========")

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# =========================================================
# 12. SAVE RESULTS
# =========================================================

results_df.to_csv("reports/cross_validation_results.csv", index=False)

print("\nResults saved to: " "reports/cross_validation_results.csv")
