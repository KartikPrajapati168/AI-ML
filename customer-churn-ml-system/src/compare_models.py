import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================================================
# 1. PATH
# =========================================================

DATA_PATH = "data/processed/featured_churn.csv"


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# =========================================================
# 3. FEATURES AND TARGET
# =========================================================

X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]


# =========================================================
# 4. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# =========================================================
# 5. LOAD TRAINED MODELS
# =========================================================

models = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
}


# =========================================================
# 6. STORE RESULTS
# =========================================================

results = []


# =========================================================
# 7. EVALUATE EACH MODEL
# =========================================================

for model_name, model_path in models.items():

    print(f"\nEvaluating {model_name}...")

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }
    )


# =========================================================
# 8. CREATE COMPARISON TABLE
# =========================================================

results_df = pd.DataFrame(results)


# =========================================================
# 9. SORT BY F1 SCORE
# =========================================================

results_df = results_df.sort_values(by="F1 Score", ascending=False)


# =========================================================
# 10. DISPLAY RESULTS
# =========================================================

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# =========================================================
# 11. BEST MODEL
# =========================================================

best_model = results_df.iloc[0]

print("\n========== BEST MODEL ==========")

print(f"Model    : {best_model['Model']}")
print(f"Accuracy : {best_model['Accuracy']:.4f}")
print(f"Precision: {best_model['Precision']:.4f}")
print(f"Recall   : {best_model['Recall']:.4f}")
print(f"F1 Score : {best_model['F1 Score']:.4f}")


# =========================================================
# 12. SAVE COMPARISON
# =========================================================

results_df.to_csv("reports/model_comparison.csv", index=False)

print("\nComparison saved to: reports/model_comparison.csv")
