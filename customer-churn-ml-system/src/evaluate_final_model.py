import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# =========================================================
# 1. PATHS
# =========================================================

DATA_PATH = "data/processed/featured_churn.csv"
MODEL_PATH = "models/tuned_decision_tree.pkl"


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
# 5. LOAD TUNED MODEL
# =========================================================

print("\nLoading tuned Decision Tree...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# =========================================================
# 6. FINAL PREDICTIONS
# =========================================================

print("\nGenerating predictions on test data...")

y_pred = model.predict(X_test)


# =========================================================
# 7. CALCULATE METRICS
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)


# =========================================================
# 8. FINAL RESULTS
# =========================================================

print("\n========================================")
print("       FINAL MODEL PERFORMANCE")
print("========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# =========================================================
# 9. CLASSIFICATION REPORT
# =========================================================

print("\n========================================")
print("       CLASSIFICATION REPORT")
print("========================================")

print(classification_report(y_test, y_pred))


# =========================================================
# 10. CONFUSION MATRIX
# =========================================================

print("\n========================================")
print("          CONFUSION MATRIX")
print("========================================")

print(confusion_matrix(y_test, y_pred))
