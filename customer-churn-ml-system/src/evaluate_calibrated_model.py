import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


DATA_PATH = "data/processed/featured_churn.csv"
MODEL_PATH = "models/calibrated_final_model.pkl"


print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Loading calibrated model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


print("\nMaking predictions...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n" + "=" * 70)
print("CALIBRATED MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


print("\n" + "=" * 70)
print("PROBABILITY ANALYSIS")
print("=" * 70)

print("\nFirst 20 churn probabilities:")

for i, probability in enumerate(y_prob[:20], start=1):
    print(f"Customer {i}: {probability:.4f}")


print("\nProbability statistics:")

print(f"Minimum probability: {y_prob.min():.4f}")
print(f"Maximum probability: {y_prob.max():.4f}")
print(f"Mean probability   : {y_prob.mean():.4f}")

unique_probabilities = len(set(y_prob))

print(f"Unique probabilities: {unique_probabilities}")