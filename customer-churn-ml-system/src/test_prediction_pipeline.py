import pandas as pd

from prediction_pipeline import predict_churn

# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = "data/processed/featured_churn.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# SELECT REAL CUSTOMERS
# ============================================================

sample_df = df.sample(n=10, random_state=42)


# ============================================================
# ORIGINAL FEATURES ONLY
# ============================================================

input_columns = [
    "CustomerID",
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
]


# ============================================================
# TEST PREDICTIONS
# ============================================================

results = []


for _, row in sample_df.iterrows():

    customer_data = row[input_columns].to_dict()

    result = predict_churn(customer_data)

    results.append(
        {
            "CustomerID": row["CustomerID"],
            "Actual Churn": int(row["Churn"]),
            "Predicted Churn": result["prediction"],
            "Prediction": result["prediction_label"],
            "Churn Probability": result["churn_probability"],
        }
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

results_df = pd.DataFrame(results)


print("\n" + "=" * 90)
print("PREDICTION PIPELINE VALIDATION")
print("=" * 90)

print(results_df.to_string(index=False))


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 90)
print("VALIDATION CHECKS")
print("=" * 90)


# Prediction values

valid_predictions = results_df["Predicted Churn"].isin([0, 1]).all()

print("Predictions are 0/1:", valid_predictions)


# Probability range

valid_probabilities = results_df["Churn Probability"].between(0, 1).all()

print("Probabilities between 0 and 1:", valid_probabilities)


# Accuracy on sampled real records

sample_accuracy = (results_df["Actual Churn"] == results_df["Predicted Churn"]).mean()


print("Sample accuracy:", round(sample_accuracy, 4))
