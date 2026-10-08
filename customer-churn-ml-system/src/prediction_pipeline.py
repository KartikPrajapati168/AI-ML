import pandas as pd
import joblib

# ============================================================
# PATH
# ============================================================

# MODEL_PATH = "models/final_model.pkl"

MODEL_PATH = "models/calibrated_final_model.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading final model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# FEATURE ENGINEERING FOR NEW CUSTOMER
# ============================================================


def create_features(customer_data):
    """
    Create the same engineered features
    that were used during model training.
    """

    data = customer_data.copy()

    # --------------------------------------------------------
    # Spend per Tenure
    # --------------------------------------------------------

    data["Spend_Per_Tenure"] = data["Total Spend"] / data["Tenure"].replace(0, 1)

    # --------------------------------------------------------
    # Support Calls per Tenure
    # --------------------------------------------------------

    data["Support_Calls_Per_Tenure"] = data["Support Calls"] / data["Tenure"].replace(
        0, 1
    )

    # --------------------------------------------------------
    # Payment Delay Category
    # --------------------------------------------------------

    data["Payment_Delay_Category"] = pd.cut(
        data["Payment Delay"], bins=[-1, 5, 15, 30], labels=["Low", "Medium", "High"]
    )

    return data


# ============================================================
# PREDICTION FUNCTION
# ============================================================


def predict_churn(customer_data):
    """
    Predict churn for one or multiple customers.
    """

    # Create DataFrame
    data = pd.DataFrame([customer_data])

    # Create engineered features
    data = create_features(data)

    # Remove CustomerID if provided
    data = data.drop(columns=["CustomerID"], errors="ignore")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(data)[0]

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = model.predict_proba(data)[0]

    churn_probability = probabilities[1]

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "prediction": int(prediction),
        "prediction_label": "Churn" if prediction == 1 else "Not Churn",
        "churn_probability": round(float(churn_probability), 4),
    }

    return result


# ============================================================
# TEST CUSTOMER
# ============================================================

if __name__ == "__main__":

    sample_customer = {
        "CustomerID": "TEST001",
        "Age": 35,
        "Gender": "Male",
        "Tenure": 12,
        "Usage Frequency": 8,
        "Support Calls": 8,
        "Payment Delay": 25,
        "Subscription Type": "Basic",
        "Contract Length": "Monthly",
        "Total Spend": 350,
        "Last Interaction": 25,
    }

    print("\n" + "=" * 70)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 70)

    result = predict_churn(sample_customer)

    print("\nPrediction:")
    print(result["prediction_label"])

    print("Churn Probability:", result["churn_probability"])
