import pandas as pd
import joblib
import shap

MODEL_PATH = "models/calibrated_final_model.pkl"

model = joblib.load(MODEL_PATH)


def create_features(customer_data):

    data = customer_data.copy()

    data["Spend_Per_Tenure"] = (
        data["Total Spend"] /
        data["Tenure"].replace(0, 1)
    )

    data["Support_Calls_Per_Tenure"] = (
        data["Support Calls"] /
        data["Tenure"].replace(0, 1)
    )

    data["Payment_Delay_Category"] = pd.cut(
        data["Payment Delay"],
        bins=[-1, 5, 15, 30],
        labels=["Low", "Medium", "High"]
    )

    return data


def get_shap_explanation(customer_data):

    data = pd.DataFrame([customer_data])

    data = create_features(data)

    data = data.drop(
        columns=["CustomerID"],
        errors="ignore"
    )

    calibrated_model = model

    base_pipeline = calibrated_model.calibrated_classifiers_[0].estimator

    preprocessor = base_pipeline.named_steps["preprocessor"]

    tree_model = base_pipeline.named_steps["model"]

    transformed_data = preprocessor.transform(data)

    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(tree_model)

    shap_values = explainer.shap_values(
        transformed_data
    )

    if isinstance(shap_values, list):

        values = shap_values[1][0]

    elif shap_values.ndim == 3:

        values = shap_values[0, :, 1]

    else:

        values = shap_values[0]

    explanation = []

    for feature, value in zip(
        feature_names,
        values
    ):

        explanation.append({
            "feature": feature,
            "impact": round(float(value), 6)
        })

    explanation.sort(
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return explanation[:10]