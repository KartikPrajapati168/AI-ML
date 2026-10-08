import pandas as pd
import joblib
import shap

from sklearn.model_selection import train_test_split

# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/featured_churn.csv"
MODEL_PATH = "models/final_model.pkl"

OUTPUT_PATH = "reports/shap_feature_importance.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df.drop(columns=["CustomerID", "Churn"])

y = df["Churn"]


# ============================================================
# SAME TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# ============================================================
# LOAD FINAL MODEL
# ============================================================

print("\nLoading final model...")

pipeline = joblib.load(MODEL_PATH)


# ============================================================
# EXTRACT PREPROCESSOR + MODEL
# ============================================================

preprocessor = pipeline.named_steps["preprocessor"]

model = pipeline.named_steps["model"]


# ============================================================
# TRANSFORM DATA
# ============================================================

print("Transforming test data...")

X_test_transformed = preprocessor.transform(X_test)


# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = preprocessor.get_feature_names_out()


print("\nNumber of transformed features:", len(feature_names))


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

print("Calculating SHAP values...")

shap_values = explainer.shap_values(X_test_transformed)


# # ============================================================
# # HANDLE BINARY CLASSIFICATION
# # ============================================================

# if isinstance(shap_values, list):

#     # Class 1 = Churn
#     shap_class_1 = shap_values[1]

# else:

#     shap_class_1 = shap_values


# # ============================================================
# # GLOBAL FEATURE IMPORTANCE
# # ============================================================

# mean_abs_shap = abs(
#     shap_class_1
# ).mean(axis=0)


# importance_df = pd.DataFrame(
#     {
#         "Feature": feature_names,
#         "Mean_Abs_SHAP": mean_abs_shap
#     }
# )


# ============================================================
# HANDLE SHAP OUTPUT
# ============================================================

if isinstance(shap_values, list):

    # Older SHAP versions
    # Class 1 = Churn
    shap_class_1 = shap_values[1]

else:

    # Newer SHAP versions
    shap_array = shap_values

    print("\nSHAP array shape:", shap_array.shape)

    if shap_array.ndim == 3:

        # Shape:
        # samples × features × classes
        #
        # Class 1 = Churn

        shap_class_1 = shap_array[:, :, 1]

    elif shap_array.ndim == 2:

        shap_class_1 = shap_array

    else:

        raise ValueError(f"Unexpected SHAP output shape: {shap_array.shape}")


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = abs(shap_class_1).mean(axis=0)


print("\nSHAP feature importance shape:", mean_abs_shap.shape)


# ============================================================
# CREATE IMPORTANCE DATAFRAME
# ============================================================

importance_df = pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP": mean_abs_shap})


importance_df = importance_df.sort_values(by="Mean_Abs_SHAP", ascending=False)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("SHAP FEATURE IMPORTANCE")
print("=" * 70)

print(importance_df.head(20).to_string(index=False))


# ============================================================
# SAVE
# ============================================================

importance_df.to_csv(OUTPUT_PATH, index=False)


print("\nSaved:")
print(OUTPUT_PATH)
