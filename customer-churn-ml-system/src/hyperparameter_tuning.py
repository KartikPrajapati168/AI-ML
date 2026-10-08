import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.tree import DecisionTreeClassifier

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


numerical_features = [
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


categorical_features = [
    "Gender",
    "Subscription Type",
    "Contract Length",
    "Payment_Delay_Category",
]


# =========================================================
# 4. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# =========================================================
# 5. PREPROCESSING
# =========================================================

numerical_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)


categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# =========================================================
# 6. DECISION TREE MODEL
# =========================================================

model = DecisionTreeClassifier(random_state=42)


# =========================================================
# 7. COMPLETE PIPELINE
# =========================================================

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


# =========================================================
# 8. HYPERPARAMETER GRID
# =========================================================

param_grid = {
    "model__max_depth": [3, 5, 10, 15, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 5],
}


# =========================================================
# 9. 5-FOLD CROSS-VALIDATION
# =========================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# =========================================================
# 10. GRID SEARCH
# =========================================================

grid_search = GridSearchCV(
    estimator=pipeline, param_grid=param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=1
)


print("\nStarting Decision Tree Hyperparameter Tuning...")

grid_search.fit(X_train, y_train)

print("\nHyperparameter tuning completed!")


# =========================================================
# 11. BEST PARAMETERS
# =========================================================

print("\n========== BEST PARAMETERS ==========")

print(grid_search.best_params_)


# =========================================================
# 12. BEST CV F1
# =========================================================

print("\n========== BEST CROSS-VALIDATION F1 ==========")

print(f"{grid_search.best_score_:.4f}")


# =========================================================
# 13. SAVE BEST MODEL
# =========================================================

best_model = grid_search.best_estimator_

joblib.dump(best_model, MODEL_PATH)

print(f"\nTuned model saved to: {MODEL_PATH}")


# xgbooost selected

# import pandas as pd
# import joblib

# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder

# from xgboost import XGBClassifier


# # ==============================
# # PATHS
# # ==============================

# DATA_PATH = "data/processed/featured_churn.csv"
# MODEL_PATH = "models/tuned_xgboost.pkl"


# # ==============================
# # LOAD DATA
# # ==============================

# df = pd.read_csv(DATA_PATH)

# print("Dataset shape:", df.shape)


# # ==============================
# # FEATURES AND TARGET
# # ==============================

# X = df.drop(columns=["CustomerID", "Churn"])
# y = df["Churn"]


# numerical_features = [
#     "Age",
#     "Tenure",
#     "Usage Frequency",
#     "Support Calls",
#     "Payment Delay",
#     "Total Spend",
#     "Last Interaction",
#     "Spend_Per_Tenure",
#     "Support_Calls_Per_Tenure"
# ]


# categorical_features = [
#     "Gender",
#     "Subscription Type",
#     "Contract Length",
#     "Payment_Delay_Category"
# ]


# # ==============================
# # TRAIN / TEST SPLIT
# # ==============================

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.20,
#     random_state=42,
#     stratify=y
# )

# print("Training samples:", X_train.shape[0])
# print("Testing samples:", X_test.shape[0])


# # ==============================
# # PREPROCESSING
# # ==============================

# numerical_pipeline = Pipeline(
#     steps=[
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler())
#     ]
# )


# categorical_pipeline = Pipeline(
#     steps=[
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("encoder", OneHotEncoder(handle_unknown="ignore"))
#     ]
# )


# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", numerical_pipeline, numerical_features),
#         ("cat", categorical_pipeline, categorical_features)
#     ]
# )


# # ==============================
# # XGBOOST MODEL
# # ==============================

# model = XGBClassifier(
#     random_state=42,
#     eval_metric="logloss"
# )


# # ==============================
# # COMPLETE PIPELINE
# # ==============================

# pipeline = Pipeline(
#     steps=[
#         ("preprocessor", preprocessor),
#         ("model", model)
#     ]
# )


# # ==============================
# # HYPERPARAMETER GRID
# # ==============================

# param_grid = {
#     "model__n_estimators": [100, 200],
#     "model__learning_rate": [0.05, 0.1],
#     "model__max_depth": [3, 5]
# }


# # ==============================
# # 5-FOLD CROSS-VALIDATION
# # ==============================

# cv = StratifiedKFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=42
# )


# # ==============================
# # GRID SEARCH
# # ==============================

# grid_search = GridSearchCV(
#     estimator=pipeline,
#     param_grid=param_grid,
#     cv=cv,
#     scoring="f1",
#     n_jobs=-1,
#     verbose=1
# )


# print("\nStarting Hyperparameter Tuning...")

# grid_search.fit(X_train, y_train)

# print("\nHyperparameter tuning completed!")


# # ==============================
# # BEST PARAMETERS
# # ==============================

# print("\n========== BEST PARAMETERS ==========")

# print(grid_search.best_params_)


# # ==============================
# # BEST CV SCORE
# # ==============================

# print("\n========== BEST CROSS-VALIDATION F1 ==========")

# print(f"{grid_search.best_score_:.4f}")


# # ==============================
# # SAVE BEST MODEL
# # ==============================

# best_model = grid_search.best_estimator_

# joblib.dump(
#     best_model,
#     MODEL_PATH
# )

# print(f"\nTuned model saved to: {MODEL_PATH}")
