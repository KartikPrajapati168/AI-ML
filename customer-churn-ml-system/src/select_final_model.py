import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# ============================================================
# PATH
# ============================================================

DATA_PATH = "data/processed/featured_churn.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading clean feature-engineered dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# REMOVE CUSTOMER ID
# ============================================================

X = df.drop(columns=["CustomerID", "Churn"])

y = df["Churn"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()


print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# NUMERIC PIPELINE
# ============================================================

numeric_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# ============================================================
# MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
    ),
}


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


results = []


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL SELECTION")
print("=" * 70)


for model_name, model in models.items():

    print(f"\nEvaluating: {model_name}")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    scores = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
    )

    result = {
        "Model": model_name,
        "Accuracy Mean": scores["test_accuracy"].mean(),
        "Precision Mean": scores["test_precision"].mean(),
        "Recall Mean": scores["test_recall"].mean(),
        "F1 Mean": scores["test_f1"].mean(),
        "F1 Std": scores["test_f1"].std(),
    }

    results.append(result)


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# Sort by F1 score

results_df = results_df.sort_values(by="F1 Mean", ascending=False)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print("Model     :", best_model["Model"])
print("F1 Score  :", round(best_model["F1 Mean"], 4))
print("F1 Std    :", round(best_model["F1 Std"], 4))


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv("reports/final_model_selection.csv", index=False)

print("\nSaved:")
print("reports/final_model_selection.csv")
