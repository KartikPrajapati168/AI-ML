import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier

# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data/processed/featured_churn.csv"
MODEL_PATH = "models/final_model.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading final dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# FEATURES AND TARGET
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
# IDENTIFY FEATURES
# ============================================================

numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "string"]
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
# FINAL DECISION TREE
# ============================================================

model = DecisionTreeClassifier(random_state=42)


# ============================================================
# FINAL PIPELINE
# ============================================================

final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


# ============================================================
# TRAIN
# ============================================================

print("\nTraining final Decision Tree...")

final_pipeline.fit(X_train, y_train)


print("Training completed.")


# ============================================================
# MODEL INFORMATION
# ============================================================

trained_tree = final_pipeline.named_steps["model"]

print("\n" + "=" * 70)
print("FINAL MODEL INFORMATION")
print("=" * 70)

print("Model       : Decision Tree")
print("Tree depth  :", trained_tree.get_depth())
print("Leaf nodes  :", trained_tree.get_n_leaves())
print("Training samples:", len(X_train))


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(final_pipeline, MODEL_PATH)


print("\nFinal model saved:")
print(MODEL_PATH)
