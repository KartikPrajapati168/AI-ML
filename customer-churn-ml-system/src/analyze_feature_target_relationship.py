import pandas as pd

# ============================================================
# 1. PATH
# ============================================================

DATA_PATH = "data/processed/featured_churn.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(df["Churn"].value_counts())

print("\nTarget percentage:")

print(df["Churn"].value_counts(normalize=True).mul(100).round(2))


# ============================================================
# 4. NUMERIC FEATURES
# ============================================================

numeric_features = [
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


# ============================================================
# 5. NUMERIC FEATURES VS CHURN
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC FEATURES VS CHURN")
print("=" * 70)

for feature in numeric_features:

    print("\n" + "-" * 70)
    print(f"FEATURE: {feature}")
    print("-" * 70)

    result = df.groupby("Churn")[feature].agg(["count", "mean", "median", "min", "max"])

    print(result)


# ============================================================
# 6. CHECK UNIQUE VALUES WITH CHURN
# ============================================================

print("\n" + "=" * 70)
print("NUMERIC FEATURE VALUE DISTRIBUTION")
print("=" * 70)

for feature in numeric_features:

    print("\n" + "-" * 70)
    print(f"FEATURE: {feature}")
    print("-" * 70)

    cross_tab = pd.crosstab(df[feature], df["Churn"], normalize="index") * 100

    print(cross_tab.round(2).head(50))


# ============================================================
# 7. CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "Gender",
    "Subscription Type",
    "Contract Length",
    "Payment_Delay_Category",
]


# ============================================================
# 8. CATEGORICAL FEATURES VS CHURN
# ============================================================

print("\n" + "=" * 70)
print("CATEGORICAL FEATURES VS CHURN")
print("=" * 70)

for feature in categorical_features:

    print("\n" + "-" * 70)
    print(f"FEATURE: {feature}")
    print("-" * 70)

    cross_tab = pd.crosstab(df[feature], df["Churn"], normalize="index") * 100

    print(cross_tab.round(2))


# ============================================================
# 9. CHECK FOR NEAR-PERFECT SEPARATION
# ============================================================

print("\n" + "=" * 70)
print("CHECK FOR NEAR-PERFECT SEPARATION")
print("=" * 70)

print(
    "\nFor each feature, we check whether a value/category "
    "contains almost only Churn=0 or Churn=1."
)


# Numeric features

for feature in numeric_features:

    grouped = df.groupby(feature)["Churn"].agg(["count", "mean"])

    suspicious = grouped[
        ((grouped["mean"] <= 0.01) | (grouped["mean"] >= 0.99))
        & (grouped["count"] >= 20)
    ]

    if not suspicious.empty:

        print("\n" + "-" * 70)
        print(f"Potentially suspicious feature: {feature}")
        print("-" * 70)

        print(suspicious)

    else:

        print(f"{feature}: " "No near-perfect separation found.")


# Categorical features

for feature in categorical_features:

    grouped = df.groupby(feature)["Churn"].agg(["count", "mean"])

    suspicious = grouped[
        ((grouped["mean"] <= 0.01) | (grouped["mean"] >= 0.99))
        & (grouped["count"] >= 20)
    ]

    if not suspicious.empty:

        print("\n" + "-" * 70)
        print(f"Potentially suspicious feature: {feature}")
        print("-" * 70)

        print(suspicious)

    else:

        print(f"{feature}: " "No near-perfect separation found.")


# ============================================================
# 10. ENGINEERED FEATURES
# ============================================================

print("\n" + "=" * 70)
print("ENGINEERED FEATURES ANALYSIS")
print("=" * 70)

engineered_features = [
    "Spend_Per_Tenure",
    "Support_Calls_Per_Tenure",
    "Payment_Delay_Category",
]

for feature in engineered_features:

    print("\n" + "-" * 70)
    print(f"ENGINEERED FEATURE: {feature}")
    print("-" * 70)

    # --------------------------------------------------------
    # Numeric engineered features
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(df[feature]):

        result = df.groupby("Churn")[feature].agg(
            ["count", "mean", "median", "min", "max"]
        )

        print(result.round(2))

    # --------------------------------------------------------
    # Categorical engineered features
    # --------------------------------------------------------

    else:

        result = pd.crosstab(df[feature], df["Churn"], normalize="index") * 100

        print(result.round(2))


# ============================================================
# 11. PAYMENT DELAY DETAILED CHECK
# ============================================================

print("\n" + "=" * 70)
print("DETAILED PAYMENT DELAY CHECK")
print("=" * 70)

payment_delay_analysis = (
    pd.crosstab(df["Payment Delay"], df["Churn"], normalize="index") * 100
)

print(payment_delay_analysis.round(2))


# ============================================================
# 12. SUPPORT CALLS DETAILED CHECK
# ============================================================

print("\n" + "=" * 70)
print("DETAILED SUPPORT CALLS CHECK")
print("=" * 70)

support_calls_analysis = (
    pd.crosstab(df["Support Calls"], df["Churn"], normalize="index") * 100
)

print(support_calls_analysis.round(2))


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE-TARGET ANALYSIS COMPLETED")
print("=" * 70)
