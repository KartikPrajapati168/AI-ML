import pandas as pd

# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/processed/featured_churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. COLUMN CHECK
# ============================================================

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)

for i, column in enumerate(df.columns):
    print(f"{i}: {column}")


# ============================================================
# 3. DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

print("Duplicate rows:", df.duplicated().sum())

if "CustomerID" in df.columns:
    print("Duplicate CustomerID:", df["CustomerID"].duplicated().sum())


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")

else:
    print(missing)


# ============================================================
# 5. TARGET CHECK
# ============================================================

print("\n" + "=" * 70)
print("TARGET CHECK")
print("=" * 70)

print(df["Churn"].value_counts())

print("\nTarget percentages:")

print((df["Churn"].value_counts(normalize=True) * 100).round(2))


# ============================================================
# 6. NUMERIC RANGE CHECK
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

print("\n" + "=" * 70)
print("NUMERIC FEATURE RANGES")
print("=" * 70)

for feature in numeric_features:

    if feature in df.columns:

        print(f"\n{feature}")

        print(f"Min    : {df[feature].min()}")

        print(f"Max    : {df[feature].max()}")

        print(f"Median : {df[feature].median()}")


# ============================================================
# 7. CATEGORICAL VALUES
# ============================================================

categorical_features = [
    "Gender",
    "Subscription Type",
    "Contract Length",
    "Payment_Delay_Category",
]

print("\n" + "=" * 70)
print("CATEGORICAL VALUES")
print("=" * 70)

for feature in categorical_features:

    if feature in df.columns:

        print(f"\n{feature}:")

        print(df[feature].value_counts(dropna=False))


# ============================================================
# 8. UNNAMED COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("CSV ARTIFACT CHECK")
print("=" * 70)

unnamed_columns = [column for column in df.columns if column.startswith("Unnamed:")]

if unnamed_columns:

    print("Found:")

    for column in unnamed_columns:
        print("-", column)

else:

    print("No Unnamed columns found.")


# ============================================================
# 9. TARGET-LIKE COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("TARGET-LIKE COLUMN CHECK")
print("=" * 70)

for column in df.columns:

    if column == "Churn":
        continue

    unique_values = df[column].nunique()

    if unique_values <= 2:

        print(f"{column} -> {unique_values} unique values")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SANITY SUMMARY")
print("=" * 70)

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])
print("Duplicates:", df.duplicated().sum())

print("Missing values:", df.isnull().sum().sum())

print("Unnamed columns:", len(unnamed_columns))

print("Churn classes:", df["Churn"].nunique())

print("\nSanity check completed.")
