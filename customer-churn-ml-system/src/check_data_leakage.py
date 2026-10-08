import pandas as pd

# =========================================================
# 1. PATH
# =========================================================

DATA_PATH = "data/processed/featured_churn.csv"


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# =========================================================
# 3. DISPLAY ALL COLUMNS
# =========================================================

print("\n========== ALL COLUMNS ==========")

for i, column in enumerate(df.columns):
    print(f"{i}: {column}")


# =========================================================
# 4. CHECK UNNAMED COLUMNS
# =========================================================

print("\n========== UNNAMED COLUMNS ==========")

unnamed_columns = [column for column in df.columns if column.startswith("Unnamed")]

if unnamed_columns:
    print("Found:", unnamed_columns)

    for column in unnamed_columns:
        print(f"\nColumn: {column}")
        print("Data type:", df[column].dtype)
        print("Unique values:", df[column].nunique())
        print(df[column].head(10))

else:
    print("No Unnamed columns found.")


# =========================================================
# 5. CHECK TARGET
# =========================================================

print("\n========== TARGET DISTRIBUTION ==========")

print(df["Churn"].value_counts())

print("\nTarget percentage:")

print(df["Churn"].value_counts(normalize=True).mul(100).round(2))


# =========================================================
# 6. CHECK DUPLICATE ROWS
# =========================================================

print("\n========== DUPLICATE ROWS ==========")

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)


# =========================================================
# 7. CHECK DUPLICATE CUSTOMER IDs
# =========================================================

print("\n========== DUPLICATE CUSTOMER IDs ==========")

if "CustomerID" in df.columns:

    duplicate_ids = df["CustomerID"].duplicated().sum()

    print("Duplicate CustomerID:", duplicate_ids)


# =========================================================
# 8. CHECK CORRELATION WITH TARGET
# =========================================================

print("\n========== NUMERIC FEATURE CORRELATION ==========")

numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()["Churn"].sort_values(ascending=False)

print(correlation)


# =========================================================
# 9. CHECK UNIQUE VALUES
# =========================================================

print("\n========== UNIQUE VALUES PER COLUMN ==========")

for column in df.columns:

    print(
        f"{column:30} " f"unique={df[column].nunique():6} " f"dtype={df[column].dtype}"
    )


# =========================================================
# 10. CHECK FEATURES WITH VERY HIGH TARGET ASSOCIATION
# =========================================================

print("\n========== HIGH CORRELATION FEATURES ==========")

for column, value in correlation.items():

    if column != "Churn" and abs(value) >= 0.90:

        print(f"{column}: correlation = {value:.4f}")


# =========================================================
# 11. CHECK TARGET-LIKE COLUMNS
# =========================================================

print("\n========== POSSIBLE TARGET-LIKE COLUMNS ==========")

target_keywords = [
    "churn",
    "status",
    "target",
    "label",
    "cancel",
    "cancelled",
    "canceled",
    "left",
    "exit",
]

for column in df.columns:

    column_lower = column.lower()

    if any(keyword in column_lower for keyword in target_keywords):

        print(f"Possible target-related column: {column}")


print("\n========== LEAKAGE CHECK COMPLETED ==========")
