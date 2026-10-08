import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

RAW_DATA_PATH = "data/raw/customer_churn.csv"

CLEAN_DATA_PATH = "data/processed/cleaned_churn.csv"


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

print("Loading raw dataset...")

df = pd.read_csv(RAW_DATA_PATH, na_values=["?", "NA", "N/A", ""], low_memory=False)

print("Original shape:", df.shape)


# ============================================================
# 3. REMOVE DUPLICATE ROWS
# ============================================================

before = len(df)

df = df.drop_duplicates()

removed = before - len(df)

print("Duplicate rows removed:", removed)


# ============================================================
# 4. REMOVE CSV ARTIFACT COLUMNS
# ============================================================

artifact_columns = [column for column in df.columns if column.startswith("Unnamed:")]

print("\nCSV artifact columns:")

if artifact_columns:

    for column in artifact_columns:
        print("-", column)

    df = df.drop(columns=artifact_columns)

else:

    print("None")


# ============================================================
# 5. CLEAN CATEGORICAL COLUMNS
# ============================================================

categorical_columns = ["Gender", "Subscription Type", "Contract Length"]

for column in categorical_columns:

    df[column] = df[column].astype("string").str.strip()


# ============================================================
# 6. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
    "Churn",
]

for column in numeric_columns:

    df[column] = pd.to_numeric(df[column], errors="coerce")


# ============================================================
# 7. TARGET VALIDATION
# ============================================================

print("\nTarget values before cleaning:")

print(df["Churn"].value_counts(dropna=False))


# Remove rows where target is missing
df = df.dropna(subset=["Churn"])


# Keep only valid binary target values
df = df[df["Churn"].isin([0, 1])]


# ============================================================
# 8. CHECK MISSING VALUES
# ============================================================

print("\nMissing values after cleaning:")

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:

    print("No missing values.")

else:

    print(missing)


# ============================================================
# 9. FINAL DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEAN DATASET")
print("=" * 70)

print("Shape:", df.shape)

print("\nColumns:")

for i, column in enumerate(df.columns):

    print(f"{i}: {column}")


# ============================================================
# 10. SAVE CLEAN DATASET
# ============================================================

df.to_csv(CLEAN_DATA_PATH, index=False)

print("\nSaved:", CLEAN_DATA_PATH)


# import pandas as pd


# RAW_PATH = "data/raw/customer_churn.csv"
# PROCESSED_PATH = "data/processed/cleaned_churn.csv"

# def load_data():
#     return pd.read_csv(RAW_PATH)

# def clean_data(df):

#     #1.Remove completely duplicated rows
#     df=df.drop_duplicates().copy()

#     #2.Standardize categorical text
#     categorical_columns=[
#         "Gender",
#         "Subscription Type",
#         "Contract Length"
#     ]
#     for column in categorical_columns:
#         df[column] = df[column].astype(str).str.strip()

#     # 3. Convert numerical columns to numeric
#     numerical_columns = [
#         "Age",
#         "Tenure",
#         "Usage Frequency",
#         "Support Calls",
#         "Payment Delay",
#         "Total Spend",
#         "Last Interaction",
#         "Churn"
#     ]
#     for column in numerical_columns:
#         df[column] = pd.to_numeric(
#             df[column],
#             errors="coerce"
#         )

#     # 4. Remove rows with missing target
#     df = df.dropna(subset=["Churn"])

#     return df

# if __name__ == "__main__":

#     df = load_data()

#     print("Before cleaning:", df.shape)

#     df = clean_data(df)

#     print("After cleaning:", df.shape)

#     df.to_csv(
#         PROCESSED_PATH,
#         index=False
#     )

#     print(
#         f"Cleaned dataset saved to: {PROCESSED_PATH}"
#     )
