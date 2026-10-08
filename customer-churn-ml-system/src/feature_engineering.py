import pandas as pd

CLEAN_DATA_PATH = "data/processed/cleaned_churn.csv"
FEATURED_DATA_PATH = "data/processed/featured_churn.csv"


print("Loading cleaned dataset...")

df = pd.read_csv(CLEAN_DATA_PATH)

print("Original shape:", df.shape)


# ============================================================
# FEATURE 1 — Spend Per Tenure
# ============================================================

df["Spend_Per_Tenure"] = df["Total Spend"] / df["Tenure"].replace(0, 1)


# ============================================================
# FEATURE 2 — Support Calls Per Tenure
# ============================================================

df["Support_Calls_Per_Tenure"] = df["Support Calls"] / df["Tenure"].replace(0, 1)


# ============================================================
# FEATURE 3 — Payment Delay Category
# ============================================================

df["Payment_Delay_Category"] = pd.cut(
    df["Payment Delay"], bins=[-1, 5, 15, 30], labels=["Low", "Medium", "High"]
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print("Final shape:", df.shape)

print("\nColumns:")
for i, column in enumerate(df.columns):
    print(f"{i}: {column}")


print("\nMissing values:")

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values.")
else:
    print(missing)


print("\nPayment Delay Category:")
print(df["Payment_Delay_Category"].value_counts())


# ============================================================
# SAVE
# ============================================================

df.to_csv(FEATURED_DATA_PATH, index=False)

print("\nSaved:", FEATURED_DATA_PATH)


# import pandas as pd


# INPUT_PATH = "data/processed/cleaned_churn.csv"
# OUTPUT_PATH = "data/processed/featured_churn.csv"


# def create_features(df):

#     df = df.copy()

#     # 1. Average spending relative to tenure
#     df["Spend_Per_Tenure"] = (
#         df["Total Spend"] /
#         df["Tenure"].replace(0, 1)
#     )

#     # 2. Support intensity relative to tenure
#     df["Support_Calls_Per_Tenure"] = (
#         df["Support Calls"] /
#         df["Tenure"].replace(0, 1)
#     )

#     # 3. Payment delay category
#     df["Payment_Delay_Category"] = pd.cut(
#         df["Payment Delay"],
#         bins=[-1, 5, 15, 30],
#         labels=["Low", "Medium", "High"]
#     )

#     return df


# if __name__ == "__main__":

#     df = pd.read_csv(INPUT_PATH)

#     print("Before feature engineering:")
#     print(df.shape)

#     df = create_features(df)

#     print("\nNew columns:")
#     print([
#         "Spend_Per_Tenure",
#         "Support_Calls_Per_Tenure",
#         "Payment_Delay_Category"
#     ])

#     print("\nAfter feature engineering:")
#     print(df.shape)

#     df.to_csv(
#         OUTPUT_PATH,
#         index=False
#     )

#     print(
#         f"\nFeatured dataset saved to: {OUTPUT_PATH}"
#     )
