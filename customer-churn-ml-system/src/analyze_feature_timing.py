import pandas as pd

# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/processed/featured_churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 2. FEATURES
# ============================================================

features = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
]


# ============================================================
# 3. FEATURE TIMING ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE TIMING / BUSINESS LEAKAGE ANALYSIS")
print("=" * 80)

print(
    "\nQuestion:"
    "\nWere these features realistically available BEFORE"
    "\nthe customer churned?"
)


for feature in features:

    print("\n" + "-" * 80)

    print("Feature:", feature)

    if feature == "Age":
        print("Business meaning: Customer age.")
        print("Usually available before churn.")

    elif feature == "Gender":
        print("Business meaning: Customer demographic information.")
        print("Usually available before churn.")

    elif feature == "Tenure":
        print("Business meaning: How long the customer has been subscribed.")
        print("Usually available before churn.")

    elif feature == "Usage Frequency":
        print("Business meaning: How frequently the customer uses the service.")
        print("Usually available before churn.")

    elif feature == "Support Calls":
        print("Business meaning: Number of support interactions.")
        print("Potential timing concern.")
        print("If measured before prediction time → valid.")
        print("If measured after churn/termination → leakage risk.")

    elif feature == "Payment Delay":
        print("Business meaning: Number of days payment is delayed.")
        print("Potential timing concern.")
        print("If known before prediction time → valid.")
        print("If generated after churn decision → leakage risk.")

    elif feature == "Subscription Type":
        print("Business meaning: Customer subscription category.")
        print("Usually available before churn.")

    elif feature == "Contract Length":
        print("Business meaning: Length/type of customer contract.")
        print("Usually available before churn.")

    elif feature == "Total Spend":
        print("Business meaning: Total amount spent by customer.")
        print("Potential timing concern.")
        print("Should represent spend accumulated BEFORE prediction time.")

    elif feature == "Last Interaction":
        print("Business meaning: Time/days since customer's last interaction.")
        print("Potential timing concern.")
        print(
            "Usually useful for pre-churn prediction if measured before prediction time."
        )


# ============================================================
# 4. IMPORTANT QUESTIONS
# ============================================================

print("\n" + "=" * 80)
print("QUESTIONS TO VERIFY FROM DATASET SOURCE")
print("=" * 80)

questions = [
    "1. When was Churn recorded?",
    "2. When were Support Calls recorded?",
    "3. When was Payment Delay recorded?",
    "4. When was Total Spend calculated?",
    "5. What does Last Interaction represent?",
    "6. Were all features collected BEFORE the churn event?",
    "7. Is there a prediction date / observation date?",
]

for question in questions:
    print(question)


# ============================================================
# 5. DATASET LIMITATION
# ============================================================

print("\n" + "=" * 80)
print("CURRENT DATASET LIMITATION")
print("=" * 80)

print(
    "The dataset does not currently provide an explicit"
    " prediction timestamp or feature observation timestamp."
)

print(
    "Therefore, we cannot mathematically prove that every"
    " feature was available before churn."
)

print(
    "This is a BUSINESS VALIDITY question rather than"
    "a normal train/test leakage problem."
)
