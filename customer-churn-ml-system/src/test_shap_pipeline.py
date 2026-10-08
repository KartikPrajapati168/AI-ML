from shap_pipeline import get_shap_explanation


sample_customer = {
    "Age": 25,
    "Gender": "Male",
    "Tenure": 2,
    "Usage Frequency": 1,
    "Support Calls": 20,
    "Payment Delay": 30,
    "Subscription Type": "Basic",
    "Contract Length": "Monthly",
    "Total Spend": 100,
    "Last Interaction": 30
}


explanation = get_shap_explanation(
    sample_customer
)


print("\nSHAP EXPLANATION")
print("=" * 60)

for item in explanation:

    print(
        f"{item['feature']:<45}"
        f"{item['impact']:+.6f}"
    )