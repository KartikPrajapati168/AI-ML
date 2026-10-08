import sys
from pathlib import Path

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .models import PredictionHistory
from .serializers import PredictionInputSerializer
from src.shap_pipeline import get_shap_explanation
import pandas as pd
from rest_framework.decorators import api_view
from rest_framework.response import Response


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Add src folder to Python path
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from prediction_pipeline import predict_churn


@api_view(["POST"])
def predict_churn_api(request):

    serializer = PredictionInputSerializer(data=request.data)

    if not serializer.is_valid():
        return Response(
            {"success": False, "errors": serializer.errors},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        customer_data = {
            "Age": request.data["age"],
            "Gender": request.data["gender"],
            "Tenure": request.data["tenure"],
            "Usage Frequency": request.data["usage_frequency"],
            "Support Calls": request.data["support_calls"],
            "Payment Delay": request.data["payment_delay"],
            "Subscription Type": request.data["subscription_type"],
            "Contract Length": request.data["contract_length"],
            "Total Spend": request.data["total_spend"],
            "Last Interaction": request.data["last_interaction"],
        }

        # ML prediction
        result = predict_churn(customer_data)

        shap_explanation = get_shap_explanation(customer_data)

        # Save prediction in database
        PredictionHistory.objects.create(
            age=customer_data["Age"],
            gender=customer_data["Gender"],
            tenure=customer_data["Tenure"],
            usage_frequency=customer_data["Usage Frequency"],
            support_calls=customer_data["Support Calls"],
            payment_delay=customer_data["Payment Delay"],
            subscription_type=customer_data["Subscription Type"],
            contract_length=customer_data["Contract Length"],
            total_spend=customer_data["Total Spend"],
            last_interaction=customer_data["Last Interaction"],
            prediction=result["prediction"],
            prediction_label=result["prediction_label"],
            churn_probability=result["churn_probability"],
        )

        # return Response({"success": True, **result}, status=status.HTTP_200_OK)

        return Response(
            {"success": True, **result, "shap_explanation": shap_explanation},
            status=200,
        )

    except KeyError as e:

        return Response(
            {"success": False, "error": f"Missing field: {str(e)}"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    except Exception as e:

        return Response(
            {"success": False, "error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
def prediction_history_api(request):

    predictions = PredictionHistory.objects.all().order_by("-created_at")

    data = []

    for prediction in predictions:
        data.append(
            {
                "id": prediction.id,
                "age": prediction.age,
                "gender": prediction.gender,
                "tenure": prediction.tenure,
                "usage_frequency": prediction.usage_frequency,
                "support_calls": prediction.support_calls,
                "payment_delay": prediction.payment_delay,
                "subscription_type": prediction.subscription_type,
                "contract_length": prediction.contract_length,
                "total_spend": prediction.total_spend,
                "last_interaction": prediction.last_interaction,
                "prediction": prediction.prediction,
                "prediction_label": prediction.prediction_label,
                "churn_probability": prediction.churn_probability,
                "created_at": prediction.created_at,
            }
        )

    return Response({"success": True, "count": len(data), "predictions": data})


@api_view(["GET"])
def feature_importance_api(request):

    df = pd.read_csv("reports/shap_feature_importance.csv")

    result = []

    for _, row in df.head(10).iterrows():
        result.append({
            "feature": row["Feature"],
            "importance": float(row["Mean_Abs_SHAP"])
        })

    return Response(result, status=200)