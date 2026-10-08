from django.urls import path
from .views import predict_churn_api,prediction_history_api,feature_importance_api


urlpatterns = [
    path("", predict_churn_api, name="predict-churn"),
    path("history/", prediction_history_api, name="prediction-history"),
    path("feature-importance/",feature_importance_api,name="feature-importance"),
]