from django.contrib import admin

from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):

    list_display = (
        "id",
        "prediction_label",
        "churn_probability_percentage",
        "age",
        "gender",
        "tenure",
        "support_calls",
        "payment_delay",
        "created_at",
    )

    list_filter = (
        "prediction_label",
        "gender",
        "subscription_type",
        "contract_length",
    )

    search_fields = (
        "prediction_label",
        "gender",
        "subscription_type",
        "contract_length",
    )

    ordering = ("-created_at",)

    @admin.display(
        description="CHURN PROBABILITY"
    )
    def churn_probability_percentage(self, obj):
        return f"{obj.churn_probability * 100:.2f}%"