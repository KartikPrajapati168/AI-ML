from django.db import models


class PredictionHistory(models.Model):

    age = models.IntegerField()
    gender = models.CharField(max_length=20)

    tenure = models.IntegerField()
    usage_frequency = models.IntegerField()
    support_calls = models.IntegerField()
    payment_delay = models.IntegerField()

    subscription_type = models.CharField(max_length=30)
    contract_length = models.CharField(max_length=30)

    total_spend = models.FloatField()
    last_interaction = models.IntegerField()

    prediction = models.IntegerField()
    prediction_label = models.CharField(max_length=30)
    churn_probability = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prediction_label} - {self.created_at}"