from rest_framework import serializers


class PredictionInputSerializer(serializers.Serializer):

    age = serializers.IntegerField(
        min_value=1,
        max_value=100
    )

    gender = serializers.ChoiceField(
        choices=["Male", "Female"]
    )

    tenure = serializers.IntegerField(
        min_value=0
    )

    usage_frequency = serializers.IntegerField(
        min_value=0
    )

    support_calls = serializers.IntegerField(
        min_value=0
    )

    payment_delay = serializers.IntegerField(
        min_value=0,
        max_value=30
    )

    subscription_type = serializers.ChoiceField(
        choices=["Basic", "Standard", "Premium"]
    )

    contract_length = serializers.ChoiceField(
        choices=["Monthly", "Quarterly", "Annual"]
    )

    total_spend = serializers.FloatField(
        min_value=0
    )

    last_interaction = serializers.IntegerField(
        min_value=0
    )