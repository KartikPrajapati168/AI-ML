from django import forms

class FarmerInputForm(forms.Form):
    breed = forms.CharField(max_length=100)
    age_years = forms.IntegerField()
    parity = forms.IntegerField()
    lactation_stage = forms.CharField(max_length=50)
    weight_kg = forms.FloatField()
    feed_kg = forms.FloatField()
    walking_km = forms.FloatField()
    rumination_min = forms.FloatField()
    temp_c = forms.FloatField()
    humidity = forms.FloatField()
    # optional confirmation fields (when farmer returns)
    # actual_milk_liters = forms.FloatField(required=False)
    # actual_disease_label = forms.IntegerField(required=False)
