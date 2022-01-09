"""from django.forms import ModelForm
from django import forms
from django.db import models
from .models import Predictions

class PredictionForm(ModelForm):
    class Meta:
        model = Predictions
        fields = ('image',)

class PredictForm(forms.Form):
    image = forms.ImageField()"""