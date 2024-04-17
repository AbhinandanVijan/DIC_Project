from django import forms
from .models import HealthData

class HealthDataForm(forms.ModelForm):
    class Meta:
        model = HealthData
        fields = ['gender','age','education','current_smoker','cigs_per_day','bp_meds','prevalent_stroke','prevalent_hypertension',
                  'diabetes','total_cholestrol','sys_bp','dia_bp','bmi','heart_rate','glucose','heart_stroke']