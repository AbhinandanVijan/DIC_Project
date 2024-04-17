from django import forms
from .models import HealthData

class HealthDataForm(forms.ModelForm):

    gender  =  forms.ChoiceField(label='gender', choices=((1, 'Male'), (0, 'Female')), widget=forms.RadioSelect)
    age = forms.IntegerField(label='Age', min_value=0, max_value=150)

    current_smoker = forms.ChoiceField(label='Current Smoker', choices=((1, 'Yes'), (0, 'No')), widget=forms.RadioSelect)
    bp_meds = forms.ChoiceField(label='BP Meds', choices=((1, 'Yes'), (0, 'No')), widget=forms.RadioSelect)
    prevalent_stroke = forms.ChoiceField(label='Prevalent Stroke', choices=((1, 'Yes'), (0, 'No')), widget=forms.RadioSelect)
    prevalent_hypertension = forms.ChoiceField(label='Prevalent Hypertension', choices=((1, 'Yes'), (0, 'No')), widget=forms.RadioSelect)
    diabetes = forms.ChoiceField(label='Diabetes', choices=((1, 'Yes'), (0, 'No')), widget=forms.RadioSelect)
    
    class Meta:
        model = HealthData
        fields = ['gender','age','education','current_smoker','cigs_per_day','bp_meds','prevalent_stroke','prevalent_hypertension',
                  'diabetes','total_cholestrol','sys_bp','dia_bp','bmi','heart_rate','glucose','heart_stroke']