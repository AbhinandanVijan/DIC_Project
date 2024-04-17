from django.db import models

# Create your models here.

class HealthData(models.Model):
    
    gender = models.CharField(max_length=10)
    age = models.IntegerField()
    education = models.CharField(max_length=20)
    current_smoker = models.IntegerField()
    cigs_per_day = models.IntegerField()
    bp_meds = models.IntegerField()
    prevalent_stroke = models.CharField(max_length=10)
    prevalent_hypertension = models.IntegerField()
    diabetes = models.IntegerField()
    total_cholestrol = models.IntegerField()
    sys_bp = models.IntegerField()
    dia_bp = models.IntegerField()
    bmi =  models.FloatField()
    heart_rate = models.IntegerField()
    glucose = models.IntegerField()
    heart_stroke = models.CharField(max_length=10)

def __str__(self):
        return f"HealthData {self.id}"