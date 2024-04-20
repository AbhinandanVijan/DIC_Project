# views.py
from django.shortcuts import render, redirect
from .forms import HealthDataForm

def home(request):
    form = HealthDataForm()
    return render(request, 'home.html', {'form': form})

def submit_health_data(request):

    if request.method == 'POST':
        form = HealthDataForm(request.POST)
        if form.is_valid():
            # Process the data, save it to the database, run the risk prediction model
            health_data = form.save(commit=False)
            print("in the view form valid")
            # Assume a function 'predict_risk' exists that takes health_data and returns a prediction
            #prediction = predict_risk(health_data)
            return render(request, 'result.html', {'prediction': 'Yes'})
        else:
            print("form not working", form.errors)
    else:
        form = HealthDataForm()
    return render(request, 'home.html', {'form': form})
