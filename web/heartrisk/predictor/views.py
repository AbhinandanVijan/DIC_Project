# views.py
from django.shortcuts import render, redirect
from .forms import HealthDataForm
# from .logistic_regression_classifier import LogRegg
from .Classification import   predictor

def home(request):
    form = HealthDataForm()
    return render(request, 'home.html', {'form': form})

def visu(request):
    return render(request, 'visu.html')

def submit_health_data(request):

    if request.method == 'POST':
        form = HealthDataForm(request.POST)
        if form.is_valid():
            # Process the data, save it to the database, run the risk prediction model
            health_data = form.save(commit=False)
            print("in the view form valid")

            pred = predictor()
            forminput_data = pred.formdata(health_data )
            print("form",forminput_data)
            pred.encoding(forminput_data)
            prediction = pred.prediction()
            print(prediction)
            # return render(request, 'result.html', {'prediction': prediction})
            return render(request, 'visu.html', {'prediction': prediction})
        else:
            print("form not working", form.errors)
    else:
        form = HealthDataForm()
    return render(request, 'home.html', {'form': form})


