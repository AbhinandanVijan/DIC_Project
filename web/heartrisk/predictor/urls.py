from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    # Assuming 'home' is the view that renders the home.html template
    path('', views.home, name='home'),
    # URL for form submission that maps to the 'submit_health_data' view
    path('result/', views.submit_health_data, name='submit_health_data'),
]