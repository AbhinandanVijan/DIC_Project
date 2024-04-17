# heartrisk/urls.py
from django.urls import path, include

urlpatterns = [
    # ... your other url patterns ...

    # Include the predictor app's URLs
    path('', include('predictor.urls')),
]
