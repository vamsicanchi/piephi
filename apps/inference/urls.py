from django.urls import path
from django.conf import settings

from .views import InferenceView

urlpatterns = [
    path('image', InferenceView.as_view(), name='upload_file'),
    path('', InferenceView.as_view(), name='upload_file')
]