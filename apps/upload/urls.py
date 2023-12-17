from django.urls import path
from django.conf import settings

from . import views

urlpatterns = [
    path('file', views.upload_file, name='upload_file'),
    path('', views.upload_file, name='upload_file')
]