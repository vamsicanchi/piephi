from django.shortcuts import render
import json

# Create your views here.
from rest_framework.views import APIView
from rest_framework.parsers import JSONParser, FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from django.http import  HttpResponse
from django.http import JsonResponse
import os
import json

from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from src.inference import run_inference_image
from src.appconfig import config
from lib.utils.log import log

# Create your views here.
class InferenceView(APIView):
    
    parser_classes = (MultiPartParser,JSONParser)

    @staticmethod
    def post(request):
        
        file = json.loads(request.body)

        run_inference_image(config, file["dataset_name"], file["image_file"], file["image_dir"], log)

        return HttpResponse("completed inference on {}!!!".format(file["image_file"]))