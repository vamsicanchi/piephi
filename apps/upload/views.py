from django.shortcuts import render

# Create your views here.

from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFile
from django.views.decorators.csrf import ensure_csrf_cookie

@ensure_csrf_cookie
def upload_file(request):
    if request.method == 'POST':
        form = UploadFile(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'])
            context = {'msg' : '<span style="color: green;">File successfully uploaded</span>'}
            return render(request, "upload.html", context)
    else:
        form = UploadFile()
    return render(request, 'upload.html', {'form': form})

def handle_uploaded_file(f):
    with open(f.name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)