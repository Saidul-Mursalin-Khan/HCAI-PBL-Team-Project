from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return render(request, 'project1/index.html')