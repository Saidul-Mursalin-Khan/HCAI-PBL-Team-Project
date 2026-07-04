from django.shortcuts import render

def index(request):
    return render(request, 'project1/index.html', {
        'title': 'Project 1: Supervised Learning Interface'
    })
