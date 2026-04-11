from django.shortcuts import render

def index(request):
    return render(request, 'project4/index.html', {
        'title': 'Project 4: Recommender System'
    })
