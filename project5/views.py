from django.shortcuts import render

def index(request):
    return render(request, 'project5/index.html', {
        'title': 'Project 5: Reinforcement Learning with Human Feedback'
    })
