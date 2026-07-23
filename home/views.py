# from django.http import HttpResponse

from django.http import HttpResponse
from django.template import loader

def index(request):
    template = loader.get_template("home/index.html")
    
    
    students = [
        {"name": "Khan, Saidul Mursalin", "matriculation": "670261"},
        {"name": "Urme, Asmaul Husna",    "matriculation": "633736"},#
        {"name": "Günes, Doga Ruken",     "matriculation": "641511"},#
        {"name": "Tabassum, Ridowana",    "matriculation": "638252"},#
        {"name": "Zeshan, Md Mehrabul Islam", "matriculation": "646871"},
    ]
    
    projects = [
        # {"name": "Home", "url_name": "home:index"},
        # {"name": "Home 2", "url_name": "home:index"},
        {"name": "Project 1: Supervised Learning",   "url_name": "project1:index"},
        {"name": "Project 2: Explainability",       "url_name": "project2:index"},
        {"name": "Project 3: Active Learning for Learning-to-Defer",      "url_name": "project3:index"},
        {"name": "Project 4: Recommender System",    "url_name": "project4:index"},
        {"name": "Project 5: RLHF",                  "url_name": "project5:index"},
    ]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))

