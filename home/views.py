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
    {
        "number": "01",
        "name": "Automated Machine Learning",
        "description": "An interactive supervised learning dashboard for data visualization and machine learning. From CSV upload to prediction, everything is streamlined into a simple and intuitive workflow.",
        "tags": ["data loader", "data visualization", "model training", "dataset"],
        "url_name": "project1:index"
    },
    {
        "number": "02",
        "name": "Explainability",
        "description": "Explore how machine learning models make decisions through interactive visual explanations. Compare interpretable models, generate counterfactuals, and analyze feature effects.",
        "tags": ["model complexity", "counterfactual explanation", "decision tree", "logistic regression"],
        "url_name": "project2:index"
    },
    
    {
    "number": "03",
    "name": "Active Learning for Learning-to-Defer",
    "description": "An intelligent human-AI collaboration platform powered by active learning and learning-to-defer. Train models, simulate experts, and optimize when AI should ask for human input.",
    "tags": ["active learning", "learning-to-defer", "text classifier", "sentiment analysis", "IMDB dataset"],
    "url_name": "project3:index"
    },
    {
        "number": "04",
        "name": "-----",
         "description": "m",
        "tags": "",
        "url_name": "project4:index"
        #"description": "movie recommendation",
        #"tags": ["active learning", "hypothesis", "case study", "matrix factorization"],
        #"url_name": "project4:index"
    },
    
]
    
    context = { 
        "students": students, 
        "projects": projects, 
    }
    
    return HttpResponse(template.render(context, request))

