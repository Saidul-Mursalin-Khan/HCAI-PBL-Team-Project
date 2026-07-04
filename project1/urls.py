from django.urls import path
from . import views

app_name = 'project1'
urlpatterns = [
    path('',                views.upload,               name='index'),
    path('upload/',         views.upload,               name='upload'),
    path('visualize/',      views.visualize,            name='visualize'),
    path('configure/',      views.configure,            name='configure'),
    path('classification/', views.classification_train, name='classification'),
    path('regression/',     views.regression_train,     name='regression'),
]
