from django.urls import path
from . import views

app_name = 'project2'
urlpatterns = [
    path('', views.index, name='index'),
    path('logistic/', views.logistic_view, name='logistic'),
    path('counterfactual/', views.counterfactual_view, name='counterfactual'),
    path('feature-effects/', views.feature_effects_view, name='feature_effects'),
]
