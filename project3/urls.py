from django.urls import path
from . import views

app_name = "project3"
urlpatterns = [
    path("", views.index, name="index"),
    path("task1/", views.task1_baseline, name="task1"),
    path("task2/", views.task2_expert, name="task2"),
    path("task3/", views.task3_deferral, name="task3"),
    path("task4/", views.task4_active_learning, name="task4"),
    path("task5/", views.task5_try_it, name="task5"),
    path("advance/<int:step>/", views.advance, name="advance"),
    path("reset/", views.reset_progress, name="reset"),
    path("report/", views.report, name="report"),
]
