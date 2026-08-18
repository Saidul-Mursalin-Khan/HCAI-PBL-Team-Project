from django.urls import path

from . import views


app_name = "project4"

urlpatterns = [
    path("", views.index, name="index"),
    path("report/", views.download_report, name="download_report"),
    path("consent/", views.consent, name="consent"),
    path("study/", views.study_dispatch, name="study"),
    path("study/pairwise/", views.pairwise_task, name="pairwise"),
    path("study/ranking/", views.ranking_task, name="ranking"),
    path(
        "study/feedback/<str:method>/", views.block_feedback, name="block_feedback"
    ),
    path("study/evaluation/", views.evaluation_task, name="evaluation"),
    path("study/final-feedback/", views.final_feedback, name="final_feedback"),
    path("study/complete/", views.complete, name="complete"),
]
