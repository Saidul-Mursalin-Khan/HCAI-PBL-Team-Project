import json
import secrets
from pathlib import Path

from django.db import transaction
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_http_methods

from .forms import BlockFeedbackForm, ConsentForm, FinalFeedbackForm
from .models import (
    BlockFeedback,
    FinalFeedback,
    PairwiseChoice,
    RankingResponse,
    StudySession,
)
from .services.movies import build_movie_plan, get_movie
from .services.study import (
    EVALUATION_TRIALS,
    PAIRWISE_TRIALS,
    RANKING_TRIALS,
    fit_and_score_session,
    next_stage_after_method,
)


SESSION_KEY = "project4_study_public_id"
REPORT_PATH = Path(__file__).resolve().parent / "docs" / "project4_report.pdf"


@require_http_methods(["GET"])
def download_report(request):
    if not REPORT_PATH.is_file():
        raise Http404("The Project 4 report is not available.")
    return FileResponse(
        REPORT_PATH.open("rb"),
        as_attachment=True,
        filename="project4_report.pdf",
        content_type="application/pdf",
    )


def _active_session(request):
    public_id = request.session.get(SESSION_KEY)
    if not public_id:
        return None
    try:
        return StudySession.objects.get(public_id=public_id)
    except (StudySession.DoesNotExist, ValueError):
        request.session.pop(SESSION_KEY, None)
        return None


def _response_time(request):
    try:
        value = int(request.POST.get("response_time_ms", "0"))
    except ValueError:
        value = 0
    return max(0, min(value, 3_600_000))


def _cards(source_ids):
    return [get_movie(source_id).as_card() for source_id in source_ids]


def _redirect_for_stage(session):
    stage_routes = {
        StudySession.Stage.PAIRWISE: "project4:pairwise",
        StudySession.Stage.PAIRWISE_FEEDBACK: "project4:block_feedback",
        StudySession.Stage.RANKING: "project4:ranking",
        StudySession.Stage.RANKING_FEEDBACK: "project4:block_feedback",
        StudySession.Stage.EVALUATION: "project4:evaluation",
        StudySession.Stage.FINAL_FEEDBACK: "project4:final_feedback",
        StudySession.Stage.COMPLETE: "project4:complete",
    }
    route = stage_routes[session.stage]
    if session.stage == StudySession.Stage.PAIRWISE_FEEDBACK:
        return redirect(route, method="pairwise")
    if session.stage == StudySession.Stage.RANKING_FEEDBACK:
        return redirect(route, method="ranking")
    return redirect(route)


@require_http_methods(["GET"])
def index(request):
    session = _active_session(request)
    return render(
        request,
        "project4/index.html",
        {
            "title": "Project 4: Preference Elicitation",
            "active_study": session if session and session.stage != session.Stage.COMPLETE else None,
        },
    )


@require_http_methods(["GET", "POST"])
def consent(request):
    form = ConsentForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            completed_or_started = StudySession.objects.count()
            condition = (
                StudySession.ConditionOrder.PAIRWISE_FIRST
                if completed_or_started % 2 == 0
                else StudySession.ConditionOrder.RANKING_FIRST
            )
            seed = secrets.randbits(63)
            plan = build_movie_plan(
                seed,
                pairwise_trials=PAIRWISE_TRIALS,
                ranking_trials=RANKING_TRIALS,
                evaluation_trials=EVALUATION_TRIALS,
            )
            first_stage = (
                StudySession.Stage.PAIRWISE
                if condition == StudySession.ConditionOrder.PAIRWISE_FIRST
                else StudySession.Stage.RANKING
            )
            session = StudySession.objects.create(
                condition_order=condition,
                stage=first_stage,
                random_seed=seed,
                movie_plan=plan,
                consented_at=timezone.now(),
            )
        request.session[SESSION_KEY] = str(session.public_id)
        return redirect("project4:study")
    return render(request, "project4/consent.html", {"form": form})


@require_http_methods(["GET"])
def study_dispatch(request):
    session = _active_session(request)
    if session is None:
        return redirect("project4:consent")
    return _redirect_for_stage(session)


@require_http_methods(["GET", "POST"])
def pairwise_task(request):
    session = _active_session(request)
    if session is None:
        return redirect("project4:consent")
    if session.stage != StudySession.Stage.PAIRWISE:
        return _redirect_for_stage(session)

    trial = session.pairwise_trial
    source_ids = session.movie_plan["pairwise"][trial]
    if request.method == "POST":
        try:
            chosen_id = int(request.POST.get("chosen_movie_id", ""))
        except ValueError:
            chosen_id = -1
        if chosen_id not in source_ids:
            return render(
                request,
                "project4/pairwise.html",
                {
                    "movies": _cards(source_ids),
                    "trial_number": trial + 1,
                    "trial_total": PAIRWISE_TRIALS,
                    "error": "Please choose one of the two movies.",
                },
                status=400,
            )
        with transaction.atomic():
            PairwiseChoice.objects.update_or_create(
                study_session=session,
                kind=PairwiseChoice.Kind.ELICITATION,
                trial_index=trial,
                defaults={
                    "left_movie_id": source_ids[0],
                    "right_movie_id": source_ids[1],
                    "chosen_movie_id": chosen_id,
                    "response_time_ms": _response_time(request),
                },
            )
            session.pairwise_trial += 1
            if session.pairwise_trial >= PAIRWISE_TRIALS:
                session.stage = StudySession.Stage.PAIRWISE_FEEDBACK
            session.save(update_fields=("pairwise_trial", "stage"))
        return redirect("project4:study")

    return render(
        request,
        "project4/pairwise.html",
        {
            "movies": _cards(source_ids),
            "trial_number": trial + 1,
            "trial_total": PAIRWISE_TRIALS,
        },
    )


@require_http_methods(["GET", "POST"])
def ranking_task(request):
    session = _active_session(request)
    if session is None:
        return redirect("project4:consent")
    if session.stage != StudySession.Stage.RANKING:
        return _redirect_for_stage(session)

    trial = session.ranking_trial
    source_ids = session.movie_plan["rankings"][trial]
    error = None
    if request.method == "POST":
        try:
            ordered_ids = [int(value) for value in json.loads(request.POST.get("ranking", "[]"))]
        except (TypeError, ValueError, json.JSONDecodeError):
            ordered_ids = []
        if len(ordered_ids) != 10 or set(ordered_ids) != set(source_ids):
            error = "Your ranking must contain every movie exactly once."
        else:
            with transaction.atomic():
                RankingResponse.objects.update_or_create(
                    study_session=session,
                    trial_index=trial,
                    defaults={
                        "presented_movie_ids": source_ids,
                        "ordered_movie_ids": ordered_ids,
                        "response_time_ms": _response_time(request),
                    },
                )
                session.ranking_trial += 1
                if session.ranking_trial >= RANKING_TRIALS:
                    session.stage = StudySession.Stage.RANKING_FEEDBACK
                session.save(update_fields=("ranking_trial", "stage"))
            return redirect("project4:study")

    return render(
        request,
        "project4/ranking.html",
        {
            "movies": _cards(source_ids),
            "trial_number": trial + 1,
            "trial_total": RANKING_TRIALS,
            "error": error,
        },
        status=400 if error else 200,
    )


@require_http_methods(["GET", "POST"])
def block_feedback(request, method):
    session = _active_session(request)
    if session is None:
        return redirect("project4:consent")
    expected_stage = {
        "pairwise": StudySession.Stage.PAIRWISE_FEEDBACK,
        "ranking": StudySession.Stage.RANKING_FEEDBACK,
    }.get(method)
    if expected_stage is None:
        raise Http404("Unknown study method.")
    if session.stage != expected_stage:
        return _redirect_for_stage(session)

    instance = BlockFeedback(study_session=session, method=method)
    form = BlockFeedbackForm(request.POST or None, instance=instance)
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            feedback = form.save(commit=False)
            feedback.study_session = session
            feedback.method = method
            feedback.save()
            session.stage = next_stage_after_method(session, method)
            session.save(update_fields=("stage",))
        return redirect("project4:study")
    return render(
        request,
        "project4/block_feedback.html",
        {"form": form, "method": method},
    )


@require_http_methods(["GET", "POST"])
def evaluation_task(request):
    session = _active_session(request)
    if session is None:
        return redirect("project4:consent")
    if session.stage != StudySession.Stage.EVALUATION:
        return _redirect_for_stage(session)

    trial = session.evaluation_trial
    source_ids = session.movie_plan["evaluation"][trial]
    if request.method == "POST":
        try:
            chosen_id = int(request.POST.get("chosen_movie_id", ""))
        except ValueError:
            chosen_id = -1
        if chosen_id not in source_ids:
            return render(
                request,
                "project4/evaluation.html",
                {
                    "movies": _cards(source_ids),
                    "trial_number": trial + 1,
                    "trial_total": EVALUATION_TRIALS,
                    "error": "Please choose one of the two movies.",
                },
                status=400,
            )
        with transaction.atomic():
            PairwiseChoice.objects.update_or_create(
                study_session=session,
                kind=PairwiseChoice.Kind.EVALUATION,
                trial_index=trial,
                defaults={
                    "left_movie_id": source_ids[0],
                    "right_movie_id": source_ids[1],
                    "chosen_movie_id": chosen_id,
                    "response_time_ms": _response_time(request),
                },
            )
            session.evaluation_trial += 1
            if session.evaluation_trial >= EVALUATION_TRIALS:
                session.stage = StudySession.Stage.FINAL_FEEDBACK
            session.save(update_fields=("evaluation_trial", "stage"))
        return redirect("project4:study")

    return render(
        request,
        "project4/evaluation.html",
        {
            "movies": _cards(source_ids),
            "trial_number": trial + 1,
            "trial_total": EVALUATION_TRIALS,
        },
    )


@require_http_methods(["GET", "POST"])
def final_feedback(request):
    session = _active_session(request)
    if session is None:
        return redirect("project4:consent")
    if session.stage != StudySession.Stage.FINAL_FEEDBACK:
        return _redirect_for_stage(session)

    instance = FinalFeedback(study_session=session)
    form = FinalFeedbackForm(request.POST or None, instance=instance)
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            feedback = form.save(commit=False)
            feedback.study_session = session
            feedback.save()
            session.stage = StudySession.Stage.COMPLETE
            session.completed_at = timezone.now()
            session.save(update_fields=("stage", "completed_at"))
        return redirect("project4:complete")
    return render(request, "project4/final_feedback.html", {"form": form})


@require_http_methods(["GET"])
def complete(request):
    session = _active_session(request)
    if session is None:
        return redirect("project4:consent")
    if session.stage != StudySession.Stage.COMPLETE:
        return _redirect_for_stage(session)
    metrics = fit_and_score_session(session)
    return render(
        request,
        "project4/complete.html",
        {"study_session": session, "metrics": metrics},
    )
