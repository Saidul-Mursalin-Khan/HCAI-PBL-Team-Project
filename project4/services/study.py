from django.utils import timezone

from .movies import get_feature_bundle
from .preference import (
    evaluate_on_pairwise_choices,
    fit_bradley_terry,
    fit_plackett_luce,
)
from ..models import PairwiseChoice


PAIRWISE_TRIALS = 10
RANKING_TRIALS = 2
EVALUATION_TRIALS = 20


def next_stage_after_method(session, method):
    if method == session.first_method:
        return session.second_method
    return session.Stage.EVALUATION


def fit_and_score_session(session):
    if session.method_metrics:
        return session.method_metrics

    bundle = get_feature_bundle()
    row_map = bundle.row_by_source_id

    pairwise = list(
        session.pairwise_choices.filter(kind=PairwiseChoice.Kind.ELICITATION)
        .order_by("trial_index")
    )
    chosen_rows = []
    rejected_rows = []
    for response in pairwise:
        chosen_rows.append(row_map[response.chosen_movie_id])
        rejected_id = (
            response.right_movie_id
            if response.chosen_movie_id == response.left_movie_id
            else response.left_movie_id
        )
        rejected_rows.append(row_map[rejected_id])

    rankings = [
        [row_map[source_id] for source_id in response.ordered_movie_ids]
        for response in session.rankings.order_by("trial_index")
    ]
    pairwise_fit = fit_bradley_terry(bundle.matrix, chosen_rows, rejected_rows)
    ranking_fit = fit_plackett_luce(bundle.matrix, rankings)

    evaluation = list(
        session.pairwise_choices.filter(kind=PairwiseChoice.Kind.EVALUATION)
        .order_by("trial_index")
    )
    left_rows = [row_map[item.left_movie_id] for item in evaluation]
    right_rows = [row_map[item.right_movie_id] for item in evaluation]
    chose_left = [item.chosen_movie_id == item.left_movie_id for item in evaluation]

    pairwise_metrics = evaluate_on_pairwise_choices(
        bundle.matrix,
        left_rows,
        right_rows,
        chose_left,
        pairwise_fit.weights,
    )
    pairwise_metrics.update(
        {
            "converged": pairwise_fit.converged,
            "training_choices": len(pairwise),
        }
    )
    ranking_metrics = evaluate_on_pairwise_choices(
        bundle.matrix,
        left_rows,
        right_rows,
        chose_left,
        ranking_fit.weights,
    )
    ranking_metrics.update(
        {
            "converged": ranking_fit.converged,
            "training_rankings": len(rankings),
            "implied_pairwise_relations": sum(
                len(ranking) * (len(ranking) - 1) // 2 for ranking in rankings
            ),
        }
    )
    session.method_metrics = {
        "pairwise": pairwise_metrics,
        "ranking": ranking_metrics,
    }
    if session.completed_at is None:
        session.completed_at = timezone.now()
    session.save(update_fields=("method_metrics", "completed_at"))
    return session.method_metrics
