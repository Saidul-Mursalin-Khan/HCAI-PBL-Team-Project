from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


@dataclass(frozen=True)
class PreferenceFit:
    weights: np.ndarray
    converged: bool
    objective: float
    iterations: int


def fit_bradley_terry(
    feature_matrix, chosen_rows, rejected_rows, l2_penalty=1.0, max_iterations=200
):
    dimension = feature_matrix.shape[1]
    if not chosen_rows:
        return PreferenceFit(np.zeros(dimension), True, 0.0, 0)

    differences = feature_matrix[chosen_rows] - feature_matrix[rejected_rows]

    def objective(weights):
        margins = np.asarray(differences @ weights).ravel()
        loss = np.logaddexp(0.0, -margins).sum()
        loss += 0.5 * l2_penalty * np.dot(weights, weights)
        gradient = np.asarray(differences.T @ (-expit(-margins))).ravel()
        gradient += l2_penalty * weights
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(dimension),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": 1e-9},
    )
    return PreferenceFit(
        weights=result.x,
        converged=bool(result.success),
        objective=float(result.fun),
        iterations=int(result.nit),
    )


def fit_plackett_luce(
    feature_matrix, rankings, l2_penalty=1.0, max_iterations=200
):
    dimension = feature_matrix.shape[1]
    if not rankings:
        return PreferenceFit(np.zeros(dimension), True, 0.0, 0)

    ranking_matrices = [feature_matrix[ranking] for ranking in rankings]

    def objective(weights):
        loss = 0.5 * l2_penalty * np.dot(weights, weights)
        gradient = l2_penalty * weights.copy()
        for ranked_movies in ranking_matrices:
            scores = np.asarray(ranked_movies @ weights).ravel()
            for position in range(len(scores) - 1):
                remaining_scores = scores[position:]
                normalizer = logsumexp(remaining_scores)
                loss += normalizer - scores[position]
                probabilities = np.exp(remaining_scores - normalizer)
                expected = np.asarray(
                    ranked_movies[position:].T @ probabilities
                ).ravel()
                observed = ranked_movies[position].toarray().ravel()
                gradient += expected - observed
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(dimension),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "ftol": 1e-9},
    )
    return PreferenceFit(
        weights=result.x,
        converged=bool(result.success),
        objective=float(result.fun),
        iterations=int(result.nit),
    )


def pairwise_probabilities(feature_matrix, left_rows, right_rows, weights):
    differences = feature_matrix[left_rows] - feature_matrix[right_rows]
    return expit(np.asarray(differences @ weights).ravel())


def evaluate_on_pairwise_choices(
    feature_matrix, left_rows, right_rows, chose_left, weights
):
    if not left_rows:
        return {
            "accuracy": None,
            "log_loss": None,
            "brier_score": None,
            "trials": 0,
        }
    probabilities = np.clip(
        pairwise_probabilities(feature_matrix, left_rows, right_rows, weights),
        1e-8,
        1.0 - 1e-8,
    )
    truth = np.asarray(chose_left, dtype=bool)
    predictions = probabilities >= 0.5
    log_loss = -np.mean(
        np.where(truth, np.log(probabilities), np.log(1.0 - probabilities))
    )
    brier_score = np.mean((probabilities - truth.astype(float)) ** 2)
    return {
        "accuracy": round(float(np.mean(predictions == truth)), 4),
        "log_loss": round(float(log_loss), 4),
        "brier_score": round(float(brier_score), 4),
        "trials": len(left_rows),
    }
