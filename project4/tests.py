import json

import numpy as np
from django.test import Client, SimpleTestCase, TestCase
from django.urls import reverse
from scipy.sparse import csr_matrix

from .models import PairwiseChoice, StudySession
from .services.movies import build_movie_plan, get_feature_bundle
from .services.preference import (
    evaluate_on_pairwise_choices,
    fit_bradley_terry,
    fit_plackett_luce,
    pairwise_probabilities,
)


class PreferenceModelTests(SimpleTestCase):
    def setUp(self):
        self.features = csr_matrix(
            np.asarray(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        )

    def test_pairwise_probability_is_symmetric(self):
        weights = np.asarray([1.2, -0.4])
        forward = pairwise_probabilities(self.features, [1], [2], weights)[0]
        reverse = pairwise_probabilities(self.features, [2], [1], weights)[0]
        self.assertAlmostEqual(forward + reverse, 1.0, places=14)

    def test_equal_features_have_half_probability(self):
        weights = np.asarray([2.0, -3.0])
        probability = pairwise_probabilities(self.features, [0], [0], weights)[0]
        self.assertEqual(probability, 0.5)

    def test_bradley_terry_learns_repeated_preference(self):
        fit = fit_bradley_terry(
            self.features,
            chosen_rows=[1, 1, 1, 1],
            rejected_rows=[0, 0, 0, 0],
        )
        self.assertTrue(fit.converged)
        self.assertGreater(fit.weights[0], 0.0)

    def test_two_item_plackett_luce_matches_bradley_terry(self):
        bt = fit_bradley_terry(self.features, [1], [2])
        pl = fit_plackett_luce(self.features, [[1, 2]])
        np.testing.assert_allclose(bt.weights, pl.weights, atol=1e-8)
        self.assertAlmostEqual(bt.objective, pl.objective, places=8)

    def test_plackett_luce_recovers_order(self):
        features = csr_matrix(np.asarray([[0.0], [1.0], [2.0]]))
        fit = fit_plackett_luce(features, [[2, 1, 0], [2, 1, 0]])
        utilities = np.asarray(features @ fit.weights).ravel()
        self.assertGreater(utilities[2], utilities[1])
        self.assertGreater(utilities[1], utilities[0])

    def test_extreme_features_produce_finite_fit_and_metrics(self):
        features = csr_matrix(np.asarray([[1e6], [0.0], [-1e6]]))
        fit = fit_plackett_luce(features, [[0, 1, 2]])
        self.assertTrue(np.isfinite(fit.weights).all())
        self.assertTrue(np.isfinite(fit.objective))
        metrics = evaluate_on_pairwise_choices(
            features, [0, 1], [1, 2], [True, True], fit.weights
        )
        self.assertTrue(np.isfinite(metrics["log_loss"]))
        self.assertTrue(np.isfinite(metrics["brier_score"]))


class MovieDataTests(SimpleTestCase):
    def test_full_dataset_builds_finite_compact_features(self):
        bundle = get_feature_bundle()
        self.assertGreater(len(bundle.movies), 4000)
        self.assertEqual(bundle.matrix.shape[0], len(bundle.movies))
        self.assertLess(bundle.matrix.shape[1], 200)
        self.assertTrue(np.isfinite(bundle.matrix.data).all())

    def test_sampling_is_deterministic_and_disjoint(self):
        first = build_movie_plan(12345)
        second = build_movie_plan(12345)
        self.assertEqual(first, second)
        flattened = [movie for group in first.values() for trial in group for movie in trial]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(len(ranking) == 10 for ranking in first["rankings"]))


class StudyViewTests(TestCase):
    consent_payload = {
        "age_confirmation": "on",
        "voluntary_consent": "on",
        "anonymous_data": "on",
    }

    def _start(self):
        response = self.client.post(reverse("project4:consent"), self.consent_payload)
        self.assertRedirects(
            response, reverse("project4:study"), fetch_redirect_response=False
        )
        public_id = self.client.session["project4_study_public_id"]
        return StudySession.objects.get(public_id=public_id)

    def _submit_pairwise_block(self, session):
        while session.stage == StudySession.Stage.PAIRWISE:
            pair = session.movie_plan["pairwise"][session.pairwise_trial]
            response = self.client.post(
                reverse("project4:pairwise"),
                {"chosen_movie_id": pair[0], "response_time_ms": 1200},
            )
            self.assertRedirects(
                response, reverse("project4:study"), fetch_redirect_response=False
            )
            session.refresh_from_db()
        response = self.client.post(
            reverse("project4:block_feedback", kwargs={"method": "pairwise"}),
            {"mental_demand": 3, "confidence": 5, "ease_of_use": 6},
        )
        self.assertRedirects(
            response, reverse("project4:study"), fetch_redirect_response=False
        )
        session.refresh_from_db()

    def _submit_ranking_block(self, session):
        while session.stage == StudySession.Stage.RANKING:
            ranking = session.movie_plan["rankings"][session.ranking_trial]
            response = self.client.post(
                reverse("project4:ranking"),
                {"ranking": json.dumps(ranking), "response_time_ms": 8500},
            )
            self.assertRedirects(
                response, reverse("project4:study"), fetch_redirect_response=False
            )
            session.refresh_from_db()
        response = self.client.post(
            reverse("project4:block_feedback", kwargs={"method": "ranking"}),
            {"mental_demand": 5, "confidence": 6, "ease_of_use": 4},
        )
        self.assertRedirects(
            response, reverse("project4:study"), fetch_redirect_response=False
        )
        session.refresh_from_db()

    def test_landing_and_report(self):
        response = self.client.get(reverse("project4:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Start study")
        self.assertContains(response, "Download report")
        report = self.client.get(reverse("project4:download_report"))
        self.assertEqual(report.status_code, 200)
        self.assertEqual(report["Content-Type"], "application/pdf")
        self.assertTrue(b"".join(report.streaming_content).startswith(b"%PDF"))

    def test_consent_post_requires_csrf(self):
        protected_client = Client(enforce_csrf_checks=True)
        response = protected_client.post(reverse("project4:consent"), self.consent_payload)
        self.assertEqual(response.status_code, 403)

    def test_consent_post_accepts_valid_csrf_token(self):
        protected_client = Client(enforce_csrf_checks=True)
        response = protected_client.get(reverse("project4:consent"))
        self.assertEqual(response.status_code, 200)
        token = protected_client.cookies["csrftoken"].value
        response = protected_client.post(
            reverse("project4:consent"),
            {**self.consent_payload, "csrfmiddlewaretoken": token},
        )
        self.assertRedirects(
            response,
            reverse("project4:study"),
            fetch_redirect_response=False,
        )

    def test_ranking_rejects_tampered_permutation(self):
        session = self._start()
        session.stage = StudySession.Stage.RANKING
        session.save(update_fields=("stage",))
        ranking = session.movie_plan["rankings"][0]
        response = self.client.post(
            reverse("project4:ranking"),
            {"ranking": json.dumps(ranking[:-1] + [999999]), "response_time_ms": 10},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(session.rankings.count(), 0)

    def test_complete_counterbalanced_study_flow(self):
        session = self._start()
        self.assertEqual(session.first_method, "pairwise")
        self._submit_pairwise_block(session)
        self._submit_ranking_block(session)
        self.assertEqual(session.stage, StudySession.Stage.EVALUATION)

        while session.stage == StudySession.Stage.EVALUATION:
            pair = session.movie_plan["evaluation"][session.evaluation_trial]
            response = self.client.post(
                reverse("project4:evaluation"),
                {"chosen_movie_id": pair[0], "response_time_ms": 900},
            )
            self.assertRedirects(
                response, reverse("project4:study"), fetch_redirect_response=False
            )
            session.refresh_from_db()

        response = self.client.post(
            reverse("project4:final_feedback"),
            {
                "preferred_method": "ranking",
                "movie_frequency": "weekly",
                "comments": "Ranking was more expressive.",
            },
        )
        self.assertRedirects(response, reverse("project4:complete"))

        response = self.client.get(reverse("project4:complete"))
        self.assertEqual(response.status_code, 200)
        session.refresh_from_db()
        self.assertEqual(session.stage, StudySession.Stage.COMPLETE)
        self.assertEqual(
            session.pairwise_choices.filter(kind=PairwiseChoice.Kind.EVALUATION).count(),
            20,
        )
        self.assertIn("pairwise", session.method_metrics)
        self.assertIn("ranking", session.method_metrics)
        self.assertIn("brier_score", session.method_metrics["pairwise"])
        self.assertIn("brier_score", session.method_metrics["ranking"])

    def test_second_session_uses_ranking_first_and_reaches_evaluation(self):
        first = self._start()
        self.assertEqual(first.first_method, "pairwise")

        second_client = Client()
        response = second_client.post(
            reverse("project4:consent"), self.consent_payload
        )
        self.assertRedirects(
            response,
            reverse("project4:study"),
            fetch_redirect_response=False,
        )
        public_id = second_client.session["project4_study_public_id"]
        second = StudySession.objects.get(public_id=public_id)
        self.assertEqual(second.first_method, "ranking")

        original_client = self.client
        self.client = second_client
        try:
            self._submit_ranking_block(second)
            self._submit_pairwise_block(second)
        finally:
            self.client = original_client
        self.assertEqual(second.stage, StudySession.Stage.EVALUATION)
