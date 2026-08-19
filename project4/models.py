import uuid

from django.db import models


class StudySession(models.Model):
    class ConditionOrder(models.TextChoices):
        PAIRWISE_FIRST = "pairwise_first", "Pairwise first"
        RANKING_FIRST = "ranking_first", "Ranking first"

    class Stage(models.TextChoices):
        PAIRWISE = "pairwise", "Pairwise elicitation"
        PAIRWISE_FEEDBACK = "pairwise_feedback", "Pairwise feedback"
        RANKING = "ranking", "Ranking elicitation"
        RANKING_FEEDBACK = "ranking_feedback", "Ranking feedback"
        EVALUATION = "evaluation", "Held-out evaluation"
        FINAL_FEEDBACK = "final_feedback", "Final feedback"
        COMPLETE = "complete", "Complete"

    public_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    condition_order = models.CharField(max_length=24, choices=ConditionOrder.choices)
    stage = models.CharField(max_length=24, choices=Stage.choices)
    random_seed = models.PositiveBigIntegerField()
    movie_plan = models.JSONField(default=dict)
    pairwise_trial = models.PositiveSmallIntegerField(default=0)
    ranking_trial = models.PositiveSmallIntegerField(default=0)
    evaluation_trial = models.PositiveSmallIntegerField(default=0)
    method_metrics = models.JSONField(default=dict, blank=True)
    consented_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    @property
    def first_method(self):
        if self.condition_order == self.ConditionOrder.PAIRWISE_FIRST:
            return "pairwise"
        return "ranking"

    @property
    def second_method(self):
        return "ranking" if self.first_method == "pairwise" else "pairwise"

    def __str__(self):
        return f"Study {self.public_id}"


class PairwiseChoice(models.Model):
    class Kind(models.TextChoices):
        ELICITATION = "elicitation", "Elicitation"
        EVALUATION = "evaluation", "Held-out evaluation"

    study_session = models.ForeignKey(
        StudySession, on_delete=models.CASCADE, related_name="pairwise_choices"
    )
    kind = models.CharField(max_length=16, choices=Kind.choices)
    trial_index = models.PositiveSmallIntegerField()
    left_movie_id = models.PositiveIntegerField()
    right_movie_id = models.PositiveIntegerField()
    chosen_movie_id = models.PositiveIntegerField()
    response_time_ms = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("study_session", "kind", "trial_index"),
                name="project4_unique_pairwise_trial",
            )
        ]
        ordering = ("created_at",)


class RankingResponse(models.Model):
    study_session = models.ForeignKey(
        StudySession, on_delete=models.CASCADE, related_name="rankings"
    )
    trial_index = models.PositiveSmallIntegerField()
    presented_movie_ids = models.JSONField()
    ordered_movie_ids = models.JSONField()
    response_time_ms = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("study_session", "trial_index"),
                name="project4_unique_ranking_trial",
            )
        ]
        ordering = ("created_at",)


class BlockFeedback(models.Model):
    class Method(models.TextChoices):
        PAIRWISE = "pairwise", "Pairwise choice"
        RANKING = "ranking", "Ten-movie ranking"

    study_session = models.ForeignKey(
        StudySession, on_delete=models.CASCADE, related_name="block_feedback"
    )
    method = models.CharField(max_length=16, choices=Method.choices)
    mental_demand = models.PositiveSmallIntegerField()
    confidence = models.PositiveSmallIntegerField()
    ease_of_use = models.PositiveSmallIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=("study_session", "method"),
                name="project4_unique_block_feedback",
            )
        ]


class FinalFeedback(models.Model):
    class PreferredMethod(models.TextChoices):
        PAIRWISE = "pairwise", "Pairwise choice"
        RANKING = "ranking", "Ten-movie ranking"
        NO_PREFERENCE = "no_preference", "No preference"

    class MovieFrequency(models.TextChoices):
        RARELY = "rarely", "Less than once a month"
        MONTHLY = "monthly", "One to three times a month"
        WEEKLY = "weekly", "One to three times a week"
        OFTEN = "often", "Four or more times a week"

    study_session = models.OneToOneField(
        StudySession, on_delete=models.CASCADE, related_name="final_feedback"
    )
    preferred_method = models.CharField(max_length=16, choices=PreferredMethod.choices)
    movie_frequency = models.CharField(max_length=12, choices=MovieFrequency.choices)
    comments = models.TextField(blank=True, max_length=1000)
    created_at = models.DateTimeField(auto_now_add=True)
