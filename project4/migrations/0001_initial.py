import django.db.models.deletion
import uuid

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="StudySession",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("public_id", models.UUIDField(default=uuid.uuid4, editable=False, unique=True)),
                ("condition_order", models.CharField(choices=[("pairwise_first", "Pairwise first"), ("ranking_first", "Ranking first")], max_length=24)),
                ("stage", models.CharField(choices=[("pairwise", "Pairwise elicitation"), ("pairwise_feedback", "Pairwise feedback"), ("ranking", "Ranking elicitation"), ("ranking_feedback", "Ranking feedback"), ("evaluation", "Held-out evaluation"), ("final_feedback", "Final feedback"), ("complete", "Complete")], max_length=24)),
                ("random_seed", models.PositiveBigIntegerField()),
                ("movie_plan", models.JSONField(default=dict)),
                ("pairwise_trial", models.PositiveSmallIntegerField(default=0)),
                ("ranking_trial", models.PositiveSmallIntegerField(default=0)),
                ("evaluation_trial", models.PositiveSmallIntegerField(default=0)),
                ("method_metrics", models.JSONField(blank=True, default=dict)),
                ("consented_at", models.DateTimeField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="PairwiseChoice",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("kind", models.CharField(choices=[("elicitation", "Elicitation"), ("evaluation", "Held-out evaluation")], max_length=16)),
                ("trial_index", models.PositiveSmallIntegerField()),
                ("left_movie_id", models.PositiveIntegerField()),
                ("right_movie_id", models.PositiveIntegerField()),
                ("chosen_movie_id", models.PositiveIntegerField()),
                ("response_time_ms", models.PositiveIntegerField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("study_session", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="pairwise_choices", to="project4.studysession")),
            ],
            options={"ordering": ("created_at",)},
        ),
        migrations.CreateModel(
            name="RankingResponse",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("trial_index", models.PositiveSmallIntegerField()),
                ("presented_movie_ids", models.JSONField()),
                ("ordered_movie_ids", models.JSONField()),
                ("response_time_ms", models.PositiveIntegerField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("study_session", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="rankings", to="project4.studysession")),
            ],
            options={"ordering": ("created_at",)},
        ),
        migrations.CreateModel(
            name="BlockFeedback",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("method", models.CharField(choices=[("pairwise", "Pairwise choice"), ("ranking", "Ten-movie ranking")], max_length=16)),
                ("mental_demand", models.PositiveSmallIntegerField()),
                ("confidence", models.PositiveSmallIntegerField()),
                ("ease_of_use", models.PositiveSmallIntegerField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("study_session", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="block_feedback", to="project4.studysession")),
            ],
        ),
        migrations.CreateModel(
            name="FinalFeedback",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("preferred_method", models.CharField(choices=[("pairwise", "Pairwise choice"), ("ranking", "Ten-movie ranking"), ("no_preference", "No preference")], max_length=16)),
                ("movie_frequency", models.CharField(choices=[("rarely", "Less than once a month"), ("monthly", "One to three times a month"), ("weekly", "One to three times a week"), ("often", "Four or more times a week")], max_length=12)),
                ("comments", models.TextField(blank=True, max_length=1000)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("study_session", models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="final_feedback", to="project4.studysession")),
            ],
        ),
        migrations.AddConstraint(
            model_name="pairwisechoice",
            constraint=models.UniqueConstraint(fields=("study_session", "kind", "trial_index"), name="project4_unique_pairwise_trial"),
        ),
        migrations.AddConstraint(
            model_name="rankingresponse",
            constraint=models.UniqueConstraint(fields=("study_session", "trial_index"), name="project4_unique_ranking_trial"),
        ),
        migrations.AddConstraint(
            model_name="blockfeedback",
            constraint=models.UniqueConstraint(fields=("study_session", "method"), name="project4_unique_block_feedback"),
        ),
    ]
