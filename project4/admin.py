from django.contrib import admin

from .models import (
    BlockFeedback,
    FinalFeedback,
    PairwiseChoice,
    RankingResponse,
    StudySession,
)


@admin.register(StudySession)
class StudySessionAdmin(admin.ModelAdmin):
    list_display = (
        "public_id",
        "condition_order",
        "stage",
        "consented_at",
        "completed_at",
    )
    list_filter = ("condition_order", "stage")
    readonly_fields = ("public_id", "created_at")


admin.site.register(PairwiseChoice)
admin.site.register(RankingResponse)
admin.site.register(BlockFeedback)
admin.site.register(FinalFeedback)
