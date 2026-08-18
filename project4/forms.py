from django import forms

from .models import BlockFeedback, FinalFeedback


LIKERT_CHOICES = [(value, str(value)) for value in range(1, 8)]


class ConsentForm(forms.Form):
    age_confirmation = forms.BooleanField(
        label="I confirm that I am at least 18 years old."
    )
    voluntary_consent = forms.BooleanField(
        label="I voluntarily agree to take part in this study prototype."
    )
    anonymous_data = forms.BooleanField(
        label=(
            "I understand that pseudonymous task responses and timings are stored "
            "locally for research analysis."
        )
    )


class BlockFeedbackForm(forms.ModelForm):
    mental_demand = forms.TypedChoiceField(
        label="How mentally demanding was this method?",
        choices=LIKERT_CHOICES,
        coerce=int,
        widget=forms.RadioSelect,
        help_text="1 = not demanding, 7 = very demanding",
    )
    confidence = forms.TypedChoiceField(
        label="How confident are you that your answers reflected your preferences?",
        choices=LIKERT_CHOICES,
        coerce=int,
        widget=forms.RadioSelect,
        help_text="1 = not confident, 7 = very confident",
    )
    ease_of_use = forms.TypedChoiceField(
        label="How easy was this method to use?",
        choices=LIKERT_CHOICES,
        coerce=int,
        widget=forms.RadioSelect,
        help_text="1 = very difficult, 7 = very easy",
    )

    class Meta:
        model = BlockFeedback
        fields = ("mental_demand", "confidence", "ease_of_use")


class FinalFeedbackForm(forms.ModelForm):
    class Meta:
        model = FinalFeedback
        fields = ("preferred_method", "movie_frequency", "comments")
        widgets = {
            "preferred_method": forms.RadioSelect,
            "movie_frequency": forms.RadioSelect,
            "comments": forms.Textarea(
                attrs={
                    "rows": 4,
                    "placeholder": "Optional: tell us what shaped your preference.",
                }
            ),
        }
