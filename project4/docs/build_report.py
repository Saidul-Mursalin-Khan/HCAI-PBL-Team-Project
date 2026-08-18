from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT = Path(__file__).resolve().parent / "project4_report.pdf"
PAGE_WIDTH, PAGE_HEIGHT = A4

INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#5F6C80")
BLUE = colors.HexColor("#2457D6")
BLUE_DARK = colors.HexColor("#173B9C")
PALE_BLUE = colors.HexColor("#EAF0FF")
CORAL = colors.HexColor("#FF7A59")
MINT = colors.HexColor("#DDF7EA")
LINE = colors.HexColor("#DCE2EA")
PAPER = colors.white


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, pagesize=A4, **kwargs)
        frame = Frame(
            20 * mm,
            18 * mm,
            PAGE_WIDTH - 40 * mm,
            PAGE_HEIGHT - 38 * mm,
            leftPadding=0,
            rightPadding=0,
            topPadding=4 * mm,
            bottomPadding=4 * mm,
        )
        self.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=self._page))

    @staticmethod
    def _page(canvas, doc):
        canvas.saveState()
        if doc.page > 1:
            canvas.setStrokeColor(LINE)
            canvas.line(20 * mm, PAGE_HEIGHT - 15 * mm, PAGE_WIDTH - 20 * mm, PAGE_HEIGHT - 15 * mm)
            canvas.setFont("Helvetica-Bold", 8)
            canvas.setFillColor(BLUE_DARK)
            canvas.drawString(20 * mm, PAGE_HEIGHT - 11 * mm, "HCAI PROJECT 4")
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(MUTED)
            canvas.drawRightString(PAGE_WIDTH - 20 * mm, PAGE_HEIGHT - 11 * mm, "Preference elicitation")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(20 * mm, 11 * mm, "Methods and planned user study - no study was conducted")
        canvas.drawRightString(PAGE_WIDTH - 20 * mm, 11 * mm, f"{doc.page:02d}")
        canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        "CoverKicker",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#FFD3C8"),
        spaceAfter=9 * mm,
        uppercase=True,
    )
)
styles.add(
    ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=34,
        leading=37,
        textColor=PAPER,
        alignment=TA_LEFT,
        spaceAfter=7 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "CoverLead",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=13,
        leading=19,
        textColor=colors.HexColor("#DFE8FF"),
        spaceAfter=8 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "Kicker",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=CORAL,
        spaceBefore=2 * mm,
        spaceAfter=2 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "H1x",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=INK,
        spaceBefore=1 * mm,
        spaceAfter=5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "H2x",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=BLUE_DARK,
        spaceBefore=5 * mm,
        spaceAfter=2.5 * mm,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        "Bodyx",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14.2,
        textColor=INK,
        spaceAfter=3.2 * mm,
        alignment=TA_LEFT,
    )
)
styles.add(
    ParagraphStyle(
        "Smallx",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11.5,
        textColor=MUTED,
        spaceAfter=2 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "Bulletx",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.8,
        leftIndent=5 * mm,
        firstLineIndent=-3 * mm,
        bulletIndent=1 * mm,
        textColor=INK,
        spaceAfter=1.8 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "Formulax",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.4,
        leading=12,
        textColor=BLUE_DARK,
        backColor=PALE_BLUE,
        borderColor=colors.HexColor("#C8D6FA"),
        borderWidth=0.6,
        borderPadding=8,
        borderRadius=5,
        spaceBefore=2 * mm,
        spaceAfter=4 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "CalloutTitle",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=BLUE_DARK,
        spaceAfter=1.5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        "Reference",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.7,
        leading=11,
        textColor=INK,
        leftIndent=5 * mm,
        firstLineIndent=-5 * mm,
        spaceAfter=1.7 * mm,
    )
)


def p(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def heading(number, title):
    return [p(f"SECTION {number}", "Kicker"), p(title, "H1x")]


def h2(title):
    return p(title, "H2x")


def bullets(items):
    return [p(f"<bullet>&bull;</bullet>{item}", "Bulletx") for item in items]


def formula(text):
    return p(text.replace(" ", "&nbsp;"), "Formulax")


def callout(title, text, background=PALE_BLUE):
    box = Table(
        [[[p(title, "CalloutTitle"), p(text, "Smallx")]]],
        colWidths=[PAGE_WIDTH - 44 * mm],
    )
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.7, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return box


def styled_table(rows, widths, header=True):
    converted = []
    for row_index, row in enumerate(rows):
        style = "CalloutTitle" if header and row_index == 0 else "Smallx"
        converted.append([p(str(cell), style) for cell in row])
    table = LongTable(converted, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BACKGROUND", (0, 0), (-1, 0), PALE_BLUE),
    ]
    for row_index in range(1, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F8FAFD")))
    table.setStyle(TableStyle(commands))
    return table


story = []

# Cover
cover = Table(
    [[[
        p("HUMAN-CENTRIC ARTIFICIAL INTELLIGENCE", "CoverKicker"),
        p("Project 4:<br/>Preference Elicitation", "CoverTitle"),
        p(
            "A reproducible movie-preference study comparing repeated pairwise choices "
            "with ten-item rankings. This report documents Tasks 1-3 and the implemented "
            "participant interface.",
            "CoverLead",
        ),
        HRFlowable(width="100%", thickness=0.7, color=colors.HexColor("#6F91EA")),
        Spacer(1, 7 * mm),
        p("METHODS AND PLANNED USER STUDY", "CoverKicker"),
        p("Version 1.2  |  18 August 2026  |  Study not conducted", "CoverLead"),
    ]]],
    colWidths=[PAGE_WIDTH - 40 * mm],
    rowHeights=[PAGE_HEIGHT - 52 * mm],
)
cover.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), BLUE_DARK),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 20 * mm),
    ("RIGHTPADDING", (0, 0), (-1, -1), 20 * mm),
    ("TOPPADDING", (0, 0), (-1, -1), 20 * mm),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 20 * mm),
]))
story.extend([cover, PageBreak()])

# Executive summary
story.extend(heading("01", "Executive summary"))
story.append(p(
    "The project asks how quickly a new-user movie recommender can learn a latent preference "
    "vector from limited interaction. It compares a low-complexity pairwise interface with a "
    "more expressive ten-movie ranking interface. Both methods use the same movie metadata, "
    "feature map, utility function, regularization and held-out evaluation set."
))
story.append(callout(
    "Deliverable status",
    "The Django prototype implements the landing page, downloadable report, consent, both "
    "elicitation interfaces, short workload questionnaires, held-out validation, server-side "
    "logging and model fitting. No participants were recruited and no empirical results are claimed.",
    MINT,
))
story.append(h2("Research question"))
story.append(p(
    "After equal exposure to 20 movies, does a Plackett-Luce model trained on two complete "
    "ten-item rankings predict new pairwise movie choices more accurately than a Bradley-Terry "
    "model trained on ten pairwise choices, and what additional cognitive burden does ranking impose?"
))
story.append(h2("Design at a glance"))
story.append(styled_table(
    [
        ["Element", "Pairwise condition", "Ranking condition"],
        ["Participant action", "Choose one of two movies", "Order ten movies from best to worst"],
        ["Movie exposure", "10 pairs = 20 unique movies", "2 lists = 20 unique movies"],
        ["Preference model", "Bradley-Terry", "Plackett-Luce"],
        ["Primary comparison", "Mean log loss on the same 20 unseen validation pairs", "Same validation responses"],
        ["Experience measures", "Time, mental demand, confidence, ease", "Time, mental demand, confidence, ease"],
    ],
    [37 * mm, 61 * mm, 61 * mm],
))
story.append(h2("Human-centric commitments"))
story.extend(bullets([
    "No IMDb score, vote count or social-media popularity is shown or learned as a taste signal.",
    "No name, email, free-text demographic detail, IP address or third-party analytics is required.",
    "Ranking supports drag-and-drop and keyboard move controls; visible focus and high contrast are retained.",
    "The study order is counterbalanced and validation responses never update either model.",
]))
story.append(PageBreak())

# Dataset and features
story.extend(heading("02", "Movie feature representation - Task 1"))
story.append(h2("Source and preprocessing"))
story.append(p(
    "The implementation uses the IMDb 5000 Movie Dataset file linked directly from the assignment "
    "brief. The bundled file contains 5,043 rows and 28 metadata fields but no individual user "
    "ratings. The documented filters retain 4,789 distinct, usable films. "
    "Titles are cleaned of trailing non-breaking spaces; rows without a title, genre, director, "
    "plausible year or plausible duration are removed; duplicate title-year pairs keep the most "
    "widely voted record. Sampling is seeded and without replacement inside a participant session."
))
story.append(h2("Feature vector x"))
story.append(styled_table(
    [
        ["Feature block", "Encoding", "Why it belongs"],
        ["Genres", "Multi-hot", "Directly represents stable thematic preferences."],
        ["Release decade", "One-hot", "Captures preference for eras without assuming a strictly linear year effect."],
        ["Duration", "Dataset z-score", "Represents pacing and time commitment on a comparable numeric scale."],
        ["Language", "Top 8 + OTHER", "Captures viewing-language preference while controlling dimensionality."],
        ["Country", "Top 10 + OTHER", "Adds broad production context without sparse country indicators."],
        ["Content rating", "Frequent levels + OTHER", "Represents content constraints relevant to watch choice."],
        ["Plot keywords", "TF-IDF, then 16-component SVD", "Adds themes beyond genre while keeping the low-data model compact."],
    ],
    [35 * mm, 43 * mm, 81 * mm],
))
story.append(h2("Deliberate exclusions"))
story.extend(bullets([
    "IMDb score, number of votes, gross, budget and Facebook counters are popularity or exposure proxies, not personal taste.",
    "Raw director and actor one-hot features create thousands of sparse dimensions that cannot be estimated reliably from 10 choices or 2 rankings; the names remain visible on cards as decision context.",
    "Movie title is an identifier, not a transferable content feature. It is never encoded into x.",
]))
story.append(callout(
    "Shared representation",
    "Both conditions use the exact same cached float64 feature matrix. Feature extraction is fit "
    "once on metadata only; participant choices never affect preprocessing. The utility has no "
    "intercept because a constant cancels in all utility differences.",
))
story.append(PageBreak())

# BT
story.extend(heading("03", "Pairwise preference model"))
story.append(p(
    "For movie i with feature vector x_i and participant preference vector w, utility is linear. "
    "The Bradley-Terry model turns the utility difference between two films into a choice probability."
))
story.append(formula("u_i = w^T x_i"))
story.append(formula("P(i preferred to j | w) = sigmoid(w^T (x_i - x_j))"))
story.append(p(
    "For each observed winner i and loser j, d = x_i - x_j. The implementation minimizes the "
    "L2-regularized negative log likelihood below. Regularization is essential because the number "
    "of observed choices is much smaller than the feature dimension."
))
story.append(formula("J_BT(w) = sum_t log(1 + exp(-d_t^T w)) + (lambda / 2) ||w||_2^2"))
story.append(formula("grad J_BT(w) = -sum_t d_t sigmoid(-d_t^T w) + lambda w"))
story.append(h2("Numerical implementation"))
story.extend(bullets([
    "SciPy L-BFGS-B with analytic gradient, w initialized at zero and lambda fixed at 1.0 before data collection.",
    "numpy.logaddexp and scipy.special.expit avoid overflow for large positive or negative utilities.",
    "The participant never sees model probabilities, which prevents prediction feedback from changing later choices.",
    "Unit tests require probability symmetry, 0.5 for equal features, finite optimization and recovery of a known synthetic direction.",
]))
story.append(h2("Interpretation"))
story.append(p(
    "A positive weight indicates that the participant tends to select films containing that feature, "
    "conditional on all other encoded attributes. Magnitude reflects model scale and regularization; "
    "it should not be interpreted as a causal effect. Correlated genres and keyword topics may share weight."
))
story.append(callout(
    "Cold-start scope",
    "The fitted w belongs to one pseudonymous participant and is estimated only from that participant's "
    "study interactions. The model does not use collaborative ratings or transfer preferences between users.",
    MINT,
))
story.append(PageBreak())

# PL
story.extend(heading("04", "Ranking extension - Task 2"))
story.append(p(
    "A ten-film ordering is not treated as 45 independent binary observations. Those implied pairs "
    "come from one correlated action. Instead, the implementation uses the Plackett-Luce model, a "
    "stagewise extension of Bradley-Terry. At each stage the participant selects the best remaining item."
))
story.append(formula("P(i_1 > ... > i_m | w) = product_(r=1)^(m-1) exp(u_(i_r)) / sum_(s=r)^m exp(u_(i_s))"))
story.append(p(
    "The first factor is the probability that i_1 is selected from all m films; the second selects "
    "i_2 from the remaining m-1; the process continues until one film remains. When m = 2, the "
    "single factor reduces exactly to the Bradley-Terry probability. This nesting gives a direct "
    "theoretical justification for comparing the two interfaces."
))
story.append(formula("J_PL(w) = sum_q sum_(r=1)^(m_q-1) [logsumexp_(s=r..m_q)(u_(i_s)) - u_(i_r)] + (lambda / 2)||w||_2^2"))
story.append(h2("Information and dependence"))
story.append(p(
    "One ten-item ranking contains 45 ordered item pairs, but the Plackett-Luce likelihood contains "
    "nine sequential choice stages. The report records both counts and never claims 45 statistically "
    "independent responses. Ranking can encode more preference structure per movie exposure, while "
    "requiring more working memory and interaction effort. That trade-off motivates the user study."
))
story.append(h2("Validation tests"))
story.extend(bullets([
    "A two-item Plackett-Luce fit must match Bradley-Terry loss and parameters within numerical tolerance.",
    "Rankings must contain every presented movie exactly once; malformed JSON, duplicates and foreign IDs are rejected server-side.",
    "The objective uses logsumexp and must remain finite on extreme synthetic utilities.",
    "Repeated synthetic orderings must produce utilities in the expected order.",
]))
story.append(PageBreak())

# Study hypotheses/design
story.extend(heading("05", "User study design - Task 3"))
story.append(h2("Hypotheses"))
story.append(styled_table(
    [
        ["ID", "Prespecified statement", "Outcome"],
        ["H1 primary", "After equal exposure to 20 movies, ranking produces lower held-out pairwise log loss than pairwise elicitation.", "Participant-level mean negative log loss on 20 common validation pairs."],
        ["H2", "Ranking tasks take longer and are more mentally demanding.", "Block time and 1-7 mental-demand response."],
        ["H3", "Ranking produces higher confidence that responses express the participant's taste.", "1-7 confidence response."],
        ["Exploratory", "Interface preference is associated with workload and predictive advantage.", "Preferred method and optional comment."],
    ],
    [23 * mm, 86 * mm, 50 * mm],
))
story.append(h2("Within-subject counterbalanced design"))
story.append(p(
    "Every participant completes both methods and therefore serves as their own control, which is "
    "important because movie preferences vary strongly between people. Server-side allocation "
    "alternates pairwise-first and ranking-first sequences to maintain AB/BA balance. In a real run, "
    "allocation should use concealed permuted blocks generated before recruitment."
))
story.append(styled_table(
    [
        ["Sequence", "Block 1", "Block 2", "Validation"],
        ["AB", "10 pairwise choices", "2 ten-film rankings", "20 unseen pairwise choices"],
        ["BA", "2 ten-film rankings", "10 pairwise choices", "20 unseen pairwise choices"],
    ],
    [30 * mm, 43 * mm, 43 * mm, 43 * mm],
))
story.append(p(
    "The two elicitation pools and the validation pool are disjoint within each participant. Each "
    "elicitation condition exposes exactly 20 unique movies. Films are sampled uniformly without "
    "replacement from cleaned records with a title, genre, director, plausible year and plausible "
    "duration, and at least 1,000 historical votes; "
    "vote count is used only as an eligibility filter to reduce unfamiliar or extremely obscure stimuli, "
    "and is neither displayed nor encoded. A sensitivity run should remove this filter."
))
story.append(PageBreak())

# Procedure/recruitment
story.extend(heading("06", "Participants, recruitment and procedure"))
story.append(h2("Recruitment and sample size"))
story.append(p(
    "The target population is adults who can read English and make movie choices on a laptop, desktop "
    "or large tablet. Recruitment would use university mailing lists and campus notices, with no course "
    "credit controlled by the research team and with compensation stated before consent. Participants "
    "must be at least 18, provide informed consent, and not have completed the study previously."
))
story.append(callout(
    "Planned sample",
    "For a paired standardized effect d_z = 0.40, two-sided alpha = 0.05 and power = 0.80, the planned "
    "minimum is 52 complete participants. Recruit 62 because ceiling(52 / 0.85) = 62, preserving the "
    "target after approximately 15 percent attrition or technical exclusion. First run a 10-12 person "
    "feasibility pilot; pilot data are not pooled with confirmation data.",
))
story.append(h2("Participant journey"))
procedure = [
    ["1", "Landing", "Read neutral purpose, expected duration and privacy summary; download the full report if desired."],
    ["2", "Consent", "Confirm age, voluntary participation and local storage of pseudonymous responses/timings."],
    ["3", "First block", "Complete the assigned elicitation method, followed by mental demand, confidence and ease ratings."],
    ["4", "Second block", "Complete the other method with a disjoint movie pool, followed by the same ratings."],
    ["5", "Validation", "Make 20 pairwise choices on unseen films; these choices update neither model."],
    ["6", "Final feedback", "Choose a preferred interface, report movie-watching frequency and optionally explain the choice."],
    ["7", "Debrief", "Receive a random study code and explanation of why model scores are hidden from participants."],
]
story.append(styled_table([["Step", "Stage", "Action"]] + procedure, [14 * mm, 35 * mm, 110 * mm]))
story.append(h2("Exclusion rules fixed before analysis"))
story.extend(bullets([
    "No valid consent or age confirmation; duplicate participation; incomplete elicitation block; or fewer than 16 of 20 validation choices.",
    "Corrupt allocation, missing movie IDs or impossible timestamps are technical exclusions.",
    "Very fast choices and repeated selection of one screen side are flagged for sensitivity analysis, not automatic exclusion.",
]))
story.append(PageBreak())

# Measures/analysis
story.extend(heading("07", "Measures and prespecified analysis"))
story.append(h2("Outcome construction"))
story.append(p(
    "After both blocks, two separate models are fit with the same feature matrix and lambda = 1.0. "
    "For every validation pair, each model outputs the probability assigned to the participant's "
    "chosen movie. Primary log loss is the negative mean log probability across the 20 trials. "
    "Lower values indicate better calibrated prediction."
))
story.append(styled_table(
    [
        ["Family", "Measure", "Role"],
        ["Predictive", "Validation log loss", "Primary"],
        ["Predictive", "Accuracy and Brier score", "Secondary"],
        ["Efficiency", "Total response time per block and per movie exposure", "Secondary"],
        ["Experience", "Mental demand, confidence, ease of use (1-7)", "Secondary"],
        ["Preference", "Preferred interface and optional explanation", "Exploratory"],
    ],
    [32 * mm, 82 * mm, 45 * mm],
))
story.append(h2("Primary analysis"))
story.append(p(
    "For each participant compute Delta = NLL_ranking - NLL_pairwise. Test the mean paired difference "
    "with a two-sided alpha of .05 and report the 95 percent confidence interval and paired effect size "
    "d_z. The directional hypothesis predicts Delta < 0, but a two-sided test avoids understating an "
    "unexpected reverse effect. A paired permutation test is the prespecified robustness analysis."
))
story.append(h2("Secondary analysis"))
story.extend(bullets([
    "Compare accuracy with a trial-level logistic mixed model containing method, period and participant random intercept.",
    "Compare response time on the log scale and Likert responses with paired ordinal or Wilcoxon analyses.",
    "Apply Holm correction across the three confirmatory secondary hypotheses.",
    "Report complete-case results plus a sensitivity analysis including participants with at least 80 percent validation completion.",
    "Do not tune features, lambda or optimizer settings on validation answers.",
]))
story.append(callout(
    "No results section",
    "This is a planned protocol and executable interface. Because the study has not been conducted, the "
    "report specifies estimands and analysis requirements without presenting outcome estimates.",
    MINT,
))
story.append(PageBreak())

# Logging/privacy/interface
story.extend(heading("08", "Implementation, logging and safeguards"))
story.append(h2("Server-side workflow"))
story.append(p(
    "Django stores a random UUID in the browser session and keeps the authoritative study stage in SQLite. "
    "A seeded movie plan is created once at consent. Every POST is CSRF-protected, IDs are checked against "
    "the current server-side stimulus set, database writes are atomic and unique constraints prevent duplicate trials."
))
story.append(styled_table(
    [
        ["Table", "Stored fields", "Purpose"],
        ["StudySession", "UUID, order, stage, seed, movie plan, timestamps", "Counterbalancing, resume and reproducibility"],
        ["PairwiseChoice", "kind, trial, left/right/chosen IDs, duration", "Elicitation and held-out validation"],
        ["RankingResponse", "presented IDs, permutation, duration", "Plackett-Luce fitting and integrity"],
        ["BlockFeedback", "method, demand, confidence, ease", "Human experience outcomes"],
        ["FinalFeedback", "preferred method, movie frequency, optional comment", "Interpretation and exploratory analysis"],
    ],
    [34 * mm, 77 * mm, 48 * mm],
))
story.append(h2("Privacy and ethics"))
story.extend(bullets([
    "The prototype collects no direct identifiers and does not intentionally log IP address, browser fingerprint or third-party analytics.",
    "Choice and timing data remain pseudonymous rather than truly anonymous; real deployment needs a named data controller, legal basis, retention period, withdrawal process and access controls.",
    "Research consent and the GDPR legal basis are separate decisions. A real study requires supervisor review and the relevant TUHH ethics/data-protection process before recruitment.",
    "Participants may stop at any time. A production study should offer a deletion request using the random completion code until a stated deadline.",
]))
story.append(h2("Interface acceptance criteria"))
story.extend(bullets([
    "Landing page exposes both required actions: download the report and start the study.",
    "Pairwise trials show exactly two equal-weight cards; ranking trials contain exactly ten unique movies.",
    "Ranking supports mouse drag interaction and keyboard up/down buttons; server validation remains authoritative.",
    "Visible progress, clear instructions, responsive layout, reduced-motion support and WCAG-oriented focus/contrast are included.",
]))
story.append(PageBreak())

# Limitations/references
story.extend(heading("09", "Limitations, reproducibility and references"))
story.append(h2("Threats to validity"))
story.extend(bullets([
    "Metadata cards cannot reproduce the rich decision context of trailers, posters or prior viewing; familiarity may dominate some choices.",
    "The historical-vote eligibility filter improves recognizability but underrepresents obscure and non-mainstream films.",
    "Linear utility and the Plackett-Luce independence-of-irrelevant-alternatives assumption may miss context effects and non-compensatory taste.",
    "Two ten-item rankings versus ten binary choices intentionally differ in information density; conclusions apply to equal movie exposure, not equal interaction time.",
    "A convenience university sample would limit generalization by age, culture and movie-consumption patterns.",
]))
story.append(h2("Reproducibility checklist"))
story.extend(bullets([
    "Dataset path and required schema are validated; the assignment-linked CSV is bundled with provenance notes.",
    "Movie sampling seed, method order and all presented IDs are stored per participant.",
    "Feature order is deterministic; keyword SVD uses random_state = 0; model lambda and optimizer are fixed.",
    "Tests cover dataset integrity, seeded disjoint sampling, model identities, malformed ranking input, CSRF and the complete study flow.",
    "Run with: python manage.py migrate; python manage.py test project4; python manage.py runserver."
]))
story.append(h2("References"))
references = [
    "[1] Human-Centric Artificial Intelligence. Project 4: Preference elicitation. Course assignment brief, pp. 1-2.",
    "[2] Bradley, R. A., and Terry, M. E. (1952). Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. Biometrika 39(3/4), 324-345. <link href='https://doi.org/10.1093/biomet/39.3-4.324'>doi:10.1093/biomet/39.3-4.324</link>.",
    "[3] Plackett, R. L. (1975). The Analysis of Permutations. Applied Statistics 24(2), 193-202. <link href='https://doi.org/10.2307/2346567'>doi:10.2307/2346567</link>.",
    "[4] Luce, R. D. (1959). Individual Choice Behavior: A Theoretical Analysis. New York: Wiley.",
    "[5] IMDb 5000 Movie Dataset file linked by the assignment. <link href='https://github.com/yash91sharma/IMDB-Movie-Dataset-Analysis/blob/master/movie_metadata.csv'>GitHub dataset page</link>.",
    "[6] European Union (2016). General Data Protection Regulation, Regulation (EU) 2016/679. <link href='https://eur-lex.europa.eu/eli/reg/2016/679/oj'>Official text</link>.",
]
for reference in references:
    story.append(p(reference, "Reference"))
story.append(Spacer(1, 5 * mm))
story.append(callout(
    "Implementation note",
    "The participant-facing interface is written in English to match the course brief. This report "
    "documents what the prototype does and clearly separates implemented behavior from recommended "
    "procedures for a future real deployment.",
))


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = ReportDocTemplate(
        str(OUTPUT),
        title="Project 4: Preference Elicitation",
        author="HCAI Project Team",
        subject="Methods and planned user study",
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    document.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
