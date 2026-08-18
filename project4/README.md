# Project 4 - Preference Elicitation

This Django app implements the HCAI Project 4 preference-elicitation assignment
within the team project.

## What is included

- IMDb 5000 metadata cleaning and deterministic feature extraction
- Bradley-Terry learning from pairwise choices
- Plackett-Luce learning from complete ten-movie rankings
- Counterbalanced within-subject participant flow
- 10 pairwise trials, 2 ten-movie rankings and 20 held-out validation trials
- Mental-demand, confidence, ease and final-preference questionnaires
- CSRF-protected server-side validation and pseudonymous study logging
- Unit, integration, CSRF and end-to-end tests
- Downloadable methods and study-design report at `/project4/report/`

## Set up on Windows

From the repository root, create a virtual environment and install the shared
requirements:

```powershell
py -m venv env
.\env\Scripts\python.exe -m pip install --upgrade pip
.\env\Scripts\python.exe -m pip install -r requirements.txt
```

Select `env\Scripts\python.exe` through **Python: Select Interpreter** in VS Code.

## Prepare and verify

```powershell
.\env\Scripts\python.exe manage.py migrate
.\env\Scripts\python.exe manage.py check
.\env\Scripts\python.exe manage.py makemigrations --check --dry-run
.\env\Scripts\python.exe manage.py test project4
```

## Run

```powershell
.\env\Scripts\python.exe manage.py runserver
```

Open <http://127.0.0.1:8000/project4/>.

## Model definitions

For movie features `x` and participant weights `w`, utility is `u = w^T x`.

Bradley-Terry pairwise probability:

```text
P(i preferred to j) = sigmoid(w^T (x_i - x_j))
```

Plackett-Luce probability for a best-to-worst ranking:

```text
P(i_1 > ... > i_m)
  = product from r=1 to m-1 of
    exp(w^T x_i_r) / sum from s=r to m exp(w^T x_i_s)
```

Both models use L2-regularized maximum likelihood and the same compact movie
representation: multi-hot genres, release decade, standardized duration,
language, country, content rating and 16 TF-IDF/SVD plot-keyword components.

## Data provenance

`data/movie_metadata.csv` is the IMDb 5000 file linked by the assignment brief:

<https://github.com/yash91sharma/IMDB-Movie-Dataset-Analysis/blob/master/movie_metadata.csv>
