# Kestrel Home: service-request router

This project routes a new service request to one of Kestrel's seven teams. For every decision it shows the reasons, in words an agent can read. When a message doesn't say what the problem is, it asks **one question** instead of guessing.

It runs entirely on your machine. There is no paid API, no API key and no per-request cost. A second engine, TypeSafe's Jev, is built in but not in use yet; see [Two engines](#two-engines-one-in-use-one-built-but-not-in-use).

## Run it (clean machine, about 3 minutes)

You need **Python 3.11 or newer**.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Put Kestrel's export files in `data/`: `train.csv`, `test_unlabelled.csv`, `resolution_log.csv`, `teams.csv` and `sample_submission.csv`. They aren't in the repository because they're client data.

```bash
python train.py                  # about 15 s: trains the model, writes predictions.csv
uvicorn app:app --port 8000      # then open http://localhost:8000
```

Optionally, rebuild the evidence:

```bash
python evaluate.py                              # evaluation/report.md (about 1 min)
pip install -r requirements-dev.txt
python experiments/compare_models.py            # evaluation/experiments.md
```

## Two engines: one in use, one built but not in use

| | **Local ML model** (in use) | **Jev** (built, not in use yet) |
|---|---|---|
| What it is | A linear SVM classifier we trained on Kestrel's own history | TypeSafe AI's pretrained "System One" decision model, reached over an API |
| What it learned from | 10,822 past requests and the team that actually resolved each one | Nothing from Kestrel. At request time it gets only the seven team descriptions and the Billing rule from ops policy §3 |
| What it returns | A team, probabilities for all seven, and the words in the message behind the choice | A team, probabilities for all seven, and two yes/no readings: does the message say what the problem is, and is the payment itself the problem |
| Tested accuracy | 85.9% on every request, 97.4% on the 84% it routes by itself (Apr-Jun 2026 backtest) | **Not measured.** We had no API key |
| Cost | ₹0. Runs on any machine, offline | About $0.04 per million input tokens; pennies a year at Kestrel's volume |
| Client data | Never leaves Kestrel | Request text goes to TypeSafe's servers, which needs Kestrel's approval (ops policy §10) |
| Status | **Produces `predictions.csv` and powers the service** | Code, page tab and evaluation script are ready. It switches on when a key is added |

### Local ML model (in use)

It reads the customer's message, plus product, cover and channel. It turns the text into word and letter-piece weights (TF-IDF) and scores each team with a linear SVM trained on real outcomes. The scores are calibrated, so its confidence is honest: at 70% or above it routes, and below that it asks one question. Every decision comes with the words that caused it. This engine produced every number in `evaluation/report.md` and the submitted `predictions.csv`.

### Jev (built, not in use yet)

Jev is a new kind of model, released in September 2026, that returns a typed decision with probabilities instead of generated text. That makes it a natural second opinion for routing. We built it in fully:

- `kestrel/jev_router.py` asks Jev for the team, whether the message describes a problem, and whether the payment is the problem. It returns the same response shape as the local model, including the follow-up question.
- The page has a **Jev tab**. With a key, each result also shows the other engine's answer, marked agree or disagree.
- `evaluate_jev.py` scores Jev on the same backtest as the local model, tries the two combined, and writes `evaluation/jev_report.md` and alternative prediction files.

**Why it isn't in use:** Jev is in limited early access and we couldn't get an API key during the assignment window. So we have no measured accuracy for it, and we don't claim one. Without a key, the Jev tab says it's off and why, and `"engine": "jev"` on the endpoint returns a readable 503. Nothing else is affected.

**Why we still think it's worth trying:** it is cheap, it needs no training, and it reads meaning rather than exact words. It could catch requests phrased in ways the history never saw. We don't expect it to fix the vague requests ("please call me about my purifier"), because the problem isn't in the text for any model to find (`evaluation/vague_requests.md`).

**To switch it on later:**

```bash
# PowerShell:  $env:TYPESAFE_API_KEY="your-key"      macOS/Linux:  export TYPESAFE_API_KEY=your-key
uvicorn app:app --port 8000
python evaluate_jev.py      # Jev vs local model vs both combined -> evaluation/jev_report.md
```

Adopt it only if the report shows it beats the local model, and only once Kestrel approves TypeSafe as a vendor.

If you start the server before training, the page still loads. It says the model isn't trained, and `POST /route` returns a plain-English 503 error.

## The endpoint

`POST /route` takes one record as JSON. Interactive docs are at `http://localhost:8000/docs`.

```json
{ "request_text": "display of air fryer gone blank pls call back",
  "product_family": "Air Fryer", "warranty_status": "in_warranty", "channel": "chat" }
```

When the router is confident (70% or more), it answers:

```json
{ "status": "routed", "team": "Repairs", "confidence": 0.91,
  "reasons": ["Words that point to Repairs: \"gone blank\", \"display\".", "Product is in warranty.",
              "Next most likely: Filters & Consumables (3%)."],
  "ranked": [{"team": "Repairs", "p": 0.91}, "..."] }
```

When it isn't, it asks one question:

```json
{ "status": "needs_clarification", "best_guess": "Installs & Demo", "confidence": 0.49,
  "question": "What is this request about?",
  "options": [{"team": "Installs & Demo", "label": "Installation, demo or wall-mounting visit"}, "..."] }
```

Send the same record again with `"clarification": "<team>"` to finish routing it. Add `"engine": "jev"` to use the Jev engine instead of the local model.

## Batch files: route a whole export

The page's **03 Batch file** tab takes a CSV export: drag it in, or choose it. Every row is routed at once, with no API. For about 2,000 rows it takes under a second. You get:

- **Tiles:** number of requests, share routed automatically, share needing one question, and average confidence.
- **Where the requests go:** a count per team. The hatched part of each bar is the best guesses that need a question first.
- **Confidence bands, plus breakdowns by channel and by product.** These show where vague requests come from.
- **A table of every request:** team, confidence, a short reason, and "ask first" flags. You can filter by status or team and search.
- **Download CSV:** the original columns plus `team`, `status`, `confidence`, `runner_up` and `reason`.
- **Accuracy**, if the file has the true team (`final_team` or `team`), shown next to the old bot if `team_label` is there. If those rows were used in training, the page says so, because the numbers are then optimistic. The honest backtest is in `evaluation/report.md`.

The same thing is available through the API:

```bash
curl -X POST http://localhost:8000/route/batch -H "Content-Type: text/csv" --data-binary @data/test_unlabelled.csv
```

The file needs a `request_text` column; `request_id`, `channel`, `product_family` and `warranty_status` are used when present. The limits are 20 MB and 50,000 rows per upload. A file without `request_text`, an empty file, or a non-CSV file gets a plain-English error.

## How it works

- **What it learns from:** the team that **actually resolved** each request (`final_team` in `resolution_log.csv`), not the old bot's first guess (`team_label`). The bot's labels were right only 77% of the time. A model that copies them copies the misroutes (see `evaluation/report.md`, section 3).
- **The model:** TF-IDF on words and character pieces, then a linear SVM with calibrated probabilities (scikit-learn). It's fast, costs nothing to run, and each decision can be explained word by word.
- **Things the data taught us:**
  - When a customer asks about two things, the last one usually decides the team, so the model gets extra features for the last thing asked.
  - Saying "I paid" doesn't make a request a Billing request (ops policy §3). This is a rule on top of the model.
  - Messages with no clue ("please call me about my purifier") get a question, not a guess.
- **Data fixes:**
  - Garbled Zoho text is repaired.
  - Renamed teams are merged (Installations = Installs & Demo, Consumables = Filters & Consumables).
  - Zoho resolution times are moved from UTC to IST (policy §9).

## Files

| Path | What it is |
|---|---|
| `train.py` | Trains on all history and writes `predictions.csv` |
| `evaluate.py` | Backtests and writes `evaluation/report.md`: accuracy, errors, rupees, team workload |
| `experiments/compare_models.py` | Everything we tried, with results (`evaluation/experiments.md`) |
| `kestrel/jev_router.py` | Second engine, not in use yet: TypeSafe Jev, same response shape |
| `evaluate_jev.py` | Jev vs local model vs both combined, on the same backtest (needs a key) |
| `app.py` | FastAPI: `POST /route` (one record), `POST /route/batch` (a CSV file), `/health`, `/api/meta` and the page |
| `static/index.html` | The screen: tabs for the local model, Jev, and batch files |
| `kestrel/` | Cleaning (`text.py`), loading (`data.py`), model and reasons (`router.py`) |
| `predictions.csv` | One team for each of the 2,178 unlabelled requests |

## Data handling

`data/*.csv` and the files in `evaluation/` that contain request text are git-ignored. Keep this repository **private** (ops policy §10).
