# Evidence: does the router work, and how often does it not?

All numbers compare against **where each request was actually resolved** (`final_team` in the resolution log), not against the old bot's labels. Every test uses only data from *before* the test window, the way the model will be used in real life. Renamed teams are merged (Installations = Installs & Demo, Consumables = Filters & Consumables).

## 1. Headline

On the most recent three months (2026-04 to 2026-06, 2,135 requests):

- The old bot sent **76.7%** of requests to the team that resolved them.
- The model, forced to answer every request, gets **85.9%**.
- With the follow-up question switched on, the model routes **84.0%** of requests by itself at **97.4%** accuracy, and asks one question on the other **16.0%**.

## 2. Backtest over three separate quarters

| Test window | Requests | Trained on | Old bot | Model, every request | Routed automatically | Accuracy when routed automatically |
|---|---|---|---|---|---|---|
| 2025-10 to 2025-12 | 2199 | 4320 | 77.5% | 85.1% | 82.7% | 98.3% |
| 2026-01 to 2026-03 | 2168 | 6519 | 77.5% | 86.3% | 84.3% | 98.0% |
| 2026-04 to 2026-06 | 2135 | 8687 | 76.7% | 85.9% | 84.0% | 97.4% |

The result is stable across quarters, including the switch from Zoho to the CRM in Oct 2025.

## 3. Why not just match the bot's labels at 90%?

A model trained on the bot's own labels matches them **94.0%** of the time, which clears the 90% bar easily. But it sends only **81.1%** of requests to the team that resolves them, because it has learned the bot's mistakes. Our model agrees with the bot only 81.6% of the time, and that disagreement is exactly the misroutes it fixes.

## 4. Why not 90% on every request?

Some requests contain no clue about the problem ("please call me about my purifier", "need help with my mixer"). Here is where the vague ones (model confidence under 40%) actually ended up:

| Resolved by | Share |
|---|---|
| Repairs | 19.7% |
| Installs & Demo | 17.3% |
| Returns & Replacement | 16.9% |
| Product Advice | 14.1% |
| Warranty Claims | 13.7% |
| Filters & Consumables | 9.6% |
| Billing | 8.8% |

They are spread across every team. No model can route these from the text, so the service asks one question instead of guessing.

| Slice | Requests | Accuracy |
|---|---|---|
| Clear enough (confidence >= 40%) | 1886 | 94.3% |
| Vague (confidence < 40%) | 249 | 22.1% |
| Asks about 2+ things | 284 | 88.0% |
| Asks about one thing | 1851 | 85.6% |
| Channel: chat | 670 | 87.3% |
| Channel: email | 228 | 86.4% |
| Channel: ivr | 579 | 84.1% |
| Channel: whatsapp | 658 | 85.9% |

## 5. Is the confidence score honest?

If the model says 80%, it should be right about 80% of the time. This is what makes the ask-a-question rule safe.

| Model confidence | Requests | Average confidence | Actually right |
|---|---|---|---|
| 0%-30% | 138 | 25.6% | 19.6% |
| 30%-40% | 111 | 34.2% | 25.2% |
| 40%-50% | 45 | 44.6% | 24.4% |
| 50%-60% | 23 | 55.1% | 39.1% |
| 60%-70% | 24 | 65.9% | 45.8% |
| 70%-80% | 41 | 75.4% | 85.4% |
| 80%-90% | 347 | 88.3% | 96.8% |
| 90%-100% | 1406 | 92.2% | 97.9% |

### Choosing when to ask

| Ask a question below | Routed automatically | Accuracy when routed automatically | Sent to one question |
|---|---|---|---|
| never ask | 100.0% | 85.9% | 0.0% |
| 40.0% | 88.3% | 94.3% | 11.7% |
| 50.0% | 86.2% | 96.0% | 13.8% |
| 60.0% | 85.2% | 96.8% | 14.8% |
| 70.0% | 84.0% | 97.4% | 16.0% |

We use **70.0%**: below that, the service asks one question. The reason is money: a misroute costs about Rs 692, and a question costs at most Rs 260. So asking pays off whenever the real chance of being right is under about 62%. The table above shows that between 50% and 70% the model is actually right less than half the time.

## 6. Where it goes wrong

| Team | Requests (truth) | When we say this team, we're right | Of this team's requests, we catch |
|---|---|---|---|
| Repairs | 498 | 79.3% | 92.2% |
| Installs & Demo | 321 | 86.5% | 83.5% |
| Filters & Consumables | 207 | 90.3% | 85.0% |
| Billing | 255 | 92.2% | 87.8% |
| Returns & Replacement | 326 | 87.2% | 83.7% |
| Warranty Claims | 259 | 86.5% | 83.8% |
| Product Advice | 269 | 88.9% | 80.7% |

Confusion matrix (rows = team that really resolved it, columns = model's choice):

| Truth \ Predicted | Repairs | Installs & Demo | Filters & Consumables | Billing | Returns & Replacement | Warranty Claims | Product Advice |
|---|---|---|---|---|---|---|---|
| Repairs | 459 | 9 | 4 | 4 | 10 | 8 | 4 |
| Installs & Demo | 25 | 268 | 4 | 3 | 10 | 6 | 5 |
| Filters & Consumables | 11 | 7 | 176 | 1 | 2 | 5 | 5 |
| Billing | 15 | 6 | 4 | 224 | 2 | 1 | 3 |
| Returns & Replacement | 31 | 5 | 2 | 3 | 273 | 7 | 5 |
| Warranty Claims | 14 | 5 | 1 | 5 | 12 | 217 | 5 |
| Product Advice | 24 | 10 | 4 | 3 | 4 | 7 | 217 |

All 301 errors from this window are listed in `holdout_errors.csv`, most confident first.

## 7. Rupees (ops policy §4: Rs 305 per transfer, Rs 260 per extra customer contact)

- Volume: about 8,540 requests a year.
- The bot misroutes about 1,992 a year, each taking 1.42 transfers on average, so about Rs 692 per misroute.
- Bot misrouting cost: **Rs 1,459,760 a year**, on top of the Rs 3.2 lakh licence.
- Model answering every request: about **Rs 833,637 a year**.
- Model with the question: about **Rs 127,400 a year** in misroutes, plus 1,364 questions. If every question cost a full extra contact (Rs 260, a pessimistic case) that adds Rs 354,640.
- Run cost: runs on any ordinary server. No per-request fee and no paid API.

## 8. Workload by team (Jan to Jun 2026, where requests really belong)

| Team | Requests per month (where they really belong) | Share | What the bot sent them per month | Median hours to resolve |
|---|---|---|---|---|
| Repairs | 165 | 23.0% | 206 | 9.0 |
| Returns & Replacement | 106 | 14.8% | 81 | 9.6 |
| Installs & Demo | 104 | 14.6% | 78 | 10.5 |
| Product Advice | 90 | 12.6% | 77 | 9.6 |
| Billing | 90 | 12.5% | 113 | 9.2 |
| Warranty Claims | 89 | 12.4% | 67 | 9.8 |
| Filters & Consumables | 73 | 10.1% | 95 | 8.9 |

Resolution times for Zoho-era rows were shifted from UTC to IST first (ops policy §9).
