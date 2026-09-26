# Can anything predict the vague requests?

About 16% of requests say nothing about the problem, for example "please call me about my purifier", "need help with my mixer" or "service request for fryer". The model's confidence on these is under 70%, and it is right only about 25% of the time. Before settling on asking a question, we checked every other field in the export for a signal.

Test: vague requests from Apr to Jun 2026 (341 requests), with models trained only on earlier data.

| What we tried | Accuracy on vague requests | Verdict |
|---|---|---|
| Always guess the most common team | 20.8% | the floor |
| Channel, warranty, product, hour of day and weekday, with no text | 19.9% | no signal, slightly worse than the floor |
| Order numbers (KO...): do they encode a purchase date, or repeat for the same customer? | - | numbers are random; the 106 repeats are coincidences (same team only 18% of the time, about chance), often for different products |
| Registration numbers (SR...): customer history? | - | only 7 of 1,396 repeat, so there is no history to link |
| Our text model | 25.2% | best of these, but still close to a guess |

Where these requests really ended up is spread across all seven teams (Repairs about 20%, the rest 9-17% each). **The information isn't anywhere in the record**, so no model can recover it, whether that's a larger model, gradient boosting or an LLM.

## What we do instead

The service asks one question. We first showed only the top 3 teams, but on vague requests the right team is in the top 3 only 49% of the time:

| Options shown | Contains the right team |
|---|---|
| Top 1 | 25.2% |
| Top 2 | 37.8% |
| Top 3 | 49.3% |
| Top 4 | 63.9% |
| Top 5 | 78.6% |
| All 7 | 100% |

So the question now lists **all seven options in plain words, most likely first**, with the top three marked "likely". The right answer is always one click away.
