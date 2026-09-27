# Screen recording script (about 1.5 minutes, no slides)

Before recording:
- Server running (`uvicorn app:app --port 8000`), browser at http://localhost:8000.
- `evaluation/report.md` open in a second tab.
- `data/test_unlabelled.csv` ready to drag in.

| Time | On screen | Say |
|---|---|---|
| 0:00-0:12 | The app, Local model tab | "Kestrel asked for a model that matches the old bot's labels 90% of the time. The first thing we checked was the labels themselves: the bot was wrong on 1 in 4 requests. Copying it would copy its mistakes." |
| 0:12-0:30 | `report.md`, backtest table | "So we trained on where each request was actually resolved. The algorithm is TF-IDF, which turns the message into word weights, and a calibrated linear SVM, which picks the team and gives an honest confidence score. On three quarters it had never seen, it scored 86%, against the bot's 77%." |
| 0:30-0:45 | Click **Clear** example, then **Two asks** | "It routes this to Repairs and shows the words behind the decision. When a customer asks two things, the last one decides, and adding that improved the model." |
| 0:45-1:05 | Click **Vague** example, then pick an answer | "Why not 90%? Sixteen percent of messages are like this one, 'please call me about my purifier'. They ended up with all seven teams equally, so no model can route them from the text. Even a perfect model on the rest tops out near 88%. So instead of guessing, it asks one question." |
| 1:05-1:15 | **Batch file** tab: drag in the CSV | "Requests arrive in files, so it routes a whole export in under a second, with a reason for each row." |
| 1:15-1:22 | Jev tab | "We threw away XGBoost and keyword lists. Jev, a new AI model, is built in but untested: we had no key." |
| 1:22-1:30 | Back to the Local model tab | "Still missing: a live trial and the automatic question to customers. Conclusion: 86% overall, 97% where it's confident, zero running cost." |
