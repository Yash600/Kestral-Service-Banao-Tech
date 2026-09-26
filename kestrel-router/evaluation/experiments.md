# What we tried

Train before 2026-04-01, test on Apr-Jun 2026 (2,135 requests). Accuracy = share sent to the team that actually resolved the request.

| Experiment | Accuracy vs real outcome | Kept? | Note |
|---|---|---|---|
| Old routing bot (for reference) | 76.7% | - |  |
| Train on the bot's labels (team_label) | 81.1% | discarded | matches the bot's labels 94.0%, but it learns the bot's mistakes |
| Logistic regression (words + character pieces) | 84.4% | runner-up |  |
| Naive Bayes (words + character pieces) | 82.6% | discarded |  |
| Linear SVM, calibrated (words + character pieces) | 85.0% | kept |  |
| LightGBM (gradient-boosted trees) | 83.3% | discarded | slower, less accurate on short text, no readable reasons |
| XGBoost (gradient-boosted trees) | 83.4% | discarded | same as LightGBM |
| Strip punctuation before modelling | 83.6% | discarded | commas separate the things a customer asks for, so removing them loses information |
| Extra weight on the FIRST thing asked | 81.9% | discarded | wrong way round: the last thing asked is what gets resolved |
| Final: SVM + last-thing-asked + cleaned IDs + Billing rule | 85.9% | kept |  |
| Final without the Billing rule (policy §3) | 85.9% | kept as a safeguard | the model already learned that 'I paid' is not Billing, so the rule rarely fires |
| Jev (TypeSafe AI, pretrained decision model, via API) | not measured | built, not in use | Engine, page tab and evaluate_jev.py are ready. No API key was available (limited early access), so it has not been scored |
