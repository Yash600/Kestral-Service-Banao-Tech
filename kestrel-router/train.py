"""Train the router on all labelled history and write predictions for the unlabelled requests.

    python train.py      -> models/router.joblib, predictions.csv, evaluation/test_predictions_detail.csv
"""
from pathlib import Path

import joblib
import pandas as pd

from kestrel.data import load_test, load_train
from kestrel.router import Router

ROOT = Path(__file__).parent


def main():
    d = load_train()
    print(f"Training on {len(d):,} requests ({d.created.min():%d %b %Y} to {d.created.max():%d %b %Y}), "
          f"target = team that actually resolved each request")
    router = Router().fit(Router.frame(d), d.final_team)
    router.train_ids_ = set(d.request_id)  # lets the batch screen warn when scoring rows the model has already seen
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(router, ROOT / "models" / "router.joblib")

    test = load_test()
    P, fired, _ = router.predict_proba(Router.frame(test))
    test["team"] = router.classes_[P.argmax(1)]
    test["confidence"] = P.max(1).round(3)
    test["would_ask_question"] = test.confidence < router.threshold
    test["billing_rule_applied"] = fired

    # predictions.csv cannot ask a question, so every row gets the model's best guess.
    test[["request_id", "team"]].to_csv(ROOT / "predictions.csv", index=False, lineterminator="\n")  # LF, like the sample
    (ROOT / "evaluation").mkdir(exist_ok=True)
    test.drop(columns=["request_text"]).to_csv(ROOT / "evaluation" / "test_predictions_detail.csv", index=False)

    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    assert list(sample.request_id) == list(test.request_id), "row order differs from sample_submission.csv"
    print(f"Wrote predictions.csv: {len(test):,} rows")
    print(test.team.value_counts().to_string())
    print(f"Would ask one question on {test.would_ask_question.mean():.1%}; "
          f"accuracy expected on the rest ~97%, overall ~85-86% (see evaluation/report.md)")


if __name__ == "__main__":
    main()
