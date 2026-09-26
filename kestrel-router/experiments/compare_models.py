"""What we tried, and what we threw away. Same backtest as evaluate.py: train before Apr 2026, test Apr-Jun 2026.

    pip install -r requirements-dev.txt
    python experiments/compare_models.py   -> evaluation/experiments.md
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_union
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kestrel.data import load_train  # noqa: E402
from kestrel.router import Router  # noqa: E402
from kestrel.text import clauses, fix_mojibake  # noqa: E402

warnings.filterwarnings("ignore")
CUT = "2026-04-01"


def tfidf():
    return make_union(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True),
    )


def main():
    d = load_train()
    tr, te = d.created < CUT, d.created >= CUT
    y, yt = d.final_team[tr].values, d.final_team[te].values
    rows = []

    def add(name, acc, verdict, note=""):
        rows.append({"Experiment": name, "Accuracy vs real outcome": f"{acc:.1%}", "Kept?": verdict, "Note": note})
        print(f"{name:55s} {acc:.3f}  {verdict}")

    rows.append({"Experiment": "Old routing bot (for reference)", "Accuracy vs real outcome": f"{d.bot_correct[te].mean():.1%}", "Kept?": "-", "Note": ""})

    # --- the target
    cp = Router().fit(Router.frame(d[tr]), d.team_label[tr]).predict(Router.frame(d[te]))[0]
    add("Train on the bot's labels (team_label)", np.mean(cp == yt), "discarded",
        f"matches the bot's labels {np.mean(cp == d.team_label[te].values):.1%}, but it learns the bot's mistakes")

    # --- algorithms on the same simple text features
    base = (d.request_text.map(fix_mojibake).str.lower() + " | " + d.product_family + " | " + d.warranty_status + " | " + d.channel).str.lower()
    U = tfidf()
    A, B = U.fit_transform(base[tr]), U.transform(base[te])
    for name, m, verdict in [
        ("Logistic regression", LogisticRegression(C=3, max_iter=4000), "runner-up"),
        ("Naive Bayes", ComplementNB(alpha=0.3), "discarded"),
        ("Linear SVM, calibrated", CalibratedClassifierCV(LinearSVC(C=0.3), cv=5), "kept"),
    ]:
        add(f"{name} (words + character pieces)", np.mean(m.fit(A, y).predict(B) == yt), verdict)

    try:
        import lightgbm as lgb
        import xgboost as xgb

        W = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=5000)
        Wa, Wb = W.fit_transform(base[tr]).astype(np.float32), W.transform(base[te]).astype(np.float32)
        le = LabelEncoder().fit(y)
        g = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31, verbose=-1).fit(Wa, le.transform(y))
        add("LightGBM (gradient-boosted trees)", np.mean(le.inverse_transform(g.predict(Wb)) == yt), "discarded", "slower, less accurate on short text, no readable reasons")
        x = xgb.XGBClassifier(n_estimators=600, learning_rate=0.1, max_depth=6, tree_method="hist").fit(Wa, le.transform(y))
        add("XGBoost (gradient-boosted trees)", np.mean(le.inverse_transform(x.predict(Wb)) == yt), "discarded", "same as LightGBM")
    except ImportError:
        print("lightgbm / xgboost not installed - skipping tree models (pip install -r requirements-dev.txt)")

    # --- text preparation
    nopunct = base.str.replace(r"[^a-z0-9&' |]+", " ", regex=True)
    U = tfidf()
    m = CalibratedClassifierCV(LinearSVC(C=0.3), cv=5).fit(U.fit_transform(nopunct[tr]), y)
    add("Strip punctuation before modelling", np.mean(m.predict(U.transform(nopunct[te])) == yt), "discarded",
        "commas separate the things a customer asks for, so removing them loses information")

    parts = base.map(lambda s: clauses(s.split(" | ")[0]))
    first = parts.map(lambda c: " ".join("f_" + w for w in re.findall(r"[a-z]+", c[0])) if c else "")
    U = tfidf()
    m = CalibratedClassifierCV(LinearSVC(C=0.3), cv=5).fit(U.fit_transform((base + " " + first)[tr]), y)
    add("Extra weight on the FIRST thing asked", np.mean(m.predict(U.transform((base + " " + first)[te])) == yt), "discarded",
        "wrong way round: the last thing asked is what gets resolved")

    final = Router().fit(Router.frame(d[tr]), y)
    add("Final: SVM + last-thing-asked + cleaned IDs + Billing rule", np.mean(final.predict(Router.frame(d[te]))[0] == yt), "kept")
    no_rule = np.mean(final.predict(Router.frame(d[te]), apply_policy=False)[0] == yt)
    add("Final without the Billing rule (policy §3)", no_rule, "kept as a safeguard",
        "the model already learned that 'I paid' is not Billing, so the rule rarely fires")

    rows.append({"Experiment": "Jev (TypeSafe AI, pretrained decision model, via API)", "Accuracy vs real outcome": "not measured",
                 "Kept?": "built, not in use",
                 "Note": "Engine, page tab and evaluate_jev.py are ready. No API key was available (limited early access), so it has not been scored"})
    out = pd.DataFrame(rows)
    lines = ["# What we tried", "", f"Train before {CUT}, test on Apr-Jun 2026 ({te.sum():,} requests). Accuracy = share sent to the team that actually resolved the request.", ""]
    lines += ["| " + " | ".join(out.columns) + " |", "|---|---|---|---|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in out.values]
    (ROOT / "evaluation" / "experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote evaluation/experiments.md")


if __name__ == "__main__":
    main()
