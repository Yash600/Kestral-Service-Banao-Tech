"""The routing model: TF-IDF features -> calibrated linear SVM, plus policy rules and readable reasons."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from .text import (
    BILLING_PROBLEM,
    PAID_MENTION,
    TEAM_QUESTION,
    clauses,
    clean_text,
    model_document,
    products_mentioned,
)

META_COLS = ["product_family", "warranty_status", "channel"]
WARRANTY_WORDS = {"in_warranty": "in warranty", "shield": "on Kestrel Shield", "out_of_warranty": "out of warranty"}
# Openers and fillers that say nothing about the problem - never shown as a reason.
FILLER = set(
    "hi hello team sir madam good morning namaste pls please thanks thank you kindly asap urgent "
    "very disappointed help regno orderno reg no did my the for of with about regarding me is a an and to i "
    "on in it this was water purifier air fryer mixer grinder induction cooktop room heater ceiling fan robot "
    "vacuum product machine need want come got order".split()
)


class Router:
    def __init__(self, C: float = 0.3, threshold: float = 0.7, billing_rule: bool = True):
        # threshold: below this confidence we ask one question. A misroute costs ~Rs 692 (1.4 transfers x Rs 305
        # + Rs 260 extra contact); a question costs at most Rs 260. Asking pays whenever the real chance of being
        # right is under ~62%, and in the backtest that is everything the model scores below 70%.
        self.C = C
        self.threshold = threshold
        self.billing_rule = billing_rule

    # ---------- data shaping ----------
    @staticmethod
    def frame(records) -> pd.DataFrame:
        df = pd.DataFrame(records).copy()
        for col in META_COLS:
            if col not in df:
                df[col] = "unknown"
            df[col] = df[col].fillna("unknown").astype(str)
        df["clean"] = df["request_text"].map(clean_text)
        df["doc"] = [
            model_document(c, p, w, ch)
            for c, p, w, ch in zip(df["clean"], df["product_family"], df["warranty_status"], df["channel"])
        ]
        return df

    # ---------- training ----------
    def fit(self, df: pd.DataFrame, y) -> "Router":
        self.features = ColumnTransformer(
            [
                ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "doc"),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True), "doc"),
            ]
        )
        X = self.features.fit_transform(df)
        self.model = CalibratedClassifierCV(LinearSVC(C=self.C), cv=5).fit(X, np.asarray(y))
        self.classes_ = self.model.classes_
        # Averaged linear weights of the 5 fold models - used only to explain decisions.
        self.coef_ = np.mean([c.estimator.coef_ for c in self.model.calibrated_classifiers_], axis=0)
        self.feature_names_ = self.features.get_feature_names_out()
        return self

    # ---------- prediction ----------
    def predict_proba(self, df: pd.DataFrame, apply_policy: bool = True):
        X = self.features.transform(df)
        P = self.model.predict_proba(X)
        fired = np.zeros(len(df), dtype=bool)
        if apply_policy and self.billing_rule:
            P, fired = self._billing_policy(df, P)
        return P, fired, X

    def _billing_policy(self, df, P):
        """Policy §3: saying you paid does not make it a Billing request."""
        P = P.copy()
        b = list(self.classes_).index("Billing")
        top = P.argmax(1) == b
        paid = df["clean"].str.contains(PAID_MENTION).to_numpy()
        problem = df["clean"].str.contains(BILLING_PROBLEM).to_numpy()
        fired = top & paid & ~problem
        P[fired, b] = 0.0
        P[fired] /= P[fired].sum(1, keepdims=True)
        return P, fired

    def predict(self, df: pd.DataFrame, apply_policy: bool = True):
        P, _, _ = self.predict_proba(df, apply_policy)
        return self.classes_[P.argmax(1)], P.max(1)

    # ---------- the live, single-record answer ----------
    def route(self, record: dict, clarification: str | None = None) -> dict:
        df = self.frame([record])
        P, fired, X = self.predict_proba(df)
        p = P[0]
        order = np.argsort(p)[::-1]
        top, conf = self.classes_[order[0]], float(p[order[0]])
        ranked = [{"team": self.classes_[i], "p": round(float(p[i]), 3)} for i in order]

        if clarification:
            if clarification not in self.classes_:
                raise ValueError(f"Unknown team: {clarification}")
            return {
                "status": "routed",
                "team": clarification,
                "confidence": None,
                "model_guess": top,
                "reasons": [
                    f'Answer to the follow-up question: "{TEAM_QUESTION[clarification]}".',
                    f"Before asking, the model's best guess was {top} at {conf:.0%}.",
                ],
                "ranked": ranked,
            }

        reasons = self._reasons(df.iloc[0], X, order, p, bool(fired[0]))
        base = {"confidence": round(conf, 3), "ranked": ranked, "reasons": reasons}
        if conf >= self.threshold:
            return {"status": "routed", "team": top, **base}
        # All seven, most likely first: on vague requests the right team is in the top 3 only about half the time,
        # so hiding the rest behind "something else" would cost a second click on every other request.
        options = [
            {"team": self.classes_[i], "label": TEAM_QUESTION[self.classes_[i]], "likely": rank < 3}
            for rank, i in enumerate(order)
        ]
        return {
            "status": "needs_clarification",
            "best_guess": top,
            "question": "What is this request about?",
            "options": options,
            **base,
        }

    # ---------- a whole file at once ----------
    def route_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Route every row of a raw export. Returns one row per request with the team, status and a short reason."""
        frame = self.frame(df.to_dict("records"))
        P, fired, X = self.predict_proba(frame)
        order = np.argsort(-P, axis=1)
        rows = []
        for i in range(len(frame)):
            top_i, second_i = order[i, 0], order[i, 1]
            conf = float(P[i, top_i])
            clean = frame["clean"].iat[i]
            phrases = self._evidence_phrases(X[i : i + 1], top_i, clean)
            notes = [("Words: " + ", ".join(phrases)) if phrases else "No specific problem described"]
            if len(clauses(clean)) > 1:
                notes.append("asks about 2+ things")
            if fired[i]:
                notes.append("paid, but not a payment problem (policy §3)")
            said = products_mentioned(clean)
            prod = frame["product_family"].iat[i]
            if said and prod not in said and prod != "unknown":
                notes.append(f"text says {said[0].lower()}, record says {prod}")
            rows.append(
                {
                    "team": self.classes_[top_i],
                    "status": "routed" if conf >= self.threshold else "needs_question",
                    "confidence": round(conf, 3),
                    "runner_up": self.classes_[second_i],
                    "runner_up_p": round(float(P[i, second_i]), 3),
                    "reason": "; ".join(notes),
                }
            )
        return pd.DataFrame(rows, index=df.index)

    def _reasons(self, row, X, order, p, billing_fired: bool) -> list[str]:
        top_i = order[0]
        top = self.classes_[top_i]
        reasons = []

        phrases = self._evidence_phrases(X, top_i, row["clean"])
        if phrases:
            quoted = ", ".join(f'"{w}"' for w in phrases)
            reasons.append(f"Words that point to {top}: {quoted}.")
        else:
            reasons.append("The message doesn't describe a specific problem, so the guess is weak.")

        if len(clauses(row["clean"])) > 1:
            reasons.append("Asks about more than one thing - the last thing asked usually decides the team.")

        if billing_fired:
            reasons.append(
                "Mentions a payment, but the problem isn't the payment itself - so not Billing (ops policy §3)."
            )

        said = products_mentioned(row["clean"])
        if said and row["product_family"] not in said and row["product_family"] != "unknown":
            reasons.append(
                f"Check the product: the message talks about {', '.join(s.lower() for s in said)}, "
                f"but the record says {row['product_family']}."
            )

        w = WARRANTY_WORDS.get(row["warranty_status"])
        if w and top in ("Warranty Claims", "Repairs"):
            reasons.append(f"Product is {w}.")

        second = order[1]
        reasons.append(f"Next most likely: {self.classes_[second]} ({p[second]:.0%}).")
        return reasons

    def _evidence_phrases(self, X, class_i: int, clean: str, k: int = 3) -> list[str]:
        """The word/phrase features in this message that pushed hardest towards the chosen team."""
        row = X[0].tocoo()
        contrib = []
        for j, v in zip(row.col, row.data):
            name = self.feature_names_[j]
            if not name.startswith("word__"):
                continue
            phrase = name[len("word__"):]
            words = phrase.split()
            # only quote what the customer wrote, and not phrases that start or end on filler
            if words[0] in FILLER or words[-1] in FILLER or words[0].startswith("l_") or words[-1].startswith("l_"):
                continue
            if not re.search(r"\b" + re.escape(phrase) + r"\b", clean):
                continue
            score = v * self.coef_[class_i, j]
            if score > 0:
                contrib.append((score, phrase))
        contrib.sort(reverse=True)
        chosen: list[str] = []
        for _, phrase in contrib:
            # skip a word already covered by a longer chosen phrase, and vice versa
            if any(re.search(rf"\b{re.escape(phrase)}\b", c) or re.search(rf"\b{re.escape(c)}\b", phrase) for c in chosen):
                continue
            chosen.append(phrase)
            if len(chosen) == k:
                break
        return chosen
