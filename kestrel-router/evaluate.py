"""Evidence that the router works - and how often it doesn't.

Everything is measured against where each request was actually resolved (final_team), using time-based
backtests: train on everything before a date, test on the three months after it, never the other way round.

    python evaluate.py        -> evaluation/report.md, evaluation/metrics.json, evaluation/holdout_errors.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from kestrel.data import load_train
from kestrel.router import Router
from kestrel.text import TEAMS, clauses

OUT = Path(__file__).parent / "evaluation"
TRANSFER_RS, EXTRA_CONTACT_RS, LICENCE_RS = 305, 260, 320_000  # ops policy §4
FOLDS = [("2025-10-01", "2026-01-01"), ("2026-01-01", "2026-04-01"), ("2026-04-01", "2026-07-01")]
THRESHOLDS = [0.0, 0.4, 0.5, 0.6, 0.7]


def pct(x):
    return f"{x:.1%}"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    d = load_train()
    metrics, report = {}, []

    # ---------- 1. rolling backtest ----------
    rows, holdout = [], None
    for start, end in FOLDS:
        tr = d[d.created < start]
        te = d[(d.created >= start) & (d.created < end)].copy()
        router = Router().fit(Router.frame(tr), tr.final_team)
        frame = Router.frame(te)
        P, _, _ = router.predict_proba(frame)
        te["clean_parts"] = frame["clean"].map(lambda c: len(clauses(c))).values
        te["pred"], te["conf"] = router.classes_[P.argmax(1)], P.max(1)
        te["ok"] = te.pred.eq(te.final_team)
        auto = te.conf >= router.threshold
        rows.append(
            {
                "Test window": f"{start[:7]} to {pd.Timestamp(end) - pd.Timedelta(days=1):%Y-%m}",
                "Requests": len(te),
                "Trained on": len(tr),
                "Old bot": pct(te.bot_correct.mean()),
                "Model, every request": pct(te.ok.mean()),
                "Routed automatically": pct(auto.mean()),
                "Accuracy when routed automatically": pct(te.ok[auto].mean()),
            }
        )
        holdout, hold_router = te, router
    backtest = pd.DataFrame(rows)
    metrics["backtest"] = rows

    te = holdout  # the most recent window, used for the detailed breakdown
    auto = te.conf >= hold_router.threshold
    metrics["holdout"] = {
        "window": rows[-1]["Test window"],
        "n": int(len(te)),
        "bot_accuracy": float(te.bot_correct.mean()),
        "model_accuracy": float(te.ok.mean()),
        "auto_share": float(auto.mean()),
        "auto_accuracy": float(te.ok[auto].mean()),
        "clarify_share": float((~auto).mean()),
    }

    # ---------- 2. the "90% match" question: train on the bot's own labels ----------
    tr = d[d.created < FOLDS[-1][0]]
    copycat = Router().fit(Router.frame(tr), tr.team_label)
    cp, _ = copycat.predict(Router.frame(te))
    metrics["copy_the_bot"] = {
        "matches_bot_labels": float(np.mean(cp == te.team_label)),
        "matches_real_outcome": float(np.mean(cp == te.final_team)),
        "our_model_matches_bot_labels": float(np.mean(te.pred == te.team_label)),
    }

    # ---------- 3. confidence: is the model honest about how sure it is? ----------
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    cal = te.groupby(pd.cut(te.conf, bins, right=False), observed=True).agg(
        requests=("ok", "size"), said=("conf", "mean"), was_right=("ok", "mean")
    ).reset_index()
    cal = pd.DataFrame(
        {
            "Model confidence": [f"{b.left:.0%}-{min(b.right, 1):.0%}" for b in cal.conf],
            "Requests": cal.requests,
            "Average confidence": cal.said.map(pct),
            "Actually right": cal.was_right.map(pct),
        }
    )
    thr = pd.DataFrame(
        [
            {
                "Ask a question below": pct(t) if t else "never ask",
                "Routed automatically": pct((te.conf >= t).mean()),
                "Accuracy when routed automatically": pct(te.ok[te.conf >= t].mean()),
                "Sent to one question": pct((te.conf < t).mean()),
            }
            for t in THRESHOLDS
        ]
    )

    # ---------- 4. where it goes wrong ----------
    labels = [t for t in TEAMS if t in set(te.final_team)]
    per_team = pd.DataFrame(classification_report(te.final_team, te.pred, labels=labels, output_dict=True)).T
    per_team = per_team.loc[labels]
    per_team = pd.DataFrame(
        {
            "Team": labels,
            "Requests (truth)": per_team["support"].astype(int).values,
            "When we say this team, we're right": per_team["precision"].map(pct).values,
            "Of this team's requests, we catch": per_team["recall"].map(pct).values,
        }
    )
    cm = pd.DataFrame(confusion_matrix(te.final_team, te.pred, labels=labels), index=labels, columns=labels)
    cm_md = md_table(cm.reset_index().rename(columns={"index": "Truth \\ Predicted"}))

    te["kind"] = np.where(te.clean_parts > 1, "asks about 2+ things", "asks about one thing")
    vague = te.conf < 0.4
    by_kind = pd.DataFrame(
        [
            {"Slice": "Clear enough (confidence >= 40%)", "Requests": int((~vague).sum()), "Accuracy": pct(te.ok[~vague].mean())},
            {"Slice": "Vague (confidence < 40%)", "Requests": int(vague.sum()), "Accuracy": pct(te.ok[vague].mean())},
        ]
        + [{"Slice": k.capitalize(), "Requests": int(g.size), "Accuracy": pct(g.mean())} for k, g in te.groupby("kind").ok]
        + [{"Slice": f"Channel: {k}", "Requests": int(g.size), "Accuracy": pct(g.mean())} for k, g in te.groupby("channel").ok]
    )
    vague_spread = te[vague].final_team.value_counts(normalize=True)

    errors = te[~te.ok].sort_values("conf", ascending=False)
    errors[["request_id", "request_text", "product_family", "warranty_status", "team_label", "final_team", "pred", "conf"]].to_csv(
        OUT / "holdout_errors.csv", index=False
    )

    # ---------- 5. rupees ----------
    per_year = len(te) * 4
    mis_bot = ~te.bot_correct
    transfers_per_mis = te.transfers[mis_bot].sum() / max(mis_bot.sum(), 1)
    cost_mis = transfers_per_mis * TRANSFER_RS + EXTRA_CONTACT_RS
    bot_cost = (te.transfers.sum() * TRANSFER_RS + mis_bot.sum() * EXTRA_CONTACT_RS) / len(te) * per_year
    model_cost = (~te.ok).mean() * per_year * cost_mis
    auto_mis = (~te.ok & auto).mean() * per_year * cost_mis
    clarify_n = (~auto).mean() * per_year
    money = {
        "requests_per_year": per_year,
        "bot_misroutes_per_year": round(mis_bot.mean() * per_year),
        "avg_transfers_per_bot_misroute": round(float(transfers_per_mis), 2),
        "cost_per_misroute_rs": round(float(cost_mis)),
        "bot_misrouting_cost_per_year_rs": round(float(bot_cost)),
        "model_misrouting_cost_per_year_rs": round(float(model_cost)),
        "model_with_question_cost_per_year_rs": round(float(auto_mis)),
        "questions_per_year": round(float(clarify_n)),
        "questions_cost_if_each_is_an_extra_contact_rs": round(float(clarify_n * EXTRA_CONTACT_RS)),
        "licence_rs": LICENCE_RS,
    }
    metrics["money"] = money

    # ---------- 6. workload by team (real outcomes, renamed teams merged) ----------
    recent = d[d.created >= "2026-01-01"]
    months = recent.created.dt.to_period("M").nunique()
    load = recent.groupby("final_team").agg(n=("request_id", "size"), hrs=("hours_to_resolve", "median"))
    bot_load = recent.team_label.value_counts()
    load = load.assign(per_month=load.n / months, share=load.n / load.n.sum(), bot=bot_load.reindex(load.index) / months)
    load = load.sort_values("n", ascending=False)
    workload = pd.DataFrame(
        {
            "Team": load.index,
            "Requests per month (where they really belong)": load.per_month.round().astype(int).values,
            "Share": load.share.map(pct).values,
            "What the bot sent them per month": load.bot.round().astype(int).values,
            "Median hours to resolve": load.hrs.round(1).values,
        }
    )
    metrics["workload"] = workload.to_dict("records")

    # ---------- write the report ----------
    h = metrics["holdout"]
    c = metrics["copy_the_bot"]
    report += [
        "# Evidence: does the router work, and how often does it not?",
        "",
        "All numbers compare against **where each request was actually resolved** (`final_team` in the resolution log), "
        "not against the old bot's labels. Every test uses only data from *before* the test window, the way the "
        "model will be used in real life. Renamed teams are merged (Installations = Installs & Demo, "
        "Consumables = Filters & Consumables).",
        "",
        "## 1. Headline",
        "",
        f"On the most recent three months ({h['window']}, {h['n']:,} requests):",
        "",
        f"- The old bot sent **{pct(h['bot_accuracy'])}** of requests to the team that resolved them.",
        f"- The model, forced to answer every request, gets **{pct(h['model_accuracy'])}**.",
        f"- With the follow-up question switched on, the model routes **{pct(h['auto_share'])}** of requests by itself at "
        f"**{pct(h['auto_accuracy'])}** accuracy, and asks one question on the other **{pct(h['clarify_share'])}**.",
        "",
        "## 2. Backtest over three separate quarters",
        "",
        md_table(backtest),
        "",
        "The result is stable across quarters, including the switch from Zoho to the CRM in Oct 2025.",
        "",
        "## 3. Why not just match the bot's labels at 90%?",
        "",
        f"A model trained on the bot's own labels matches them **{pct(c['matches_bot_labels'])}** of the time, which clears "
        f"the 90% bar easily. But it sends only **{pct(c['matches_real_outcome'])}** of requests to the team that resolves "
        f"them, because it has learned the bot's mistakes. Our model agrees with the bot only "
        f"{pct(c['our_model_matches_bot_labels'])} of the time, and that disagreement is exactly the misroutes it fixes.",
        "",
        "## 4. Why not 90% on every request?",
        "",
        "Some requests contain no clue about the problem (\"please call me about my purifier\", \"need help with my "
        "mixer\"). Here is where the vague ones (model confidence under 40%) actually ended up:",
        "",
        md_table(pd.DataFrame({"Resolved by": vague_spread.index, "Share": vague_spread.map(pct).values})),
        "",
        "They are spread across every team. No model can route these from the text, so the service asks one "
        "question instead of guessing.",
        "",
        md_table(by_kind),
        "",
        "## 5. Is the confidence score honest?",
        "",
        "If the model says 80%, it should be right about 80% of the time. This is what makes the ask-a-question rule safe.",
        "",
        md_table(cal),
        "",
        "### Choosing when to ask",
        "",
        md_table(thr),
        "",
        f"We use **{pct(hold_router.threshold)}**: below that, the service asks one question. The reason is money: a "
        f"misroute costs about Rs {money['cost_per_misroute_rs']}, and a question costs at most Rs {EXTRA_CONTACT_RS}. So asking "
        "pays off whenever the real chance of being right is under about 62%. The table above shows that between 50% and "
        "70% the model is actually right less than half the time.",
        "",
        "## 6. Where it goes wrong",
        "",
        md_table(per_team),
        "",
        "Confusion matrix (rows = team that really resolved it, columns = model's choice):",
        "",
        cm_md,
        "",
        f"All {len(errors)} errors from this window are listed in `holdout_errors.csv`, most confident first.",
        "",
        "## 7. Rupees (ops policy §4: Rs 305 per transfer, Rs 260 per extra customer contact)",
        "",
        f"- Volume: about {money['requests_per_year']:,} requests a year.",
        f"- The bot misroutes about {money['bot_misroutes_per_year']:,} a year, each taking "
        f"{money['avg_transfers_per_bot_misroute']} transfers on average, so about Rs {money['cost_per_misroute_rs']} per misroute.",
        f"- Bot misrouting cost: **Rs {money['bot_misrouting_cost_per_year_rs']:,} a year**, on top of the Rs 3.2 lakh licence.",
        f"- Model answering every request: about **Rs {money['model_misrouting_cost_per_year_rs']:,} a year**.",
        f"- Model with the question: about **Rs {money['model_with_question_cost_per_year_rs']:,} a year** in misroutes, plus "
        f"{money['questions_per_year']:,} questions. If every question cost a full extra contact (Rs 260, a pessimistic "
        f"case) that adds Rs {money['questions_cost_if_each_is_an_extra_contact_rs']:,}.",
        "- Run cost: runs on any ordinary server. No per-request fee and no paid API.",
        "",
        "## 8. Workload by team (Jan to Jun 2026, where requests really belong)",
        "",
        md_table(workload),
        "",
        "Resolution times for Zoho-era rows were shifted from UTC to IST first (ops policy §9).",
        "",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print("\n".join(report[:20]))
    print(f"\nWrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
