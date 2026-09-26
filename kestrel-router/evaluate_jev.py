"""Score the Jev engine on the same backtest as the local model, and try combining the two.

    set TYPESAFE_API_KEY=...        (PowerShell: $env:TYPESAFE_API_KEY="...")
    python evaluate_jev.py          -> evaluation/jev_report.md, predictions_jev.csv, predictions_combined.csv

Answers are cached in evaluation/jev_cache.jsonl, so a re-run doesn't pay twice. Every request sent goes to
TypeSafe's servers: that is client data leaving Kestrel (ops policy §10), so run this only with Kestrel's OK.
Without a key it explains what's missing and exits cleanly.
"""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kestrel.data import load_test, load_train
from kestrel.jev_router import MODEL, USD_PER_MILLION_INPUT_TOKENS, JevRouter, _questions, _state, ts
from kestrel.router import Router

ROOT = Path(__file__).parent
OUT = ROOT / "evaluation"
CACHE = OUT / "jev_cache.jsonl"
CUT, THRESHOLD, CONCURRENCY = "2026-04-01", 0.7, 16


def load_cache() -> dict:
    if not CACHE.exists():
        return {}
    return {r["request_id"]: r for r in map(json.loads, CACHE.read_text(encoding="utf-8").splitlines())}


async def ask_jev(frame: pd.DataFrame, cache: dict) -> dict:
    todo = frame[~frame.request_id.isin(cache)]
    if todo.empty:
        return cache
    print(f"Asking Jev about {len(todo):,} requests ({len(frame) - len(todo):,} already cached)...")
    sem = asyncio.Semaphore(CONCURRENCY)
    retry = ts.RetryPolicy(max_retries=3, backoff_initial=0.5, backoff_max=4.0, timeout=30.0)
    async with ts.AsyncTypeSafeClient(model=MODEL, retry=retry, timeout=15.0) as client:
        with CACHE.open("a", encoding="utf-8") as f:
            async def one(rec):
                async with sem:
                    try:
                        res = await client.system_one(_state(rec), _questions())
                    except ts.TypeSafeAPIError as e:
                        print(f"  {rec['request_id']}: Jev error {getattr(e, 'status', '?')} - skipped")
                        return
                    row = {"request_id": rec["request_id"], **JevRouter.parse(res)}
                    cache[row["request_id"]] = row
                    f.write(json.dumps(row) + "\n")
            await asyncio.gather(*(one(r) for r in todo.to_dict("records")))
    return cache


def pct(x):
    return f"{x:.1%}"


def main():
    st = JevRouter.status()
    if not st["available"]:
        print(f"Jev evaluation skipped: {st['reason']}")
        print("The local model's evidence is in evaluation/report.md.")
        return 0

    OUT.mkdir(exist_ok=True)
    d = load_train()
    tr, te = d[d.created < CUT], d[d.created >= CUT].copy()
    test = load_test()
    cache = asyncio.run(ask_jev(pd.concat([te, test])[["request_id", "request_text", "product_family", "warranty_status", "channel"]], load_cache()))

    # --- local model on the same backtest
    ml = Router().fit(Router.frame(tr), tr.final_team)
    P_ml, _, _ = ml.predict_proba(Router.frame(te))
    classes = list(ml.classes_)

    te = te[te.request_id.isin(cache)].copy()
    keep = d[d.created >= CUT].request_id.isin(cache).to_numpy()
    P_ml = P_ml[keep]
    P_jev = np.array([[cache[i]["probabilities"].get(c, 0.0) for c in classes] for i in te.request_id])
    y = te.final_team.to_numpy()

    def score(P, name):
        pred, conf = np.array(classes)[P.argmax(1)], P.max(1)
        auto = conf >= THRESHOLD
        return {
            "Engine": name,
            "Every request": pct(np.mean(pred == y)),
            "Routed automatically": pct(auto.mean()),
            "Accuracy when routed automatically": pct(np.mean(pred[auto] == y[auto])) if auto.any() else "-",
        }, pred

    rows, preds = [], {}
    rows.append({"Engine": "Old routing bot", "Every request": pct(te.bot_correct.mean()), "Routed automatically": "-", "Accuracy when routed automatically": "-"})
    for name, P in [("Local model (ML)", P_ml), ("Jev", P_jev)]:
        r, preds[name] = score(P, name)
        rows.append(r)
    # combination: weighted average of the two probability sets, weight chosen on the *training* side is not possible
    # for Jev without paying for the training rows, so we report the plain 50/50 average and say so.
    r, preds["combined"] = score((P_ml + P_jev) / 2, "Combined (average of both)")
    rows.append(r)
    agree = np.mean(preds["Local model (ML)"] == preds["Jev"])
    both_right = np.mean((preds["Local model (ML)"] == y) & (preds["Jev"] == y))
    either_right = np.mean((preds["Local model (ML)"] == y) | (preds["Jev"] == y))

    tokens = sum(cache[i]["input_tokens"] for i in te.request_id)
    per_req_usd = tokens / len(te) * USD_PER_MILLION_INPUT_TOKENS / 1e6
    per_year = len(te) * 4

    # --- predictions for the unlabelled requests
    test = test[test.request_id.isin(cache)].copy()
    full = Router().fit(Router.frame(d), d.final_team)
    Pt_ml, _, _ = full.predict_proba(Router.frame(test))
    Pt_jev = np.array([[cache[i]["probabilities"].get(c, 0.0) for c in classes] for i in test.request_id])
    pd.DataFrame({"request_id": test.request_id, "team": np.array(classes)[Pt_jev.argmax(1)]}).to_csv(ROOT / "predictions_jev.csv", index=False)
    pd.DataFrame({"request_id": test.request_id, "team": np.array(classes)[((Pt_ml + Pt_jev) / 2).argmax(1)]}).to_csv(ROOT / "predictions_combined.csv", index=False)

    table = pd.DataFrame(rows)
    lines = [
        "# Jev vs the local model",
        "",
        f"Same backtest as `report.md`: Apr-Jun 2026, {len(te):,} requests, scored against the team that actually "
        f"resolved each one. Jev model: `{cache[te.request_id.iloc[0]]['model']}`. Jev was not trained on Kestrel's "
        "data; it only got the seven team descriptions and the routing rules from ops policy §3.",
        "",
        "| " + " | ".join(table.columns) + " |",
        "|---|---|---|---|",
        *("| " + " | ".join(map(str, r)) + " |" for r in table.values),
        "",
        f"- The two engines pick the same team on {pct(agree)} of requests. Both are right on {pct(both_right)}, "
        f"and at least one is right on {pct(either_right)}.",
        f"- Jev cost: {tokens:,} input tokens for {len(te):,} requests, about ${per_req_usd * 1e6:.1f} per million requests, "
        f"or about ${per_req_usd * per_year:.4f} a year at Kestrel's volume ({per_year:,} requests).",
        "- The local model costs nothing to run and never sends data outside Kestrel.",
        "",
        "Files: `predictions_jev.csv` (Jev alone), `predictions_combined.csv` (average of both). The submitted "
        "`predictions.csv` stays with whichever engine scores best above.",
    ]
    (OUT / "jev_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
