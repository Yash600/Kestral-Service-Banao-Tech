"""Kestrel service-request router: one JSON endpoint and one screen.

    uvicorn app:app --port 8000      then open http://localhost:8000
"""
import io
import json
import logging
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from kestrel.jev_router import JevRouter, JevUnavailable
from kestrel.text import PRODUCT_WORDS, TEAM_QUESTION, TEAMS, normalize_team

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "router.joblib"
METRICS_PATH = ROOT / "evaluation" / "metrics.json"
log = logging.getLogger("kestrel")

app = FastAPI(
    title="Kestrel Home - service request router",
    description="Routes a service request to one of seven teams, with reasons. Runs locally, no paid API.",
    version="1.0",
)

try:
    router = joblib.load(MODEL_PATH)
except FileNotFoundError:
    router = None
    log.warning("No trained model at %s - run `python train.py`. The page still loads.", MODEL_PATH)

jev = JevRouter()  # optional second engine; reports itself unavailable without TYPESAFE_API_KEY

Product = Literal[tuple(PRODUCT_WORDS)]  # type: ignore[valid-type]
Team = Literal[tuple(TEAMS)]  # type: ignore[valid-type]


class RouteRequest(BaseModel):
    request_text: str = Field(..., min_length=2, max_length=4000, examples=["display of air fryer gone blank pls call back"])
    product_family: Optional[Product] = Field(None, examples=["Air Fryer"])
    warranty_status: Optional[Literal["in_warranty", "shield", "out_of_warranty"]] = Field(None, examples=["in_warranty"])
    channel: Optional[Literal["ivr", "chat", "whatsapp", "email"]] = Field(None, examples=["chat"])
    request_id: Optional[str] = Field(None, max_length=40)
    clarification: Optional[Team] = Field(
        None, description="The team chosen from the follow-up question, when the first answer was needs_clarification."
    )
    engine: Literal["ml", "jev"] = Field(
        "ml", description="ml = the local model (default, free, offline). jev = TypeSafe's Jev API (needs TYPESAFE_API_KEY)."
    )


@app.post("/route")
def route(req: RouteRequest):
    record = req.model_dump(exclude={"clarification", "engine"})
    if req.engine == "jev":
        try:
            result = jev.route(record, clarification=req.clarification)
        except JevUnavailable as e:
            raise HTTPException(503, str(e))
        return {"request_id": req.request_id, **result}
    if router is None:
        raise HTTPException(503, "The routing model hasn't been trained yet. Run `python train.py`, then restart.")
    result = router.route(record, clarification=req.clarification)
    return {"request_id": req.request_id, "engine": "ml", **result}


MAX_BATCH_BYTES, MAX_BATCH_ROWS = 20 * 1024 * 1024, 50_000
TRUTH_COLUMNS = ["final_team", "team", "expected_team", "true_team"]


@app.post(
    "/route/batch",
    summary="Route a whole CSV export",
    openapi_extra={"requestBody": {"content": {"text/csv": {"schema": {"type": "string"}}}, "required": True}},
)
async def route_batch(request: Request):
    """Send a CSV file as the request body (Content-Type: text/csv).

    Needs a `request_text` column. `request_id`, `product_family`, `warranty_status` and `channel` are used when
    present. If the file also has the true team (`final_team` or `team`), the summary includes accuracy against it.
    """
    if router is None:
        raise HTTPException(503, "The routing model hasn't been trained yet. Run `python train.py`, then restart.")
    raw = await request.body()
    if not raw.strip():
        raise HTTPException(400, "The file is empty.")
    if len(raw) > MAX_BATCH_BYTES:
        raise HTTPException(413, f"The file is over {MAX_BATCH_BYTES // 2**20} MB. Split it and upload the parts.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except Exception:
        raise HTTPException(400, "That doesn't look like a CSV file. Export it from the CRM as CSV and try again.")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "request_text" not in df:
        found = ", ".join(df.columns[:12]) or "none"
        raise HTTPException(422, f"The file needs a 'request_text' column. Columns found: {found}.")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(413, f"The file has {len(df):,} rows; the limit is {MAX_BATCH_ROWS:,} per upload.")

    # Pull the true team (if any) aside before the model adds its own "team" column.
    truth_col = next((c for c in TRUTH_COLUMNS if c in df and df[c].str.strip().ne("").any()), None)
    df["_truth"] = df.pop(truth_col).str.strip() if truth_col else ""
    df.attrs["truth_col"] = truth_col

    df["request_text"] = df["request_text"].str.strip()
    skipped = int((df["request_text"].str.len() < 2).sum())
    df = df[df["request_text"].str.len() >= 2].reset_index(drop=True)
    if df.empty:
        raise HTTPException(422, "No rows have any request text to route.")
    if "request_id" not in df:
        df["request_id"] = [f"row {i + 2}" for i in range(len(df))]  # spreadsheet row number
    for col in ["product_family", "warranty_status", "channel"]:
        if col not in df:
            df[col] = ""
        df[col] = df[col].replace("", None)

    out = pd.concat([df, router.route_batch(df)], axis=1)
    out.attrs["truth_col"] = truth_col
    return {"summary": batch_summary(out), "rows": batch_rows(out), "skipped_empty": skipped}


def batch_summary(out: pd.DataFrame) -> dict:
    n = len(out)
    routed = out["status"].eq("routed")
    teams = []
    for t in TEAMS:
        g = out[out["team"] == t]
        teams.append({"team": t, "count": int(len(g)), "share": len(g) / n, "needs_question": int((g["status"] != "routed").sum())})

    def breakdown(col):
        res = []
        for key, g in out.groupby(out[col].fillna("unknown")):
            res.append({
                "key": key, "count": int(len(g)),
                "needs_question_share": float((g["status"] != "routed").mean()),
                "top_team": g["team"].value_counts().index[0],
            })
        return sorted(res, key=lambda r: -r["count"])

    bands = [(0.9, 1.01, "90-100%"), (0.7, 0.9, "70-90%"), (0.5, 0.7, "50-70%"), (0.0, 0.5, "under 50%")]
    summary = {
        "rows": n,
        "routed": int(routed.sum()),
        "needs_question": int((~routed).sum()),
        "avg_confidence": float(out["confidence"].mean()),
        "threshold": router.threshold,
        "teams": teams,
        "by_channel": breakdown("channel"),
        "by_product": breakdown("product_family"),
        "confidence_bands": [
            {"band": label, "count": int(out["confidence"].between(lo, hi, inclusive="left").sum())} for lo, hi, label in bands
        ],
        "accuracy": None,
    }

    truth_col = out.attrs.get("truth_col")
    if truth_col:
        truth = out["_truth"].map(normalize_team)
        known = truth.isin(TEAMS)
        ok = out["team"].eq(truth) & known
        acc = {
            "column": truth_col,
            "rows_with_truth": int(known.sum()),
            "overall": float(ok[known].mean()) if known.any() else None,
            "when_routed": float(ok[known & routed].mean()) if (known & routed).any() else None,
            "when_asked": float(ok[known & ~routed].mean()) if (known & ~routed).any() else None,
            "per_team": [
                {"team": t, "rows": int((truth == t).sum()), "caught": float(ok[truth == t].mean()) if (truth == t).any() else None}
                for t in TEAMS
            ],
        }
        seen = getattr(router, "train_ids_", set())
        acc["rows_used_in_training"] = int(out["request_id"].isin(seen).sum())
        if "team_label" in out:
            bot = out["team_label"].map(normalize_team)
            acc["old_bot"] = float(bot[known].eq(truth[known]).mean())
        summary["accuracy"] = acc
    return summary


def batch_rows(out: pd.DataFrame) -> list[dict]:
    cols = ["request_id", "channel", "product_family", "warranty_status", "request_text",
            "team", "status", "confidence", "runner_up", "runner_up_p", "reason"]
    rows = out[cols].copy()
    for c in ["channel", "product_family", "warranty_status"]:
        rows[c] = rows[c].fillna("")
    if out.attrs.get("truth_col"):
        rows["expected"] = out["_truth"].map(normalize_team)
    return rows.to_dict("records")


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": router is not None, "jev": JevRouter.status()}


@app.get("/api/meta")
def meta():
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
    return {
        "teams": TEAMS,
        "team_question": TEAM_QUESTION,
        "products": list(PRODUCT_WORDS),
        "threshold": getattr(router, "threshold", None),
        "model_loaded": router is not None,
        "holdout": metrics and metrics["holdout"],
        "jev": JevRouter.status(),
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
