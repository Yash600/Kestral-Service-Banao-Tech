"""Second engine: TypeSafe AI's Jev, a pretrained "System One" decision model reached over an API.

Jev returns a typed choice with probabilities instead of generated text, so it slots in beside the local model
with the same response shape. It needs TYPESAFE_API_KEY. Without the key or the SDK, everything else still
works and this engine reports itself as unavailable.
"""
from __future__ import annotations

import os

from .text import TEAM_QUESTION, TEAMS, clean_text

try:
    import typesafe_sdk as ts
except ImportError:  # optional dependency
    ts = None

USD_PER_MILLION_INPUT_TOKENS = 0.042  # Jev list price; output tokens are free
MODEL = os.environ.get("JEV_MODEL", "jev-latest")

# What each team handles (teams.csv) plus the routing rules from ops policy §3.
TEAM_CRITERIA = {
    "repairs": "Product faults, breakdowns, error codes, noise, leaks, not turning on - anything that needs a technician to fix.",
    "installs": "New-product installation, demo and wall-mounting visits, including installer no-shows.",
    "filters": "Buying or fitting filters, candles, membranes, jars, brushes, blades, AMC kits and other spares. Not faults.",
    "billing": "Invoices, GST, double charges, refunds of payments, EMI conversion, coupons - only when the problem IS the payment. Saying 'I paid' does not make it billing.",
    "returns": "Damaged, wrong, used or incomplete deliveries; returns, cancellations and exchanges within the return window.",
    "warranty": "Warranty and Kestrel Shield registration, coverage questions, extensions, certificates and claim status.",
    "advice": "Pre- and post-purchase questions on how to use, clean or choose a product. Nothing is broken.",
}
SLUG_TO_TEAM = dict(zip(TEAM_CRITERIA, TEAMS))
INSTRUCTIONS = (
    "Which Kestrel Home service team should handle this customer request first? Route by what the customer needs "
    "solved. If the customer asks about several things, the last thing they ask about usually decides the team."
)
QUESTIONS_SPEC = {
    "team": ("choice", INSTRUCTIONS, TEAM_CRITERIA),
    "describes_problem": (
        "noul",
        "The message says what the problem or need is, rather than only asking for a call back or for help.",
        None,
    ),
    "payment_is_problem": (
        "noul",
        "The problem the customer wants solved is a payment itself: an invoice, GST, a double charge, a refund of a "
        "payment, EMI conversion or a coupon.",
        None,
    ),
}


def _questions():
    out = {}
    for key, (kind, instructions, criteria) in QUESTIONS_SPEC.items():
        out[key] = ts.Choice(instructions=instructions, criteria=criteria) if kind == "choice" else ts.Noul(instructions=instructions)
    return out


def _state(record: dict) -> str:
    return (
        f"Customer message: {clean_text(record.get('request_text', ''))}\n"
        f"Product on the record: {record.get('product_family') or 'unknown'}\n"
        f"Cover: {(record.get('warranty_status') or 'unknown').replace('_', ' ')}\n"
        f"Channel: {record.get('channel') or 'unknown'}"
    )


class JevUnavailable(Exception):
    """Raised with a message a Kestrel employee can read."""


class JevRouter:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self._client = None

    # ---------- availability ----------
    @staticmethod
    def status() -> dict:
        if ts is None:
            return {"available": False, "reason": "Jev SDK not installed (pip install typesafe-sdk)."}
        if not os.environ.get("TYPESAFE_API_KEY"):
            return {"available": False, "reason": "No Jev API key. Set TYPESAFE_API_KEY to switch this engine on."}
        return {"available": True, "reason": f"Jev ready ({MODEL})."}

    def _get_client(self):
        st = self.status()
        if not st["available"]:
            raise JevUnavailable(st["reason"])
        if self._client is None:
            self._client = ts.TypeSafeClient(model=MODEL, timeout=15.0)
        return self._client

    # ---------- raw call, shared by the service and the evaluation ----------
    @staticmethod
    def parse(response) -> dict:
        team = response.answers["team"]
        probs = {SLUG_TO_TEAM[k]: float(v) for k, v in team.probabilities.items()}
        return {
            "team": SLUG_TO_TEAM[team.choice],
            "probabilities": probs,
            "jev_confidence": float(team.confidence),
            "describes_problem": float(response.answers["describes_problem"].noul),
            "payment_is_problem": float(response.answers["payment_is_problem"].noul),
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
        }

    def decide(self, record: dict) -> dict:
        client = self._get_client()
        try:
            return self.parse(client.system_one(_state(record), _questions()))
        except ts.TypeSafeAuthenticationError:
            raise JevUnavailable("Jev rejected the API key. Check TYPESAFE_API_KEY.")
        except ts.TypeSafeRateLimitError:
            raise JevUnavailable("Jev is rate-limiting requests right now. The local model still works.")
        except (ts.TypeSafeAPITimeoutError, ts.TypeSafeAPIConnectionError):
            raise JevUnavailable("Couldn't reach Jev (timeout or no connection). The local model still works.")
        except ts.TypeSafeAPIError as e:
            raise JevUnavailable(f"Jev returned an error ({getattr(e, 'status', '?')}). The local model still works.")

    # ---------- same response shape as the local model ----------
    def route(self, record: dict, clarification: str | None = None) -> dict:
        d = self.decide(record)
        ranked = sorted(({"team": t, "p": round(p, 3)} for t, p in d["probabilities"].items()), key=lambda r: -r["p"])
        top, conf = ranked[0]["team"], ranked[0]["p"]
        meta = {
            "engine": "jev",
            "model": d["model"],
            "cost_usd": round(d["input_tokens"] * USD_PER_MILLION_INPUT_TOKENS / 1e6, 8),
        }

        if clarification:
            return {
                "status": "routed", "team": clarification, "confidence": None, "model_guess": top, "ranked": ranked,
                "reasons": [
                    f'Answer to the follow-up question: "{TEAM_QUESTION[clarification]}".',
                    f"Before asking, Jev's best guess was {top} at {conf:.0%}.",
                ],
                **meta,
            }

        reasons = [f"Jev picked {top} with probability {conf:.0%} (Jev's own confidence score: {d['jev_confidence']:.0%})."]
        if d["describes_problem"] < 0.5:
            reasons.append(f"Jev thinks the message doesn't say what the problem is ({d['describes_problem']:.0%} that it does).")
        else:
            reasons.append(f"Jev reads the message as describing a specific need ({d['describes_problem']:.0%}).")
        if top == "Billing" or d["payment_is_problem"] >= 0.5:
            reasons.append(f"Is the payment itself the problem (policy §3)? Jev says {d['payment_is_problem']:.0%}.")
        reasons.append(f"Next most likely: {ranked[1]['team']} ({ranked[1]['p']:.0%}).")

        base = {"confidence": conf, "ranked": ranked, "reasons": reasons, **meta}
        if conf >= self.threshold and d["describes_problem"] >= 0.5:
            return {"status": "routed", "team": top, **base}
        return {
            "status": "needs_clarification",
            "best_guess": top,
            "question": "What is this request about?",
            "options": [
                {"team": r["team"], "label": TEAM_QUESTION[r["team"]], "likely": i < 3} for i, r in enumerate(ranked)
            ],
            **base,
        }
