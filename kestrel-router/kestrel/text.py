"""Text cleaning and domain constants shared by training and the live service."""
import re
import unicodedata

# Current team names (after the 15 Jan 2026 rename, policy §5).
TEAMS = [
    "Repairs",
    "Installs & Demo",
    "Filters & Consumables",
    "Billing",
    "Returns & Replacement",
    "Warranty Claims",
    "Product Advice",
]
RENAMES = {"Installations": "Installs & Demo", "Consumables": "Filters & Consumables"}

# Plain-language wording used when the service has to ask one question.
TEAM_QUESTION = {
    "Repairs": "It's not working, shows an error, leaks or makes noise",
    "Installs & Demo": "Installation, demo or wall-mounting visit",
    "Filters & Consumables": "Needs a filter, spare part or AMC kit",
    "Billing": "A problem with a payment, invoice, EMI or refund",
    "Returns & Replacement": "Delivery was damaged, wrong or incomplete, or wants to return it",
    "Warranty Claims": "Warranty or Kestrel Shield registration, coverage or claim status",
    "Product Advice": "How to use it, or which model to choose - nothing is broken",
}

PRODUCT_WORDS = {
    "Water Purifier": ["purifier"],
    "Air Fryer": ["fryer"],
    "Mixer Grinder": ["mixer", "grinder"],
    "Induction Cooktop": ["cooktop", "induction"],
    "Room Heater": ["heater"],
    "Ceiling Fan": ["fan"],
    "Robot Vacuum": ["vacuum"],
}

# Policy §3: Billing only when the problem IS the payment.
BILLING_PROBLEM = re.compile(
    r"invoice|gst|double|twice|refund|emi|coupon|charged|deducted|payment (?:failed|deducted|not)|bill\b|receipt"
)
PAID_MENTION = re.compile(r"\bpaid\b|\bpayment done\b")


def normalize_team(name: str) -> str:
    return RENAMES.get(name, name)


def fix_mojibake(s: str) -> str:
    """Undo UTF-8 text that was decoded as cp1252 (possibly more than once) in the Zoho export."""
    for _ in range(3):
        try:
            fixed = s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if fixed == s:
            break
        s = fixed
    return s


GREETING = re.compile(
    r"^(hi|hello|hello team|dear team|sir|madam|namaste|good morning|good evening|urgent|pls help|please help)$"
)


def clean_text(s: str) -> str:
    """Fix encoding, fold accents, lowercase. Punctuation is kept: commas separate the things a customer asks for."""
    s = fix_mojibake(str(s or ""))
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"\bsr\d+\b", "regno", s)      # registration / request numbers carry no meaning
    s = re.sub(r"\bko\d+\b", "orderno", s)    # order numbers: keep the fact, drop the digits
    return re.sub(r"\s+", " ", s).strip()


def clauses(clean: str) -> list[str]:
    """Split a message into the separate things it asks about, dropping greetings."""
    parts = (p.strip(" .!?") for p in re.split(r",|\s-\s|:", clean))
    return [p for p in parts if p and not GREETING.match(p)]


def model_document(clean: str, product: str, warranty: str, channel: str) -> str:
    """What the model reads: the message, the record's fields, and the last thing asked about tagged separately.

    In the history, when a customer asks about two things the team that closes the request is usually the one
    for the last thing asked - so the last clause gets its own features.
    """
    parts = clauses(clean)
    last = " ".join("l_" + w for w in re.findall(r"[a-z]+", parts[-1])) if len(parts) > 1 else ""
    return f"{clean} | {product} | {warranty} | {channel} {last}".lower()


def products_mentioned(clean: str) -> list[str]:
    return [p for p, words in PRODUCT_WORDS.items() if any(re.search(rf"\b{w}\b", clean) for w in words)]
