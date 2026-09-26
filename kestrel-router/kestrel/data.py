"""Loading the Kestrel export and fixing the known problems described in the email thread and ops policy."""
from pathlib import Path

import pandas as pd

from .text import normalize_team

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_train(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    train = pd.read_csv(data_dir / "train.csv")
    log = pd.read_csv(data_dir / "resolution_log.csv")
    df = train.merge(log, on="request_id", how="left", validate="one_to_one")

    # Two teams were renamed on 15 Jan 2026 with no change in responsibilities (policy §5).
    for col in ["team_label", "first_team", "final_team"]:
        df[col] = df[col].map(normalize_team)

    df["created"] = pd.to_datetime(df["created_at_ist"])
    df["resolved"] = pd.to_datetime(df["resolved_at"])
    # Zoho resolution events were stored in UTC and never converted (policy §9).
    zoho = df["source"].eq("legacy_zoho")
    df.loc[zoho, "resolved"] += pd.Timedelta(hours=5, minutes=30)
    df["hours_to_resolve"] = (df["resolved"] - df["created"]).dt.total_seconds() / 3600

    df["bot_correct"] = df["team_label"].eq(df["final_team"])
    return df.sort_values("created").reset_index(drop=True)


def load_test(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "test_unlabelled.csv")
