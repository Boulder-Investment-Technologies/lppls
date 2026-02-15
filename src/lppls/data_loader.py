from __future__ import annotations

from importlib.resources import files
import pandas as pd


def nasdaq_dotcom() -> pd.DataFrame:
    """Load the bundled Nasdaq Dot-com bubble dataset."""
    source = files("lppls.data").joinpath("nasdaq_dotcom.csv")
    with source.open("r", encoding="utf-8") as f:
        return pd.read_csv(f)
