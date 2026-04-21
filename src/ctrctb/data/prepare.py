from __future__ import annotations

from pathlib import Path
import pandas as pd

from ..utils.io import ensure_dir


def price_panel_from_long(df_long: pd.DataFrame) -> pd.DataFrame:
    panel = df_long.pivot(index='date', columns='ticker', values='adj_close').sort_index()
    panel = panel.dropna(how='all')
    panel = panel.ffill()
    return panel


def compute_daily_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    returns = price_panel.pct_change().dropna(how='all')
    return returns.dropna(axis=0, how='any')


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).max().tolist())


def save_processed_data(price_panel: pd.DataFrame, returns: pd.DataFrame, processed_dir: str | Path) -> None:
    processed_dir = ensure_dir(processed_dir)
    price_panel.to_csv(Path(processed_dir) / 'price_panel.csv')
    returns.to_csv(Path(processed_dir) / 'returns_panel.csv')
