from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd
from ..utils.io import ensure_dir, timestamp_utc


def download_adjusted_close(tickers: Iterable[str], start: str, end: str, raw_dir: str | Path) -> pd.DataFrame:
    import yfinance as yf

    tickers = list(tickers)
    raw_dir = ensure_dir(raw_dir)
    data = yf.download(tickers=tickers, start=start, end=end, auto_adjust=False, progress=False, group_by='ticker')
    frames = []
    ts = timestamp_utc()
    for t in tickers:
        if len(tickers) == 1:
            df_t = data.copy()
        else:
            df_t = data[t].copy()
        df_t = df_t.rename(columns={c: c.lower().replace(' ', '_') for c in df_t.columns})
        if 'adj_close' not in df_t.columns and 'adj close' in df_t.columns:
            df_t['adj_close'] = df_t['adj close']
        df_t = df_t.reset_index().rename(columns={'Date': 'date', 'date': 'date'})
        df_t['ticker'] = t
        df_t['source'] = 'yfinance'
        df_t['download_timestamp'] = ts
        keep = [c for c in ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'source', 'download_timestamp'] if c in df_t.columns]
        out = df_t[keep].copy()
        out.to_csv(raw_dir / f'{t}.csv', index=False)
        frames.append(out[['date', 'ticker', 'adj_close']])
    panel = pd.concat(frames, ignore_index=True)
    return panel


def load_or_download_adjusted_close(tickers: Iterable[str], start: str, end: str, raw_dir: str | Path, force_download: bool = False) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    tickers = list(tickers)
    existing = all((raw_dir / f'{t}.csv').exists() for t in tickers)
    if force_download or not existing:
        return download_adjusted_close(tickers, start, end, raw_dir)
    frames = []
    for t in tickers:
        df = pd.read_csv(raw_dir / f'{t}.csv', parse_dates=['date'])
        df = df[(df['date'] >= start) & (df['date'] <= end)]
        frames.append(df[['date', 'ticker', 'adj_close']])
    return pd.concat(frames, ignore_index=True)
