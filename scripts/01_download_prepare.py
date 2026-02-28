#!/usr/bin/env python3
"""Step 1-2: Download ASX data (Open/Adj Close) and create daily direction column."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


ASX_LISTED_CSV = "https://www.asx.com.au/asx/research/ASXListedCompanies.csv"


def get_asx_tickers(limit: int | None = None) -> list[str]:
    raw = pd.read_csv(ASX_LISTED_CSV, skiprows=1)
    raw = raw.dropna(subset=["ASX code"])
    codes = raw["ASX code"].astype(str).str.strip().str.upper()
    tickers = [f"{c}.AX" for c in codes if c and c != "N/A"]
    unique_tickers = sorted(set(tickers))
    if limit is not None:
        return unique_tickers[:limit]
    return unique_tickers


def chunked(seq: list[str], size: int) -> list[list[str]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def download_history(tickers: list[str], start: str, end: str, batch_size: int = 200) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for batch in chunked(tickers, batch_size):
        data = yf.download(
            tickers=batch,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            if "Ticker" not in data.columns.names:
                data.columns.names = ["Price", "Ticker"]
            stacked = data.stack(level="Ticker").reset_index()
            stacked = stacked.rename(columns={"level_0": "Date"})
        else:
            stacked = data.reset_index()
            stacked["Ticker"] = batch[0]

        keep = [c for c in ["Date", "Ticker", "Open", "Adj Close"] if c in stacked.columns]
        stacked = stacked[keep].dropna(subset=["Open", "Adj Close"])
        frames.append(stacked)

    if not frames:
        raise RuntimeError("No market data downloaded. Try smaller universe or check network.")

    long_df = pd.concat(frames, ignore_index=True)
    long_df["Date"] = pd.to_datetime(long_df["Date"])  # type: ignore[arg-type]
    long_df = long_df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    long_df["up"] = (long_df["Adj Close"] > long_df["Open"]).astype(int)
    return long_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of tickers.")
    parser.add_argument("--outdir", type=Path, default=Path("data"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    end_dt = datetime.utcnow().date() + timedelta(days=1)
    start_dt = end_dt - timedelta(days=365 * args.years)

    tickers = get_asx_tickers(limit=args.limit)
    print(f"Downloading {len(tickers)} ASX tickers from {start_dt} to {end_dt}...")
    long_df = download_history(tickers=tickers, start=str(start_dt), end=str(end_dt))

    long_df.to_parquet(args.outdir / "asx_ohlc_up_long.parquet", index=False)

    close_wide = long_df.pivot(index="Date", columns="Ticker", values="Adj Close").sort_index()
    up_wide = long_df.pivot(index="Date", columns="Ticker", values="up").sort_index()

    close_wide.to_parquet(args.outdir / "adj_close_wide.parquet")
    up_wide.to_parquet(args.outdir / "up_wide.parquet")

    print("Saved:")
    print(f" - {args.outdir / 'asx_ohlc_up_long.parquet'}")
    print(f" - {args.outdir / 'adj_close_wide.parquet'}")
    print(f" - {args.outdir / 'up_wide.parquet'}")


if __name__ == "__main__":
    main()
