#!/usr/bin/env python3
"""Step 10-13: Build MA/z-score signals and run 1-unit long/short backtests for common pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_signals(metric: pd.Series, entry: float = 1.25, exit_band: float = 1.0) -> pd.DataFrame:
    ma5 = metric.rolling(5).mean()
    ma20 = metric.rolling(20).mean()
    std20 = metric.rolling(20).std(ddof=0)
    z = (ma5 - ma20) / std20

    pos = pd.Series(0, index=metric.index, dtype=int)
    current = 0
    for i, val in enumerate(z):
        if np.isnan(val):
            pos.iat[i] = current
            continue

        if current == 0:
            if val > entry:
                current = -1  # short metric
            elif val < -entry:
                current = 1  # long metric
        else:
            if -exit_band <= val <= exit_band:
                current = 0
        pos.iat[i] = current

    return pd.DataFrame({"metric": metric, "ma5": ma5, "ma20": ma20, "std20": std20, "zscore": z, "position": pos})


def backtest_pair(price1: pd.Series, price2: pd.Series, position: pd.Series) -> pd.Series:
    ret1 = price1.diff().fillna(0.0)
    ret2 = price2.diff().fillna(0.0)
    held = position.shift(1).fillna(0)

    # held=+1 => long stock1 short stock2 ; held=-1 => short stock1 long stock2
    pnl = held * (ret1 - ret2)
    return pnl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--close-file", type=Path, default=Path("data/adj_close_wide.parquet"))
    parser.add_argument("--stationarity-file", type=Path, default=Path("results/common_pairs_stationarity.csv"))
    parser.add_argument("--initial-capital", type=float, default=10_000_000_000.0)
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    close = pd.read_parquet(args.close_file).sort_index()
    stationarity = pd.read_csv(args.stationarity_file)
    selected = stationarity[stationarity["preferred_metric"].isin(["spread", "ratio"])]

    portfolio_daily_pnl = pd.Series(0.0, index=close.index)
    summary_rows = []

    for _, row in selected.iterrows():
        s1, s2, metric_name = row["stock_1"], row["stock_2"], row["preferred_metric"]
        if s1 not in close.columns or s2 not in close.columns:
            continue

        pair = close[[s1, s2]].dropna()
        if len(pair) < 30:
            continue

        metric = pair[s1] - pair[s2] if metric_name == "spread" else pair[s1] / pair[s2]
        sig = build_signals(metric)
        pnl = backtest_pair(pair[s1], pair[s2], sig["position"])

        pair_equity = args.initial_capital + pnl.cumsum()
        total_pnl = float(pnl.sum())
        n_trades = int((sig["position"].diff().abs() > 0).sum())

        summary_rows.append(
            {
                "stock_1": s1,
                "stock_2": s2,
                "metric_used": metric_name,
                "total_pnl": total_pnl,
                "final_equity": float(pair_equity.iloc[-1]),
                "trade_switches": n_trades,
            }
        )

        aligned = pnl.reindex(portfolio_daily_pnl.index).fillna(0.0)
        portfolio_daily_pnl = portfolio_daily_pnl.add(aligned, fill_value=0.0)

        debug = pd.concat([pair, sig, pnl.rename("daily_pnl")], axis=1)
        debug.to_parquet(args.outdir / f"signals_{s1}_{s2}_{metric_name}.parquet")

    summary = pd.DataFrame(summary_rows).sort_values("total_pnl", ascending=False)
    summary.to_csv(args.outdir / "pair_backtest_summary.csv", index=False)
    equity_curve = args.initial_capital + portfolio_daily_pnl.cumsum()
    equity_curve.to_frame("portfolio_equity").to_csv(args.outdir / "portfolio_equity_curve.csv")

    print(f"Pairs traded: {len(summary)}")
    print(f"Saved {args.outdir / 'pair_backtest_summary.csv'}")
    print(f"Saved {args.outdir / 'portfolio_equity_curve.csv'}")


if __name__ == "__main__":
    main()
