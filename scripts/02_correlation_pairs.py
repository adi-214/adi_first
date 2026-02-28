#!/usr/bin/env python3
"""Step 3: Correlation matrix on close prices + top 100 pairs."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--close-file", type=Path, default=Path("data/adj_close_wide.parquet"))
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--min-overlap", type=int, default=252)
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    close = pd.read_parquet(args.close_file).sort_index()

    corr = close.corr(min_periods=args.min_overlap)
    corr.to_parquet(args.outdir / "close_corr_matrix.parquet")

    rows = []
    cols = corr.columns.tolist()
    for a, b in combinations(cols, 2):
        val = corr.at[a, b]
        if pd.notna(val):
            rows.append((a, b, float(val), abs(float(val))))

    corr_pairs = pd.DataFrame(rows, columns=["stock_1", "stock_2", "correlation", "abs_correlation"])
    top = corr_pairs.sort_values("abs_correlation", ascending=False).head(args.top_n).reset_index(drop=True)
    top.to_csv(args.outdir / "top100_corr_pairs.csv", index=False)

    print(f"Saved {args.outdir / 'close_corr_matrix.parquet'}")
    print(f"Saved {args.outdir / 'top100_corr_pairs.csv'}")


if __name__ == "__main__":
    main()
