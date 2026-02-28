#!/usr/bin/env python3
"""Step 4: Chi-square independence tests for all pairs using the up/down column."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency


def pair_chi_square(x: pd.Series, y: pd.Series) -> tuple[float, float] | None:
    pair = pd.concat([x, y], axis=1).dropna()
    if pair.empty:
        return None
    table = pd.crosstab(pair.iloc[:, 0].astype(int), pair.iloc[:, 1].astype(int))
    if table.shape != (2, 2):
        return None
    chi2, pval, _, _ = chi2_contingency(table)
    return float(chi2), float(pval)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--up-file", type=Path, default=Path("data/up_wide.parquet"))
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    up = pd.read_parquet(args.up_file).sort_index()

    rows = []
    cols = up.columns.tolist()
    for a, b in combinations(cols, 2):
        res = pair_chi_square(up[a], up[b])
        if res is None:
            continue
        chi2, pval = res
        rows.append((a, b, chi2, pval))

    chi_df = pd.DataFrame(rows, columns=["stock_1", "stock_2", "chi2", "p_value"])
    top = chi_df.sort_values("chi2", ascending=False).head(args.top_n).reset_index(drop=True)
    top.to_csv(args.outdir / "top100_chi_square_pairs.csv", index=False)

    print(f"Saved {args.outdir / 'top100_chi_square_pairs.csv'}")


if __name__ == "__main__":
    main()
