#!/usr/bin/env python3
"""Step 5-9: Intersect pair lists, compute spread/ratio, run ADF tests, select preferred metric."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series, min_obs: int = 100) -> tuple[float | None, float | None, bool]:
    s = series.dropna()
    if len(s) < min_obs:
        return None, None, False
    stat, pval, *_ = adfuller(s, autolag="AIC")
    return float(stat), float(pval), bool(pval < 0.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--close-file", type=Path, default=Path("data/adj_close_wide.parquet"))
    parser.add_argument("--corr-file", type=Path, default=Path("results/top100_corr_pairs.csv"))
    parser.add_argument("--chi-file", type=Path, default=Path("results/top100_chi_square_pairs.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    close = pd.read_parquet(args.close_file).sort_index()
    corr_top = pd.read_csv(args.corr_file)
    chi_top = pd.read_csv(args.chi_file)

    corr_set = set(zip(corr_top["stock_1"], corr_top["stock_2"]))
    chi_set = set(zip(chi_top["stock_1"], chi_top["stock_2"]))
    common = sorted(corr_set & chi_set)

    print(f"Common pairs between top-correlation and top-chi-square: {len(common)}")

    rows = []
    for s1, s2 in common:
        if s1 not in close.columns or s2 not in close.columns:
            continue
        pair = close[[s1, s2]].dropna()
        if pair.empty:
            continue

        spread = pair[s1] - pair[s2]
        ratio = pair[s1] / pair[s2]

        sp_stat, sp_p, sp_stationary = adf_test(spread)
        rt_stat, rt_p, rt_stationary = adf_test(ratio)

        preferred = "none"
        if sp_stationary and rt_stationary:
            preferred = "spread" if (sp_p is not None and rt_p is not None and sp_p <= rt_p) else "ratio"
        elif sp_stationary:
            preferred = "spread"
        elif rt_stationary:
            preferred = "ratio"

        rows.append(
            {
                "stock_1": s1,
                "stock_2": s2,
                "spread_adf_stat": sp_stat,
                "spread_adf_p": sp_p,
                "spread_stationary": sp_stationary,
                "ratio_adf_stat": rt_stat,
                "ratio_adf_p": rt_p,
                "ratio_stationary": rt_stationary,
                "preferred_metric": preferred,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(args.outdir / "common_pairs_stationarity.csv", index=False)

    if not out.empty:
        spread_cnt = int(out["spread_stationary"].sum())
        ratio_cnt = int(out["ratio_stationary"].sum())
        both_cnt = int((out["spread_stationary"] & out["ratio_stationary"]).sum())
    else:
        spread_cnt = ratio_cnt = both_cnt = 0

    print(f"Stationary spreads: {spread_cnt}")
    print(f"Stationary ratios: {ratio_cnt}")
    print(f"Pairs passing both: {both_cnt}")
    print(f"Saved {args.outdir / 'common_pairs_stationarity.csv'}")


if __name__ == "__main__":
    main()
