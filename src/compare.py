"""Base vs fine-tuned comparison tables and paired statistics (CPU only).

    python -m src.compare results/baseline results/finetuned results/
writes results/comparison.csv, results/flips.csv and results/statistics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

LEVELS = ["easy", "medium", "hard", "extra", "all"]
METRICS = ["exec_acc", "exact_match", "valid_sql"]


def load_predictions(folder) -> pd.DataFrame:
    return pd.read_csv(Path(folder) / "predictions.csv", keep_default_na=False)


def comparison_table(base: pd.DataFrame, ft: pd.DataFrame) -> pd.DataFrame:
    """One row per level: count + base / fine-tuned / delta for each metric (percent)."""
    cols = {"exec_acc": "exec_match", "exact_match": "exact_match", "valid_sql": "valid_sql"}
    rows = []
    for level in LEVELS:
        b = base if level == "all" else base[base.hardness == level]
        f = ft if level == "all" else ft[ft.hardness == level]
        row = {"level": level, "count": len(b)}
        for m, c in cols.items():
            bv, fv = 100 * b[c].mean(), 100 * f[c].mean()
            row[f"{m}_base"], row[f"{m}_ft"], row[f"{m}_delta"] = round(bv, 2), round(fv, 2), round(fv - bv, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def mcnemar_exact(b_only: int, f_only: int) -> float:
    """Exact McNemar test = two-sided binomial test on the discordant pairs (p = 0.5)."""
    n = b_only + f_only
    return 1.0 if n == 0 else binomtest(f_only, n, 0.5).pvalue


def bootstrap_ci(base_correct, ft_correct, n_boot=10_000, seed=42):
    """95% paired bootstrap CI for the accuracy difference (FT - base), in percentage points."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(ft_correct, float) - np.asarray(base_correct, float)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1) * 100
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_stats(base: pd.DataFrame, ft: pd.DataFrame, col="exec_match") -> list[dict]:
    assert (base["id"].values == ft["id"].values).all(), "predictions are not aligned"
    out = []
    for level in LEVELS:
        mask = np.ones(len(base), bool) if level == "all" else (base.hardness == level).values
        b, f = base[col].values[mask], ft[col].values[mask]
        w2c, c2w = int(((b == 0) & (f == 1)).sum()), int(((b == 1) & (f == 0)).sum())
        lo, hi = bootstrap_ci(b, f)
        out.append({
            "level": level, "n": int(mask.sum()),
            "both_correct": int(((b == 1) & (f == 1)).sum()), "both_wrong": int(((b == 0) & (f == 0)).sum()),
            "wrong_to_correct": w2c, "correct_to_wrong": c2w, "net_gain": w2c - c2w,
            "mcnemar_exact_p": mcnemar_exact(c2w, w2c),
            "delta_pp": round(100 * (f.mean() - b.mean()), 2), "ci95_low_pp": round(lo, 2), "ci95_high_pp": round(hi, 2),
        })
    return out


def flips_table(base: pd.DataFrame, ft: pd.DataFrame) -> pd.DataFrame:
    df = base[["id", "db_id", "hardness", "question", "gold_sql"]].copy()
    df["base_sql"], df["ft_sql"] = base.pred_sql.values, ft.pred_sql.values
    df["base_exec"], df["ft_exec"] = base.exec_match.values, ft.exec_match.values
    df["base_error"], df["ft_error"] = base.pred_error.values, ft.pred_error.values
    df["flip"] = np.select(
        [(df.base_exec == 0) & (df.ft_exec == 1), (df.base_exec == 1) & (df.ft_exec == 0),
         (df.base_exec == 1) & (df.ft_exec == 1)],
        ["wrong_to_correct", "correct_to_wrong", "both_correct"], "both_wrong")
    return df


def main(base_dir, ft_dir, out_dir):
    base, ft = load_predictions(base_dir), load_predictions(ft_dir)
    out = Path(out_dir)
    comparison_table(base, ft).to_csv(out / "comparison.csv", index=False)
    flips_table(base, ft).to_csv(out / "flips.csv", index=False)
    stats = {"exec_match": paired_stats(base, ft, "exec_match"), "exact_match": paired_stats(base, ft, "exact_match")}
    (out / "statistics.json").write_text(json.dumps(stats, indent=2))
    print(pd.read_csv(out / "comparison.csv").to_string(index=False))
    print(pd.DataFrame(stats["exec_match"]).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
