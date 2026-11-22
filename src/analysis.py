"""Phase 4 analysis: charts + automatic error-category pre-labels (CPU only).

    python -m src.analysis results/baseline results/finetuned results/training results/

Writes into results/: exec_acc_by_level.png, exact_match_by_level.png, valid_sql_rate.png,
loss_curve.png (copied/re-drawn from training logs), error_types.csv (category counts base vs FT,
all failures) and error_sample.csv (50 FT failures, stratified by level, with pre-labels for
manual review).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

LEVELS = ["easy", "medium", "hard", "extra", "all"]
# Validated categorical palette (dataviz reference palette, slots 1-2): base = blue, fine-tuned = orange.
C_BASE, C_FT = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

CATEGORIES = [
    "invalid SQL", "nested query / set operation", "wrong table", "missing / incorrect JOIN",
    "GROUP BY / aggregation", "wrong column", "ORDER BY / LIMIT", "wrong condition / value", "other",
]


# ---------------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------------
def _bar_chart(table: pd.DataFrame, metric: str, title: str, ylabel: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    t = table.set_index("level").loc[LEVELS]
    x = np.arange(len(LEVELS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for off, col, color, name in [(-w / 2, f"{metric}_base", C_BASE, "Base model"),
                                  (w / 2, f"{metric}_ft", C_FT, "Fine-tuned (LoRA)")]:
        # 2 px gap between neighbouring bars: shrink the width slightly and draw a white edge
        bars = ax.bar(x + off, t[col], w * 0.96, color=color, label=name, edgecolor="white", linewidth=1)
        for b, v in zip(bars, t[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"{lv}\n(n={n})" for lv, n in zip(LEVELS, t["count"])], color=INK2)
    ax.set_ylim(0, 105)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.legend(frameon=False, loc="upper right", ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_charts(comparison: pd.DataFrame, out: Path):
    _bar_chart(comparison, "exec_acc", "Execution accuracy on Spider dev (1,034 held-out questions)",
               "execution accuracy (%)", out / "exec_acc_by_level.png")
    _bar_chart(comparison, "exact_match", "Exact set match (official Spider metric) on Spider dev",
               "exact match (%)", out / "exact_match_by_level.png")
    _bar_chart(comparison, "valid_sql", "Valid-SQL rate: prediction executes without error",
               "valid SQL (%)", out / "valid_sql_rate.png")


# ---------------------------------------------------------------------------------------------
# Error categories (automatic pre-label; reviewed by hand for the 50-example sample)
# ---------------------------------------------------------------------------------------------
def _structure(sql: str):
    """Coarse structural features of a query, via sqlglot (SQLite dialect)."""
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(sql, read="sqlite")
    tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    columns = {c.name.lower() for c in tree.find_all(exp.Column)}
    aggs = sorted(type(a).__name__ for a in tree.find_all(exp.AggFunc))
    return {
        "tables": tables, "columns": columns,
        "joins": len(list(tree.find_all(exp.Join))),
        "subqueries": len(list(tree.find_all(exp.Subquery))) + sum(
            1 for s in tree.find_all(exp.Select) if s is not tree and not isinstance(s.parent, exp.Subquery)),
        "setops": sorted(type(s).__name__ for s in tree.find_all(exp.Union, exp.Intersect, exp.Except)),
        "group": sorted(c.sql().lower() for g in tree.find_all(exp.Group) for c in g.expressions),
        "having": tree.find(exp.Having) is not None,
        "aggs": aggs,
        "order": [o.sql().lower() for o in tree.find_all(exp.Ordered)],
        "limit": tree.find(exp.Limit) is not None,
        "distinct": tree.find(exp.Distinct) is not None,
    }


def categorize(gold: str, pred: str, valid: int) -> str:
    """First matching category in a fixed priority order (structure before details)."""
    if not valid:
        return "invalid SQL"
    try:
        g, p = _structure(gold), _structure(pred)
    except Exception:
        return "other"
    if g["setops"] != p["setops"] or (g["subqueries"] > 0) != (p["subqueries"] > 0):
        return "nested query / set operation"
    if g["tables"] != p["tables"]:
        return "wrong table"
    if g["joins"] != p["joins"]:
        return "missing / incorrect JOIN"
    if g["group"] != p["group"] or g["having"] != p["having"] or g["aggs"] != p["aggs"]:
        return "GROUP BY / aggregation"
    if g["columns"] != p["columns"]:
        return "wrong column"
    if g["order"] != p["order"] or g["limit"] != p["limit"]:
        return "ORDER BY / LIMIT"
    return "wrong condition / value"  # same skeleton: literals, comparison operators, DISTINCT, ...


def error_tables(base: pd.DataFrame, ft: pd.DataFrame, out: Path, n_sample=50, seed=42):
    rows = []
    for name, df in [("base", base), ("finetuned", ft)]:
        wrong = df[df.exec_match == 0]
        cats = [categorize(g, p, v) for g, p, v in zip(wrong.gold_sql, wrong.pred_sql, wrong.valid_sql)]
        counts = pd.Series(cats).value_counts()
        for c in CATEGORIES:
            rows.append({"model": name, "category": c, "count": int(counts.get(c, 0)),
                         "share_of_failures_pct": round(100 * counts.get(c, 0) / max(len(wrong), 1), 1)})
        if name == "finetuned":
            wrong = wrong.assign(auto_category=cats)
    types = pd.DataFrame(rows).pivot(index="category", columns="model", values=["count", "share_of_failures_pct"])
    types = types.loc[CATEGORIES]
    types.columns = [f"{a}_{b}" for a, b in types.columns]
    types.reset_index().to_csv(out / "error_types.csv", index=False)

    # Stratified sample of FT failures (proportional to failures per level), for manual review.
    frac = n_sample / len(wrong)
    parts = [wrong[wrong.hardness == lv] for lv in LEVELS[:4]]
    sample = pd.concat([d.sample(max(1, round(len(d) * frac)), random_state=seed) for d in parts if len(d)])
    # rounding per level can leave us 1-2 short (or over): top up from the remaining failures
    if len(sample) < n_sample:
        rest = wrong.drop(sample.index)
        sample = pd.concat([sample, rest.sample(n_sample - len(sample), random_state=seed)])
    sample = sample.head(n_sample)
    keep = ["id", "db_id", "hardness", "question", "gold_sql", "pred_sql", "pred_error", "auto_category"]
    sample[keep].assign(manual_category="", note="").to_csv(out / "error_sample.csv", index=False)
    return types


def main(base_dir, ft_dir, train_dir, out_dir):
    out = Path(out_dir)
    base = pd.read_csv(Path(base_dir) / "predictions.csv", keep_default_na=False)
    ft = pd.read_csv(Path(ft_dir) / "predictions.csv", keep_default_na=False)
    make_charts(pd.read_csv(out / "comparison.csv"), out)
    hist = Path(train_dir) / "loss_history.csv"
    if hist.exists():
        from .train import plot_loss
        plot_loss(pd.read_csv(hist).astype(object).where(lambda d: d.notna(), None).to_dict("records"),
                  out / "loss_curve.png")
    print(error_tables(base, ft, out).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
