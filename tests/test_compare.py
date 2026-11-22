"""CPU tests for the base-vs-fine-tuned comparison and statistics."""
import pandas as pd

from src.compare import bootstrap_ci, comparison_table, mcnemar_exact, paired_stats


def _preds(exec_vals, levels):
    n = len(exec_vals)
    return pd.DataFrame({"id": range(n), "hardness": levels, "exec_match": exec_vals,
                         "exact_match": exec_vals, "valid_sql": [1] * n})


def test_identical_models_have_zero_delta():
    p = _preds([1, 0, 1, 1], ["easy", "easy", "hard", "extra"])
    t = comparison_table(p, p)
    assert (t[t["count"] > 0].filter(like="_delta") == 0).all().all()  # levels with examples
    s = {r["level"]: r for r in paired_stats(p, p)}["all"]
    assert s["wrong_to_correct"] == s["correct_to_wrong"] == 0 and s["mcnemar_exact_p"] == 1.0


def test_flip_counts_and_mcnemar():
    base = _preds([0, 0, 0, 1, 1], ["easy"] * 5)
    ft = _preds([1, 1, 0, 1, 0], ["easy"] * 5)
    s = paired_stats(base, ft)[-1]
    assert (s["wrong_to_correct"], s["correct_to_wrong"], s["both_correct"], s["both_wrong"]) == (2, 1, 1, 1)
    assert s["delta_pp"] == 20.0
    # exact McNemar: 10 vs 0 discordant pairs -> p = 2 * 0.5**10
    assert abs(mcnemar_exact(0, 10) - 2 * 0.5 ** 10) < 1e-12


def test_bootstrap_ci_contains_true_difference():
    lo, hi = bootstrap_ci([0] * 50 + [1] * 50, [1] * 70 + [0] * 30)
    assert lo < 20 < hi
