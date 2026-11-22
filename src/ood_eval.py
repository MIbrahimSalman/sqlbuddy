"""Out-of-domain test: 30 hand-written questions on our own university.db (never seen by either model).

    python -m src.ood_eval                      # CPU or GPU, ~10 min on a 4-core CPU
    python -m src.ood_eval --adapter path/to/adapter

Why: Spider dev databases are unseen, but they were written in the same style as the training data.
A database we designed ourselves (schema naming, question wording) checks whether the improvement
carries over to a realistic "new user" database. The questions and gold SQL in
sample_db/university_test.json were fixed *before* either model was run on them.

Same protocol as the main evaluation: the identical prompt (schema as CREATE TABLE + question),
greedy decoding, SQL extraction, execution accuracy (row multiset; order only if gold has ORDER BY),
official Spider hardness. Writes results/ood/{predictions.csv, metrics.csv, statistics.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from .compare import bootstrap_ci, mcnemar_exact  # noqa: E402
from .evaluate import has_order_by, results_match  # noqa: E402
from .execution import execute_sql  # noqa: E402
from .spider_eval.evaluation import Evaluator  # noqa: E402
from .spider_eval.process_sql import Schema, get_schema, get_sql  # noqa: E402

DB = ROOT / "sample_db" / "university.db"
QUESTIONS = ROOT / "sample_db" / "university_test.json"
LEVELS = ["easy", "medium", "hard", "extra", "all"]


def score(pred_sql: str, gold_sql: str) -> tuple[int, int, str]:
    """(exec_match, valid_sql, error) with the same rules as src/evaluate.py."""
    _, gold_rows, gerr = execute_sql(DB, gold_sql)
    assert gerr is None, gerr
    _, pred_rows, perr = execute_sql(DB, pred_sql) if pred_sql else (None, None, "empty prediction")
    if perr is not None:
        return 0, 0, perr
    return int(results_match(pred_rows, gold_rows, has_order_by(gold_sql))), 1, ""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None, help="adapter repo id or folder (default: the app's ADAPTER)")
    ap.add_argument("--out", default=str(ROOT / "results" / "ood"))
    a = ap.parse_args(argv)
    if a.adapter:
        import os
        os.environ["SQLBUDDY_ADAPTER"] = a.adapter
    import sqlbuddy_core as core  # the demo back-end: same prompt, greedy decoding, SQL extraction

    qs = json.loads(QUESTIONS.read_text())
    schema_text = core.schema_of(DB)
    spider_schema, evaluator = Schema(get_schema(str(DB))), Evaluator()
    models = core.Models()
    models.load()
    assert models.has_adapter, f"adapter not loaded: {models.adapter_error}"

    rows = []
    for i, q in enumerate(qs):
        hardness = evaluator.eval_hardness(get_sql(spider_schema, q["gold"]))
        for name, ft in [("base", False), ("finetuned", True)]:
            sql, raw, secs = models.generate(schema_text, q["question"], ft)
            ex, valid, err = score(sql, q["gold"])
            rows.append({"id": f"ood_{i}", "model": name, "hardness": hardness, "question": q["question"],
                         "gold_sql": q["gold"], "pred_sql": sql, "exec_match": ex, "valid_sql": valid,
                         "pred_error": err, "seconds": round(secs, 2), "raw_output": raw})
            print(f"[{i:2d}] {name:9s} {'OK ' if ex else 'BAD'} {secs:5.1f}s  {sql}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "predictions.csv", index=False)

    # metrics per hardness level, base vs fine-tuned
    met = []
    for lv in LEVELS:
        d = df if lv == "all" else df[df.hardness == lv]
        row = {"level": lv, "count": int(len(d) // 2)}
        for name in ["base", "finetuned"]:
            m = d[d.model == name]
            row[f"exec_acc_{name}"] = round(100 * m.exec_match.mean(), 2) if len(m) else float("nan")
            row[f"valid_sql_{name}"] = round(100 * m.valid_sql.mean(), 2) if len(m) else float("nan")
        met.append(row)
    pd.DataFrame(met).to_csv(out / "metrics.csv", index=False)

    b = df[df.model == "base"].exec_match.to_numpy()
    f = df[df.model == "finetuned"].exec_match.to_numpy()
    b_only, f_only = int(((b == 1) & (f == 0)).sum()), int(((b == 0) & (f == 1)).sum())
    lo, hi = bootstrap_ci(b, f)
    stats = {"n": int(len(b)), "exec_acc_base": round(100 * b.mean(), 2), "exec_acc_ft": round(100 * f.mean(), 2),
             "wrong_to_correct": f_only, "correct_to_wrong": b_only,
             "mcnemar_exact_p": mcnemar_exact(b_only, f_only), "ci95_pp": [lo, hi],
             "mean_seconds_base": round(df[df.model == "base"].seconds.mean(), 2),
             "mean_seconds_ft": round(df[df.model == "finetuned"].seconds.mean(), 2),
             "device": models.describe()}
    (out / "statistics.json").write_text(json.dumps(stats, indent=2))
    print(pd.DataFrame(met).to_string(index=False))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
