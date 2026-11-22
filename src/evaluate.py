"""Scoring predictions on Spider.

Per example we compute:
  * exec_match  - execution accuracy (PRIMARY): run predicted and gold SQL on the database and
                  compare the results. Invalid SQL or a timeout counts as wrong.
  * exact_match - official Spider exact-set-match (compares SQL components such as SELECT columns,
                  WHERE conditions, GROUP BY..., ignoring literal values). Uses vendored official code.
  * valid_sql   - the prediction executes without error.
  * hardness    - official Spider difficulty of the GOLD query: easy / medium / hard / extra.

Result comparison rules for exec_match (documented in docs/DECISIONS.md):
  * rows are compared as multisets (order ignored) unless the gold query has ORDER BY;
  * column order may differ (e.g. "SELECT name, age" vs "SELECT age, name") as long as
    some permutation of the predicted columns matches the gold columns;
  * floats are rounded to 6 decimals before comparing.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from pathlib import Path

from .data import db_path
from .execution import execute_sql
from .spider_eval import evaluation as E
from .spider_eval.process_sql import Schema, get_schema, get_sql

LEVELS = ["easy", "medium", "hard", "extra"]


# ---------------------------------------------------------------------------------------------
# Execution accuracy
# ---------------------------------------------------------------------------------------------
def _norm(v):
    if isinstance(v, float):
        return round(v, 6)
    return v


def _rows(rows, perm=None):
    if perm is None:
        return [tuple(_norm(v) for v in r) for r in rows]
    return [tuple(_norm(r[i]) for i in perm) for r in rows]


def results_match(pred_rows, gold_rows, order_matters: bool) -> bool:
    if len(pred_rows) != len(gold_rows):
        return False
    if not gold_rows:
        return True  # both empty
    n_pred, n_gold = len(pred_rows[0]), len(gold_rows[0])
    if n_pred != n_gold:
        return False
    gold = _rows(gold_rows)
    gold_key = gold if order_matters else Counter(gold)
    # Try the identity first, then other column permutations (only for few columns: k! grows fast).
    perms = itertools.permutations(range(n_pred)) if n_pred <= 6 else [tuple(range(n_pred))]
    for perm in perms:
        pred = _rows(pred_rows, perm)
        if (pred if order_matters else Counter(pred)) == gold_key:
            return True
    return False


def has_order_by(sql: str) -> bool:
    return "order by" in " ".join(sql.lower().split())


# ---------------------------------------------------------------------------------------------
# Exact set match (official Spider logic, same steps as evaluation.evaluate())
# ---------------------------------------------------------------------------------------------
def _official_rebuild(sql_dict, schema, kmap):
    valid_col_units = E.build_valid_col_units(sql_dict["from"]["table_units"], schema)
    sql_dict = E.rebuild_sql_val(sql_dict)
    return E.rebuild_sql_col(valid_col_units, sql_dict, kmap)


class SpiderScorer:
    """Scores predictions against gold SQL. Caches schemas and gold results per example."""

    def __init__(self, spider_root: str | Path, timeout_s: float = 30.0):
        self.root = Path(spider_root)
        self.timeout_s = timeout_s
        # Foreign-key maps from tables.json: the official EM treats joined key columns as equivalent.
        self.kmaps = E.build_foreign_key_map_from_json(str(self.root / "tables.json"))
        self._schemas: dict[str, Schema] = {}
        self.evaluator = E.Evaluator()

    def _schema(self, db_id: str) -> Schema:
        if db_id not in self._schemas:
            self._schemas[db_id] = Schema(get_schema(str(db_path(self.root, db_id))))
        return self._schemas[db_id]

    def hardness(self, db_id: str, gold_sql: str) -> str:
        return self.evaluator.eval_hardness(get_sql(self._schema(db_id), gold_sql))

    def exact_match(self, db_id: str, pred_sql: str, gold_sql: str) -> int:
        schema, kmap = self._schema(db_id), self.kmaps[db_id]
        g = _official_rebuild(get_sql(schema, gold_sql), schema, kmap)
        try:
            p = _official_rebuild(get_sql(schema, pred_sql), schema, kmap)
        except Exception:
            return 0  # the official parser can't parse it (invalid SQL or unknown column/table)
        try:
            return int(bool(self.evaluator.eval_exact_match(p, g)))
        except Exception:
            return 0

    def score(self, example: dict, pred_sql: str) -> dict:
        """example needs db_id and query (gold). Returns a dict of per-example results."""
        db_id, gold = example["db_id"], example["query"]
        dbf = db_path(self.root, db_id)
        _, gold_rows, gold_err = execute_sql(dbf, gold, self.timeout_s)
        _, pred_rows, pred_err = execute_sql(dbf, pred_sql, self.timeout_s) if pred_sql else ([], [], "empty")
        exec_match = (
            pred_err is None and gold_err is None
            and results_match(pred_rows, gold_rows, has_order_by(gold))
        )
        return {
            "hardness": self.hardness(db_id, gold),
            "exec_match": int(exec_match),
            "exact_match": self.exact_match(db_id, pred_sql, gold) if pred_sql else 0,
            "valid_sql": int(pred_err is None),
            "pred_error": pred_err or "",
            "gold_error": gold_err or "",
        }


# ---------------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------------
def aggregate(per_example: list[dict]) -> list[dict]:
    """Overall + per-hardness metrics (percentages). Input: rows with hardness/exec_match/... ."""
    out = []
    for level in LEVELS + ["all"]:
        rows = [r for r in per_example if level == "all" or r["hardness"] == level]
        n = len(rows)

        def pct(key):
            return round(100.0 * sum(r[key] for r in rows) / n, 2) if n else float("nan")

        out.append({
            "level": level, "count": n,
            "exec_acc": pct("exec_match"), "exact_match": pct("exact_match"), "valid_sql": pct("valid_sql"),
        })
    return out


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))
