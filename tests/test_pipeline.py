"""CPU tests for the shared code. Run: SPIDER_DIR=/path/to/spider pytest -q tests/

Tests that need the Spider databases are skipped if SPIDER_DIR is not set.
"""
import os
import sqlite3
from pathlib import Path

import pytest

from src.data import split_train_val
from src.evaluate import results_match
from src.execution import execute_sql
from src.prompts import SYSTEM_PROMPT, build_messages, normalize_gold_sql
from src.schema import get_schema_ddl
from src.sql_extract import extract_sql

SPIDER = os.environ.get("SPIDER_DIR")
needs_spider = pytest.mark.skipif(not SPIDER, reason="SPIDER_DIR not set")


# ---------------------------------------------------------------- SQL extraction
@pytest.mark.parametrize("raw, expected", [
    ("SELECT count(*) FROM singer", "SELECT count(*) FROM singer"),
    ("SELECT count(*) FROM singer;", "SELECT count(*) FROM singer"),
    ("```sql\nSELECT name\nFROM singer;\n```", "SELECT name FROM singer"),
    ("```\nSELECT name FROM singer\n```\nThis query returns names.", "SELECT name FROM singer"),
    ("SQL: SELECT a FROM t", "SELECT a FROM t"),
    ("Here is the query:\n\nSELECT a FROM t WHERE b = 'x;y';\nExplanation...", "SELECT a FROM t WHERE b = 'x;y'"),
    ("SELECT a FROM t\n\nThis selects a.", "SELECT a FROM t"),
    ("SELECT a FROM t; SELECT b FROM t;", "SELECT a FROM t"),
    ("WITH x AS (SELECT 1) SELECT * FROM x", "WITH x AS (SELECT 1) SELECT * FROM x"),
    ("select   a ,\n  b from t", "select a , b from t"),
    ("```sql\nSELECT \"my col\" FROM t\n```", 'SELECT "my col" FROM t'),
    ("I don't know.", "I don't know."),
    ("", ""),
])
def test_extract_sql(raw, expected):
    assert extract_sql(raw) == expected


# ---------------------------------------------------------------- result comparison
def test_results_match_order_and_columns():
    gold = [(1, "a"), (2, "b")]
    assert results_match([(2, "b"), (1, "a")], gold, order_matters=False)
    assert not results_match([(2, "b"), (1, "a")], gold, order_matters=True)
    assert results_match([("a", 1), ("b", 2)], gold, order_matters=True)  # swapped columns
    assert not results_match([(1, "a")], gold, order_matters=False)  # missing row
    assert not results_match([(1, "a"), (1, "a")], [(1, "a")], order_matters=False)  # duplicates count
    assert results_match([], [], order_matters=False)
    assert results_match([(0.1 + 0.2,)], [(0.3,)], order_matters=False)  # float rounding
    assert not results_match([(1,)], [(1, "a")], order_matters=False)


# ---------------------------------------------------------------- prompt
def test_prompt_and_gold_normalisation():
    msgs = build_messages("CREATE TABLE t (a INT);", "  How many? ", "SELECT count(*)  FROM t ;")
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["content"].endswith("Question: How many?")
    assert msgs[2] == {"role": "assistant", "content": "SELECT count(*) FROM t"}
    assert normalize_gold_sql("SELECT a ,  b FROM t") == "SELECT a, b FROM t"


# ---------------------------------------------------------------- schema + read-only execution
@pytest.fixture()
def tiny_db(tmp_path):
    p = tmp_path / "tiny.sqlite"
    con = sqlite3.connect(p)
    con.executescript("""
        CREATE TABLE dept (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE "emp data" (eid INT, dept_id INT, salary REAL,
            PRIMARY KEY (eid), FOREIGN KEY (dept_id) REFERENCES dept(id));
        INSERT INTO dept VALUES (1, 'CS'), (2, 'Math');
        INSERT INTO "emp data" VALUES (10, 1, 100.0), (11, 2, 50.5);
    """)
    con.commit()
    con.close()
    return p


def test_schema_ddl(tiny_db):
    ddl = get_schema_ddl(str(tiny_db))
    assert "CREATE TABLE dept (\n  id INTEGER,\n  name TEXT,\n  PRIMARY KEY (id)\n);" in ddl
    assert 'CREATE TABLE "emp data"' in ddl
    assert "FOREIGN KEY (dept_id) REFERENCES dept(id)" in ddl
    # the DDL must itself be valid SQLite
    sqlite3.connect(":memory:").executescript(ddl)


def test_execution_is_read_only_and_reports_errors(tiny_db):
    cols, rows, err = execute_sql(tiny_db, "SELECT name FROM dept ORDER BY id")
    assert err is None and cols == ["name"] and rows == [("CS",), ("Math",)]
    _, _, err = execute_sql(tiny_db, "DELETE FROM dept")
    assert err and "readonly" in err
    _, _, err = execute_sql(tiny_db, "SELECT nope FROM dept")
    assert err and "no such column" in err


def test_execution_timeout(tiny_db):
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c"
    _, _, err = execute_sql(tiny_db, slow, timeout_s=0.5)
    assert err and "timeout" in err


# ---------------------------------------------------------------- splits
def test_val_split_is_database_disjoint():
    train = [{"db_id": f"db{i % 40}", "question": "q", "query": "s"} for i in range(1000)]
    tr, va, val_dbs = split_train_val(train, val_frac=0.05, seed=42)
    assert len(tr) + len(va) == 1000 and len(va) >= 50
    assert not ({e["db_id"] for e in tr} & {e["db_id"] for e in va})
    assert split_train_val(train, 0.05, 42)[2] == val_dbs  # deterministic


# ---------------------------------------------------------------- Spider-dependent checks
@needs_spider
def test_gold_as_prediction_scores_100_and_hardness_counts():
    from src.data import load_spider
    from src.evaluate import SpiderScorer, aggregate

    dev = load_spider("test")
    sc = SpiderScorer(SPIDER)
    rows = [sc.score(e, normalize_gold_sql(e["query"])) for e in dev]
    m = {r["level"]: r for r in aggregate(rows)}
    assert m["all"]["count"] == 1034
    assert m["all"]["exec_acc"] == m["all"]["exact_match"] == m["all"]["valid_sql"] == 100.0
    # official Spider dev difficulty counts
    assert [m[k]["count"] for k in ["easy", "medium", "hard", "extra"]] == [248, 446, 174, 166]


@needs_spider
def test_wrong_predictions_score_zero():
    from src.evaluate import SpiderScorer

    sc = SpiderScorer(SPIDER)
    ex = {"db_id": "concert_singer", "query": "SELECT count(*) FROM singer"}
    # note: "SELECT count(*) FROM concert" would ALSO return 6 here - an accidental execution match
    assert sc.score(ex, "SELECT count(*) FROM stadium")["exec_match"] == 0
    assert sc.score(ex, "SELECT count(*) FROM concert")["exact_match"] == 0
    r = sc.score(ex, "SELECT count(*) FROM no_such_table")
    assert (r["exec_match"], r["valid_sql"], r["exact_match"]) == (0, 0, 0)
    r = sc.score(ex, "")
    assert (r["exec_match"], r["valid_sql"], r["exact_match"]) == (0, 0, 0)


@needs_spider
def test_no_dev_database_in_train():
    from src.data import load_spider

    assert not ({e["db_id"] for e in load_spider("train")} & {e["db_id"] for e in load_spider("test")})


@needs_spider
def test_every_spider_schema_is_valid_sql():
    for d in sorted(Path(SPIDER, "database").iterdir()):
        f = d / f"{d.name}.sqlite"
        if f.exists():
            ddl = get_schema_ddl(str(f))
            assert ddl.startswith("CREATE TABLE")
            sqlite3.connect(":memory:").executescript(ddl)
