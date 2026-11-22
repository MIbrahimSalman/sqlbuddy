"""The demo must only ever run a single read-only SELECT."""
import pytest

from app.sqlbuddy_core import SAMPLE_DB, UnsafeQuery, check_select_only, run_select


@pytest.mark.parametrize("sql", [
    "SELECT name FROM students", "select count(*) from courses;", "  WITH t AS (SELECT 1) SELECT * FROM t",
    "SELECT name FROM courses WHERE title = 'Drop; Create'",   # keywords/semicolon inside a string are fine
])
def test_allowed(sql):
    check_select_only(sql)


@pytest.mark.parametrize("sql", [
    "", "DELETE FROM students", "DROP TABLE students", "SELECT 1; DROP TABLE students",
    "PRAGMA table_info(students)", "ATTACH DATABASE 'x.db' AS x", "UPDATE grades SET score = 100",
    "WITH t AS (SELECT 1) INSERT INTO students(name) VALUES ('x')", "insert into students values (1)",
])
def test_rejected(sql):
    with pytest.raises(UnsafeQuery):
        check_select_only(sql)


def test_runs_on_sample_db_and_caps_rows():
    cols, rows, truncated = run_select(SAMPLE_DB, "SELECT name FROM students ORDER BY student_id", max_rows=5)
    assert cols == ["name"] and len(rows) == 5 and truncated


def test_authorizer_blocks_even_if_check_bypassed():
    import sqlite3
    from app import sqlbuddy_core as core
    conn = sqlite3.connect(":memory:")
    conn.set_authorizer(core._authorizer)
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("CREATE TABLE x (a INT)")
