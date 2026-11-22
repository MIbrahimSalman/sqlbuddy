"""Safe, read-only SQL execution on SQLite with a time limit."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def connect_readonly(db_file: str | Path) -> sqlite3.Connection:
    """Open a SQLite file in read-only mode (writes raise 'attempt to write a readonly database')."""
    uri = Path(db_file).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    # Some Spider databases contain bytes that are not valid UTF-8; don't crash on them.
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    return conn


def execute_sql(db_file: str | Path, sql: str, timeout_s: float = 30.0, max_rows: int | None = None):
    """Run `sql` and return (columns, rows, error). On success error is None.

    The timeout is enforced with SQLite's progress handler, which aborts long-running queries
    (e.g. accidental cross joins in model predictions).
    """
    conn = connect_readonly(db_file)
    deadline = time.monotonic() + timeout_s
    # Returning non-zero from the handler interrupts the query.
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows) if max_rows else cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        return cols, rows, None
    except Exception as e:  # sqlite3.Error, but also e.g. "You can only execute one statement"
        msg = "timeout" if isinstance(e, sqlite3.OperationalError) and "interrupted" in str(e) else str(e)
        return [], [], f"{type(e).__name__}: {msg}"
    finally:
        conn.close()
