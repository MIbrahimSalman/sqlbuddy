"""Turn a SQLite database file into CREATE TABLE statements for the prompt.

We rebuild a *canonical* DDL from SQLite's PRAGMA metadata instead of copying the original
`CREATE TABLE` text. Reasons: the original Spider DDL is very inconsistent (comments, odd quoting,
CHECK constraints, mixed formatting), and the canonical form works the same for any uploaded
database in the demo. The model still sees column types, primary keys and foreign keys.

Example output:
    CREATE TABLE singer (
      Singer_ID INT,
      Name TEXT,
      PRIMARY KEY (Singer_ID)
    );
    CREATE TABLE singer_in_concert (
      concert_ID INT,
      Singer_ID TEXT,
      PRIMARY KEY (concert_ID, Singer_ID),
      FOREIGN KEY (Singer_ID) REFERENCES singer(Singer_ID)
    );
"""
from __future__ import annotations

import re
from functools import lru_cache

from .execution import connect_readonly

_SIMPLE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# SQL keywords that appear as table/column names in real databases (e.g. a column called "From").
_KEYWORDS = {
    "abort", "all", "alter", "and", "as", "asc", "between", "by", "case", "check", "column", "create",
    "cross", "default", "delete", "desc", "distinct", "drop", "else", "end", "except", "exists", "from",
    "full", "group", "having", "in", "index", "inner", "insert", "intersect", "into", "is", "join", "key",
    "left", "like", "limit", "natural", "not", "null", "of", "on", "or", "order", "outer", "primary",
    "references", "right", "select", "set", "table", "then", "to", "union", "unique", "update", "using",
    "values", "when", "where", "with",
}


def quote_ident(name: str) -> str:
    """Quote an identifier only when needed (spaces, symbols, leading digit, SQL keyword)."""
    if _SIMPLE_IDENT.match(name) and name.lower() not in _KEYWORDS:
        return name
    return '"' + name.replace('"', '""') + '"'


def _table_ddl(conn, table: str) -> str:
    cols = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    # table_info rows: (cid, name, type, notnull, default, pk_position)
    lines = []
    for _cid, name, ctype, _nn, _dflt, _pk in cols:
        lines.append(f"  {quote_ident(name)} {ctype.strip() or 'TEXT'}".rstrip())

    pk_cols = [c[1] for c in sorted(cols, key=lambda c: c[5]) if c[5] > 0]
    if pk_cols:
        lines.append("  PRIMARY KEY (" + ", ".join(quote_ident(c) for c in pk_cols) + ")")

    # foreign_key_list rows: (id, seq, ref_table, from_col, to_col, on_update, on_delete, match)
    fks: dict[int, list] = {}
    for fid, _seq, ref_table, from_col, to_col, *_ in conn.execute(
        f"PRAGMA foreign_key_list({quote_ident(table)})"
    ).fetchall():
        fks.setdefault(fid, [ref_table, [], []])
        fks[fid][1].append(from_col)
        fks[fid][2].append(to_col)
    for ref_table, from_cols, to_cols in fks.values():
        to_txt = ", ".join(quote_ident(c) for c in to_cols if c) if any(to_cols) else ""
        lines.append(
            "  FOREIGN KEY (" + ", ".join(quote_ident(c) for c in from_cols) + ") REFERENCES "
            + quote_ident(ref_table) + (f"({to_txt})" if to_txt else "")
        )
    return f"CREATE TABLE {quote_ident(table)} (\n" + ",\n".join(lines) + "\n);"


def list_tables(conn) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
    ).fetchall()
    return [r[0] for r in rows]


@lru_cache(maxsize=512)
def get_schema_ddl(db_file: str) -> str:
    """CREATE TABLE statements for every user table in the database (cached per file)."""
    conn = connect_readonly(db_file)
    try:
        return "\n".join(_table_ddl(conn, t) for t in list_tables(conn))
    finally:
        conn.close()
