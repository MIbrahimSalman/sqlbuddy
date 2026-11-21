"""Extract a single SQL query from raw model output.

Models (especially the base model) sometimes wrap SQL in ```sql fences, prefix it with "SQL:",
or add an explanation afterwards. We keep only the first SQL statement.
"""
from __future__ import annotations

import re

_FENCE = re.compile(r"```[ \t]*(?:sql|sqlite)?[ \t]*\n?(.*?)(?:```|$)", re.IGNORECASE | re.DOTALL)
_SQL_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)


def _cut_at_semicolon(sql: str) -> str:
    """Return the text before the first ';' that is not inside a quoted string."""
    quote = None
    for i, ch in enumerate(sql):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch == ";":
            return sql[:i]
    return sql


def extract_sql(text: str) -> str:
    text = text.strip()
    # 1) Prefer the content of the first code fence, if there is one.
    m = _FENCE.search(text)
    if m and _SQL_START.search(m.group(1)):
        text = m.group(1)
    # 2) Start at the first SELECT / WITH keyword (drops "SQL:", "Here is the query:", ...).
    m = _SQL_START.search(text)
    if not m:
        return re.sub(r"\s+", " ", text).strip()  # no SQL found; returned as-is, will fail to execute
    text = text[m.start():]
    # 3) Keep the first statement only.
    text = _cut_at_semicolon(text)
    # 4) Without a ';', an explanation usually follows after a blank line.
    text = re.split(r"\n\s*\n", text, maxsplit=1)[0]
    return re.sub(r"\s+", " ", text).strip()
