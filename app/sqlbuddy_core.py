"""Demo back-end: model loading, SQL generation and *safe* query execution.

Uses the same prompt, schema builder and SQL extraction as the evaluation (src/), so the demo
behaves exactly like the evaluated models.
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prompts import build_prompt  # noqa: E402
from src.schema import get_schema_ddl  # noqa: E402
from src.sql_extract import _cut_at_semicolon, extract_sql  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# LoRA adapter: HF repo id (private -> needs HF_TOKEN / `hf auth login`) or a local folder.
ADAPTER = os.environ.get("SQLBUDDY_ADAPTER", "ibrahimsalman022/sqlbuddy-qwen2.5-coder-1.5b-lora")
SAMPLE_DB = ROOT / "sample_db" / "university.db"
MAX_ROWS = 200


# ---------------------------------------------------------------------------------------------
# Safe execution: read-only connection + SELECT-only checks + SQLite authorizer + timeout
# ---------------------------------------------------------------------------------------------
class UnsafeQuery(ValueError):
    pass


_FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|create|alter|replace|attach|detach|pragma|vacuum|reindex)\b",
                        re.IGNORECASE)


def _strip_strings(sql: str) -> str:
    """Remove quoted literals so keywords inside strings ('Drop shipping') are not flagged."""
    return re.sub(r"'([^']|'')*'|\"([^\"]|\"\")*\"", "''", sql)


def check_select_only(sql: str) -> str:
    """Return the cleaned query or raise UnsafeQuery. Only a single SELECT (or WITH ... SELECT)."""
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise UnsafeQuery("empty query")
    if _cut_at_semicolon(sql).strip() != sql:
        raise UnsafeQuery("only one statement is allowed")
    if not re.match(r"(?is)^\s*(select|with)\b", sql):
        raise UnsafeQuery("only SELECT queries are allowed")
    bad = _FORBIDDEN.search(_strip_strings(sql))
    if bad:
        raise UnsafeQuery(f"keyword not allowed: {bad.group(1).upper()}")
    return sql


def _authorizer(action, *_):
    # Second line of defence inside SQLite itself: allow reading only.
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
    if hasattr(sqlite3, "SQLITE_RECURSIVE"):
        allowed.add(sqlite3.SQLITE_RECURSIVE)
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def run_select(db_file: str | Path, sql: str, timeout_s: float = 10.0, max_rows: int = MAX_ROWS):
    """Execute a checked SELECT read-only. Returns (columns, rows, truncated)."""
    sql = check_select_only(sql)
    uri = Path(db_file).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    conn.set_authorizer(_authorizer)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        cols = [d[0] for d in cur.description or []]
        return cols, rows[:max_rows], len(rows) > max_rows
    finally:
        conn.close()


def schema_of(db_file: str | Path) -> str:
    return get_schema_ddl(str(Path(db_file).resolve()))


# ---------------------------------------------------------------------------------------------
# Models: one copy of the base weights; the LoRA adapter is switched on/off per request.
# ---------------------------------------------------------------------------------------------
class Models:
    def __init__(self):
        self.tok = self.model = None
        self.adapter_error: str | None = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = torch.cuda.is_available()
        self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        # CPU: fp32 (fp16 is slow/unsupported on CPU); GPU: fp16.
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16 if cuda else torch.float32)
        base = base.to("cuda" if cuda else "cpu").eval()
        try:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base, ADAPTER).eval()
        except Exception as e:  # e.g. no HF token for the private adapter -> base model still works
            self.adapter_error = f"{type(e).__name__}: {e}"
            self.model = base

    @property
    def has_adapter(self) -> bool:
        return hasattr(self.model, "disable_adapter")

    def generate(self, schema: str, question: str, fine_tuned: bool, max_new_tokens: int = 256):
        """Greedy decoding with the evaluation prompt. Returns (sql, raw_output, seconds)."""
        import contextlib
        import torch

        self.load()
        if fine_tuned and not self.has_adapter:
            raise RuntimeError("fine-tuned adapter not available: " + (self.adapter_error or "?"))
        prompt = build_prompt(self.tok, schema, question)
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        # Base model = same weights with the LoRA adapter temporarily disabled.
        ctx = self.model.disable_adapter() if (self.has_adapter and not fine_tuned) else contextlib.nullcontext()
        t0 = time.time()
        with ctx, torch.inference_mode():
            out = self.model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tok.pad_token_id, temperature=None, top_p=None, top_k=None)
        raw = self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return extract_sql(raw), raw, time.time() - t0
