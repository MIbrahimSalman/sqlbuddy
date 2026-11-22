"""Demo back-end: model loading, SQL generation and *safe* query execution.

Uses the same prompt, schema builder and SQL extraction as the evaluation (src/), so the demo
behaves exactly like the evaluated models (greedy decoding, same prompt, same post-processing).

Speed notes (details in docs/PERFORMANCE.md):
* The model is loaded once, in the background, when the app starts (not on the first click).
* The system prompt + schema part of the prompt is identical for every question about the same
  database, so its attention cache (KV cache) is computed once and reused -> only the question
  tokens are processed again. This does not change the output, it only skips repeated work.
* Tokens are streamed to the UI as they are generated; a Stop button ends generation early.
* On a CPU, each generated token costs ~0.1-0.3 s (the whole 1.5 B-parameter network runs once
  per token), so a 50-token query takes ~5-15 s. This is a hardware limit, not a bug.
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
import sqlite3
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prompts import build_prompt  # noqa: E402
from src.schema import get_schema_ddl  # noqa: E402
from src.sql_extract import _cut_at_semicolon, extract_sql  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# LoRA adapter: public HF repo id (downloaded automatically, no login needed) or a local folder.
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


MAX_UPLOAD_MB = 50


def validate_sqlite(path: str | Path) -> None:
    """Raise ValueError with a readable message if the file is not a usable SQLite database."""
    path = Path(path)
    if path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"file is larger than {MAX_UPLOAD_MB} MB")
    with open(path, "rb") as f:
        if f.read(16) != b"SQLite format 3\x00":
            raise ValueError("this is not a SQLite database file (.sqlite / .db)")
    if not db_overview(path):
        raise ValueError("the database has no tables")


def db_overview(db_file: str | Path) -> list[tuple[str, int, int]]:
    """[(table, n_columns, n_rows)] for the database card in the UI (read-only connection)."""
    uri = Path(db_file).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        out = []
        for n in names:
            q = '"' + n.replace('"', '""') + '"'
            ncol = len(conn.execute(f"PRAGMA table_info({q})").fetchall())
            nrow = conn.execute(f"SELECT count(*) FROM {q}").fetchone()[0]
            out.append((n, ncol, nrow))
        return out
    finally:
        conn.close()


# ---------------------------------------------------------------------------------------------
# Models: one copy of the base weights; the LoRA adapter is switched on/off per request.
# ---------------------------------------------------------------------------------------------
def pick_device():
    """(device, dtype name). Override with SQLBUDDY_DEVICE=cpu|cuda|mps and SQLBUDDY_DTYPE=fp32|bf16|fp16."""
    import torch

    dev = os.environ.get("SQLBUDDY_DEVICE")
    if not dev:
        if torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"  # Apple-silicon GPU
        else:
            dev = "cpu"
    # GPU: fp16 (as in the evaluation). CPU: fp32 by default (fast and exact on every CPU;
    # bf16 halves RAM but is only fast on CPUs with native bf16 instructions).
    dtype = os.environ.get("SQLBUDDY_DTYPE") or ("fp32" if dev == "cpu" else "fp16")
    return dev, dtype


class Cancelled(Exception):
    pass


class Models:
    """Loads base model + LoRA adapter once and generates SQL (streamed, cancellable)."""

    def __init__(self):
        self.tok = self.model = None
        self.adapter_error: str | None = None
        self.state = "not loaded"         # not loaded -> loading -> ready | error
        self.load_error: str | None = None
        self.load_seconds: float | None = None
        self.device, self.dtype_name = None, None
        self._load_lock = threading.Lock()
        self.gen_lock = threading.Lock()   # one generation at a time (shared weights)
        self._prefix_cache: OrderedDict = OrderedDict()

    # ---- loading -------------------------------------------------------------------------
    def load(self):
        """Load once (thread-safe). Other callers wait until loading has finished."""
        with self._load_lock:
            if self.model is not None or self.state == "error":
                return
            self.state = "loading"
            t0 = time.time()
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                self.device, self.dtype_name = pick_device()
                dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[self.dtype_name]
                self.tok = AutoTokenizer.from_pretrained(BASE_MODEL)
                base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype).to(self.device).eval()
                try:
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(base, ADAPTER).eval()
                except Exception as e:  # e.g. no internet -> the base model still works
                    self.adapter_error = f"{type(e).__name__}: {e}"
                    self.model = base
                self.state = "ready"
            except Exception as e:
                self.state, self.load_error = "error", f"{type(e).__name__}: {e}"
            self.load_seconds = time.time() - t0

    def load_in_background(self, warm_schema: str | None = None):
        """Start loading in a background thread; optionally pre-compute the prompt cache for one schema."""
        def _job():
            self.load()
            if warm_schema and self.state == "ready":
                try:
                    self.warm_up(warm_schema)
                except Exception:
                    pass  # warm-up is only an optimisation
        threading.Thread(target=_job, daemon=True).start()

    def warm_up(self, schema: str):
        """Fill the prompt-prefix cache for this schema (both modes) so the first question is fast too."""
        prompt = build_prompt(self.tok, schema, "warm-up")
        ids = self.tok(prompt, return_tensors="pt")["input_ids"].to(self.device)
        with self.gen_lock:
            for ft in ([False, True] if self.has_adapter else [False]):
                self._prefix_kv(prompt, ids, ft)

    @property
    def has_adapter(self) -> bool:
        return hasattr(self.model, "disable_adapter")

    def describe(self) -> str:
        """Short hardware/model description for the UI, e.g. 'CPU (4 threads) · fp32'."""
        if self.device is None:
            return "not loaded yet"
        import torch

        if self.device == "cuda":
            hw = f"GPU: {torch.cuda.get_device_name(0)}"
        elif self.device == "mps":
            hw = "Apple GPU (MPS)"
        else:
            hw = f"CPU ({torch.get_num_threads()} threads)"
        return f"{hw} · {self.dtype_name}"

    # ---- prompt-prefix cache ---------------------------------------------------------------
    def _ctx(self, fine_tuned: bool):
        import contextlib
        # Base model = same weights with the LoRA adapter temporarily disabled.
        return self.model.disable_adapter() if (self.has_adapter and not fine_tuned) else contextlib.nullcontext()

    def _prefix_kv(self, prompt: str, input_ids, fine_tuned: bool):
        """KV cache of everything before 'Question:' (system prompt + schema), or None.

        The cached part depends on the adapter (LoRA changes the keys/values), so base and
        fine-tuned get separate entries. Returns a *copy*, because generate() extends the cache.
        """
        import torch

        cut = prompt.rfind("Question: ")
        if cut <= 0:
            return None
        prefix_ids = self.tok(prompt[:cut], return_tensors="pt")["input_ids"].to(self.device)
        n = prefix_ids.shape[1]
        if n >= input_ids.shape[1] or not torch.equal(input_ids[0, :n], prefix_ids[0]):
            return None  # tokenisation boundary differs -> just don't use the cache
        key = (fine_tuned, hashlib.sha1(prompt[:cut].encode()).hexdigest())
        if key not in self._prefix_cache:
            with self._ctx(fine_tuned), torch.inference_mode():
                self._prefix_cache[key] = self.model(prefix_ids, use_cache=True).past_key_values
            while len(self._prefix_cache) > 6:           # keep the 6 most recent (db, mode) pairs
                self._prefix_cache.popitem(last=False)
        self._prefix_cache.move_to_end(key)
        return copy.deepcopy(self._prefix_cache[key])

    # ---- generation ------------------------------------------------------------------------
    def generate_stream(self, schema: str, question: str, fine_tuned: bool, stop_event=None,
                        max_new_tokens: int = 256):
        """Yield (partial_raw_text, stats) while generating; the last yield has stats['done'] = True.

        Greedy decoding with the evaluation prompt. stop_event (threading.Event) cancels early.
        """
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

        self.load()
        if self.state != "ready":
            raise RuntimeError("model could not be loaded: " + (self.load_error or self.state))
        if fine_tuned and not self.has_adapter:
            raise RuntimeError("fine-tuned adapter not available: " + (self.adapter_error or "?"))

        class _Stop(StoppingCriteria):
            def __call__(self, *_a, **_k):
                return bool(stop_event and stop_event.is_set())

        t0 = time.time()
        prompt = build_prompt(self.tok, schema, question)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        n_prompt = enc["input_ids"].shape[1]
        past = self._prefix_kv(prompt, enc["input_ids"], fine_tuned)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        result: dict = {}

        def _run():
            try:
                with self._ctx(fine_tuned), torch.inference_mode():
                    out = self.model.generate(
                        **enc, past_key_values=past, do_sample=False, max_new_tokens=max_new_tokens,
                        pad_token_id=self.tok.pad_token_id, temperature=None, top_p=None, top_k=None,
                        streamer=streamer, stopping_criteria=StoppingCriteriaList([_Stop()]))
                result["n_new"] = int(out.shape[1] - n_prompt)
            except Exception as e:
                result["error"] = e
                streamer.end()

        th = threading.Thread(target=_run, daemon=True)
        th.start()
        raw, first = "", None
        for piece in streamer:
            if first is None:
                first = time.time() - t0
            raw += piece
            yield raw, {"done": False, "seconds": time.time() - t0}
        th.join()
        if "error" in result:
            raise result["error"]
        secs = time.time() - t0
        n_new = result.get("n_new", 0)
        stats = {"done": True, "seconds": secs, "first_token_s": first or secs, "new_tokens": n_new,
                 "prompt_tokens": n_prompt, "cached_prefix": past is not None,
                 "tokens_per_s": n_new / max(secs - (first or 0), 1e-6),
                 "cancelled": bool(stop_event and stop_event.is_set())}
        yield raw, stats

    def generate(self, schema: str, question: str, fine_tuned: bool, max_new_tokens: int = 256):
        """Non-streaming helper (used by notebook 04 self-test). Returns (sql, raw_output, seconds)."""
        raw, stats = "", {"seconds": 0.0}
        for raw, stats in self.generate_stream(schema, question, fine_tuned, max_new_tokens=max_new_tokens):
            pass
        return extract_sql(raw), raw, stats["seconds"]
