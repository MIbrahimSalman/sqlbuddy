"""SQLBuddy demo: ask questions about a SQLite database in plain English.

    python app/app.py            # local, http://127.0.0.1:7860
    python app/app.py --share    # also create a public gradio.live link (used on Kaggle)

Flow: (1) pick a database (sample university.db or upload your own .sqlite) and read its schema,
(2) type a question and choose Fine-tuned / Base / Compare both, (3) get the generated SQL, the result
table and the measured generation time. SQL runs read-only and only single SELECT statements are
executed (see sqlbuddy_core.run_select).

UI states shown for every model panel: loading model -> queued -> generating (tokens stream in) ->
running SQL -> done / error / cancelled.
"""
from __future__ import annotations

import argparse
import html
import sqlite3
import sys
import threading
import time
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sqlbuddy_core import (  # noqa: E402
    ADAPTER, SAMPLE_DB, Models, UnsafeQuery, db_overview, extract_sql, run_select, schema_of, validate_sqlite)

MODELS = Models()
SAMPLE_LABEL, UPLOAD_LABEL = "Sample: university.db", "Upload my own"
MODES = ["Fine-tuned", "Base", "Compare both"]
# Example buttons for the sample university.db. Each one is checked (results/ood/predictions.csv): the
# fine-tuned model answers all of them correctly; the base model fails the 2nd, 3rd, 5th and 6th, which
# makes them good for "Compare both".
EXAMPLES = [
    "How many students are there?",
    "List the names of students from Lahore.",
    "What is the average score in each course? Show the course title.",
    "Which teacher teaches the most courses?",
    "Which course has the most enrollments? Give its title.",
    "Which students scored below 50 in at least one course?",
]
NO_RESULT = pd.DataFrame({"(no result)": []})  # empty table with a readable header
NO_SQL = "-- the generated SQL will appear here"
STOP_EVENTS: dict[str, threading.Event] = {}   # one stop flag per browser session

# ---------------------------------------------------------------------------------------------
# Styling (Gradio 6 takes theme/css in launch(); notebook 04 uses LAUNCH_KW too)
# ---------------------------------------------------------------------------------------------
CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto; }
#hero h1 { margin-bottom: 2px; }
#hero p { margin-top: 0; color: var(--body-text-color-subdued); }
.step-title h3 { margin: 0 0 4px 0; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.85em; font-weight: 600;
         border: 1px solid transparent; }
.b-idle      { background: #eef0f3; color: #4a4f57; }
.b-loading, .b-queued { background: #fff4dc; color: #7a4d00; border-color: #f1d08a; }
.b-running   { background: #e3efff; color: #0b4fa8; border-color: #9cc3f5; }
.b-done      { background: #e2f6e9; color: #146c34; border-color: #94d4aa; }
.b-error     { background: #fde7e7; color: #a11d1d; border-color: #f0a3a3; }
.b-cancelled { background: #eef0f3; color: #4a4f57; border-color: #c9ccd1; }
.panel-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }
.panel-head .title { font-weight: 700; font-size: 1.05em; }
.metrics { font-size: 0.88em; color: var(--body-text-color-subdued); }
.msg { margin-top: 6px; font-size: 0.92em; }
.sqlbox .cm-editor, .sqlbox .codemirror-wrapper, .sqlbox .wrap { min-height: 0 !important; max-height: 200px; }
/* phones: let long example questions wrap instead of widening the page */
.gallery-item, .gallery button { white-space: normal !important; text-align: left; }
.statusbar { font-size: 0.9em; padding: 8px 12px; border-radius: 8px; background: var(--block-background-fill);
             border: 1px solid var(--border-color-primary); }
"""
THEME = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")
LAUNCH_KW = {"css": CSS, "theme": THEME}


def badge(state: str, text: str) -> str:
    return f'<span class="badge b-{state}">{html.escape(text)}</span>'


def panel_header(title: str, state: str, text: str, msg: str = "") -> str:
    """Panel title + coloured state badge + optional message line (all HTML-escaped)."""
    m = f'<div class="msg">{msg}</div>' if msg else ""
    return f'<div class="panel-head"><span class="title">{html.escape(title)}</span>{badge(state, text)}</div>{m}'


def model_status() -> str:
    """Status bar: model loading state, device, adapter availability."""
    s = MODELS.state
    if s == "ready":
        b = badge("done", "Model ready")
        extra = f" · loaded in {MODELS.load_seconds:.0f} s"
    elif s == "error":
        b, extra = badge("error", "Model failed to load"), f" · {html.escape(MODELS.load_error or '')}"
    else:
        b, extra = badge("loading", "Loading model…"), " · first start downloads ~3 GB, later starts take seconds"
    dev = html.escape(MODELS.describe())
    if s == "ready" and not MODELS.has_adapter:
        ad = badge("error", "Fine-tuned adapter unavailable") + \
            " <small>(could not download it from Hugging Face: check the internet connection; Base mode still works)</small>"
    elif s == "ready":
        ad = badge("done", "Fine-tuned adapter loaded")
    else:
        ad = ""
    return f'<div class="statusbar">{b} &nbsp; <b>Device:</b> {dev}{extra} &nbsp; {ad}</div>'


# ---------------------------------------------------------------------------------------------
# Database selection
# ---------------------------------------------------------------------------------------------
def pick_db(choice: str, upload) -> str:
    if choice == UPLOAD_LABEL and upload:
        return upload if isinstance(upload, str) else upload.name
    return str(SAMPLE_DB)


def db_card(choice, upload):
    """(summary markdown, schema DDL, upload-visible) for the chosen database."""
    show_upload = gr.update(visible=choice == UPLOAD_LABEL)
    if choice == UPLOAD_LABEL and not upload:
        return "⬆️ Upload a `.sqlite` / `.db` file to use your own database.", "", show_upload
    db = pick_db(choice, upload)
    try:
        validate_sqlite(db)
        tables = db_overview(db)
        schema = schema_of(db)
    except (ValueError, sqlite3.DatabaseError, OSError) as e:
        return f"❌ **Cannot use this file:** {e}", "", show_upload
    name = "university.db (sample)" if db == str(SAMPLE_DB) else Path(db).name
    rows = "\n".join(f"| `{t}` | {c} | {n:,} |" for t, c, n in tables)
    md = (f"**{name}** · {len(tables)} tables · {sum(n for *_, n in tables):,} rows\n\n"
          f"| table | columns | rows |\n|---|---|---|\n{rows}")
    return md, schema, show_upload


# ---------------------------------------------------------------------------------------------
# Generation + safe execution, streamed to one result panel
# ---------------------------------------------------------------------------------------------
def friendly_sql_error(e: Exception) -> str:
    """Translate SQLite errors into explanations a presentation audience understands."""
    msg = str(e)
    if isinstance(e, UnsafeQuery):
        return f"🛡️ <b>Blocked for safety:</b> {html.escape(msg)}. Only a single read-only SELECT is executed."
    low = msg.lower()
    if "no such column" in low:
        return (f"The model referred to a column that does not exist (<code>{html.escape(msg)}</code>). "
                "Typical <i>schema-linking</i> error, e.g. a column taken from the wrong table alias.")
    if "no such table" in low:
        return f"The model used a table that is not in this database (<code>{html.escape(msg)}</code>)."
    if "interrupted" in low:
        return "The query ran longer than 10 s and was stopped."
    if "syntax error" in low or "incomplete input" in low:
        return f"The generated SQL is not valid SQLite (<code>{html.escape(msg)}</code>)."
    return f"The SQL failed to run: <code>{html.escape(msg)}</code>"


def round_floats(df: pd.DataFrame) -> pd.DataFrame:
    return df.round(2) if len(df) else df


def answer_stream(db: str, question: str, fine_tuned: bool, stop: threading.Event, queued_note: str = ""):
    """Yield (header_html, sql, table, metrics_md) for one model, through every UI state."""
    title = "Fine-tuned (LoRA)" if fine_tuned else "Base model"
    mode_txt = "Fine-tuned (LoRA adapter on)" if fine_tuned else "Base (LoRA adapter off)"
    # 1) model loading
    while MODELS.state in ("not loaded", "loading"):
        if stop.is_set():
            yield panel_header(title, "cancelled", "Cancelled"), NO_SQL, NO_RESULT, ""
            return
        yield panel_header(title, "loading", "Loading model…",
                           "The model is loaded once; this takes about a minute on the first start."), NO_SQL, NO_RESULT, ""
        time.sleep(0.5)
    if MODELS.state == "error":
        yield panel_header(title, "error", "Error", html.escape(MODELS.load_error or "")), NO_SQL, NO_RESULT, ""
        return
    # 2) queued (another generation is running: one at a time, the weights are shared)
    while not MODELS.gen_lock.acquire(timeout=0.3):
        if stop.is_set():
            yield panel_header(title, "cancelled", "Cancelled"), NO_SQL, NO_RESULT, ""
            return
        yield panel_header(title, "queued", "Queued", queued_note or "Waiting for another question to finish…"), \
            NO_SQL, NO_RESULT, ""
    try:
        # 3) generating: stream the raw text into the SQL box
        raw, stats = "", {}
        try:
            for raw, stats in MODELS.generate_stream(schema_of(db), question, fine_tuned, stop_event=stop):
                if not stats["done"]:
                    yield panel_header(title, "running", f"Generating… {stats['seconds']:.1f} s"), \
                        extract_sql(raw) or raw or NO_SQL, NO_RESULT, ""
        except Exception as e:
            yield panel_header(title, "error", "Error", html.escape(f"Generation failed: {e}")), NO_SQL, NO_RESULT, ""
            return
    finally:
        MODELS.gen_lock.release()
    sql = extract_sql(raw)
    metrics = (f"**Device:** {MODELS.describe()} · **Mode:** {mode_txt}  \n"
               f"**Generation:** {stats['seconds']:.1f} s · {stats['new_tokens']} tokens "
               f"({stats['tokens_per_s']:.1f} tokens/s) · first token after {stats['first_token_s']:.1f} s"
               f" · prompt {stats['prompt_tokens']} tokens{' (schema part cached)' if stats['cached_prefix'] else ''}")
    if stats.get("cancelled"):
        yield panel_header(title, "cancelled", "Cancelled", "Stopped before the query was finished."), \
            sql, NO_RESULT, metrics
        return
    if not sql:
        yield panel_header(title, "error", "No SQL", "The model did not produce a SQL query."), raw, NO_RESULT, metrics
        return
    # 4) run the SQL (read-only, SELECT-only, 10 s limit)
    yield panel_header(title, "running", "Running SQL…"), sql, NO_RESULT, metrics
    t0 = time.time()
    try:
        cols, rows, truncated = run_select(db, sql)
    except (UnsafeQuery, sqlite3.Error) as e:
        yield panel_header(title, "error", "SQL error", friendly_sql_error(e)), sql, NO_RESULT, metrics
        return
    ms = (time.time() - t0) * 1000
    df = round_floats(pd.DataFrame(rows, columns=cols))
    note = f" (first {len(rows)} shown)" if truncated else ""
    yield (panel_header(title, "done", f"Done · {len(rows)} row(s){note}"), sql, df if len(df.columns) else NO_RESULT,
           metrics + f" · SQL ran in {ms:.0f} ms")


# ---------------------------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------------------------
IDLE = {"base": panel_header("Base model", "idle", "Idle"), "ft": panel_header("Fine-tuned (LoRA)", "idle", "Idle")}


def panel_visibility(mode):
    """Show only the panel(s) the chosen mode uses (Compare = side by side)."""
    return gr.update(visible=mode in ("Base", "Compare both")), gr.update(visible=mode in ("Fine-tuned", "Compare both"))


def lock_ui():
    """Disable Generate while a question is being answered (no duplicate submissions); enable Stop."""
    return gr.update(interactive=False, value="Working…"), gr.update(interactive=True)


def unlock_ui():
    return gr.update(interactive=True, value="Generate SQL and run"), gr.update(interactive=False)


def ask(choice, upload, question, mode, request: gr.Request):
    """Generator feeding both panels: (b_head, b_sql, b_tab, b_met, f_head, f_sql, f_tab, f_met)."""
    stop = STOP_EVENTS.setdefault(request.session_hash if request else "local", threading.Event())
    stop.clear()
    base = [IDLE["base"], NO_SQL, NO_RESULT, ""]
    ft = [IDLE["ft"], NO_SQL, NO_RESULT, ""]
    if not question or not question.strip():
        msg = "Type a question first (or click an example)."
        tgt = base if mode == "Base" else ft
        tgt[0] = panel_header("Base model" if mode == "Base" else "Fine-tuned (LoRA)", "error", "No question", msg)
        yield (*base, *ft)
        return
    if choice == UPLOAD_LABEL and not upload:
        ft[0] = panel_header("Fine-tuned (LoRA)", "error", "No database", "Upload a SQLite file first.")
        base[0] = panel_header("Base model", "error", "No database", "Upload a SQLite file first.")
        yield (*base, *ft)
        return
    db = pick_db(choice, upload)
    try:
        validate_sqlite(db)
    except (ValueError, sqlite3.DatabaseError, OSError) as e:
        ft[0] = base[0] = panel_header("Database", "error", "Invalid file", html.escape(str(e)))
        yield (*base, *ft)
        return
    if mode == "Compare both":
        # Two generations, one after the other (same weights, adapter off then on).
        ft[0] = panel_header("Fine-tuned (LoRA)", "queued", "Queued",
                             "Runs after the base model (comparison = two generations).")
        yield (*base, *ft)
        for base in answer_stream(db, question, False, stop):
            yield (*base, *ft)
        if stop.is_set():
            yield (*base, panel_header("Fine-tuned (LoRA)", "cancelled", "Cancelled"), NO_SQL, NO_RESULT, "")
            return
        for ft in answer_stream(db, question, True, stop):
            yield (*base, *ft)
    elif mode == "Base":
        for base in answer_stream(db, question, False, stop):
            yield (*base, *ft)
    else:
        for ft in answer_stream(db, question, True, stop):
            yield (*base, *ft)


def request_stop(request: gr.Request):
    """Stop button: signal the running generation to end after the current token."""
    STOP_EVENTS.setdefault(request.session_hash if request else "local", threading.Event()).set()


def clear():
    return ("", IDLE["base"], NO_SQL, NO_RESULT, "", IDLE["ft"], NO_SQL, NO_RESULT, "")


# ---------------------------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------------------------
def result_panel(title_html: str, visible: bool):
    with gr.Column(visible=visible, min_width=280, variant="panel") as col:
        head = gr.HTML(title_html)
        sql = gr.Code(value=NO_SQL, language="sql", label="Generated SQL", wrap_lines=True, lines=3, max_lines=8,
                      elem_classes="sqlbox", interactive=False)
        table = gr.Dataframe(value=NO_RESULT, label="Result (read-only)", interactive=False, max_height=320,
                             wrap=True)
        metrics = gr.Markdown(elem_classes="metrics")
    return col, head, sql, table, metrics


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SQLBuddy") as demo:
        gr.Markdown(
            "# SQLBuddy: ask your database in plain English\n"
            "Qwen2.5-Coder-1.5B-Instruct, base vs. LoRA fine-tuned on Spider (execution accuracy on 1,034 "
            "held-out questions: **57.0 % → 73.1 %**). SQL runs **read-only**; only a single `SELECT` is executed.",
            elem_id="hero")
        status = gr.HTML(model_status())
        timer = gr.Timer(1.0)

        with gr.Row(equal_height=False):
            # ---- step 1: database ----
            with gr.Column(scale=4, min_width=280):
                gr.Markdown("### 1 · Choose a database", elem_classes="step-title")
                db_choice = gr.Radio([SAMPLE_LABEL, UPLOAD_LABEL], value=SAMPLE_LABEL, show_label=False)
                upload = gr.File(label="SQLite file (.sqlite / .db, max 50 MB)", type="filepath",
                                 file_types=[".sqlite", ".db", ".sqlite3"], visible=False)
                md0, schema0, _ = db_card(SAMPLE_LABEL, None)
                db_info = gr.Markdown(md0)
                with gr.Accordion("Schema the model sees (CREATE TABLE statements)", open=True):
                    schema = gr.Code(value=schema0, language="sql", show_label=False, lines=10, max_lines=16, wrap_lines=True,
                                     interactive=False)
            # ---- step 2: question ----
            with gr.Column(scale=6, min_width=280):
                gr.Markdown("### 2 · Ask a question", elem_classes="step-title")
                question = gr.Textbox(show_label=False, lines=2, max_lines=4,
                                      placeholder="e.g. Which course has the highest average score?")
                gr.Examples(EXAMPLES, inputs=question, label="Example questions (click to use)")
                mode = gr.Radio(MODES, value="Fine-tuned", label="Model",
                                info="Compare both runs two generations one after the other "
                                     "(base, then fine-tuned), so it takes about twice as long.")
                with gr.Row():
                    go = gr.Button("Generate SQL and run", variant="primary", scale=3)
                    stop_btn = gr.Button("Stop", variant="stop", interactive=False, scale=1)
                    clear_btn = gr.Button("Clear", variant="secondary", scale=1)

        # ---- step 3: results (side by side in Compare mode) ----
        gr.Markdown("### 3 · Results", elem_classes="step-title")
        with gr.Row(equal_height=False):
            b_col, b_head, b_sql, b_tab, b_met = result_panel(IDLE["base"], visible=False)
            f_col, f_head, f_sql, f_tab, f_met = result_panel(IDLE["ft"], visible=True)
        gr.Markdown(f"<small>Adapter: `{ADAPTER}` · greedy decoding, same prompt as in the evaluation · "
                    "safety: read-only connection, single SELECT only, SQLite authorizer, 10 s limit, 200 rows max.</small>")

        # ---- events ----
        def tick():
            done = MODELS.state in ("ready", "error")
            return model_status(), gr.Timer(active=not done)
        timer.tick(tick, outputs=[status, timer], show_progress="hidden")
        demo.load(tick, outputs=[status, timer], show_progress="hidden")

        db_choice.change(db_card, [db_choice, upload], [db_info, schema, upload])
        upload.change(db_card, [db_choice, upload], [db_info, schema, upload])
        mode.change(panel_visibility, mode, [b_col, f_col])

        outs = [b_head, b_sql, b_tab, b_met, f_head, f_sql, f_tab, f_met]
        for trigger in (go.click, question.submit):
            ev = trigger(lock_ui, None, [go, stop_btn], queue=False, trigger_mode="once")
            run = ev.then(ask, [db_choice, upload, question, mode], outs, show_progress="hidden",
                          concurrency_limit=8)
            run.then(unlock_ui, None, [go, stop_btn], queue=False)
        stop_btn.click(request_stop, None, None, queue=False).then(unlock_ui, None, [go, stop_btn], queue=False)
        clear_btn.click(clear, None, [question, *outs], queue=False)
    return demo


# Kept for notebook 04's self-test: one non-streaming answer as (sql, dataframe, status text).
def answer_one(db: str, question: str, fine_tuned: bool):
    last = None
    for last in answer_stream(db, question, fine_tuned, threading.Event()):
        pass
    head, sql, table, metrics = last
    import re
    return sql, table, re.sub(r"<[^>]+>", " ", head) + " " + metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    MODELS.load_in_background(warm_schema=schema_of(SAMPLE_DB))   # load now, not on the first click
    build_ui().queue(default_concurrency_limit=8).launch(share=a.share, server_port=a.port, **LAUNCH_KW)
