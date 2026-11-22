"""SQLBuddy demo: ask questions about a SQLite database in plain English.

    python app/app.py            # local, http://127.0.0.1:7860
    python app/app.py --share    # also create a public gradio.live link (used on Kaggle)

Pick a database (sample university.db or upload your own .sqlite), read its schema, type a question,
choose Base / Fine-tuned / Compare, and get the generated SQL + the result table.
Queries run read-only and only single SELECT statements are allowed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sqlbuddy_core import ADAPTER, SAMPLE_DB, Models, UnsafeQuery, run_select, schema_of  # noqa: E402

MODELS = Models()
EXAMPLES = [
    "How many students are there?",
    "List the names of students from Lahore.",
    "What is the average score in each course? Show the course title.",
    "Which teacher teaches the most courses?",
    "Show the names of students who got an A in Databases.",
    "How many students are enrolled in each department's courses?",
]


def pick_db(choice: str, upload) -> str:
    if choice == "Upload my own" and upload is not None:
        return upload if isinstance(upload, str) else upload.name
    return str(SAMPLE_DB)


def show_schema(choice, upload):
    try:
        return schema_of(pick_db(choice, upload))
    except Exception as e:
        return f"-- could not read database: {e}"


def answer_one(db: str, question: str, fine_tuned: bool):
    """Returns (sql, dataframe, status markdown) for one model."""
    label = "Fine-tuned" if fine_tuned else "Base"
    try:
        sql, _raw, secs = MODELS.generate(schema_of(db), question, fine_tuned)
    except Exception as e:
        return "", pd.DataFrame(), f"**{label}:** generation failed - {e}"
    try:
        cols, rows, truncated = run_select(db, sql)
        df = pd.DataFrame(rows, columns=cols)
        note = f" (first {len(rows)} rows shown)" if truncated else ""
        return sql, df, f"**{label}:** {len(rows)} row(s){note} · generated in {secs:.1f} s"
    except UnsafeQuery as e:
        return sql, pd.DataFrame(), f"**{label}:** blocked - {e}"
    except Exception as e:
        return sql, pd.DataFrame(), f"**{label}:** the SQL failed to run - `{e}`"


def ask(choice, upload, question, mode):
    if not question.strip():
        empty = ("", pd.DataFrame(), "Type a question first.")
        return (*empty, "", pd.DataFrame(), "")
    db = pick_db(choice, upload)
    if mode == "Compare both":
        return (*answer_one(db, question, False), *answer_one(db, question, True))
    first = answer_one(db, question, mode == "Fine-tuned")
    return (*first, "", pd.DataFrame(), "")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SQLBuddy") as demo:
        gr.Markdown(
            "# SQLBuddy: ask your database in plain English\n"
            "Qwen2.5-Coder-1.5B-Instruct, base vs. LoRA fine-tuned on Spider. "
            "Queries run **read-only** and only single `SELECT` statements are executed.")
        with gr.Row():
            with gr.Column(scale=2):
                db_choice = gr.Radio(["Sample: university.db", "Upload my own"],
                                     value="Sample: university.db", label="Database")
                upload = gr.File(label="SQLite file (.sqlite / .db)", file_types=[".sqlite", ".db", ".sqlite3"],
                                 type="filepath")
                schema = gr.Code(value=show_schema("Sample: university.db", None), language="sql",
                                 label="Schema (what the model sees)", lines=18)
            with gr.Column(scale=3):
                question = gr.Textbox(label="Question", placeholder="e.g. Which course has the highest average score?")
                mode = gr.Radio(["Fine-tuned", "Base", "Compare both"], value="Fine-tuned", label="Model")
                go = gr.Button("Generate SQL and run", variant="primary")
                gr.Examples(EXAMPLES, inputs=question)
                sql1 = gr.Code(language="sql", label="Generated SQL")
                status1 = gr.Markdown()
                table1 = gr.Dataframe(label="Result", interactive=False)
                with gr.Accordion("Second model (Compare both)", open=False):
                    sql2 = gr.Code(language="sql", label="Generated SQL (fine-tuned)")
                    status2 = gr.Markdown()
                    table2 = gr.Dataframe(label="Result (fine-tuned)", interactive=False)

        db_choice.change(show_schema, [db_choice, upload], schema)
        upload.change(lambda u: ("Upload my own", show_schema("Upload my own", u)), upload, [db_choice, schema])
        outs = [sql1, table1, status1, sql2, table2, status2]
        go.click(ask, [db_choice, upload, question, mode], outs)
        question.submit(ask, [db_choice, upload, question, mode], outs)
        gr.Markdown(f"Adapter: `{ADAPTER}` · model loads on the first question (CPU: allow ~1 min).")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build_ui().launch(share=a.share, server_port=a.port)
