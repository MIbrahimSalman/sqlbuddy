"""The single prompt format used everywhere: baseline eval, training, fine-tuned eval and demo.

Using one function guarantees that base and fine-tuned models are compared on identical inputs.
"""
from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are an expert SQLite assistant. Given a database schema and a question, "
    "write one SQLite SELECT query that answers the question. "
    "Return only the SQL query, with no explanation."
)


def build_messages(schema: str, question: str, answer_sql: str | None = None) -> list[dict]:
    """Chat messages for one example. If `answer_sql` is given (training), it becomes the assistant turn."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Database schema:\n{schema}\n\nQuestion: {question.strip()}"},
    ]
    if answer_sql is not None:
        messages.append({"role": "assistant", "content": normalize_gold_sql(answer_sql)})
    return messages


def build_prompt(tokenizer, schema: str, question: str) -> str:
    """Prompt text ending with the assistant header, ready for generation."""
    return tokenizer.apply_chat_template(
        build_messages(schema, question), tokenize=False, add_generation_prompt=True
    )


def normalize_gold_sql(sql: str) -> str:
    """Spider gold SQL has irregular spacing ('a ,  b'); collapse whitespace for a clean target."""
    sql = re.sub(r"\s+", " ", sql).strip()
    sql = re.sub(r"\s+,", ",", sql)
    return sql.rstrip(";").strip()
