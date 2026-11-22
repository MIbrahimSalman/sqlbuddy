"""Demo UI helpers that do not need the model: upload validation, error messages, panel logic."""
import sqlite3

import pytest

gr = pytest.importorskip("gradio")
from app import app as ui  # noqa: E402
from app.sqlbuddy_core import SAMPLE_DB, db_overview, validate_sqlite  # noqa: E402


def test_sample_db_overview():
    tables = dict((t, (c, n)) for t, c, n in db_overview(SAMPLE_DB))
    assert set(tables) == {"teachers", "students", "courses", "enrollments", "grades"}
    assert tables["students"] == (5, 25)


def test_validate_rejects_non_sqlite(tmp_path):
    bad = tmp_path / "notes.db"
    bad.write_text("hello, not a database")
    with pytest.raises(ValueError, match="not a SQLite"):
        validate_sqlite(bad)


def test_validate_rejects_empty_db(tmp_path):
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).execute("PRAGMA user_version = 1").connection.commit()
    with pytest.raises(ValueError, match="no tables"):
        validate_sqlite(empty)


def test_db_card_for_sample_and_bad_upload(tmp_path):
    md, schema, _ = ui.db_card(ui.SAMPLE_LABEL, None)
    assert "5 tables" in md and "CREATE TABLE students" in schema
    bad = tmp_path / "x.sqlite"
    bad.write_bytes(b"\x00" * 64)
    md, schema, _ = ui.db_card(ui.UPLOAD_LABEL, str(bad))
    assert "Cannot use this file" in md and schema == ""


@pytest.mark.parametrize("err, expect", [
    (sqlite3.OperationalError("no such column: T2.title"), "column that does not exist"),
    (sqlite3.OperationalError("no such table: cities"), "table that is not in this database"),
    (sqlite3.OperationalError("interrupted"), "longer than 10 s"),
    (sqlite3.OperationalError('near "FROM": syntax error'), "not valid SQLite"),
    (ui.UnsafeQuery("only SELECT queries are allowed"), "Blocked for safety"),  # the class the app uses
])
def test_friendly_errors(err, expect):
    assert expect in ui.friendly_sql_error(err)


def test_error_message_is_html_escaped():
    assert "<script>" not in ui.friendly_sql_error(sqlite3.OperationalError("no such column: <script>"))


@pytest.mark.parametrize("mode, base_vis, ft_vis", [
    ("Fine-tuned", False, True), ("Base", True, False), ("Compare both", True, True)])
def test_panel_visibility(mode, base_vis, ft_vis):
    b, f = ui.panel_visibility(mode)
    assert b["visible"] is base_vis and f["visible"] is ft_vis


def test_empty_question_needs_no_model():
    out = list(ui.ask(ui.SAMPLE_LABEL, None, "   ", "Fine-tuned", None))
    assert len(out) == 1 and "No question" in out[0][4]
    assert ui.MODELS.state == "not loaded"   # validation happens before any model work


def test_fine_tuned_is_default_mode():
    assert ui.MODES[0] == "Fine-tuned"
