"""Step 10: store paper notes. Run python step10_notes.py selftest offline."""
import json
import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.environ.get("PAPERPILOT_DB", Path(__file__).parent / "data" / "paperpilot.db"))


def connect(path=DB_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS notes (
        id INTEGER PRIMARY KEY, paper_id TEXT NOT NULL, text TEXT NOT NULL,
        question TEXT, answer TEXT, sources TEXT, created_at TEXT NOT NULL
    )""")
    connection.commit()
    return connection


def note_record(row):
    note = dict(row)
    note["sources"] = json.loads(note["sources"]) if note["sources"] else []
    return note


def add_note(paper_id, text, question=None, answer=None, sources=None, path=DB_PATH):
    with closing(connect(path)) as connection, connection:
        cursor = connection.execute(
            "INSERT INTO notes (paper_id, text, question, answer, sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (paper_id, text, question, answer, json.dumps(sources) if sources is not None else None,
             datetime.now(timezone.utc).isoformat()),
        )
        return note_record(connection.execute("SELECT * FROM notes WHERE id = ?", (cursor.lastrowid,)).fetchone())


def list_notes(paper_id=None, path=DB_PATH):
    with closing(connect(path)) as connection:
        if paper_id is None:
            rows = connection.execute("SELECT * FROM notes ORDER BY created_at DESC, id DESC")
        else:
            rows = connection.execute(
                "SELECT * FROM notes WHERE paper_id = ? ORDER BY created_at DESC, id DESC", (paper_id,))
        return [note_record(row) for row in rows]


def delete_note(note_id, path=DB_PATH):
    with closing(connect(path)) as connection, connection:
        return connection.execute("DELETE FROM notes WHERE id = ?", (note_id,)).rowcount > 0


def delete_notes_for_paper(paper_id, path=DB_PATH):
    with closing(connect(path)) as connection, connection:
        connection.execute("DELETE FROM notes WHERE paper_id = ?", (paper_id,))


def export_markdown(paper_id, title, path=DB_PATH):
    parts = [f"# Notes: {title}"]
    for note in reversed(list_notes(paper_id, path=path)):
        parts.extend([f"## {note['created_at'][:10]}", note["text"]])
        if note["question"]:
            parts.append(f"Question: {note['question']}")
        if note["answer"]:
            parts.append(f"Answer:\n\n{note['answer']}")
        if note["sources"]:
            parts.append("Sources:\n" + "\n".join(
                f"- [{source['n']}] {source['label']}" for source in note["sources"]))
    return "\n\n".join(parts) + "\n"


def selftest():
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "notes.db"
        first = add_note("attention", "It's useful'); DROP TABLE notes; --", "Why?", "Because [1].",
                         [{"n": 1, "label": "Attention, p. 1"}], path=path)
        second = add_note("attention", "Second note", path=path)
        third = add_note("bert", "BERT note", path=path)
        assert list_notes(path=path) == [third, second, first]
        assert list_notes("attention", path=path) == [second, first]
        assert first["sources"] == [{"n": 1, "label": "Attention, p. 1"}]
        exported = export_markdown("attention", "Attention Is All You Need", path=path)
        assert "# Notes: Attention Is All You Need" in exported
        assert first["text"] in exported and "Sources:\n- [1] Attention, p. 1" in exported
        assert "Question: Why?" in exported and "Because [1]." in exported
        assert exported.index(first["text"]) < exported.index(second["text"])
        assert delete_note(first["id"], path=path)
        assert not delete_note(first["id"], path=path)
        assert list_notes("attention", path=path) == [second]


if __name__ == "__main__":
    if sys.argv[1:] != ["selftest"]:
        sys.exit(__doc__)
    selftest()
    print("selftest ok")
