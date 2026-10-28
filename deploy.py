"""Build the demo index and enforce daily paid-answer budgets."""
from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

from step10_notes import DB_PATH

DEPLOYED = os.environ.get("PAPERPILOT_DEPLOYED") == "1"
DAILY_LLM_LIMIT = 40
SESSION_DAILY_LIMIT = 10
PAPERS = {
    "attention": "https://arxiv.org/pdf/1706.03762v7",
    "bert": "https://arxiv.org/pdf/1810.04805v2",
    "rag": "https://arxiv.org/pdf/2005.11401v4",
    "sbert": "https://arxiv.org/pdf/1908.10084v1",
}
PAPERS_DIR = Path(__file__).parent / "papers"


def ensure_index() -> list[str]:
    from step3_embed import ingest, list_papers

    indexed = {paper["paper_id"] for paper in list_papers()}
    ingested = []
    for paper_id, url in PAPERS.items():
        if paper_id in indexed:
            continue
        path = PAPERS_DIR / f"{paper_id}.pdf"
        if not path.exists():
            response = httpx.get(url, follow_redirects=True, timeout=120)
            response.raise_for_status()
            if not response.content.startswith(b"%PDF"):
                raise ValueError(f"Download for {paper_id} is not a PDF")
            PAPERS_DIR.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=PAPERS_DIR, suffix=".tmp", delete=False) as saved:
                temporary = Path(saved.name)
                try:
                    saved.write(response.content)
                    saved.close()
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        ingest(path)
        ingested.append(paper_id)
        print(f"Ingested {paper_id}")
    return ingested


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE IF NOT EXISTS llm_calls (
        day TEXT PRIMARY KEY, count INTEGER NOT NULL
    )""")
    connection.execute("""CREATE TABLE IF NOT EXISTS llm_session_calls (
        day TEXT, session TEXT, count INTEGER NOT NULL, PRIMARY KEY (day, session)
    )""")
    connection.commit()
    return connection


def take_llm_call(session_id, path=DB_PATH) -> str | None:
    day = datetime.now(timezone.utc).date().isoformat()
    with closing(connect(path)) as connection, connection:
        # Reserve both allowances under the same write lock, including the reads.
        connection.execute("BEGIN IMMEDIATE")
        global_row = connection.execute("SELECT count FROM llm_calls WHERE day = ?", (day,)).fetchone()
        session_row = connection.execute("SELECT count FROM llm_session_calls WHERE day = ? AND session = ?",
                                         (day, session_id)).fetchone()
        if global_row and global_row[0] >= DAILY_LLM_LIMIT:
            return "global"
        if session_row and session_row[0] >= SESSION_DAILY_LIMIT:
            return "session"
        connection.execute("""INSERT INTO llm_calls (day, count) VALUES (?, 1)
            ON CONFLICT(day) DO UPDATE SET count = count + 1""", (day,))
        connection.execute("""INSERT INTO llm_session_calls (day, session, count) VALUES (?, ?, 1)
            ON CONFLICT(day, session) DO UPDATE SET count = count + 1""", (day, session_id))
    return None


def llm_calls_today(path=DB_PATH) -> int:
    day = datetime.now(timezone.utc).date().isoformat()
    with closing(connect(path)) as connection:
        row = connection.execute("SELECT count FROM llm_calls WHERE day = ?", (day,)).fetchone()
        return row[0] if row else 0


def session_calls_today(session_id, path=DB_PATH) -> int:
    day = datetime.now(timezone.utc).date().isoformat()
    with closing(connect(path)) as connection:
        row = connection.execute("SELECT count FROM llm_session_calls WHERE day = ? AND session = ?",
                                 (day, session_id)).fetchone()
        return row[0] if row else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        from step3_embed import get_collection, retrieve

        ensure_index()
        # Query once too: serving must never need a model download on cold start.
        retrieve("What is attention?", k=1)
        print(f"Index ready: {get_collection().count()} chunks")
    elif sys.argv[1:] == ["check"]:
        for name in ("ANTHROPIC_API_KEY", "APP_PASSWORD", "SESSION_SECRET"):
            print(f"{name}: {'set' if os.environ.get(name) else 'not set'}")
        print(f"DB path: {DB_PATH}")
    else:
        sys.exit("Usage: python deploy.py build|check")
