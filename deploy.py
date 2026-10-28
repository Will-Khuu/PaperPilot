"""Build the demo index and enforce daily paid-answer budgets."""
from __future__ import annotations

import os
import shutil
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

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
IMAGE_DIR = Path(__file__).parent
INDEX_DIR = Path(os.environ.get("PAPERPILOT_INDEX_DIR", IMAGE_DIR / "chroma_db"))
PAPERS_DIR = Path(os.environ.get("PAPERPILOT_PAPERS_DIR", IMAGE_DIR / "papers"))


def seed_data():
    for source, target in ((IMAGE_DIR / "chroma_db", INDEX_DIR), (IMAGE_DIR / "papers", PAPERS_DIR)):
        if target.exists() or target.resolve() == source.resolve() or not source.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # A killed process cannot clean its TemporaryDirectory; reclaim it before copying.
        for stale in target.parent.glob(f".{target.name}-seed-*"):
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
            else:
                stale.unlink()
        # Rename only a complete copy, so a crash cannot leave a half-seeded index.
        with TemporaryDirectory(prefix=f".{target.name}-seed-", dir=target.parent) as directory:
            copied = Path(directory) / "data"
            shutil.copytree(source, copied)
            copied.replace(target)


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


def take_llm_call(session_id, path=DB_PATH) -> tuple[str | None, str | None]:
    """Return the charged UTC day and any rejection reason."""
    day = datetime.now(timezone.utc).date().isoformat()
    with closing(connect(path)) as connection, connection:
        # Reserve both allowances under the same write lock, including the reads.
        connection.execute("BEGIN IMMEDIATE")
        global_row = connection.execute("SELECT count FROM llm_calls WHERE day = ?", (day,)).fetchone()
        session_row = connection.execute("SELECT count FROM llm_session_calls WHERE day = ? AND session = ?",
                                         (day, session_id)).fetchone()
        if global_row and global_row[0] >= DAILY_LLM_LIMIT:
            return None, "global"
        if session_row and session_row[0] >= SESSION_DAILY_LIMIT:
            return None, "session"
        connection.execute("""INSERT INTO llm_calls (day, count) VALUES (?, 1)
            ON CONFLICT(day) DO UPDATE SET count = count + 1""", (day,))
        connection.execute("""INSERT INTO llm_session_calls (day, session, count) VALUES (?, ?, 1)
            ON CONFLICT(day, session) DO UPDATE SET count = count + 1""", (day, session_id))
    return day, None


def llm_calls_today(path=DB_PATH) -> int:
    day = datetime.now(timezone.utc).date().isoformat()
    with closing(connect(path)) as connection:
        row = connection.execute("SELECT count FROM llm_calls WHERE day = ?", (day,)).fetchone()
        return row[0] if row else 0


def release_llm_call(session_id, day, path=DB_PATH):
    with closing(connect(path)) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("UPDATE llm_calls SET count = MAX(0, count - 1) WHERE day = ?", (day,))
        connection.execute("""UPDATE llm_session_calls SET count = MAX(0, count - 1)
            WHERE day = ? AND session = ?""", (day, session_id))


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
    elif sys.argv[1:] == ["seed"]:
        seed_data()
    elif sys.argv[1:] == ["check"]:
        for name in ("ANTHROPIC_API_KEY", "APP_PASSWORD", "OWNER_PASSWORD", "SESSION_SECRET"):
            print(f"{name}: {'set' if os.environ.get(name) else 'not set'}")
        if os.environ.get("OWNER_PASSWORD") and os.environ["OWNER_PASSWORD"] == os.environ.get("APP_PASSWORD"):
            print("Warning: OWNER_PASSWORD matches APP_PASSWORD; owner access is disabled.")
        print(f"DB path: {DB_PATH}")
        print(f"Index path: {INDEX_DIR}")
        print(f"Papers path: {PAPERS_DIR}")
    else:
        sys.exit("Usage: python deploy.py build|seed|check")
