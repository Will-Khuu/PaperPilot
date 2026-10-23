"""Community Cloud startup and the shared daily answer budget."""
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

from step10_notes import DB_PATH

DEPLOYED = os.environ.get("PAPERPILOT_DEPLOYED") == "1"
DAILY_LLM_LIMIT = 40
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
    # Community Cloud resets this counter with the disk on reboot; the spending cap is the backstop.
    connection.execute("""CREATE TABLE IF NOT EXISTS llm_calls (
        day TEXT PRIMARY KEY, count INTEGER NOT NULL
    )""")
    connection.commit()
    return connection


def take_llm_call(path=DB_PATH) -> bool:
    day = datetime.now(timezone.utc).date().isoformat()
    with closing(connect(path)) as connection, connection:
        connection.execute("""INSERT INTO llm_calls (day, count) VALUES (?, 1)
            ON CONFLICT(day) DO UPDATE SET count = count + 1 WHERE count < ?""",
                           (day, DAILY_LLM_LIMIT))
        return connection.execute("SELECT changes()").fetchone()[0] == 1


def llm_calls_today(path=DB_PATH) -> int:
    day = datetime.now(timezone.utc).date().isoformat()
    with closing(connect(path)) as connection:
        row = connection.execute("SELECT count FROM llm_calls WHERE day = ?", (day,)).fetchone()
        return row[0] if row else 0
