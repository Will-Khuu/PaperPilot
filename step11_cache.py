"""Step 11: exact-match answer and summary cache in the notes database.

Usage:
    python step11_cache.py clear
    python step11_cache.py selftest  # temporary database only
"""
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from step10_notes import DB_PATH


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def cache_key(kind, question="", paper_ids=None, *, model, prompt, fingerprint, k=None, k_per_paper=None,
              retrieval=None):
    inputs = {
        "kind": kind,
        "question": " ".join(question.split()),
        "paper_ids": sorted(paper_ids) if kind == "ask" and paper_ids else paper_ids,
        "model": model,
        "prompt": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "index": fingerprint,
    }
    if kind == "ask":
        inputs["paper_ids"] = inputs["paper_ids"] or None
        inputs["k"] = k
    elif kind == "compare":
        inputs["k_per_paper"] = k_per_paper
    if kind in ("ask", "compare"):
        inputs["retrieval"] = retrieval
    return digest(inputs)


def index_fingerprint(paper_ids=None, collection=None):
    from step3_embed import get_collection

    collection = get_collection() if collection is None else collection
    filters = {"where": {"paper_id": {"$in": list(paper_ids)}}} if paper_ids else {}
    found = collection.get(include=["documents", "metadatas"], **filters)
    chunks = sorted(
        (chunk_id, document, meta.get("title"), meta.get("section"), meta.get("page"), meta.get("page_end"))
        for chunk_id, document, meta in zip(found["ids"], found["documents"], found["metadatas"])
    )
    metadata = collection.metadata or {}
    return digest({"chunks": chunks, "chunk_size": metadata.get("chunk_size"),
                   "chunk_overlap": metadata.get("chunk_overlap")})


def connect(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE IF NOT EXISTS cache (
        key TEXT PRIMARY KEY, response TEXT NOT NULL, created_at TEXT NOT NULL
    )""")
    connection.commit()
    return connection


def get(key, path=DB_PATH):
    with closing(connect(path)) as connection:
        row = connection.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row is not None else None


def put(key, response, path=DB_PATH):
    with closing(connect(path)) as connection, connection:
        connection.execute("INSERT OR REPLACE INTO cache (key, response, created_at) VALUES (?, ?, ?)",
                           (key, json.dumps(response), datetime.now(timezone.utc).isoformat()))


def clear(path=DB_PATH):
    with closing(connect(path)) as connection, connection:
        return connection.execute("DELETE FROM cache").rowcount


def selftest():
    from tempfile import TemporaryDirectory
    from unittest.mock import Mock
    import step10_notes as notes

    inputs = {"kind": "ask", "question": "How many heads?", "paper_ids": ["bert", "attention"],
              "k": 5, "model": "model", "prompt": "system prompt", "fingerprint": "index",
              "retrieval": "hybrid-rrf60"}
    key = cache_key(**inputs)
    assert len(key) == 64 and key == cache_key(**inputs)
    assert key == cache_key(**{**inputs, "question": "  How  many\nheads?  "})
    assert key == cache_key(**{**inputs, "paper_ids": ["attention", "bert"]})
    for field, value in (("kind", "compare"), ("question", "how many heads?"), ("paper_ids", ["bert"]),
                         ("k", 3), ("model", "other"), ("prompt", "other"), ("fingerprint", "other"),
                         ("retrieval", "vector")):
        assert key != cache_key(**{**inputs, field: value}), field
    comparison = {**inputs, "kind": "compare", "k_per_paper": 3}
    assert cache_key(**comparison) != cache_key(**{**comparison, "paper_ids": ["attention", "bert"]})
    assert cache_key(**comparison) != cache_key(**{**comparison, "k_per_paper": 4})
    assert cache_key(**comparison) != cache_key(**{**comparison, "retrieval": "vector"})
    unfiltered = {**inputs, "paper_ids": None}
    assert cache_key(**unfiltered) == cache_key(**{**unfiltered, "paper_ids": []})
    summary = {**inputs, "kind": "summary", "question": "", "paper_ids": ["attention"]}
    assert cache_key(**summary) != cache_key(**{**summary, "paper_ids": ["bert"]})
    assert cache_key(**summary) == cache_key(**{**summary, "retrieval": "vector"})

    collection = Mock()
    collection.metadata = {"chunk_size": 220, "chunk_overlap": 80}
    found = {"ids": ["b", "a"], "documents": ["Second chunk", "First chunk"],
             "metadatas": [{"title": "Paper", "section": "Results", "page": 2, "page_end": 3},
                           {"title": "Paper", "section": "Introduction", "page": 1, "page_end": 1}]}
    collection.get.return_value = found
    fingerprint = index_fingerprint(collection=collection)
    collection.get.assert_called_once_with(include=["documents", "metadatas"])
    assert fingerprint == index_fingerprint(collection=collection)
    collection.get.return_value = {field: list(reversed(values)) for field, values in found.items()}
    assert fingerprint == index_fingerprint(["attention"], collection=collection)
    collection.get.assert_called_with(include=["documents", "metadatas"],
                                      where={"paper_id": {"$in": ["attention"]}})
    collection.get.return_value = found
    for field, value in (("ids", "c"), ("documents", "Changed chunk text")):
        original = found[field][0]
        found[field][0] = value
        assert fingerprint != index_fingerprint(collection=collection), field
        found[field][0] = original
    for field, value in (("title", "New title"), ("section", "Discussion"), ("page", 4), ("page_end", 5)):
        original = found["metadatas"][0][field]
        found["metadatas"][0][field] = value
        assert fingerprint != index_fingerprint(collection=collection), field
        found["metadatas"][0][field] = original
    assert fingerprint == index_fingerprint(collection=collection)
    for field in ("chunk_size", "chunk_overlap"):
        collection.metadata[field] += 1
        assert fingerprint != index_fingerprint(collection=collection)
        collection.metadata[field] -= 1

    with TemporaryDirectory() as directory:
        path = Path(directory) / "nested" / "cache.db"
        assert get(key, path=path) is None
        response = {"answer": "It's eight [1].", "sources": [{"n": 1, "label": "Page 1"}]}
        put(key, response, path=path)
        assert get(key, path=path) == response
        put(key, {"answer": "Updated"}, path=path)
        assert get(key, path=path) == {"answer": "Updated"}
        note = notes.add_note("attention", "Keep this note", path=path)
        injected = "'); DROP TABLE cache; --"
        put(injected, response, path=path)
        assert get(injected, path=path) == response
        assert clear(path=path) == 2 and get(key, path=path) is None
        assert clear(path=path) == 0
        assert notes.list_notes(path=path) == [note]


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
        print("selftest ok")
    elif sys.argv[1:] == ["clear"]:
        print(f"cleared {clear()} cached responses")
    else:
        sys.exit(__doc__)
