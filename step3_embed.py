"""Step 3: embed the chunks, store them in Chroma, and search them. No LLM yet.

Usage:
    python step3_embed.py ingest papers/attention.pdf      # (re)build this paper's chunks in chroma_db/
    python step3_embed.py ask "how many attention heads?"  # top 5 with distances
    python step3_embed.py ask "..." --k 10 --with-references
    python step3_embed.py demo                             # the eight test questions

Retrieval is tested on its own here, before any LLM is attached, so a bad
answer later can be traced to either "found the wrong chunks" or "used the
right chunks badly", never a mix of both.
"""
import sys
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

from step1_parse import clean, parse
from step2_chunk import chunk

DB_PATH = "chroma_db"
COLLECTION = "papers"

TEST_QUESTIONS = [
    "How many attention heads does the Transformer use?",            # exact fact
    "What optimizer was used for training?",                         # exact fact
    "What is scaled dot-product attention?",                         # concept
    "Why use self-attention instead of recurrence?",                 # concept
    "What BLEU score did the big model get on English-to-German?",   # number in a table
    "How does the model know the order of the words?",               # paraphrase: "positional encoding"
    "Who proposed multi-head attention?",                            # answer is in a footnote
    "What is BERT?",                                                 # not in the paper
]


def get_collection():
    """One collection for every paper; compare mode filters by paper_id.

    The embedding function is attached to the collection, so the same model
    embeds chunks at ingest and questions at query time. Mixing models would
    put the two in different vector spaces and the distances would mean nothing.
    Cosine distance, because Chroma defaults to L2 (straight-line distance),
    which is also affected by vector length rather than only direction.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        COLLECTION,
        embedding_function=ONNXMiniLM_L6_V2(),
        configuration={"hnsw": {"space": "cosine"}},
    )


def ingest(pdf_path):
    """Delete this paper's old chunks, then add the new ones.

    Re-chunking can change how many chunks a paper has. A plain overwrite by
    id would leave stale ones behind (60 old, 55 new: 5 ghosts that still show
    up in search). Delete-then-add gives the same result however often it runs.
    """
    path = Path(pdf_path)
    pages = clean(parse(path))
    title = pages[0]["text"].splitlines()[0].strip()
    chunks = chunk(pages, paper_id=path.stem, title=title)

    collection = get_collection()
    collection.delete(where={"paper_id": path.stem})
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{k: c[k] for k in ("paper_id", "title", "section", "page", "page_end", "chunk_index")}
                   for c in chunks],
    )
    return len(chunks)


def retrieve(question, k=5, paper_ids=None, include_references=False):
    """Return the k chunks closest to the question, closest first.

    Each result: {"id", "text", "distance", and the chunk's metadata}.
    References are left out by default: their titles share words with real
    questions ("attention", "translation") and would take top-k slots from
    chunks that actually answer.
    """
    conditions = []
    if paper_ids:
        conditions.append({"paper_id": {"$in": list(paper_ids)}})
    if not include_references:
        conditions.append({"section": {"$ne": "References"}})
    where = None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}

    found = get_collection().query(query_texts=[question], n_results=k, where=where)
    return [
        {"id": id_, "text": text, "distance": dist, **meta}
        for id_, text, dist, meta in zip(found["ids"][0], found["documents"][0],
                                         found["distances"][0], found["metadatas"][0])
    ]


def show(question, results):
    print(f"\nQ: {question}")
    for rank, r in enumerate(results, start=1):
        pages = f"p{r['page']}" if r["page"] == r["page_end"] else f"p{r['page']}-{r['page_end']}"
        body = " ".join(r["text"].split("\n", 1)[1].split())
        print(f"  {rank}. {r['distance']:.3f}  {r['id']}  {pages:<6} {r['section']}")
        print(f"       {body[:100]}...")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    command = sys.argv[1]

    if command == "ingest":
        n = ingest(sys.argv[2])
        collection = get_collection()
        print(f"ingested {n} chunks from {sys.argv[2]}; collection now holds {collection.count()}")
        print(f"distance: {collection.configuration['hnsw']['space']}")
    elif command == "ask":
        k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 5
        question = sys.argv[2]
        show(question, retrieve(question, k=k, include_references="--with-references" in sys.argv))
    elif command == "demo":
        for question in TEST_QUESTIONS:
            show(question, retrieve(question))
            # Would references have pushed into the top 5 without the filter?
            ranked = retrieve(question, include_references=True)
            refs = [rank for rank, r in enumerate(ranked, start=1) if r["section"] == "References"]
            print(f"     with references included: {f'references at rank {refs}' if refs else 'no change'}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
