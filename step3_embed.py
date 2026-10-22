"""Step 3: embed the chunks, store them in Chroma, and search them. No LLM yet.

Usage:
    python step3_embed.py ingest papers/attention.pdf      # (re)build this paper's chunks in chroma_db/
    python step3_embed.py ingest papers/attention.pdf --size 160 --overlap 80
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
from step2_chunk import CHUNK_OVERLAP, CHUNK_SIZE, chunk

DB_PATH = str(Path(__file__).parent / "chroma_db")
EMPTY_INDEX = "the index is empty; run `python step3_embed.py ingest papers/attention.pdf` first"
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
    Cosine is explicit so ranking does not depend on the model normalizing
    its vectors, and distances read on a fixed 0-2 scale. This model returns
    unit-length vectors, so cosine and L2 rank results identically here.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        COLLECTION,
        embedding_function=ONNXMiniLM_L6_V2(),
        configuration={"hnsw": {"space": "cosine"}},
    )


CHUNK_FIELDS = ("paper_id", "title", "section", "page", "page_end", "chunk_index")


def paper_chunks(pdf_path, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    path = Path(pdf_path)
    pages = clean(parse(path))
    title = pages[0]["text"].splitlines()[0].strip()
    return chunk(pages, paper_id=path.stem, title=title, size=size, overlap=overlap)


def add_chunks(collection, chunks):
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{k: c[k] for k in CHUNK_FIELDS} for c in chunks],
    )


def ingest(pdf_path, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Delete this paper's old chunks, then add the new ones.

    Re-chunking can change how many chunks a paper has. A plain overwrite by
    id would leave stale ones behind (60 old, 55 new: 5 ghosts that still show
    up in search). Delete-then-add gives the same result however often it runs.

    The chunk settings go on the collection itself, so anything reading the
    index (Step 5's eval) learns how it was built from the index, not from
    chunk()'s current defaults. They are per collection, not per paper: every
    paper is meant to be ingested with the same settings.
    """
    chunks = paper_chunks(pdf_path, size=size, overlap=overlap)
    paper_id = Path(pdf_path).stem
    collection = get_collection()
    collection.delete(where={"paper_id": paper_id})
    add_chunks(collection, chunks)
    collection.modify(metadata={"chunk_size": size, "chunk_overlap": overlap})
    return len(chunks)


def retrieve(question, k=5, paper_ids=None, include_references=False, collection=None):
    """Return the k chunks closest to the question, closest first.

    Each result: {"id", "text", "distance", and the chunk's metadata}.
    References are left out by default: their titles share words with real
    questions ("attention", "translation") and would take top-k slots from
    chunks that actually answer.

    `collection` defaults to the real index; Step 6 passes throwaway ones.
    """
    conditions = []
    if paper_ids:
        conditions.append({"paper_id": {"$in": list(paper_ids)}})
    if not include_references:
        conditions.append({"section": {"$ne": "References"}})
    where = None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}

    collection = get_collection() if collection is None else collection
    if collection.count() == 0:
        raise RuntimeError(EMPTY_INDEX)
    found = collection.query(query_texts=[question], n_results=k, where=where)
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
        size = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else CHUNK_SIZE
        overlap = int(sys.argv[sys.argv.index("--overlap") + 1]) if "--overlap" in sys.argv else CHUNK_OVERLAP
        n = ingest(sys.argv[2], size=size, overlap=overlap)
        collection = get_collection()
        print(f"ingested {n} chunks from {sys.argv[2]}; collection now holds {collection.count()}")
        print(f"distance: {collection.configuration['hnsw']['space']}; chunk settings: {collection.metadata}")
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
    try:
        main()
    except RuntimeError as e:
        sys.exit(str(e))
