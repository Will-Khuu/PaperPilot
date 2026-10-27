"""Step 14: compare LangChain retrieval with our pipeline. Measurement only.

Setup:
    /opt/homebrew/bin/python3.12 -m venv .venv-langchain
    .venv-langchain/bin/python -m pip install -r requirements.txt -r requirements-langchain.txt

Usage:
    .venv-langchain/bin/python step14_langchain.py run
    .venv-langchain/bin/python step14_langchain.py show attention lc-defaults
    .venv-langchain/bin/python step14_langchain.py selftest

The whole script, including control, runs in .venv-langchain in the same
process. Every index is throwaway and in memory; chroma_db/ is never opened.
All four pipelines use the app's ONNX MiniLM vectors. This step adopts nothing.
"""
import argparse
import inspect
import json
import os
import sys
import uuid
from importlib.metadata import distributions, version
from math import sqrt
from pathlib import Path

# Force tracing off even if the shell enabled it. Set this before LC imports.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from step1_parse import clean, paper_title, parse
from step2_chunk import CHUNK_OVERLAP, CHUNK_SIZE, TOKENIZER, chunk, count_tokens, split_sections
from step3_embed import add_chunks, retrieve
from step5_eval import (CUTOFFS, HIT_TESTS, K, checked_questions, first_hit_rank, normalize,
                        score_questions, summarize)
from step6_sweep import MIN_GAIN_AT_1, qualifies

REPO_DIR = Path(__file__).parent
PAPERS = ("attention", "bert", "rag", "sbert")
LC_RUNS = ("lc-defaults", "lc-matched", "lc-our-pages")
CONTESTANTS = ("control",) + LC_RUNS
PACKAGES = ("langchain-core", "langchain-community", "langchain-text-splitters", "langchain-chroma")
OUT_FILE = REPO_DIR / "out/eval/langchain.json"
FIRST_RUN_NAIVE_LENGTH = {
    "label": "length_function=count_tokens (adds 2 markers per piece)",
    "scope": "attention",
    "lc-matched": {"chunks": 98, "mean_tokens": 144.1, "max_tokens": 191,
                   "phrase": {"hit@1": 10, "hit@5": 18, "hit@10": 19}},
    "lc-our-pages": {"chunks": 90, "mean_tokens": 149.1, "max_tokens": 191,
                     "phrase": {"hit@1": 10, "hit@5": 18, "hit@10": 19}},
}


class MiniLMEmbeddings(Embeddings):
    # LangChain has no default embedding model. Identical vectors isolate
    # loading and splitting effects instead of introducing a model change.
    def __init__(self):
        self.model = ONNXMiniLM_L6_V2()

    def embed_documents(self, texts):
        return [vector.tolist() for vector in self.model(texts)]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def load_documents(pdf, contestant, pages=None):
    title = paper_title(pdf)
    if contestant == "lc-our-pages":
        pages = clean(parse(pdf)) if pages is None else pages
        # Append footnotes because our chunker indexes them too. Keeping the
        # page as one Document leaves only splitting different from our path.
        return [Document(page_content=p["text"] + "\n" + p.get("footnotes", ""),
                         metadata={"paper_id": pdf.stem, "title": title, "page": p["page"]})
                for p in pages]
    documents = PyMuPDFLoader(str(pdf)).load()
    for doc in documents:
        doc.metadata.update(paper_id=pdf.stem, title=title)
    return documents


def content_tokens(text):
    return len(TOKENIZER.encode(text, add_special_tokens=False).ids)


def split_documents(documents, contestant):
    if contestant == "lc-defaults":
        splitter = RecursiveCharacterTextSplitter()
    else:
        # The splitter assumes lengths are additive over pieces. count_tokens
        # adds two markers to every piece, even whitespace separators.
        # Reserve those markers once so a finished chunk counts at most 220, like ours.
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE - 2, chunk_overlap=CHUNK_OVERLAP,
                                                 length_function=content_tokens)
    return splitter.split_documents(documents)


def make_vectorstore(client, name, contestant, chunks):
    options = {} if contestant == "lc-defaults" else {
        "collection_configuration": {"hnsw": {"space": "cosine"}}}
    store = Chroma(client=client, collection_name=name, embedding_function=MiniLMEmbeddings(), **options)
    store.add_documents(chunks)
    return store


def retrieve_documents(vectorstore, question):
    return vectorstore.as_retriever(search_kwargs={"k": K}).invoke(question)


def to_result(doc, zero_based=True):
    # PyMuPDFLoader numbers its first page 0; our cleaned pages start at 1.
    page = doc.metadata["page"] + int(zero_based)
    # Splitting each page Document separately means chunks never cross pages.
    return {"text": doc.page_content, "paper_id": doc.metadata["paper_id"],
            "page": page, "page_end": page}


def question_row(question, results):
    return {"id": question["id"], "type": question["type"], "question": question["question"],
            "ranks": {test: first_hit_rank(results, question, hit) for test, hit in HIT_TESTS.items()}}


def references_text(pages):
    return normalize(" ".join(word for section in split_sections(pages)
                              if section["section"].lower() == "references"
                              for word, _ in section["words"]))


def is_references(result, references):
    if "section" in result:
        return result["section"] == "References"
    text = normalize(result["text"])
    start = max(0, (len(text) - 60) // 2)
    probe = text[start:start + 60]
    return bool(probe) and probe in references[result["paper_id"]]


def require_results(results, contestant, scope, question):
    if not results:
        raise RuntimeError(f"{contestant}/{scope}: {question['id']} returned 0 results; stopping")


def score_langchain(questions, store, contestant, scope):
    rows, unconverted, retrieved = [], [], []
    for question in questions:
        documents = retrieve_documents(store, question["question"])
        require_results(documents, contestant, scope, question)
        results = [to_result(doc, zero_based=contestant != "lc-our-pages") for doc in documents]
        rows.append(question_row(question, results))
        retrieved.extend(results)
        if contestant == "lc-defaults":
            unconverted.append(question_row(question, [to_result(d, zero_based=False) for d in documents]))
    return rows, retrieved, unconverted


def chunk_extras(collection, retrieved, references):
    stored = collection.get(include=["documents", "metadatas"])
    lengths = [count_tokens(text) for text in stored["documents"]]
    return {"max_tokens": max(lengths), "mean_tokens": sum(lengths) / len(lengths),
            "over_256_tokens": sum(n > 256 for n in lengths),
            "distance_metric": collection.configuration["hnsw"]["space"],
            "metadata_keys": sorted(stored["metadatas"][0]),
            "references_results": sum(is_references(r, references) for r in retrieved),
            "total_results": len(retrieved)}


def check_baseline(result):
    expected = {"attention": {"page": (15, 19, 20), "phrase": (12, 19, 20)},
                "all": {"page": (14, 18, 19), "phrase": (11, 18, 19)}}
    actual = {test: tuple(at[f"hit@{k}"] for k in CUTOFFS)
              for test, at in result["summary"]["overall"].items()}
    if actual != expected[result["scope"]]:
        raise RuntimeError(f"control {result['scope']} differs from Step 13: "
                           f"expected {expected[result['scope']]}, got {actual}; stopping")


def count_loc(source):
    return sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in source.splitlines())


def pipeline_loc():
    # Rough measure: ours includes CLI and selftests, with nothing subtracted;
    # LC counts our adapter and load/split/store/retrieve glue, not library code.
    functions = (MiniLMEmbeddings, load_documents, content_tokens, split_documents,
                 make_vectorstore, retrieve_documents)
    ours = ("step1_parse.py", "step2_chunk.py", "step3_embed.py")
    return {"langchain": sum(count_loc(inspect.getsource(f)) for f in functions),
            "ours": sum(count_loc((REPO_DIR / name).read_text(encoding="utf-8")) for name in ours),
            "langchain_functions": [f.__name__ for f in functions], "our_files": ours,
            "note": "Rough LOC: non-blank, non-comment lines; ours includes CLI and selftests; "
                    "LangChain counts local pipeline glue and adapter, not library internals."}


def probe_no_embedding(client):
    name = f"lc-no-embedding-{uuid.uuid4().hex[:8]}"
    try:
        store = Chroma(client=client, collection_name=name, embedding_function=None)
        text = "one offline embedding probe"
        try:
            store.add_texts([text], ids=["probe"])
            documents = store.similarity_search(text, k=1)
        except ValueError as error:
            return f"no: embedding_function=None raised ValueError: {error}"
        if not documents:
            raise RuntimeError("no-embedding probe returned 0 results")
        vector = client.get_collection(name).get(include=["embeddings"])["embeddings"][0]
        matches = list(vector) == MiniLMEmbeddings().embed_query(text)
        return (f"yes: add_texts and similarity_search succeeded with embedding_function=None; "
                f"{len(vector)}-d stored vector matches ONNXMiniLM_L6_V2: {matches}")
    finally:
        client.delete_collection(name)


def format_comparison(results):
    n = results[0]["summary"]["n"]
    lines = [f"{'run':<13} {'scope':<9} {'chunks':>6}   "
             f"{'page @1':>7} {'@5':>3} {'@10':>3}   "
             f"{'phrase @1':>9} {'@5':>3} {'@10':>3}   (counts out of {n})"]
    for result in results:
        overall = result["summary"]["overall"]
        counts = [overall[test][f"hit@{k}"] for test in HIT_TESTS for k in CUTOFFS]
        lines.append(f"{result['contestant']:<13} {result['scope']:<9} {result['chunks']:>6}   "
                     f"{counts[0]:>7} {counts[1]:>3} {counts[2]:>3}   "
                     f"{counts[3]:>9} {counts[4]:>3} {counts[5]:>3}")
    return "\n".join(lines)


def format_by_type(results):
    lines = ["hit@5 by type:", f"{'run':<13} {'scope':<9} {'type':<12} {'page':>6} {'phrase':>6}"]
    for result in results:
        for kind, scores in result["summary"]["hit@5_by_type"].items():
            lines.append(f"{result['contestant']:<13} {result['scope']:<9} {kind:<12} "
                         f"{scores['page']:>4}/{scores['n']} {scores['phrase']:>4}/{scores['n']}")
    return "\n".join(lines)


def format_ranks(results):
    lines = []
    for scope in ("attention", "all"):
        selected = {r["contestant"]: r["questions"] for r in results if r["scope"] == scope}
        lines += [f"{scope} ranks (page/phrase; - means outside top {K}):",
                  f"{'id':<4} " + " ".join(f"{name:>13}" for name in CONTESTANTS) + "  question"]
        for i, question in enumerate(selected["control"]):
            cells = ["/".join(str(rank) if rank is not None else "-"
                              for rank in selected[name][i]["ranks"].values()) for name in CONTESTANTS]
            lines.append(f"{question['id']:<4} " + " ".join(f"{cell:>13}" for cell in cells)
                         + f"  {question['question']}")
    return "\n".join(lines)


def run():
    questions = checked_questions()
    pages, titles = {}, {}
    for paper in PAPERS:
        pdf = REPO_DIR / "papers" / f"{paper}.pdf"
        pages[paper], titles[paper] = clean(parse(pdf)), paper_title(pdf)
    references = {paper: references_text(pages[paper]) for paper in PAPERS}
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    results = []
    for contestant in CONTESTANTS:
        for scope, papers in (("attention", ("attention",)), ("all", PAPERS)):
            if contestant == "control":
                chunks = [c for paper in papers for c in chunk(
                    pages[paper], paper_id=paper, title=titles[paper], size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)]
            else:
                documents = [d for paper in papers for d in load_documents(
                    REPO_DIR / "papers" / f"{paper}.pdf", contestant, pages[paper])]
                chunks = split_documents(documents, contestant)
            if not chunks:
                raise RuntimeError(f"{contestant}/{scope}: 0 chunks; stopping")
            name = f"langchain-{contestant}-{scope}-{uuid.uuid4().hex[:8]}"
            try:
                unconverted = []
                if contestant == "control":
                    collection = client.create_collection(
                        name, embedding_function=ONNXMiniLM_L6_V2(),
                        configuration={"hnsw": {"space": "cosine"}},
                        metadata={"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP})
                    add_chunks(collection, chunks)
                    retrieved = []
                    for question in questions:
                        found = retrieve(question["question"], k=K, collection=collection, method="vector")
                        require_results(found, contestant, scope, question)
                        retrieved.extend(found)
                    rows = score_questions(questions, collection, method="vector")
                else:
                    store = make_vectorstore(client, name, contestant, chunks)
                    collection = client.get_collection(name)
                    rows, retrieved, unconverted = score_langchain(questions, store, contestant, scope)
                result = {"contestant": contestant, "scope": scope, "chunks": collection.count(),
                          "questions": rows, "summary": summarize(rows),
                          "extras": chunk_extras(collection, retrieved, references)}
                if contestant == "lc-defaults" and scope == "attention":
                    vectors = collection.get(limit=5, include=["embeddings"])["embeddings"]
                    norms = [sqrt(sum(float(value) ** 2 for value in vector)) for vector in vectors]
                if unconverted:
                    result["unconverted_page_summary"] = summarize(unconverted)["overall"]["page"]
                if contestant == "control":
                    check_baseline(result)
            finally:
                client.delete_collection(name)
            results.append(result)
            print(f"scored {contestant}/{scope}: {result['chunks']} chunks", flush=True)

    base = results[0]["summary"]["overall"]["phrase"]
    qualifying = [r["contestant"] for r in results if r["scope"] == "attention"
                  and r["contestant"] != "control"
                  and qualifies(r["summary"]["overall"]["phrase"], base)]
    extras = {"lines_of_code": pipeline_loc(), "installed_packages": sum(1 for _ in distributions()),
              "first_run_naive_length": FIRST_RUN_NAIVE_LENGTH,
              "minilm_vector_norms": {"source": "lc-defaults/attention", "values": norms,
                                      "min": min(norms), "max": max(norms)},
              "chroma_wrapper_no_embedding": probe_no_embedding(client)}
    versions = {package: version(package) for package in PACKAGES}
    print("\n" + format_comparison(results))
    print("\n" + format_by_type(results))
    print("\n" + format_ranks(results))
    print("\nchunk diagnostics (MiniLM tokens, including markers):")
    for result in results:
        e = result["extras"]
        label = f"{result['contestant']}/{result['scope']}"
        print(f"{label}: chunks={result['chunks']}, max={e['max_tokens']}, mean={e['mean_tokens']:.1f}, "
              f"over 256={e['over_256_tokens']}, distance={e['distance_metric']}, "
              f"References={e['references_results']}/{e['total_results']}")
        print("  stored metadata keys: " + ", ".join(e["metadata_keys"]))
        if "unconverted_page_summary" in result:
            counts = result["unconverted_page_summary"]
            print("  WITHOUT page +1: page hit@1/5/10 = "
                  + "/".join(str(counts[f"hit@{k}"]) for k in CUTOFFS))
    print("References detection: section when present; otherwise text-based: "
          "middle of the chunk found in our References section.")
    print("LC defaults: 4000/200 characters, len; matched: "
          f"{CHUNK_SIZE}/{CHUNK_OVERLAP} MiniLM tokens; our pages append footnotes.")
    print("LC chunks stay within pages and have no section metadata or References filter.")
    # For unit vectors, squared L2 distance = 2 - 2*cosine similarity.
    print(f"MiniLM vectors have norm ~{sum(norms) / len(norms):.3f} "
          f"({min(norms):.6f}..{max(norms):.6f}), so L2 and cosine rank identically; "
          "lc-defaults' gap is not the metric.")
    loc = extras["lines_of_code"]
    print(f"LOC: LangChain pipeline={loc['langchain']}, ours={loc['ours']}. {loc['note']}")
    print("Chroma wrapper without embeddings: " + extras["chroma_wrapper_no_embedding"])
    print(f"Installed packages in this venv: {extras['installed_packages']}")
    print("Versions: " + ", ".join(f"{p}={v}" for p, v in versions.items()))
    print(f"\nrule: Attention phrase hit@1 at least +{MIN_GAIN_AT_1} over control, and phrase hit@5 not lower")
    for contestant in LC_RUNS:
        print(f"{contestant}: " + ("qualifies" if contestant in qualifying else "does not qualify"))
    print("This step adopts nothing either way.")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "settings": {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "k": K,
                     "method": "vector", "cutoffs": CUTOFFS, "scopes": {"attention": ["attention"], "all": PAPERS},
                     "lc_defaults": {"chunk_size": 4000, "chunk_overlap": 200, "length_function": "len"},
                     "lc_matched": {"chunk_size": CHUNK_SIZE - 2, "chunk_overlap": CHUNK_OVERLAP,
                                    "length_function": "content_tokens"},
                     "embedding_model": "ONNXMiniLM_L6_V2", "environment": ".venv-langchain for all runs",
                     "our_pages_footnotes": "appended", "langsmith_tracing": False,
                     "references_detection": "section when present; otherwise text-based: "
                                             "middle of the chunk found in our References section"},
        "rule": {"scope": "attention", "baseline": "control", "min_gain_phrase_hit@1": MIN_GAIN_AT_1,
                 "phrase_hit@5": "not lower than baseline"},
        "results": results, "extras": extras, "versions": versions,
        "qualifying": qualifying, "adopted": None,
    }, indent=2), encoding="utf-8")
    print(f"results written to {OUT_FILE}")


def show(paper, contestant):
    documents = load_documents(REPO_DIR / "papers" / f"{paper}.pdf", contestant)
    chunks = split_documents(documents, contestant)
    if not chunks:
        raise RuntimeError(f"{paper}/{contestant}: 0 chunks; stopping")
    for i, doc in enumerate(chunks):
        result = to_result(doc, zero_based=contestant != "lc-our-pages")
        print(f"{i:03d}  p{result['page']}  {count_tokens(result['text'])} tokens  "
              + " ".join(result["text"].split())[:200])


def selftest():
    doc = Document(page_content="body", metadata={"paper_id": "attention", "page": 0})
    assert to_result(doc) == {"text": "body", "paper_id": "attention", "page": 1, "page_end": 1}
    doc.metadata["page"] = 7
    assert to_result(doc)["page"] == 8
    assert to_result(doc, zero_based=False)["page_end"] == 7
    adapter, text = MiniLMEmbeddings(), "The Transformer uses eight attention heads."
    expected = ONNXMiniLM_L6_V2()([text])[0].tolist()
    assert len(expected) == 384
    assert adapter.embed_documents([text]) == [expected]
    assert adapter.embed_query(text) == expected
    sample = Document(page_content=("Attention learns relations between words.\n\n" * 100), metadata={})
    pieces = split_documents([sample], "lc-matched")
    assert len(pieces) > 1
    assert CHUNK_SIZE - 20 <= max(count_tokens(d.page_content) for d in pieces) <= CHUNK_SIZE
    assert content_tokens("\n") == 0
    assert content_tokens(" ") == 0
    assert count_loc("\n# comment\n  # indented\nx = 1  # inline\nprint(x)\n") == 2
    entry = "An author. A paper about attention and its applications. Published in 2017."
    references = {"attention": references_text([
        {"page": 3, "text": "body on the same page\nReferences\n" + entry},
        {"page": 4, "text": "Attention Visualizations\nappendix body"}])}
    assert references["attention"] == entry
    assert is_references({"paper_id": "attention", "page": 3, "text": entry}, references)
    assert is_references({"paper_id": "attention", "page": 3, "text": "An\nauthor."}, references)
    assert not is_references({"paper_id": "attention", "page": 3,
                              "text": "body on the same page"}, references)
    assert not is_references({"paper_id": "attention", "page": 4, "text": "appendix body"}, references)
    assert is_references({"section": "References"}, references)
    assert not is_references({"section": "Appendix"}, references)
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    print("Chroma wrapper without embeddings: " + probe_no_embedding(client))
    print("selftest passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run")
    commands.add_parser("selftest")
    show_command = commands.add_parser("show")
    show_command.add_argument("paper", choices=PAPERS)
    show_command.add_argument("run", choices=LC_RUNS)
    args = parser.parse_args()
    if args.command == "run":
        run()
    elif args.command == "show":
        show(args.paper, args.run)
    else:
        selftest()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        sys.exit(str(error))
