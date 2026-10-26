"""Step 13: compare PDF parsers on Step 5's questions. Measurement only.

Setup:
    # Separate Python 3.11+ env; the deployed app does not use it.
    brew install poppler tesseract
    .venv-unstructured/bin/python -m pip install -r requirements-unstructured.txt

Usage:
    .venv-unstructured/bin/python step13_parsers.py parse
    .venv-unstructured/bin/python step13_parsers.py parse --strategy fast --force
    .venv/bin/python step13_parsers.py run
    .venv/bin/python step13_parsers.py show attention hi_res
    .venv/bin/python step13_parsers.py selftest

parse runs fast then hi_res by default, caching elements in out/unstructured/.
run uses the app's Python 3.9 env and throwaway in-memory collections. Cleaning,
chunking, embeddings and scoring stay the same. This step adopts nothing.
"""
import argparse
import json
import os
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from step5_eval import CUTOFFS, checked_questions, pages_with_phrase, score_questions, summarize
from step6_sweep import MIN_GAIN_AT_1, qualifies

REPO_DIR = Path(__file__).parent
PAPERS = ("attention", "bert", "rag", "sbert")
STRATEGIES = ("fast", "hi_res")
CONTESTANTS = ("pymupdf",) + STRATEGIES
PARSE_DIR = REPO_DIR / "out/unstructured"
OUT_FILE = REPO_DIR / "out/eval/parsers.json"
PARSE_PYTHON = ".venv-unstructured/bin/python"

# Checked on all four papers with 0.27.34: the Footnote key is never hit;
# layout footnotes serialize as Footer and are dropped. On these papers every
# dropped Header/Footer is a page number, venue line, arXiv stamp or figure-label
# words, so no body text or footnote is lost.
ROLE = {"Header": "drop", "Footer": "drop", "PageNumber": "drop", "Footnote": "footnotes"}


def elements_to_pages(elements, page_count=None):
    """Map serialized elements to Step 1 pages, preserving empty pages and order."""
    grouped = {}
    for element in elements:
        page = element.get("page")
        if type(page) is not int or page < 1:
            raise ValueError(f"element needs a positive page number: {element!r}")
        grouped.setdefault(page, {"page": page, "text": "", "footnotes": ""})
        role = ROLE.get(element["category"], "text")
        text = element["text"].strip()
        if role != "drop" and text:
            grouped[page][role] += text + "\n"
    last = max(grouped, default=0)
    if page_count is not None:
        if type(page_count) is not int or page_count < last:
            raise ValueError("page_count must cover all element page numbers")
        last = page_count
    return [grouped.get(page, {"page": page, "text": "", "footnotes": ""})
            for page in range(1, last + 1)]


def partition_path(paper, strategy):
    return PARSE_DIR / f"{paper}.{strategy}.json"


def has_text(elements):
    return any(isinstance(el, dict) and isinstance(el.get("text"), str) and el["text"].strip()
               for el in elements)


def load_partition(paper, strategy):
    path = partition_path(paper, strategy)
    command = f"{PARSE_PYTHON} step13_parsers.py parse --strategy {strategy} --force"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RuntimeError(f"cannot load {path}: {e}; run:\n{command}") from e
    if (not isinstance(data, dict) or data.get("paper") != paper
            or data.get("strategy") != strategy or not isinstance(data.get("elements"), list)
            or not has_text(data["elements"])):
        raise RuntimeError(f"invalid {path}: expected {paper}/{strategy} with non-empty text; "
                           f"run:\n{command}")
    return data


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # An interrupted write must not look like a completed cached parse.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def parse_papers(strategy=None, force=False):
    from importlib.metadata import version

    # Avoid telemetry requests and notices during the parser comparison.
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("UNSTRUCTURED_DISABLE_NOTICE", "1")
    from unstructured.partition.pdf import partition_pdf

    for selected_strategy in (strategy,) if strategy else STRATEGIES:
        first_in_process = True
        for paper in PAPERS:
            path = partition_path(paper, selected_strategy)
            cached = path.exists() and not force
            if cached:
                data = load_partition(paper, selected_strategy)
            else:
                options = {"infer_table_structure": True} if selected_strategy == "hi_res" else {}
                start = time.perf_counter()
                elements = partition_pdf(filename=str(REPO_DIR / "papers" / f"{paper}.pdf"),
                                         strategy=selected_strategy, languages=["eng"], **options)
                seconds = time.perf_counter() - start
                data = {"paper": paper, "strategy": selected_strategy,
                        "unstructured_version": version("unstructured"), "seconds": seconds,
                        "first_in_process": first_in_process,
                        "elements": [{"category": el.category, "text": el.text,
                                      "page": el.metadata.page_number,
                                      "html": getattr(el.metadata, "text_as_html", None)}
                                     for el in elements]}
                if not has_text(data["elements"]):
                    raise RuntimeError(f"{paper}/{selected_strategy}: partition returned no non-empty text; "
                                       "rerun with logging to diagnose PDF text extraction failure")
                first_in_process = False
                write_json(path, data)
            mark = " (cached)" if cached else ""
            print(f"{paper:<9} {selected_strategy:<6} {data['seconds']:.3f}s  "
                  f"{len(data['elements'])} elements{mark}", flush=True)


def format_comparison(results):
    n = results[0]["summary"]["n"]
    lines = [f"{'contestant':<10} {'scope':<9} {'chunks':>6}   "
             f"{'page @1':>7} {'@5':>3} {'@10':>3}   "
             f"{'phrase @1':>9} {'@5':>3} {'@10':>3}   (counts out of {n})"]
    for result in results:
        overall = result["summary"]["overall"]
        counts = [overall[test][f"hit@{k}"] for test in ("page", "phrase") for k in CUTOFFS]
        lines.append(f"{result['contestant']:<10} {result['scope']:<9} {result['chunks']:>6}   "
                     f"{counts[0]:>7} {counts[1]:>3} {counts[2]:>3}   "
                     f"{counts[3]:>9} {counts[4]:>3} {counts[5]:>3}")
    return "\n".join(lines)


def format_by_type(results):
    lines = ["phrase hit@5 by type:",
             f"{'contestant':<10} {'scope':<9} {'type':<12} hit@5"]
    for result in results:
        for kind, scores in result["summary"]["hit@5_by_type"].items():
            lines.append(f"{result['contestant']:<10} {result['scope']:<9} {kind:<12} "
                         f"{scores['phrase']}/{scores['n']}")
    return "\n".join(lines)


def format_ranks(results):
    attention = {r["contestant"]: r["questions"] for r in results if r["scope"] == "attention"}
    lines = ["Attention phrase ranks:",
             f"{'id':<4} " + " ".join(f"{name:>8}" for name in CONTESTANTS) + "  question"]
    for i, question in enumerate(attention["pymupdf"]):
        ranks = [attention[name][i]["ranks"]["phrase"] for name in CONTESTANTS]
        lines.append(f"{question['id']:<4} "
                     + " ".join(f"{str(rank) if rank is not None else '-':>8}" for rank in ranks)
                     + f"  {question['question']}")
    return "\n".join(lines)


def check_baseline(result):
    expected = {"attention": {"page": (15, 19, 20), "phrase": (12, 19, 20)},
                "all": {"page": (14, 18, 19), "phrase": (11, 18, 19)}}
    actual = {test: tuple(at[f"hit@{k}"] for k in CUTOFFS)
              for test, at in result["summary"]["overall"].items()}
    if actual != expected[result["scope"]]:
        raise RuntimeError(f"PyMuPDF {result['scope']} differs from Step 5: "
                           f"expected {expected[result['scope']]}, got {actual}; stopping")


def count_hyphen_breaks(pages):
    return sum(len(re.findall(r"[a-z]- [a-z]", p[field]))
               for p in pages for field in ("text", "footnotes"))


def run():
    partitions = {strategy: {paper: load_partition(paper, strategy) for paper in PAPERS}
                  for strategy in STRATEGIES}

    import chromadb
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    from step1_parse import clean, paper_title, parse
    from step2_chunk import CHUNK_OVERLAP, CHUNK_SIZE, chunk
    from step3_embed import add_chunks

    questions = checked_questions()
    pages, timings, titles = {"pymupdf": {}}, {"pymupdf": {}}, {}
    first_in_process = {name: {paper: False for paper in PAPERS} for name in CONTESTANTS}
    for paper in PAPERS:
        pdf = REPO_DIR / "papers" / f"{paper}.pdf"
        start = time.perf_counter()
        raw = parse(pdf)
        timings["pymupdf"][paper] = time.perf_counter() - start
        pages["pymupdf"][paper] = clean(raw)
        titles[paper] = paper_title(pdf)
    for strategy in STRATEGIES:
        pages[strategy], timings[strategy] = {}, {}
        for paper, data in partitions[strategy].items():
            pages[strategy][paper] = clean(elements_to_pages(
                data["elements"], page_count=len(pages["pymupdf"][paper])))
            timings[strategy][paper] = data["seconds"]
            first_in_process[strategy][paper] = data.get("first_in_process", False)

    hyphen_breaks = {name: {paper: count_hyphen_breaks(pages[name][paper]) for paper in PAPERS}
                     for name in CONTESTANTS}
    presence = {name: [{"id": q["id"], "phrase": q["phrase"],
                        "pages": pages_with_phrase(pages[name]["attention"], q["phrase"])}
                       for q in questions] for name in CONTESTANTS}
    client = chromadb.EphemeralClient()
    results = []
    for contestant in CONTESTANTS:
        for scope, papers in (("attention", ("attention",)), ("all", PAPERS)):
            chunks = [c for paper in papers for c in chunk(
                pages[contestant][paper], paper_id=paper, title=titles[paper],
                size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)]
            name = f"parsers-{contestant}-{scope}-{uuid.uuid4().hex[:8]}"
            collection = client.create_collection(
                name, embedding_function=ONNXMiniLM_L6_V2(),
                configuration={"hnsw": {"space": "cosine"}},
                metadata={"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP},
            )
            try:
                add_chunks(collection, chunks)
                rows = score_questions(questions, collection, method="vector")
            finally:
                client.delete_collection(name)
            result = {"contestant": contestant, "scope": scope, "chunks": len(chunks),
                      "questions": rows, "summary": summarize(rows)}
            results.append(result)
            print(f"scored {contestant}/{scope}: {len(chunks)} chunks", flush=True)
            if contestant == "pymupdf":
                check_baseline(result)

    base = results[0]["summary"]["overall"]["phrase"]
    qualifying = [r["contestant"] for r in results if r["scope"] == "attention"
                  and r["contestant"] != "pymupdf"
                  and qualifies(r["summary"]["overall"]["phrase"], base)]
    print("\n" + format_comparison(results))
    print("\n" + format_by_type(results))
    print("\n" + format_ranks(results))
    print("\nAttention phrases missing from cleaned text:")
    for contestant, entries in presence.items():
        missing = [f"{entry['id']} ({entry['phrase']})" for entry in entries if not entry["pages"]]
        print(f"{contestant}: " + (", ".join(missing) if missing else "none"))
    print("\nin-line hyphen breaks after cleaning ([a-z]- [a-z]):")
    print(f"{'paper':<9} " + " ".join(f"{name:>10}" for name in CONTESTANTS))
    for paper in PAPERS:
        print(f"{paper:<9} " + " ".join(f"{hyphen_breaks[name][paper]:>10}" for name in CONTESTANTS))
    print("\nparse seconds per paper:")
    print(f"{'paper':<9} " + " ".join(f"{name:>10}" for name in CONTESTANTS))
    for paper in PAPERS:
        cells = [f"{timings[name][paper]:.3f}" + ("*" if first_in_process[name][paper] else "")
                 for name in CONTESTANTS]
        print(f"{paper:<9} " + " ".join(f"{cell:>10}" for cell in cells))
    print("* First paper partitioned in that process for the strategy; includes one-time spaCy/model loads.")
    print(f"\nrule: Attention phrase hit@1 at least +{MIN_GAIN_AT_1} over pymupdf, "
          "and phrase hit@5 not lower")
    for strategy in STRATEGIES:
        print(f"{strategy}: " + ("qualifies" if strategy in qualifying else "does not qualify"))
    print("This step adopts nothing either way.")

    write_json(OUT_FILE, {
        "settings": {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
                     "method": "vector", "cutoffs": CUTOFFS},
        "rule": {"scope": "attention", "baseline": "pymupdf",
                 "min_gain_phrase_hit@1": MIN_GAIN_AT_1, "phrase_hit@5": "not lower than baseline"},
        "results": results, "phrase_presence": presence, "timings": timings,
        "first_in_process": first_in_process, "hyphen_breaks": hyphen_breaks,
        "unstructured_version": {s: {p: data["unstructured_version"] for p, data in items.items()}
                                 for s, items in partitions.items()},
        "qualifying": qualifying, "adopted": None,
    })
    print(f"results written to {OUT_FILE}")


def write_pages(path, pages):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(f"=== page {p['page']} ===\n{p['text']}"
                  + (f"--- footnotes ---\n{p['footnotes']}" if p["footnotes"] else "")
                  for p in pages), encoding="utf-8")


def show(paper, strategy):
    data = load_partition(paper, strategy)
    from step1_parse import clean, parse
    from step2_chunk import split_sections

    elements = data["elements"]
    baseline = clean(parse(REPO_DIR / "papers" / f"{paper}.pdf"))
    pages = clean(elements_to_pages(elements, page_count=len(baseline)))
    print(f"{paper}/{strategy}, Unstructured {data['unstructured_version']}")
    print("element counts by category:")
    for category, count in Counter(el["category"] for el in elements).most_common():
        print(f"  {category}: {count}")
    print("\nTitle elements:")
    for el in elements:
        if el["category"] == "Title":
            print(f"  page {el['page']}: {el['text']}")
    for name, version in ((strategy, pages), ("pymupdf", baseline)):
        print(f"\n{name} in-line hyphen breaks after cleaning: {count_hyphen_breaks(version)}")
        print(f"\n{name} sections:")
        for section in split_sections(version):
            print(f"  {section['section']}")
        path = PARSE_DIR / f"{paper}.{name}.clean.txt"
        write_pages(path, version)
        print(f"cleaned pages written to {path}")
    tables = [el for el in elements if el["category"] == "Table"]
    print(f"\nTable elements: {len(tables)}")
    if tables:
        print("first Table HTML:")
        print(tables[0].get("html") or "(none)")


def selftest():
    assert not has_text([])
    assert not has_text([{"text": " \n "}, {"text": None}])
    assert has_text([{"text": ""}, {"text": " body "}])
    assert count_hyphen_breaks([
        {"text": "transduc- tion x- y a- B A- b a-\nb", "footnotes": "foot- note"},
        {"text": "whole words", "footnotes": ""},
    ]) == 3, "count only in-line breaks with lowercase letters, including footnotes"
    elements = [
        {"category": "Title", "text": "  Heading  ", "page": 2},
        {"category": "NarrativeText", "text": " first ", "page": 1},
        {"category": "Footnote", "text": " note one ", "page": 1},
        {"category": "Unknown", "text": " second ", "page": 1},
        {"category": "Footnote", "text": "note two", "page": 1},
        {"category": "Table", "text": "cells", "page": 2},
        {"category": "NarrativeText", "text": " \n ", "page": 2},
        {"category": "Header", "text": "running header", "page": 1},
        {"category": "Footer", "text": "running footer", "page": 2},
        {"category": "PageNumber", "text": "4", "page": 4},
    ]
    expected = [
        {"page": 1, "text": "first\nsecond\n", "footnotes": "note one\nnote two\n"},
        {"page": 2, "text": "Heading\ncells\n", "footnotes": ""},
        {"page": 3, "text": "", "footnotes": ""},
        {"page": 4, "text": "", "footnotes": ""},
    ]
    assert elements_to_pages(elements) == expected, "roles, order, whitespace and empty pages"
    assert elements_to_pages(elements, 5) == expected + [{"page": 5, "text": "", "footnotes": ""}]
    assert elements_to_pages([]) == []
    assert elements_to_pages([], 2) == [{"page": p, "text": "", "footnotes": ""} for p in (1, 2)]
    assert elements[0]["text"] == "  Heading  ", "input is unchanged"
    for page in (None, 0, -1, "1", 1.5, True):
        for category in ("NarrativeText", "Header"):
            try:
                elements_to_pages([{"category": category, "text": "", "page": page}])
            except ValueError:
                pass
            else:
                raise AssertionError("even dropped or empty elements need a valid page")
    try:
        elements_to_pages([{"category": "Title", "text": "x"}])
    except ValueError:
        pass
    else:
        raise AssertionError("a missing page must raise")
    for count in (3, -1, "5", True):
        try:
            elements_to_pages(elements, count)
        except ValueError:
            pass
        else:
            raise AssertionError("page_count must cover all pages")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    parse_command = commands.add_parser("parse")
    parse_command.add_argument("--strategy", choices=STRATEGIES)
    parse_command.add_argument("--force", action="store_true")
    commands.add_parser("run")
    show_command = commands.add_parser("show")
    show_command.add_argument("paper", choices=PAPERS)
    show_command.add_argument("strategy", choices=STRATEGIES)
    commands.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "parse":
        parse_papers(args.strategy, args.force)
    elif args.command == "run":
        run()
    elif args.command == "show":
        show(args.paper, args.strategy)
    elif args.command == "selftest":
        selftest()
        print("selftest ok")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        sys.exit(str(e))
