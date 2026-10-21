"""Step 5: score retrieval on 20 labeled questions. No LLM.

Usage:
    python step5_eval.py run                    # results to out/eval/<timestamp>.json
    python step5_eval.py run --label baseline   # results to out/eval/baseline.json
    python step5_eval.py selftest               # pure functions only: no Chroma, no network

Two hit tests, because each misses something the other catches. A page hit
means a retrieved chunk covers a gold page; it ignores chunking, but a chunk
on the right page may still not contain the answer. A phrase hit means the
chunk's text contains the answer phrase; stricter, it proves the answer was
actually in front of the model.
"""
import inspect
import json
import sys
import time
from pathlib import Path

# step3_embed (Chroma, ONNX model) and step1_parse (PyMuPDF) are imported inside
# main()'s branches, so selftest and the pure functions load with the stdlib only.

QUESTIONS = Path("eval/questions.json")
PAPER = "papers/attention.pdf"
OUT_DIR = Path("out/eval")
K = 10
CUTOFFS = (1, 5, 10)


def normalize(text):
    """Collapse all whitespace to single spaces. Chunks join words with one
    space while the cleaned page keeps line breaks, so raw text never matches."""
    return " ".join(text.split())


def page_hit(result, question):
    """Any gold page inside the chunk's [page, page_end]."""
    return any(result["page"] <= p <= result["page_end"] for p in question["pages"])


def phrase_hit(result, question):
    return normalize(question["phrase"]) in normalize(result["text"])


HIT_TESTS = {"page": page_hit, "phrase": phrase_hit}


def first_hit_rank(results, question, hit):
    """1-based rank of the first result that hits, or None."""
    return next((rank for rank, r in enumerate(results, start=1) if hit(r, question)), None)


def hits_at(ranks, k):
    """How many ranks are at or above k. None is a miss."""
    return sum(1 for r in ranks if r is not None and r <= k)


def pages_with_phrase(pages, phrase):
    """Page numbers whose normalized text contains the phrase."""
    target = normalize(phrase)
    return [p["page"] for p in pages if target in normalize(p["text"])]


def bad_labels(questions, pages):
    """[(id, labeled pages, found pages)] for every label that doesn't match the paper.

    Exact match, not "found on at least one": a phrase that also appears on an
    unlabeled page would let a page hit there count as a miss, or the reverse.
    """
    bad = []
    for q in questions:
        found = pages_with_phrase(pages, q["phrase"])
        if sorted(q["pages"]) != found:
            bad.append((q["id"], sorted(q["pages"]), found))
    return bad


def summarize(rows):
    """hit@k overall for each test, plus hit@5 per question type.

    rows: [{"id", "type", "question", "ranks": {test: rank or None}}].
    """
    overall = {test: {f"hit@{k}": hits_at([r["ranks"][test] for r in rows], k) for k in CUTOFFS}
               for test in HIT_TESTS}
    by_type = {}
    for t in dict.fromkeys(r["type"] for r in rows):
        group = [r for r in rows if r["type"] == t]
        by_type[t] = {"n": len(group),
                      **{test: hits_at([r["ranks"][test] for r in group], 5) for test in HIT_TESTS}}
    return {"n": len(rows), "overall": overall, "hit@5_by_type": by_type}


def score(n, total):
    return f"{n}/{total} ({100 * n / total:.0f}%)"


def format_table(rows):
    shown = lambda rank: "-" if rank is None else str(rank)
    lines = [f"{'id':<4} {'type':<10} {'page':>4} {'phrase':>6}  question"]
    for r in rows:
        lines.append(f"{r['id']:<4} {r['type']:<10} {shown(r['ranks']['page']):>4} "
                     f"{shown(r['ranks']['phrase']):>6}  {r['question']}")
    return "\n".join(lines)


def format_summary(summary):
    n = summary["n"]
    lines = [f"{'':<8}" + "".join(f"{f'hit@{k}':<14}" for k in CUTOFFS)]
    for test, at in summary["overall"].items():
        lines.append(f"{test:<8}" + "".join(f"{score(at[f'hit@{k}'], n):<14}" for k in CUTOFFS))
    lines += ["", "hit@5 by type:", f"  {'type':<10} {'page':<12} phrase"]
    for t, s in summary["hit@5_by_type"].items():
        lines.append(f"  {t:<10} {score(s['page'], s['n']):<12} {score(s['phrase'], s['n'])}")
    return "\n".join(lines)


def selftest():
    one_page = {"page": 5, "page_end": 5, "text": "3.2 Attention\nfoo bar"}
    two_page = {"page": 5, "page_end": 6, "text": "3.2 Attention\nfoo bar"}
    q6 = {"pages": [6], "phrase": "x"}
    assert not page_hit(one_page, q6), "one-page chunk on p5 misses gold p6"
    assert page_hit(two_page, q6), "chunk spanning p5-6 hits gold p6"
    assert page_hit(one_page, {"pages": [1, 5], "phrase": "x"}), "any of several gold pages counts"

    chunk = {"page": 1, "page_end": 1, "text": "Training\nwe used  N = 6\nidentical layers here"}
    assert phrase_hit(chunk, {"pages": [1], "phrase": "N = 6 identical layers"}), \
        "line breaks and double spaces in the chunk are normalized"
    assert phrase_hit(chunk, {"pages": [1], "phrase": "N  =\n6"}), "phrase whitespace is normalized too"
    assert not phrase_hit(chunk, {"pages": [1], "phrase": "N = 8"}), "different text is a miss"

    results = [{"page": p, "page_end": p, "text": ""} for p in (2, 7, 7, 3)]
    assert first_hit_rank(results, {"pages": [7]}, page_hit) == 2, "rank is 1-based, first hit wins"
    assert first_hit_rank(results, {"pages": [9]}, page_hit) is None, "no hit is None, not 0"
    assert first_hit_rank([], {"pages": [7]}, page_hit) is None, "empty results is None"

    ranks = [1, 3, None, 5, 10, 6]
    assert [hits_at(ranks, k) for k in (1, 5, 10)] == [1, 3, 5], "hit@k counts rank <= k, skips None"

    pages = [{"page": 1, "text": "a\nb c"}, {"page": 2, "text": "b c d"}]
    assert pages_with_phrase(pages, "b c") == [1, 2], "phrase found across a line break"
    assert bad_labels([{"id": "x", "pages": [1, 2], "phrase": "b c"}], pages) == [], "exact pages pass"
    assert bad_labels([{"id": "x", "pages": [2], "phrase": "b c"}], pages) == [("x", [2], [1, 2])], \
        "a phrase also on an unlabeled page is bad"

    rows = [{"id": "a", "type": "fact", "question": "?", "ranks": {"page": 1, "phrase": None}},
            {"id": "b", "type": "table", "question": "?", "ranks": {"page": 7, "phrase": 4}}]
    s = summarize(rows)
    assert s["overall"]["page"] == {"hit@1": 1, "hit@5": 1, "hit@10": 2}, "page hit@k"
    assert s["overall"]["phrase"] == {"hit@1": 0, "hit@5": 1, "hit@10": 1}, "phrase hit@k"
    assert s["hit@5_by_type"]["table"] == {"n": 1, "page": 0, "phrase": 1}, "per-type hit@5"
    assert "-" in format_table(rows).splitlines()[1], "missing rank shown as -"


def run(label):
    from step1_parse import clean, parse
    from step2_chunk import chunk
    from step3_embed import get_collection, retrieve

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    bad = bad_labels(questions, clean(parse(PAPER)))
    if bad:
        for qid, labeled, found in bad:
            print(f"bad label {qid}: labeled pages {labeled}, phrase found on {found}")
        sys.exit(f"{len(bad)} bad labels; fix eval/questions.json before scoring")
    print(f"labels: {len(questions)} checked, 0 bad\n")

    rows = []
    for q in questions:
        results = retrieve(q["question"], k=K)
        rows.append({"id": q["id"], "type": q["type"], "question": q["question"],
                     "ranks": {test: first_hit_rank(results, q, hit) for test, hit in HIT_TESTS.items()}})
    summary = summarize(rows)
    print(format_table(rows))
    print()
    print(format_summary(summary))

    # Read from chunk()'s defaults, which assumes the index was built with them
    # (step3_embed.ingest passes none). Step 6 must re-ingest after changing them.
    defaults = inspect.signature(chunk).parameters
    settings = {"k": K, "chunks": get_collection().count(),
                "chunk_size": defaults["size"].default, "chunk_overlap": defaults["overlap"].default}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{label or time.strftime('%Y%m%d-%H%M%S')}.json"
    out_file.write_text(json.dumps({"settings": settings, "questions": rows, "summary": summary}, indent=2),
                        encoding="utf-8")
    print(f"\nsettings: {settings}")
    print(f"results written to {out_file}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    command = sys.argv[1]
    if command == "run":
        run(sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else None)
    elif command == "selftest":
        selftest()
        print("selftest ok")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
