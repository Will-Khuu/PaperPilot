"""Step 6: rerun the 20 questions across chunk sizes and overlaps. No LLM.

Usage:
    python step6_sweep.py run        # 12 settings, table + verdict, out/eval/sweep.json
    python step6_sweep.py selftest   # the decision rule only: no Chroma, no network

Each setting gets its own throwaway in-memory collection, so the real index in
chroma_db/ is never touched. Scoring is Step 5's own code, so a number here
means the same thing as a number in out/eval/baseline.json.
"""
import json
import sys
import uuid

from step5_eval import CUTOFFS, OUT_DIR, PAPER, checked_questions, score_questions, summarize

# step3_embed (Chroma, ONNX model) is imported inside run(), so selftest loads
# with the stdlib only.

SIZES = (100, 160, 220, 256)  # 256 is the model's limit, its two markers included
OVERLAPS = (0, 40, 80)
BASELINE = (220, 40)
OUT_FILE = OUT_DIR / "sweep.json"

# Phrase hit@1 is the target because hit@5 is near ceiling at the baseline
# (18/20). +2 questions, not +1: one question moving could be a single lucky
# tie in distance, two is a pattern. hit@5 must not drop, so a winner doesn't
# buy rank-1 hits by losing answers from the top 5 the LLM actually sees.
MIN_GAIN_AT_1 = 2


def settings_of(result):
    return result["chunk_size"], result["chunk_overlap"]


def qualifies(phrase, base_phrase):
    """phrase, base_phrase: {"hit@1": n, "hit@5": n, ...} for a setting and the baseline."""
    return (phrase["hit@1"] >= base_phrase["hit@1"] + MIN_GAIN_AT_1
            and phrase["hit@5"] >= base_phrase["hit@5"])


def winners(results):
    """[(size, overlap)] of every setting that beats the baseline by the rule above."""
    base = next(r for r in results if settings_of(r) == BASELINE)["summary"]["overall"]["phrase"]
    return [settings_of(r) for r in results if qualifies(r["summary"]["overall"]["phrase"], base)]


def format_sweep(results):
    n = results[0]["summary"]["n"]
    lines = [f"{'size':>4} {'overlap':>7} {'chunks':>6}   {'page @1':>7} {'@5':>3}   "
             f"{'phrase @1':>9} {'@5':>3} {'@10':>3}   (counts out of {n})"]
    for r in results:
        page, phrase = r["summary"]["overall"]["page"], r["summary"]["overall"]["phrase"]
        mark = "  <- baseline" if settings_of(r) == BASELINE else ""
        lines.append(f"{r['chunk_size']:>4} {r['chunk_overlap']:>7} {r['chunks']:>6}   "
                     f"{page['hit@1']:>7} {page['hit@5']:>3}   "
                     f"{phrase['hit@1']:>9} {phrase['hit@5']:>3} {phrase['hit@10']:>3}{mark}")
    return "\n".join(lines)


def selftest():
    base = {"hit@1": 9, "hit@5": 18}
    assert qualifies({"hit@1": 11, "hit@5": 18}, base), "+2 at @1 with @5 held qualifies"
    assert qualifies({"hit@1": 12, "hit@5": 19}, base), "@5 going up is fine"
    assert not qualifies({"hit@1": 10, "hit@5": 20}, base), "+1 at @1 is not enough"
    assert not qualifies({"hit@1": 13, "hit@5": 17}, base), "any drop at @5 disqualifies"
    assert not qualifies(base, base), "the baseline never beats itself"

    def result(size, overlap, at1, at5):
        return {"chunk_size": size, "chunk_overlap": overlap,
                "summary": {"overall": {"phrase": {"hit@1": at1, "hit@5": at5}}}}

    results = [result(100, 0, 12, 18), result(220, 40, 9, 18), result(256, 80, 11, 17)]
    assert winners(results) == [(100, 0)], "only settings passing both conditions are listed"
    assert winners([result(220, 40, 9, 18), result(160, 0, 10, 19)]) == [], "no winner is an empty list"


def run():
    import chromadb
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    from step3_embed import add_chunks, paper_chunks

    questions = checked_questions()
    client = chromadb.EphemeralClient()
    results = []
    for size in SIZES:
        for overlap in OVERLAPS:
            chunks = paper_chunks(PAPER, size=size, overlap=overlap)
            # A unique name, so if a delete ever fails, the next setting can't
            # quietly score against the previous setting's chunks.
            name = f"sweep-{size}-{overlap}-{uuid.uuid4().hex[:8]}"
            collection = client.create_collection(
                name,
                embedding_function=ONNXMiniLM_L6_V2(),
                configuration={"hnsw": {"space": "cosine"}},
                metadata={"chunk_size": size, "chunk_overlap": overlap},
            )
            try:
                add_chunks(collection, chunks)
                rows = score_questions(questions, collection, method="vector")
            finally:
                client.delete_collection(name)
            results.append({"chunk_size": size, "chunk_overlap": overlap, "chunks": len(chunks),
                            "questions": rows, "summary": summarize(rows)})
            print(f"scored {size}/{overlap}: {len(chunks)} chunks")

    qualifying = winners(results)
    print()
    print(format_sweep(results))
    print(f"\nrule: phrase hit@1 at least +{MIN_GAIN_AT_1} over {BASELINE[0]}/{BASELINE[1]}, "
          f"and phrase hit@5 not lower")
    if qualifying:
        print("qualifying: " + ", ".join(f"{s}/{o}" for s, o in qualifying))
    else:
        print(f"no clear winner; keep {BASELINE[0]}/{BASELINE[1]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({
        "grid": {"sizes": SIZES, "overlaps": OVERLAPS, "baseline": BASELINE, "cutoffs": CUTOFFS},
        "rule": {"min_gain_phrase_hit@1": MIN_GAIN_AT_1, "phrase_hit@5": "not lower than baseline"},
        "results": results,
        "qualifying": qualifying,
    }, indent=2), encoding="utf-8")
    print(f"results written to {OUT_FILE}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    command = sys.argv[1]
    if command == "run":
        run()
    elif command == "selftest":
        selftest()
        print("selftest ok")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
