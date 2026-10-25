"""Step 12: combine keyword and vector rankings without an LLM.

BM25 gives rare words more weight. Repeated words help less as their count
grows, controlled by K1; B penalizes long chunks so length alone cannot win.
Reciprocal rank fusion (RRF) uses positions, because BM25 scores and cosine
distances have different scales and cannot sensibly be added directly.

Usage:
    python step12_hybrid.py selftest  # stdlib only, no index or network
    python step12_hybrid.py show "question" --paper attention
"""
import argparse
import math
import re
from collections import Counter

# Standard, untuned Okapi defaults balance repeated words and chunk length.
K1 = 1.5
B = 0.75
RRF_K = 60


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_scores(question, texts):
    """Score each text using word rarity within this set of texts only."""
    counts = [Counter(tokenize(text)) for text in texts]
    if not counts:
        return []
    lengths = [sum(count.values()) for count in counts]
    avglen = sum(lengths) / len(lengths)
    if avglen == 0:
        return [0.0] * len(texts)
    df = Counter(token for count in counts for token in count)
    terms = sorted(set(tokenize(question)))
    idf = {t: math.log(1 + (len(texts) - df[t] + 0.5) / (df[t] + 0.5)) for t in terms}
    scores = []
    for count, length in zip(counts, lengths):
        score = 0.0
        for t in terms:
            tf = count[t]
            score += idf[t] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avglen))
        scores.append(score)
    return scores


def fuse(vector_ids_in_order, keyword_scores_by_id):
    """Return (id, RRF score, vector rank, keyword rank), best first.

    Every candidate has both ranks, even when its keyword score is zero.
    Missing vector candidates rank last, in keyword_scores_by_id insertion order.
    Vector order breaks keyword ties and then breaks fusion ties.
    """
    vector_ranks = {id_: rank for rank, id_ in enumerate(vector_ids_in_order, start=1)}
    for id_ in keyword_scores_by_id:
        if id_ not in vector_ranks:
            vector_ranks[id_] = len(vector_ranks) + 1
    keyword_order = sorted(vector_ranks,
                           key=lambda id_: (-keyword_scores_by_id[id_], vector_ranks[id_]))
    keyword_ranks = {id_: rank for rank, id_ in enumerate(keyword_order, start=1)}
    fused = [(id_, 1 / (RRF_K + vr) + 1 / (RRF_K + keyword_ranks[id_]), vr, keyword_ranks[id_])
             for id_, vr in vector_ranks.items()]
    return sorted(fused, key=lambda row: (-row[1], row[2]))


def selftest():
    assert tokenize("BLEU 28.4 EN-DE, ByteNet!") == ["bleu", "28", "4", "en", "de", "bytenet"]
    assert tokenize("a_b/c\nD") == ["a", "b", "c", "d"]
    assert tokenize("") == []
    texts = ["rare common", "common extra", "common extra"]
    assert bm25_scores("rare", texts)[0] > bm25_scores("common", texts)[0]
    assert all(score > 0 for score in bm25_scores("common", texts))
    scores = bm25_scores("word", ["word " + "filler " * 9, "word " * 10])
    assert scores[1] < 10 * scores[0], "repeated words saturate"
    short, long = bm25_scores("word", ["word", "word filler filler filler"])
    assert short > long, "same tf, longer chunk scores lower"
    assert bm25_scores("missing", texts) == [0.0, 0.0, 0.0]
    assert bm25_scores("rare missing", texts) == bm25_scores("rare", texts)
    assert bm25_scores("rare rare", texts) == bm25_scores("rare", texts)
    assert bm25_scores("word", []) == []
    assert bm25_scores("word", ["", "!"]) == [0.0, 0.0]

    ranked = {row[0]: row for row in fuse(["a", "b", "c"], {"a": 0, "b": 2, "c": 1})}
    assert ranked["a"][1:] == (1 / 61 + 1 / 63, 1, 3)
    assert ranked["b"][1:] == (1 / 62 + 1 / 61, 2, 1)
    tied = fuse(["a", "b"], {"a": 0, "b": 1})
    assert tied[0][1] == tied[1][1] and tied[0][0] == "a", "fusion ties use vector rank"
    zeros = fuse(["a", "b", "c"], {"a": 0, "b": 0, "c": 0})
    assert [(row[2], row[3]) for row in zeros] == [(1, 1), (2, 2), (3, 3)]
    assert fuse([], {}) == []
    ranked = fuse(["a", "b", "c", "d"], {"a": 2, "b": 1, "c": 3, "d": 4})
    assert ranked[1][0] == "d" and ranked[1][2:] == (4, 1), "last in vectors reaches the top two"
    ranked = {row[0]: row for row in fuse(["b"], {"a": 3, "b": 0, "c": 1})}
    assert {id_: row[2] for id_, row in ranked.items()} == {"a": 2, "b": 1, "c": 3}
    assert ranked["a"][1:] == (1 / 62 + 1 / 61, 2, 1), "keyword-only candidates participate"
    assert [row[0] for row in fuse([], {"c": 0, "a": 0})] == ["c", "a"]


def show(question, paper_ids=None):
    from step3_embed import get_collection, retrieve

    collection = get_collection()
    vector = retrieve(question, k=10, paper_ids=paper_ids, collection=collection, method="vector")
    # Full results supply ranks for vector hits outside the hybrid top 10.
    all_hybrid = retrieve(question, k=collection.count(), paper_ids=paper_ids, collection=collection,
                          method="hybrid")
    ranks = {hit["id"]: (hit["vector_rank"], hit["keyword_rank"]) for hit in all_hybrid}
    hybrid = all_hybrid[:10]

    def cell(hit):
        vr, kr = ranks[hit["id"]]
        distance = "-" if hit["distance"] is None else f"{hit['distance']:.3f}"
        return (f"{hit['id']}  v={vr} kw={kr} d={distance} "
                f"p{hit['page']}-{hit['page_end']} {hit['section']}")

    print(f"Q: {question}")
    width = max([len("Vector")] + [len(cell(hit)) for hit in vector])
    print(f"    {'Vector':<{width}} | Hybrid")
    for i in range(max(len(vector), len(hybrid))):
        left = cell(vector[i]) if i < len(vector) else ""
        right = cell(hybrid[i]) if i < len(hybrid) else ""
        print(f"{i + 1:>2}. {left:<{width}} | {right}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("selftest")
    demo = commands.add_parser("show")
    demo.add_argument("question")
    demo.add_argument("--paper", action="append")
    args = parser.parse_args()
    if args.command == "selftest":
        selftest()
        print("selftest ok")
    else:
        show(args.question, args.paper)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(str(e))
