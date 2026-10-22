"""Step 9: summarize one paper with citations to its own chunks.

Usage:
    python step9_summary.py <paper_id>
    python step9_summary.py selftest  # real index, fake answerer, no API calls
"""
import sys

from step4_answer import MODEL, SUMMARY_PROMPT, ask_claude, build_prompt, check_answer, source_label


def paper_excerpts(paper_id, collection=None):
    from step3_embed import get_collection

    collection = get_collection() if collection is None else collection
    found = collection.get(
        where={"$and": [{"paper_id": paper_id}, {"section": {"$ne": "References"}}]},
        include=["documents", "metadatas"],
    )
    # There is no query vector in a summary, so distance has no value.
    chunks = [{"id": id_, "text": text, "distance": None, **meta}
              for id_, text, meta in zip(found["ids"], found["documents"], found["metadatas"])]
    if not chunks:
        raise RuntimeError(f"Unknown or empty paper {paper_id!r}: no non-References chunks in the index")
    return sorted(chunks, key=lambda chunk: chunk["chunk_index"])


def summarize(paper_id, answerer=ask_claude, collection=None):
    chunks = paper_excerpts(paper_id, collection=collection)
    title = chunks[0]["title"]
    summary, model = answerer(build_prompt(f"Summarize this paper: {title}", chunks), system=SUMMARY_PROMPT)
    checked = check_answer(chunks, summary)
    return {
        "paper_id": paper_id, "title": title, "summary": summary,
        "model": model, "fallback": model != MODEL,
        "sources": [{"n": n, "id": chunks[n - 1]["id"], "label": source_label(chunks[n - 1])}
                    for n in checked["cited"]],
        "invalid": checked["invalid"], "uncited": checked["uncited"], "chunks_used": len(chunks),
    }


def selftest():
    import re
    from unittest.mock import Mock
    from step3_embed import list_papers

    for paper in list_papers():
        pid = paper["paper_id"]
        chunks = paper_excerpts(pid)
        assert all(c["paper_id"] == pid and c["section"] != "References" for c in chunks), \
            "summaries use only this paper's non-References chunks"
        assert [c["chunk_index"] for c in chunks] == sorted(c["chunk_index"] for c in chunks), \
            "excerpts follow reading order"
        calls = []

        def fake_answerer(prompt, system):
            calls.append(prompt)
            assert system == SUMMARY_PROMPT, "summaries use their own formatting rules"
            labels = re.findall(r"^\[(\d+)\] \((.*)\)$", prompt, re.M)
            assert labels == [(str(n), source_label(c)) for n, c in enumerate(chunks, 1)], \
                "the prompt numbers every chunk and uses its source label"
            return "Overview [2][1][2].\n\nTakeaway [999].", "fake-model"

        result = summarize(pid, answerer=fake_answerer)
        assert calls == [build_prompt(f"Summarize this paper: {paper['title']}", chunks)], \
            "one answerer call summarizes the whole paper"
        assert result["sources"] == [{"n": n, "id": chunks[n - 1]["id"],
                                      "label": source_label(chunks[n - 1])} for n in (1, 2)], \
            "citations map to the numbered chunks, sorted and deduplicated"
        assert result["invalid"] == [999], "out-of-range citations are flagged"
        assert result["chunks_used"] == len(chunks) and result["fallback"], "count and fallback are reported"
        assert "refused" not in result and not result["uncited"], "summaries omit the refusal flag"
        plain = summarize(pid, answerer=lambda prompt, system: ("Plain prose.", MODEL))
        assert plain["uncited"] and not plain["fallback"], "uncited summaries and primary model are reported"
    assert list_papers(), "selftest requires a populated real index"
    empty = Mock()
    empty.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    for collection in (None, empty):
        try:
            paper_excerpts("unknown-paper", collection=collection)
        except RuntimeError as e:
            assert "unknown-paper" in str(e) and "no non-References chunks" in str(e), \
                "unknown and empty papers have a clear error"
        else:
            assert False, "unknown and empty papers must raise"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    if sys.argv[1] == "selftest":
        selftest()
        print("selftest ok")
        return
    result = summarize(sys.argv[1])
    print(result["summary"])
    print("\nSOURCES")
    for source in result["sources"]:
        print(f"  [{source['n']}] {source['label']}  ({source['id']})")
    if result["invalid"]:
        print(f"WARNING: invalid citation numbers: {result['invalid']}")
    if result["uncited"]:
        print("WARNING: no citations. This may come from outside knowledge.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        sys.exit(str(e))
