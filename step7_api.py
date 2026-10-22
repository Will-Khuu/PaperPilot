"""Step 7: serve retrieval and cited answers on this machine.

Usage:
    python step7_api.py serve
    python step7_api.py selftest  # requires the real index; no LLM calls
"""
import sys
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from step3_embed import EMPTY_INDEX, get_collection, retrieve
from step4_answer import MODEL, ask_claude, build_prompt, check_answer, page_label, source_label

app = FastAPI()


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=10)

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value):
        return value.strip() if isinstance(value, str) else value


class Health(BaseModel):
    chunks: int
    chunk_size: int
    chunk_overlap: int


class Excerpt(BaseModel):
    n: int
    id: str
    section: str
    title: str
    page: int
    page_end: int
    pages: str
    distance: float
    text: str


class Source(BaseModel):
    n: int
    id: str
    label: str


class SearchResult(BaseModel):
    question: str
    retrieved: List[Excerpt]


class AnswerResult(SearchResult):
    answer: str
    model: str
    fallback: bool
    sources: List[Source]
    invalid: List[int]
    refused: bool
    uncited: bool


def get_answerer():
    return ask_claude


def get_chunks(request: Question):
    try:
        return retrieve(request.question, k=request.k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None


def excerpts(chunks):
    return [Excerpt(n=n, pages=page_label(chunk), **chunk)
            for n, chunk in enumerate(chunks, start=1)]


@app.get("/health", response_model=Health)
def health():
    collection = get_collection()
    count = collection.count()
    if count == 0:
        raise HTTPException(status_code=503, detail=EMPTY_INDEX)
    return Health(chunks=count, **collection.metadata)


@app.post("/search", response_model=SearchResult)
def search(request: Question, chunks=Depends(get_chunks)):
    return SearchResult(question=request.question, retrieved=excerpts(chunks))


@app.post("/ask", response_model=AnswerResult)
def ask(request: Question, chunks=Depends(get_chunks), answerer=Depends(get_answerer)):
    try:
        answer, model = answerer(build_prompt(request.question, chunks))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from None
    checked = check_answer(chunks, answer)
    sources = [Source(n=n, id=chunks[n - 1]["id"], label=source_label(chunks[n - 1]))
               for n in checked.pop("cited")]
    return AnswerResult(question=request.question, answer=answer, model=model,
                        fallback=model != MODEL, retrieved=excerpts(chunks), sources=sources, **checked)


def selftest():
    """Offline local server tests; requires the real index built by step3 ingest."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient
    from step4_answer import NOT_COVERED

    question = "How many attention heads does the Transformer use?"
    payload = {"question": question, "k": 3}
    reply = ["Eight heads [2][1][2][99].", "fake-model"]
    prompts = []

    def fake_answerer(prompt):
        prompts.append(prompt)
        return tuple(reply)

    def fail_answerer(prompt):
        raise RuntimeError("The answer was cut off.")

    previous = app.dependency_overrides.copy()
    app.dependency_overrides[get_answerer] = lambda: fake_answerer
    try:
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200, "health reads the real index"
            assert response.json() == {"chunks": get_collection().count(), **get_collection().metadata}, \
                "health reports index count and stored settings"
            with patch(__name__ + ".get_collection") as empty_collection:
                empty_collection.return_value.count.return_value = 0
                empty_collection.return_value.metadata = None
                response = client.get("/health")
                assert response.status_code == 503 and response.json() == {"detail": EMPTY_INDEX}, \
                    "health returns 503 for an empty index without metadata or changing the real index"
            response = client.post("/search", json=payload)
            assert response.status_code == 200, "search succeeds without an answerer call"
            found = response.json()["retrieved"]
            assert [c["n"] for c in found] == [1, 2, 3], "search numbers k excerpts in retrieval order"
            assert all(c["pages"] == page_label(c) for c in found), "pages use the CLI label"
            assert not prompts, "search never calls the answerer"
            response = client.post("/ask", json=payload)
            assert response.status_code == 200, "fake answerer makes ask offline"
            result = response.json()
            assert result["sources"] == [
                {"n": n, "id": found[n - 1]["id"], "label": source_label(found[n - 1])}
                for n in (1, 2)
            ], "citations map to the same numbered excerpts, sorted and deduplicated"
            assert result["invalid"] == [99] and result["fallback"], "invalid citations and fallback are flagged"
            assert prompts[-1] == build_prompt(question, found), "answerer gets the existing numbered prompt"
            reply[:] = [NOT_COVERED, MODEL]
            result = client.post("/ask", json=payload).json()
            assert result["refused"] and not result["sources"] and not result["uncited"], "refusal has no sources"
            assert not result["fallback"], "primary model is not a fallback"
            reply[0] = "Plain prose."
            assert client.post("/ask", json=payload).json()["uncited"], "plain prose is uncited"
            for endpoint in ("/search", "/ask"):
                for invalid in ({"question": "x" * 501}, {"question": " "},
                                {"question": question, "k": 11}, {"question": question, "k": 0}):
                    assert client.post(endpoint, json=invalid).status_code == 422, "invalid input is rejected"
            result = client.post("/search", json={"question": "  " + question + "  "}).json()
            assert result["question"] == question and len(result["retrieved"]) == 5, "strip question; default k is five"
            app.dependency_overrides[get_answerer] = lambda: fail_answerer
            response = client.post("/ask", json=payload)
            assert response.status_code == 502 and response.json() == {"detail": "The answer was cut off."}, \
                "answerer RuntimeError becomes 502 with its message"
            with patch(__name__ + ".retrieve", side_effect=RuntimeError(EMPTY_INDEX)):
                for endpoint in ("/search", "/ask"):
                    response = client.post(endpoint, json=payload)
                    assert response.status_code == 503 and response.json() == {"detail": EMPTY_INDEX}, \
                        "empty index becomes 503 without changing the real index"
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    if sys.argv[1] == "serve":
        import uvicorn

        uvicorn.run(app, host="127.0.0.1", port=8000)
    elif sys.argv[1] == "selftest":
        selftest()
        print("selftest ok")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
