"""Step 7: serve retrieval and cited answers on this machine.

Usage:
    python step7_api.py serve
    python step7_api.py selftest  # requires the real index; no LLM calls
"""
import re
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, field_validator

from step3_embed import EMPTY_INDEX, get_collection, ingest, list_papers, retrieve
from step4_answer import COMPARE_PROMPT, MODEL, ask_claude, build_prompt, check_answer, page_label, source_label

app = FastAPI()
PAPERS_DIR = Path(__file__).parent / "papers"
MAX_UPLOAD = 20 * 1024 * 1024


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=10)
    paper_ids: Optional[List[str]] = None

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value):
        return value.strip() if isinstance(value, str) else value


class Comparison(Question):
    paper_ids: List[str] = Field(min_length=2, max_length=4)
    k_per_paper: int = Field(default=3, ge=1, le=5)

    @field_validator("paper_ids")
    @classmethod
    def distinct_papers(cls, value):
        if len(set(value)) != len(value):
            raise ValueError("Choose distinct papers")
        return value


class Paper(BaseModel):
    paper_id: str
    title: str
    chunks: int


class Health(BaseModel):
    chunks: int
    chunk_size: int
    chunk_overlap: int


class Excerpt(BaseModel):
    paper_id: str
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
    paper_id: str
    title: str
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


def validate_papers(paper_ids):
    known = {paper["paper_id"] for paper in list_papers()}
    if paper_ids and set(paper_ids) - known:
        raise HTTPException(status_code=422, detail=f"Unknown paper ids; known ids: {', '.join(sorted(known))}")


def get_chunks(request: Question):
    validate_papers(request.paper_ids)
    try:
        return retrieve(request.question, k=request.k, paper_ids=request.paper_ids)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from None


def excerpts(chunks):
    return [Excerpt(n=n, pages=page_label(chunk), **chunk)
            for n, chunk in enumerate(chunks, start=1)]


@app.get("/papers", response_model=List[Paper])
def papers():
    return list_papers()


@app.post("/papers", response_model=Paper)
async def upload_paper(file: UploadFile = File(...)):
    # Read one extra byte to detect oversized files without loading the whole body.
    data = await file.read(MAX_UPLOAD + 1)
    await file.close()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(status_code=422, detail="PDF must be at most 20 MB")
    name = (file.filename or "").replace("\\", "/").split("/")[-1]
    if not name.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
        raise HTTPException(status_code=422, detail="Upload a PDF filename and PDF contents")
    paper_id = re.sub(r"[^a-z0-9-]", "", Path(name).stem.lower())
    if not paper_id:
        raise HTTPException(status_code=422, detail="Filename must contain a letter, digit or hyphen")
    if any(p["paper_id"] == paper_id for p in list_papers()):
        raise HTTPException(status_code=409, detail=f"Paper {paper_id} is already indexed")
    path = PAPERS_DIR / f"{paper_id}.pdf"
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive creation also prevents overwriting a file or following a symlink.
        with path.open("xb") as saved:
            saved.write(data)
    except FileExistsError:
        raise HTTPException(status_code=409, detail=f"File for {paper_id} already exists") from None
    try:
        ingest(path)
        return next(p for p in list_papers() if p["paper_id"] == paper_id)
    except Exception:
        path.unlink(missing_ok=True)
        get_collection().delete(where={"paper_id": paper_id})
        raise HTTPException(status_code=422, detail="Could not parse or index this PDF") from None


@app.post("/compare", response_model=AnswerResult)
def compare(request: Comparison, answerer=Depends(get_answerer)):
    validate_papers(request.paper_ids)
    chunks = []
    for pid in request.paper_ids:
        try:
            found = retrieve(request.question, k=request.k_per_paper, paper_ids=[pid])
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from None
        if not found:
            raise HTTPException(status_code=422, detail=f"Paper {pid} has no searchable chunks")
        chunks.extend(found)
    return answer_result(request, chunks, answerer, system=COMPARE_PROMPT)


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
    return answer_result(request, chunks, answerer)


def answer_result(request, chunks, answerer, system=None):
    try:
        prompt = build_prompt(request.question, chunks)
        answer, model = answerer(prompt, system=system) if system else answerer(prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from None
    checked = check_answer(chunks, answer)
    sources = [Source(n=n, id=chunks[n - 1]["id"], label=source_label(chunks[n - 1]),
                      paper_id=chunks[n - 1]["paper_id"], title=chunks[n - 1]["title"])
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

    def fake_answerer(prompt, system=None):
        prompts.append(prompt)
        if system is not None:
            assert system == COMPARE_PROMPT
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
                {"n": n, "id": found[n - 1]["id"], "label": source_label(found[n - 1]),
                 "paper_id": found[n - 1]["paper_id"], "title": found[n - 1]["title"]}
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
            indexed = client.get("/papers").json()
            assert indexed == list_papers()
            pid = indexed[0]["paper_id"]
            filtered = client.post("/search", json={**payload, "paper_ids": [pid]}).json()
            assert all(c["paper_id"] == pid for c in filtered["retrieved"])
            for endpoint in ("/search", "/ask", "/compare"):
                response = client.post(endpoint, json={**payload, "paper_ids": [pid, "unknown"]})
                assert response.status_code == 422 and pid in response.json()["detail"]
            ids = [p["paper_id"] for p in indexed[:2]]
            reply[0] = "First [1]. Second [4]."
            result = client.post("/compare", json={"question": question, "paper_ids": ids}).json()
            assert [c["n"] for c in result["retrieved"]] == list(range(1, 7))
            assert [c["paper_id"] for c in result["retrieved"]] == [ids[0]] * 3 + [ids[1]] * 3
            assert [c["paper_id"] for c in result["sources"]] == ids
            for bad_ids in ([pid], [pid] * 5, [pid, pid]):
                assert client.post("/compare", json={"question": question, "paper_ids": bad_ids}).status_code == 422
            for k in (0, 6):
                assert client.post("/compare", json={"question": question, "paper_ids": ids,
                                                    "k_per_paper": k}).status_code == 422
            with patch(__name__ + ".retrieve", return_value=[]):
                assert client.post("/compare", json={"question": question, "paper_ids": ids}).status_code == 422
            test_uploads(client, indexed)
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


def test_uploads(client, indexed):
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    import fitz

    with TemporaryDirectory() as directory:
        save_dir = Path(directory) / "papers"
        records = list(indexed)

        def fake_ingest(path):
            assert path.parent == save_dir
            with fitz.open(path) as pdf:
                assert pdf.page_count == 1
            records.append({"paper_id": path.stem, "title": "Uploaded paper", "chunks": 1})

        with patch(__name__ + ".PAPERS_DIR", save_dir), \
             patch(__name__ + ".list_papers", side_effect=lambda: records), \
             patch(__name__ + ".ingest", side_effect=fake_ingest) as ingest_mock, \
             patch(__name__ + ".get_collection") as collection:
            def upload(name, data):
                return client.post("/papers", files={"file": (name, data, "application/pdf")})

            assert upload("wrong.pdf", b"not a pdf").status_code == 422
            assert upload("large.pdf", b"%PDF" + b"x" * MAX_UPLOAD).status_code == 422
            assert upload(indexed[0]["paper_id"] + ".pdf", b"%PDF").status_code == 409
            assert upload("!!!.pdf", b"%PDF").status_code == 422
            assert upload("wrong.txt", b"%PDF").status_code == 422
            assert not ingest_mock.called
            with fitz.open() as pdf:
                pdf.new_page().insert_text((72, 72), "Uploaded paper")
                data = pdf.tobytes()
            response = upload("../../Uploaded Paper.PDF", data)
            assert response.status_code == 200, response.text
            assert response.json() == records[-1]
            assert (save_dir / "uploadedpaper.pdf").read_bytes() == data
            ingest_mock.side_effect = ValueError("bad PDF")
            assert upload("broken.pdf", b"%PDF broken").status_code == 422
            assert not (save_dir / "broken.pdf").exists()
            collection.return_value.delete.assert_called_once_with(where={"paper_id": "broken"})


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
