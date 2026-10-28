"""Step 7: serve retrieval and cited answers on this machine.

Usage:
    python step7_api.py serve
    python step7_api.py selftest  # requires the real index; no LLM calls
"""
import re
import os
import math
import secrets
import sys
import time
from pathlib import Path
from threading import Lock
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
from anthropic import APIConnectionError, APIStatusError, APITimeoutError
import fitz

from step3_embed import EMPTY_INDEX, RETRIEVAL, get_collection, ingest, list_papers, retrieve
from step4_answer import COMPARE_PROMPT, SUMMARY_PROMPT, SYSTEM_PROMPT, MODEL, ask_claude, build_prompt, check_answer, page_label, source_label

from step9_summary import summarize
import step10_notes as notes
import step11_cache as cache
import deploy
import auth
from deploy import PAPERS_DIR
from step11_cache import index_fingerprint

app = FastAPI()
paper_lock = Lock()
BUSY_LIBRARY = "Another upload is in progress. Try again in a minute."


def configured_upload_mb():
    try:
        value = float(os.environ.get("PAPERPILOT_MAX_UPLOAD_MB", "4"))
        if not math.isfinite(value * 1024 * 1024) or value <= 0:
            raise ValueError
        return value
    except ValueError:
        print("Warning: invalid PAPERPILOT_MAX_UPLOAD_MB; using 4 MB.", file=sys.stderr)
        return 4.0


DEPLOYED_UPLOAD_MB = configured_upload_mb()
DEPLOYED_UPLOAD_BYTES = int(DEPLOYED_UPLOAD_MB * 1024 * 1024)


def max_upload_mb():
    return DEPLOYED_UPLOAD_MB if deploy.DEPLOYED else 20


@app.middleware("http")
async def block_demo_uploads(request, call_next):
    # Reject before FastAPI's multipart parser reads the upload body.
    path = request.url.path.rstrip("/")
    changes_papers = ((request.method == "POST" and path == "/papers") or
                      (request.method == "DELETE" and re.fullmatch(r"/papers/[^/]+", path)))
    if deploy.DEPLOYED and changes_papers and request.state.role != "owner":
        return JSONResponse(status_code=403, content={"detail": "Only the owner can add or delete papers."})
    return await call_next(request)


@app.middleware("http")
async def require_session(request, call_next):
    session = auth.read_session(request.cookies.get(auth.COOKIE_NAME))
    request.state.session_id, request.state.role = session if session else (None, None)
    if deploy.DEPLOYED and request.url.path not in ("/health", "/login", "/session"):
        if not os.environ.get("APP_PASSWORD"):
            return JSONResponse(status_code=503, content={"detail": "This demo is not configured."})
        if request.state.session_id is None:
            return JSONResponse(status_code=401, content={"detail": "Sign in first."})
    return await call_next(request)


class Login(BaseModel):
    password: str = Field(max_length=256)


@app.post("/login")
def login(body: Login, response: Response):
    if not deploy.DEPLOYED:
        return {"authenticated": True}
    if not os.environ.get("APP_PASSWORD"):
        raise HTTPException(status_code=503, detail="This demo is not configured.")
    role, error = auth.check_login(body.password)
    if error == "limited":
        raise HTTPException(status_code=429, detail="Too many failed sign-ins. Wait a minute.")
    if error == "incorrect":
        raise HTTPException(status_code=401, detail="Incorrect password.")
    cookie = auth.sign(secrets.token_urlsafe(16), role, int(time.time()) + auth.SESSION_SECONDS)
    response.set_cookie(auth.COOKIE_NAME, cookie, max_age=auth.SESSION_SECONDS,
                        httponly=True, secure=True, samesite="lax", path="/")
    return {"authenticated": True}


@app.post("/logout")
def logout(response: Response):
    response.delete_cookie(auth.COOKIE_NAME, path="/", httponly=True, secure=True, samesite="lax")
    return {"authenticated": False}


@app.get("/session")
def session(request: Request):
    return {"deployed": deploy.DEPLOYED,
            "authenticated": not deploy.DEPLOYED or request.state.session_id is not None,
            "role": request.state.role,
            "can_upload": not deploy.DEPLOYED or request.state.role == "owner",
            "max_upload_mb": max_upload_mb()}


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
    pages: int
    builtin: bool


class Health(BaseModel):
    chunks: int
    chunk_size: int
    chunk_overlap: int
    answers_left: Optional[int] = None
    daily_limit: Optional[int] = None
    session_limit: Optional[int] = None


class Excerpt(BaseModel):
    paper_id: str
    n: int
    id: str
    section: str
    title: str
    page: int
    page_end: int
    pages: str
    distance: Optional[float]  # None when a hybrid candidate had no vector hit
    text: str


class Source(BaseModel):
    paper_id: str
    title: str
    n: int
    id: str
    label: str
    page: Optional[int] = None
    url: Optional[str] = None


class NoteSource(BaseModel):
    n: int
    label: str


class NoteRequest(BaseModel):
    paper_id: str
    text: str = Field(min_length=1, max_length=5000)
    question: Optional[str] = None
    answer: Optional[str] = None
    sources: Optional[List[NoteSource]] = None

    @field_validator("text", mode="before")
    @classmethod
    def strip_text(cls, value):
        return value.strip() if isinstance(value, str) else value


class SummaryRequest(BaseModel):
    paper_id: str


class SummarySource(BaseModel):
    n: int
    id: str
    label: str
    paper_id: Optional[str] = None
    page: Optional[int] = None
    url: Optional[str] = None


class SummaryResult(BaseModel):
    paper_id: str
    title: str
    summary: str
    model: str
    fallback: bool
    sources: List[SummarySource]
    invalid: List[int]
    uncited: bool
    chunks_used: int
    cached: bool


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
    cached: bool


def get_answerer(request: Request):
    if deploy.DEPLOYED:
        def demo_answerer(prompt, system=None):
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise HTTPException(status_code=503, detail="This demo is not configured.")
            day, reason = deploy.take_llm_call(request.state.session_id, path=notes.DB_PATH)
            if reason == "global":
                raise HTTPException(status_code=429, detail=(
                    f"Daily demo limit of {deploy.DAILY_LLM_LIMIT} answers reached. "
                    "Search only, cached answers and notes still work."))
            if reason == "session":
                raise HTTPException(status_code=429, detail=(
                    "You've used your 10 paid answers for today. "
                    "Search, cached answers and notes still work."))
            try:
                return ask_claude(prompt, system=system) if system else ask_claude(prompt)
            except RuntimeError as error:
                # Timeouts may have generated billed tokens on the server, as do
                # refusals and cut-offs. Refund only other API/connection failures.
                cause = error.__cause__
                if isinstance(cause, (APIConnectionError, APIStatusError)) and not isinstance(cause, APITimeoutError):
                    deploy.release_llm_call(request.state.session_id, day, path=notes.DB_PATH)
                raise

        return demo_answerer
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


def with_source_links(result):
    # Enrich at response time so old cache records get links without a new cache key.
    metadata = {}
    if "retrieved" not in result and result["sources"]:
        found = get_collection().get(ids=[s["id"] for s in result["sources"]], include=["metadatas"])
        metadata = dict(zip(found["ids"], found["metadatas"]))
    sources = []
    for source in result["sources"]:
        chunk = (result["retrieved"][source["n"] - 1] if "retrieved" in result
                 else metadata.get(source["id"], {}))
        paper_id = chunk.get("paper_id") or source.get("paper_id") or result.get("paper_id")
        page = chunk.get("page")
        url = None
        if paper_id in deploy.PAPERS and page is not None:
            url = f"{deploy.PAPERS[paper_id]}#page={page}"
        elif paper_id and page is not None:
            url = f"/papers/{paper_id}/pdf#page={page}"
        sources.append({**source, "paper_id": paper_id, "page": page, "url": url})
    return {**result, "sources": sources}


@app.get("/papers", response_model=List[Paper])
def papers():
    return list_papers()


@app.post("/papers", response_model=Paper)
async def upload_paper(file: UploadFile = File(...)):
    if not paper_lock.acquire(blocking=False):
        await file.close()
        raise HTTPException(status_code=409, detail=BUSY_LIBRARY)
    try:
        # Read one extra byte to detect oversized files without loading the whole body.
        limit_mb = max_upload_mb()
        limit = DEPLOYED_UPLOAD_BYTES if deploy.DEPLOYED else 20 * 1024 * 1024
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(status_code=422, detail=f"PDF must be at most {limit_mb:g} MB")
        name = (file.filename or "").replace("\\", "/").split("/")[-1]
        if not name.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
            raise HTTPException(status_code=422, detail="Upload a PDF filename and PDF contents")
        paper_id = re.sub(r"[^a-z0-9-]", "", Path(name).stem.lower())
        if not paper_id:
            raise HTTPException(status_code=422, detail="Filename must contain a letter, digit or hyphen")
        # Parsing and embedding can take seconds; keep health and search requests responsive.
        return await run_in_threadpool(save_paper, paper_id, data)
    finally:
        paper_lock.release()
        await file.close()


def save_paper(paper_id, data):
    indexed = list_papers()
    if any(p["paper_id"] == paper_id for p in indexed):
        raise HTTPException(status_code=409, detail=f"Paper {paper_id} is already indexed")
    if len(indexed) >= 20:
        raise HTTPException(status_code=409, detail="The library is full (20 papers). Delete one first.")
    try:
        with fitz.open(stream=data, filetype="pdf") as pdf:
            if pdf.page_count > 60:
                raise HTTPException(status_code=422, detail="PDFs over 60 pages aren't supported here.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=422, detail="Could not parse or index this PDF") from None
    path = PAPERS_DIR / f"{paper_id}.pdf"
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    # A crash after saving but before indexing can leave an orphan PDF. The index
    # check above protects real papers; reclaim this name while holding paper_lock.
    path.unlink(missing_ok=True)
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


@app.delete("/papers/{paper_id}", status_code=204)
def delete_paper(paper_id: str):
    if not paper_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=BUSY_LIBRARY)
    try:
        if not re.fullmatch(r"[a-z0-9-]+", paper_id) or not any(
                paper["paper_id"] == paper_id for paper in list_papers()):
            raise HTTPException(status_code=404, detail="Paper not found")
        if paper_id in deploy.PAPERS:
            raise HTTPException(status_code=403, detail="Built-in papers can't be deleted.")
        get_collection().delete(where={"paper_id": paper_id})
        (PAPERS_DIR / f"{paper_id}.pdf").unlink(missing_ok=True)
        notes.delete_notes_for_paper(paper_id, path=notes.DB_PATH)
        return Response(status_code=204)
    finally:
        paper_lock.release()


@app.get("/papers/{paper_id}/pdf")
def paper_pdf(paper_id: str):
    if not re.fullmatch(r"[a-z0-9-]+", paper_id) or not any(
            paper["paper_id"] == paper_id for paper in list_papers()):
        raise HTTPException(status_code=404, detail="Paper not found")
    path = PAPERS_DIR / f"{paper_id}.pdf"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Paper not found")
    return FileResponse(path, media_type="application/pdf", filename=f"{paper_id}.pdf",
                        content_disposition_type="inline", headers={"X-Content-Type-Options": "nosniff"})


@app.post("/compare", response_model=AnswerResult)
def compare(request: Comparison, answerer=Depends(get_answerer)):
    validate_papers(request.paper_ids)
    key = cache.cache_key("compare", request.question, request.paper_ids, k_per_paper=request.k_per_paper,
                          model=MODEL, prompt=COMPARE_PROMPT, fingerprint=index_fingerprint(request.paper_ids),
                          retrieval=RETRIEVAL)
    stored = cache.get(key, path=notes.DB_PATH)
    if stored is not None:
        return with_source_links({**stored, "cached": True})
    chunks = []
    for pid in request.paper_ids:
        try:
            found = retrieve(request.question, k=request.k_per_paper, paper_ids=[pid])
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from None
        if not found:
            raise HTTPException(status_code=422, detail=f"Paper {pid} has no searchable chunks")
        chunks.extend(found)
    result = answer_result(request, chunks, answerer, system=COMPARE_PROMPT)
    cache.put(key, result.model_dump(), path=notes.DB_PATH)
    return with_source_links(result.model_dump())


@app.get("/health", response_model=Health)
def health(request: Request):
    collection = get_collection()
    count = collection.count()
    if count == 0:
        raise HTTPException(status_code=503, detail=EMPTY_INDEX)
    answers_left = None
    if deploy.DEPLOYED:
        answers_left = max(0, deploy.DAILY_LLM_LIMIT - deploy.llm_calls_today(path=notes.DB_PATH))
        if request.state.session_id is not None:
            session_left = max(0, deploy.SESSION_DAILY_LIMIT - deploy.session_calls_today(
                request.state.session_id, path=notes.DB_PATH))
            answers_left = min(answers_left, session_left)
    daily_limit = deploy.DAILY_LLM_LIMIT if deploy.DEPLOYED else None
    session_limit = deploy.SESSION_DAILY_LIMIT if deploy.DEPLOYED else None
    return Health(chunks=count, answers_left=answers_left, daily_limit=daily_limit,
                  session_limit=session_limit, **collection.metadata)


@app.post("/search", response_model=SearchResult)
def search(request: Question, chunks=Depends(get_chunks)):
    return SearchResult(question=request.question, retrieved=excerpts(chunks))


@app.post("/ask", response_model=AnswerResult)
def ask(request: Question, answerer=Depends(get_answerer)):
    validate_papers(request.paper_ids)
    key = cache.cache_key("ask", request.question, request.paper_ids, k=request.k,
                          model=MODEL, prompt=SYSTEM_PROMPT, fingerprint=index_fingerprint(request.paper_ids),
                          retrieval=RETRIEVAL)
    stored = cache.get(key, path=notes.DB_PATH)
    if stored is not None:
        return with_source_links({**stored, "cached": True})
    result = answer_result(request, get_chunks(request), answerer)
    cache.put(key, result.model_dump(), path=notes.DB_PATH)
    return with_source_links(result.model_dump())


@app.post("/notes")
def save_note(request: NoteRequest):
    validate_papers([request.paper_id])
    return notes.add_note(**request.model_dump(), path=notes.DB_PATH)


@app.get("/notes")
def get_notes(paper_id: Optional[str] = None):
    return notes.list_notes(paper_id, path=notes.DB_PATH)


@app.get("/notes/export")
def export_notes(paper_id: str):
    validate_papers([paper_id])
    title = next(p["title"] for p in list_papers() if p["paper_id"] == paper_id)
    return Response(notes.export_markdown(paper_id, title, path=notes.DB_PATH), media_type="text/markdown",
                    headers={"Content-Disposition": f'attachment; filename="notes-{paper_id}.md"'})


@app.delete("/notes/{note_id}", status_code=204)
def remove_note(note_id: int):
    if not notes.delete_note(note_id, path=notes.DB_PATH):
        raise HTTPException(status_code=404, detail="Note not found")
    return Response(status_code=204)


@app.post("/summarize", response_model=SummaryResult)
def summarize_paper(request: SummaryRequest, answerer=Depends(get_answerer)):
    collection = get_collection()
    if collection.count() == 0:
        raise HTTPException(status_code=503, detail=EMPTY_INDEX)
    validate_papers([request.paper_id])
    key = cache.cache_key("summary", paper_ids=[request.paper_id], model=MODEL, prompt=SUMMARY_PROMPT,
                          fingerprint=index_fingerprint([request.paper_id], collection=collection))
    stored = cache.get(key, path=notes.DB_PATH)
    if stored is not None:
        return with_source_links({**stored, "cached": True})
    try:
        result = SummaryResult(**summarize(request.paper_id, answerer=answerer, collection=collection), cached=False)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from None
    cache.put(key, result.model_dump(), path=notes.DB_PATH)
    return with_source_links(result.model_dump())


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
                        fallback=model != MODEL, retrieved=excerpts(chunks), sources=sources, cached=False, **checked)


def selftest():
    from shutil import copytree
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    import step3_embed

    # Chroma opens SQLite read-write even for queries. Test a copy, never the shared index.
    with TemporaryDirectory() as directory:
        index = Path(directory) / "chroma_db"
        copytree(step3_embed.DB_PATH, index)
        with patch.object(step3_embed, "DB_PATH", str(index)), \
             patch(__name__ + ".ask_claude", side_effect=AssertionError("Selftests must never call the paid answerer")):
            test_local_api()


def test_local_api():
    """Offline local server tests; requires the real index built by step3 ingest."""
    from unittest.mock import patch
    from tempfile import TemporaryDirectory

    from fastapi.testclient import TestClient
    from step4_answer import NOT_COVERED

    question = "How many attention heads does the Transformer use?"
    assert len(retrieve(question, paper_ids=[])) == len(retrieve(question, paper_ids=None)), \
        "an empty paper filter searches all papers in the real index"
    payload = {"question": question, "k": 3}
    reply = ["Eight heads [2][1][2][99].", "fake-model"]
    prompts = []

    def fake_answerer(prompt, system=None):
        prompts.append(prompt)
        if system is not None:
            assert system in (COMPARE_PROMPT, SUMMARY_PROMPT)
        return tuple(reply)

    def fail_answerer(prompt, system=None):
        raise RuntimeError("The answer was cut off.")

    previous = app.dependency_overrides.copy()
    app.dependency_overrides[get_answerer] = lambda: fake_answerer
    try:
        with TemporaryDirectory() as directory, patch.object(notes, "DB_PATH", Path(directory) / "notes.db"), \
             patch.object(deploy, "DEPLOYED", False), TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200, "health reads the real index"
            assert response.json() == {"chunks": get_collection().count(), **get_collection().metadata,
                                       "answers_left": None, "daily_limit": None, "session_limit": None}, \
                "health reports index count and stored settings"
            assert client.get("/session").json() == {"deployed": False, "authenticated": True,
                "role": None, "can_upload": True, "max_upload_mb": 20}
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
                 "paper_id": found[n - 1]["paper_id"], "title": found[n - 1]["title"],
                 "page": found[n - 1]["page"],
                 "url": f"{deploy.PAPERS[found[n - 1]['paper_id']]}#page={found[n - 1]['page']}"}
                for n in (1, 2)
            ], "citations map to the same numbered excerpts, sorted and deduplicated"
            assert result["invalid"] == [99] and result["fallback"], "invalid citations and fallback are flagged"
            assert prompts[-1] == build_prompt(question, found), "answerer gets the existing numbered prompt"
            assert result["cached"] is False
            legacy = {**result, "sources": [{k: v for k, v in s.items() if k not in ("page", "url")}
                                            for s in result["sources"]]}
            with patch.object(cache, "get", return_value=legacy):
                assert client.post("/ask", json=payload).json() == {**result, "cached": True}
                with patch.object(deploy, "PAPERS", {}):
                    unlinked = client.post("/ask", json=payload).json()
                    assert all(s["url"] == f"/papers/{s['paper_id']}/pdf#page={s['page']}"
                               for s in unlinked["sources"]), "uploaded sources link to authenticated PDFs"
            with patch(__name__ + ".retrieve", side_effect=AssertionError("cache hit retrieved chunks")):
                repeated = client.post("/ask", json=payload).json()
                assert repeated == {**result, "cached": True} and len(prompts) == 1
                spaced = client.post("/ask", json={**payload, "question": "  How  many attention heads does the Transformer use?  "})
                assert spaced.json()["cached"] is True and len(prompts) == 1
            for changed in ({**payload, "question": "What optimizer was used?"}, {**payload, "k": 2},
                            {**payload, "paper_ids": ["attention"]}):
                before = len(prompts)
                response = client.post("/ask", json=changed)
                assert response.status_code == 200 and response.json()["cached"] is False
                assert len(prompts) == before + 1, "question, k and filter each miss"
            with patch(__name__ + ".index_fingerprint", return_value="changed-index") as fingerprint:
                before = len(prompts)
                assert client.post("/ask", json=payload).json()["cached"] is False
                fingerprint.assert_called_once_with(None)
                assert len(prompts) == before + 1, "all-papers fingerprint changes miss"
            cache.clear(path=notes.DB_PATH)
            reply[:] = [NOT_COVERED, MODEL]
            result = client.post("/ask", json=payload).json()
            assert result["refused"] and not result["sources"] and not result["uncited"], "refusal has no sources"
            assert not result["fallback"], "primary model is not a fallback"
            cache.clear(path=notes.DB_PATH)
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
            for source in result["sources"]:
                chunk = result["retrieved"][source["n"] - 1]
                assert source["page"] == chunk["page"]
                assert source["url"] == f"{deploy.PAPERS[chunk['paper_id']]}#page={chunk['page']}"
            legacy = {**result, "sources": [{k: v for k, v in s.items() if k not in ("page", "url")}
                                            for s in result["sources"]]}
            with patch.object(cache, "get", return_value=legacy):
                assert client.post("/compare", json={"question": question, "paper_ids": ids}).json() == {**result, "cached": True}
            assert result["cached"] is False
            before = len(prompts)
            assert client.post("/compare", json={"question": question, "paper_ids": ids}).json()["cached"] is True
            assert len(prompts) == before, "repeated comparisons skip the answerer"
            reversed_result = client.post("/compare", json={"question": question, "paper_ids": ids[::-1]}).json()
            assert reversed_result["cached"] is False and len(prompts) == before + 1
            assert [c["paper_id"] for c in reversed_result["retrieved"]] == [ids[1]] * 3 + [ids[0]] * 3
            before = len(prompts)
            assert client.post("/ask", json={**payload, "paper_ids": ids}).json()["cached"] is False
            assert client.post("/ask", json={**payload, "paper_ids": ids[::-1]}).json()["cached"] is True
            assert len(prompts) == before + 1, "ask filters are unordered"
            with patch(__name__ + ".RETRIEVAL", "changed-retrieval"):
                for endpoint, body in (("/ask", {**payload, "paper_ids": ids}),
                                       ("/compare", {"question": question, "paper_ids": ids})):
                    before = len(prompts)
                    assert client.post(endpoint, json=body).json()["cached"] is False
                    assert client.post(endpoint, json=body).json()["cached"] is True
                    assert len(prompts) == before + 1, "retrieval changes invalidate answers"
            for bad_ids in ([pid], [pid] * 5, [pid, pid]):
                assert client.post("/compare", json={"question": question, "paper_ids": bad_ids}).status_code == 422
            for k in (0, 6):
                assert client.post("/compare", json={"question": question, "paper_ids": ids,
                                                    "k_per_paper": k}).status_code == 422
            cache.clear(path=notes.DB_PATH)
            with patch(__name__ + ".retrieve", return_value=[]):
                assert client.post("/compare", json={"question": question, "paper_ids": ids}).status_code == 422
            from step9_summary import paper_excerpts

            reply[:] = ["Overview [1].\n\nTakeaway [2].\nTakeaway [999].", "fake-model"]
            response = client.post("/summarize", json={"paper_id": pid})
            assert response.status_code == 200, "summarize uses the injected offline answerer"
            result = response.json()
            chunks = paper_excerpts(pid)
            assert result["sources"] == [
                {"n": n, "id": chunks[n - 1]["id"], "label": source_label(chunks[n - 1]),
                 "paper_id": pid, "page": chunks[n - 1]["page"],
                 "url": f"{deploy.PAPERS[pid]}#page={chunks[n - 1]['page']}"}
                for n in (1, 2)
            ], "summary citations map to the paper's reading order"
            legacy = {**result, "sources": [{k: v for k, v in s.items() if k not in ("paper_id", "page", "url")}
                                            for s in result["sources"]]}
            with patch.object(cache, "get", return_value=legacy):
                assert client.post("/summarize", json={"paper_id": pid}).json() == {**result, "cached": True}
                for source in legacy["sources"]:
                    source["label"] = "A label with no page number"
                linked = client.post("/summarize", json={"paper_id": pid}).json()
                assert [(s["page"], s["url"]) for s in linked["sources"]] == [
                    (s["page"], s["url"]) for s in result["sources"]], "summary links use chunk metadata, not labels"
            assert result["invalid"] == [999] and result["fallback"], "summary warnings survive serialization"
            assert result["chunks_used"] == len(chunks) and "refused" not in result
            assert prompts[-1] == build_prompt(f"Summarize this paper: {result['title']}", chunks)
            assert result["cached"] is False
            before = len(prompts)
            with patch(__name__ + ".summarize", side_effect=AssertionError("cache hit summarized")), \
                 patch(__name__ + ".RETRIEVAL", "changed-retrieval"):
                repeated = client.post("/summarize", json={"paper_id": pid}).json()
                assert repeated == {**result, "cached": True} and len(prompts) == before
            cache.clear(path=notes.DB_PATH)
            reply[:] = ["Plain prose.", MODEL]
            result = client.post("/summarize", json={"paper_id": pid}).json()
            assert result["uncited"] and not result["fallback"], "uncited summaries are flagged"
            response = client.post("/summarize", json={"paper_id": "unknown"})
            assert response.status_code == 422 and all(p["paper_id"] in response.json()["detail"] for p in indexed), \
                "unknown summary paper lists known ids"
            for invalid in ({}, {"paper_id": None}, {"paper_id": 1}):
                assert client.post("/summarize", json=invalid).status_code == 422, "summary needs a string paper id"
            with patch(__name__ + ".get_collection") as empty_collection:
                empty_collection.return_value.count.return_value = 0
                response = client.post("/summarize", json={"paper_id": pid})
                assert response.status_code == 503 and response.json() == {"detail": EMPTY_INDEX}, \
                    "empty summary index returns 503 before paper validation"
            note_payload = {"paper_id": pid, "text": "  It's a note  ", "question": question,
                            "answer": "Eight [1].", "sources": [{"n": 1, "label": "Page 1"}]}
            saved = client.post("/notes", json=note_payload).json()
            assert saved["text"] == "It's a note" and saved["sources"] == note_payload["sources"]
            assert client.get("/notes", params={"paper_id": pid}).json() == [saved]
            assert client.get("/notes").json() == [saved]
            for text in (" ", "x" * 5001):
                assert client.post("/notes", json={**note_payload, "text": text}).status_code == 422
            assert client.post("/notes", json={**note_payload, "paper_id": "unknown"}).status_code == 422
            assert client.get("/notes/export", params={"paper_id": "unknown"}).status_code == 422
            exported = client.get("/notes/export", params={"paper_id": pid})
            assert exported.status_code == 200 and exported.headers["content-type"].startswith("text/markdown")
            assert f'notes-{pid}.md' in exported.headers["content-disposition"]
            assert "It's a note" in exported.text and "Sources:" in exported.text
            assert client.delete(f"/notes/{saved['id']}").status_code == 204
            assert client.delete(f"/notes/{saved['id']}").status_code == 404
            assert client.get("/notes").json() == []
            test_uploads(client, indexed)
            cache.clear(path=notes.DB_PATH)
            failures = []

            def counting_failure(prompt, system=None):
                failures.append(prompt)
                return fail_answerer(prompt, system=system)

            app.dependency_overrides[get_answerer] = lambda: counting_failure
            for endpoint, body in (("/ask", payload), ("/compare", {"question": question, "paper_ids": ids}),
                                   ("/summarize", {"paper_id": pid})):
                before = len(failures)
                for _ in range(2):
                    response = client.post(endpoint, json=body)
                    assert response.status_code == 502 and response.json() == {"detail": "The answer was cut off."}
                assert len(failures) == before + 2, "answerer errors retry instead of being cached"
                app.dependency_overrides[get_answerer] = lambda: fake_answerer
                assert client.post(endpoint, json=body).json()["cached"] is False, "success after error is a miss"
                assert client.post(endpoint, json=body).json()["cached"] is True
                app.dependency_overrides[get_answerer] = lambda: counting_failure
            cache.clear(path=notes.DB_PATH)
            app.dependency_overrides[get_answerer] = lambda: fail_answerer
            response = client.post("/ask", json=payload)
            assert response.status_code == 502 and response.json() == {"detail": "The answer was cut off."}, \
                "answerer RuntimeError becomes 502 with its message"
            response = client.post("/summarize", json={"paper_id": pid})
            assert response.status_code == 502 and response.json() == {"detail": "The answer was cut off."}, \
                "summary answerer errors return 502 with their message"
            with patch(__name__ + ".retrieve", side_effect=RuntimeError(EMPTY_INDEX)):
                for endpoint in ("/search", "/ask"):
                    response = client.post(endpoint, json=payload)
                    assert response.status_code == 503 and response.json() == {"detail": EMPTY_INDEX}, \
                        "empty index becomes 503 without changing the real index"
            test_deployed(client, payload, ids, pid, fake_answerer)
            test_auth(payload, fake_answerer)
            test_session_budget(payload, fake_answerer)
            test_owner_keys()
            test_upload_configuration()
            test_owner_library()
            test_seed_data()
            test_refunds(payload)
            test_midnight_refund()
            test_web_deletion()
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)


def test_deployed(client, payload, ids, pid, fake_answerer):
    from concurrent.futures import ThreadPoolExecutor
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    with TemporaryDirectory() as directory:
        path = Path(directory) / "budget.db"
        with patch.object(deploy, "DEPLOYED", True), patch.object(deploy, "DAILY_LLM_LIMIT", 2), \
             patch.object(notes, "DB_PATH", path), patch.dict(os.environ, {"APP_PASSWORD": "test-password", "ANTHROPIC_API_KEY": "test-key"}), \
             patch(__name__ + ".ask_claude", side_effect=fake_answerer) as paid:
            client.cookies.set(auth.COOKIE_NAME, auth.sign("budget-tests", "visitor", int(time.time()) + 3600))
            app.dependency_overrides.pop(get_answerer, None)
            assert client.post("/papers", content=b"not even multipart").status_code == 403
            response = client.post("/papers", files={"file": ("demo.pdf", b"%PDF", "application/pdf")})
            assert response.json() == {"detail": "Only the owner can add or delete papers."}
            for endpoint, body in (("/ask", payload), ("/compare", {"question": payload["question"], "paper_ids": ids}),
                                   ("/summarize", {"paper_id": pid})):
                cache.clear(path=notes.DB_PATH)
                path.unlink(missing_ok=True)
                assert deploy.llm_calls_today(path=path) == 0
                assert client.get("/health").json()["answers_left"] == 2
                assert client.get("/health").json()["daily_limit"] == 2, "health uses the configured budget"
                first = client.post(endpoint, json=body)
                assert first.status_code == 200 and not first.json()["cached"]
                assert deploy.llm_calls_today(path=path) == 1
                assert client.get("/health").json()["answers_left"] == 1
                before = paid.call_count
                assert client.post(endpoint, json=body).json()["cached"]
                assert paid.call_count == before and deploy.llm_calls_today(path=path) == 1
                cache.clear(path=notes.DB_PATH)
                assert client.post(endpoint, json=body).status_code == 200
                assert deploy.llm_calls_today(path=path) == 2
                assert client.get("/health").json()["answers_left"] == 0
                assert client.get("/health").json()["daily_limit"] == 2, "the limit stays fixed as answers are spent"
                assert client.post(endpoint, json=body).json()["cached"], "cached answers work at the limit"
                cache.clear(path=notes.DB_PATH)
                before = paid.call_count
                response = client.post(endpoint, json=body)
                assert response.status_code == 429 and response.json() == {"detail": (
                    "Daily demo limit of 2 answers reached. Search only, cached answers and notes still work.")}
                assert paid.call_count == before and deploy.llm_calls_today(path=path) == 2
            assert client.post("/search", json=payload).status_code == 200
            assert client.get("/health").status_code == 200
            with patch.object(deploy, "llm_calls_today", return_value=3):
                assert client.get("/health").json()["answers_left"] == 0, "remaining answers never go negative"
            assert client.get("/papers").status_code == 200
            note = client.post("/notes", json={"paper_id": pid, "text": "Still available"}).json()
            assert client.get("/notes").status_code == 200
            assert client.delete(f"/notes/{note['id']}").status_code == 204
            assert deploy.llm_calls_today(path=path) == 2
            path.unlink()
            with ThreadPoolExecutor(max_workers=8) as pool:
                accepted = list(pool.map(lambda _: deploy.take_llm_call("budget-tests", path=path)[1], range(16)))
            assert accepted.count(None) == 2 and deploy.llm_calls_today(path=path) == 2
            client.cookies.clear()


def test_auth(payload, fake_answerer):
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    from fastapi.testclient import TestClient

    with TemporaryDirectory() as directory, patch.object(notes, "DB_PATH", Path(directory) / "auth.db"), \
         patch.object(deploy, "DEPLOYED", True), patch.object(auth, "failed_logins", deque()), \
         patch.dict(os.environ, {"APP_PASSWORD": "", "OWNER_PASSWORD": "", "SESSION_SECRET": "",
                                 "PAPERPILOT_MAX_UPLOAD_MB": "4"}), \
         patch(__name__ + ".ask_claude", side_effect=fake_answerer), \
         TestClient(app, base_url="https://testserver") as client:
        for endpoint, body in (("/search", payload), ("/login", {"password": "test-password"})):
            response = client.post(endpoint, json=body)
            assert response.status_code == 503 and response.json()["detail"] == "This demo is not configured."
        assert client.get("/health").status_code == 200
        assert client.get("/session").json() == {"deployed": True, "authenticated": False,
            "role": None, "can_upload": False, "max_upload_mb": 4}
        os.environ["APP_PASSWORD"] = "test-password"
        assert client.post("/login", json={"password": "x" * 257}).status_code == 422
        assert not auth.failed_logins, "oversized passwords are rejected before login attempts"
        for _ in range(10):
            response = client.post("/login", json={"password": "wrong"})
            assert response.status_code == 401 and response.json()["detail"] == "Incorrect password."
        response = client.post("/login", json={"password": "test-password"})
        assert response.status_code == 429
        assert response.json()["detail"] == "Too many failed sign-ins. Wait a minute."
        with patch.object(auth.time, "monotonic", return_value=time.monotonic() + 61):
            response = client.post("/login", json={"password": "test-password"})
        assert response.status_code == 200 and response.json() == {"authenticated": True}
        header = response.headers["set-cookie"].lower()
        assert all(part in header for part in ("httponly", "secure", "samesite=lax", "path=/", "max-age=604800"))
        cookie = client.cookies.get(auth.COOKIE_NAME)
        session = auth.read_session(cookie)
        assert session and session[0] != "streamlit" and session[1] == "visitor"
        assert client.get("/session").json()["role"] == "visitor"
        assert client.get("/session").json()["can_upload"] is False
        assert client.get("/session").json()["authenticated"]
        assert client.post("/search", json=payload).status_code == 200, "password-derived signing key works"
        with patch.dict(os.environ, {"APP_PASSWORD": "changed-password"}):
            assert auth.read_session(cookie) is None, "changing the fallback password signs everyone out"
        os.environ["SESSION_SECRET"] = "test-signing-secret"
        assert auth.read_session(cookie) is None, "a new signing key rejects old signatures"
        response = client.post("/login", json={"password": "test-password"})
        cookie = client.cookies.get(auth.COOKIE_NAME)
        assert response.status_code == 200 and auth.read_session(cookie)
        assert client.post("/search", json=payload).status_code == 200
        response = client.post("/logout")
        assert response.json() == {"authenticated": False} and "max-age=0" in response.headers["set-cookie"].lower()
        assert client.cookies.get(auth.COOKIE_NAME) is None
        assert client.get("/session").json()["authenticated"] is False
        bad_signature = cookie[:-1] + ("0" if cookie[-1] != "0" else "1")
        old_cookie = cookie.replace(".visitor", "")
        invalid_cookies = (None, bad_signature, auth.sign("expired", "visitor", int(time.time()) - 1), "malformed",
                           old_cookie, cookie.replace(".visitor.", ".owner."),
                           "a.not-a-time.signature", "a." + "9" * 100 + "." + "a" * 64)
        for invalid in invalid_cookies:
            client.cookies.clear()
            if invalid is not None:
                client.cookies.set(auth.COOKIE_NAME, invalid)
            response = client.post("/search", json=payload)
            assert response.status_code == 401 and response.json()["detail"] == "Sign in first."
            assert client.get("/health").status_code == 200
            assert client.get("/session").json()["authenticated"] is False
        auth.failed_logins.clear()
        with ThreadPoolExecutor(max_workers=12) as pool:
            errors = list(pool.map(lambda _: auth.check_login("wrong")[1], range(24)))
        assert errors.count("incorrect") == 10 and errors.count("limited") == 14

        # The cached client must still work after its first cookie would have expired.
        from step7_ui import api_client, request

        api_client.clear()
        try:
            streamlit = api_client()
            result, error = request("GET", "/session")
            assert error is None and result["role"] == "visitor" and result["can_upload"] is False
            now = time.time()
            for moment in (now, now + auth.SESSION_SECONDS + 86400):
                with patch.object(time, "time", return_value=moment):
                    result, error = request("POST", "/search", json=payload)
                    assert error is None and result["retrieved"], "Streamlit renews its cookie on each request"
                    assert api_client() is streamlit, "cookie renewal keeps the cached client"
        finally:
            api_client.clear()


def test_session_budget(payload, fake_answerer):
    from concurrent.futures import ThreadPoolExecutor
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    from fastapi.testclient import TestClient

    with TemporaryDirectory() as directory, patch.object(notes, "DB_PATH", Path(directory) / "sessions.db"), \
         patch.object(deploy, "DEPLOYED", True), \
         patch.dict(os.environ, {"APP_PASSWORD": "test-password", "SESSION_SECRET": "test-signing-secret",
                                 "ANTHROPIC_API_KEY": "test-key"}), \
         patch(__name__ + ".ask_claude", side_effect=fake_answerer) as paid, \
         TestClient(app, base_url="https://testserver") as client:
        def use_session(session_id):
            client.cookies.clear()
            client.cookies.set(auth.COOKIE_NAME, auth.sign(session_id, "visitor", int(time.time()) + 3600))

        def ask_uncached():
            cache.clear(path=notes.DB_PATH)
            return client.post("/ask", json=payload)

        assert client.get("/health").json()["answers_left"] == 40
        use_session("A")
        assert client.get("/health").json()["answers_left"] == 10
        assert client.get("/health").json()["session_limit"] == 10
        for _ in range(10):
            assert ask_uncached().status_code == 200
        assert deploy.session_calls_today("A", path=notes.DB_PATH) == 10
        assert client.get("/health").json()["answers_left"] == 0
        before = paid.call_count
        assert client.post("/ask", json=payload).json()["cached"], "cache is free at the session limit"
        response = ask_uncached()
        assert response.status_code == 429 and response.json()["detail"] == (
            "You've used your 10 paid answers for today. Search, cached answers and notes still work.")
        assert paid.call_count == before and deploy.llm_calls_today(path=notes.DB_PATH) == 10
        for session_id in ("B", "C"):
            use_session(session_id)
            for _ in range(10):
                assert ask_uncached().status_code == 200
        use_session("D")
        for _ in range(6):
            assert ask_uncached().status_code == 200
        use_session("E")
        health = client.get("/health").json()
        assert health["answers_left"] == 4 and health["daily_limit"] == 40
        for _ in range(4):
            assert ask_uncached().status_code == 200
        use_session("F")
        assert client.get("/health").json()["answers_left"] == 0
        before = paid.call_count
        response = ask_uncached()
        assert response.status_code == 429 and response.json()["detail"].startswith("Daily demo limit of 40 answers reached.")
        assert paid.call_count == before
        assert deploy.session_calls_today("F", path=notes.DB_PATH) == 0, "global rejection does not charge the session"
        client.cookies.clear()
        assert client.get("/health").json()["answers_left"] == 0, "anonymous health reports only global allowance"

        concurrent_path = Path(directory) / "concurrent.db"
        with ThreadPoolExecutor(max_workers=8) as pool:
            accepted = list(pool.map(lambda _: deploy.take_llm_call("same", path=concurrent_path)[1], range(24)))
        assert accepted.count(None) == 10 and accepted.count("session") == 14
        assert deploy.llm_calls_today(path=concurrent_path) == deploy.session_calls_today("same", path=concurrent_path) == 10
        concurrent_path = Path(directory) / "global.db"
        with ThreadPoolExecutor(max_workers=8) as pool:
            accepted = list(pool.map(lambda i: deploy.take_llm_call(str(i), path=concurrent_path)[1], range(64)))
        assert accepted.count(None) == 40 and accepted.count("global") == 24
        assert deploy.llm_calls_today(path=concurrent_path) == 40


def test_uploads(client, indexed):
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    with TemporaryDirectory() as directory:
        save_dir = Path(directory) / "papers"
        records = list(indexed)

        def fake_ingest(path):
            assert path.parent == save_dir
            with fitz.open(path) as pdf:
                assert pdf.page_count == 1
            records.append({"paper_id": path.stem, "title": "Uploaded paper", "chunks": 1, "pages": 1,
                            "builtin": False})

        with patch(__name__ + ".PAPERS_DIR", save_dir), \
             patch(__name__ + ".list_papers", side_effect=lambda: records), \
             patch(__name__ + ".ingest", side_effect=fake_ingest) as ingest_mock, \
             patch(__name__ + ".get_collection") as collection:
            def upload(name, data):
                return client.post("/papers", files={"file": (name, data, "application/pdf")})

            assert upload("wrong.pdf", b"not a pdf").status_code == 422
            response = upload("large.pdf", b"%PDF" + b"x" * (20 * 1024 * 1024))
            assert response.status_code == 422 and response.json()["detail"] == "PDF must be at most 20 MB"
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
            assert upload("broken.pdf", data).status_code == 422
            assert not (save_dir / "broken.pdf").exists()
            collection.return_value.delete.assert_called_once_with(where={"paper_id": "broken"})


def test_owner_keys():
    from collections import deque
    from contextlib import redirect_stdout
    from io import StringIO
    import runpy
    from unittest.mock import patch
    from fastapi.testclient import TestClient

    with patch.dict(os.environ, {"APP_PASSWORD": "visitor-password", "OWNER_PASSWORD": "owner-password",
                                 "SESSION_SECRET": "private-secret"}), \
         patch.object(auth, "failed_logins", deque()), patch.object(deploy, "DEPLOYED", True), \
         TestClient(app, base_url="https://testserver") as client:
        expires = int(time.time()) + 3600
        owner = auth.sign("owner-test", "owner", expires)
        visitor = auth.sign("visitor-test", "visitor", expires)
        expected_key = auth.hmac.new(b"private-secret", b"owner:owner-password", auth.hashlib.sha256).digest()
        assert auth.signing_key("owner") == expected_key
        assert auth.read_session(owner) == ("owner-test", "owner")
        with patch.dict(os.environ, {"OWNER_PASSWORD": "new-owner-password"}):
            assert auth.read_session(owner) is None, "password rotation signs owners out"
            assert auth.read_session(visitor) == ("visitor-test", "visitor")
            assert client.post("/login", json={"password": "owner-password"}).status_code == 401
            assert client.post("/login", json={"password": "new-owner-password"}).status_code == 200

        for disabled in ({"SESSION_SECRET": ""}, {"OWNER_PASSWORD": ""},
                         {"OWNER_PASSWORD": "visitor-password"}, {"SESSION_SECRET": "visitor-password"}):
            with patch.dict(os.environ, disabled):
                assert not auth.owner_enabled()
                assert auth.read_session(owner) is None
                message = f"forged.owner.{expires}"
                signature = auth.hmac.new(auth.signing_key(), message.encode(), auth.hashlib.sha256).hexdigest()
                assert auth.read_session(f"{message}.{signature}") is None, "visitor key cannot grant owner access"
                assert client.post("/login", json={"password": "owner-password"}).status_code == 401
                assert client.post("/login", json={"password": "visitor-password"}).status_code == 200
                assert client.get("/session").json()["role"] == "visitor"
                assert client.get("/session").json()["can_upload"] is False

        output = StringIO()
        with patch.dict(os.environ, {"OWNER_PASSWORD": "visitor-password"}), \
             patch.object(sys, "argv", ["deploy.py", "check"]), redirect_stdout(output):
            runpy.run_path(str(Path(__file__).with_name("deploy.py")), run_name="__main__")
        assert "Warning: OWNER_PASSWORD matches APP_PASSWORD" in output.getvalue()
        assert "visitor-password" not in output.getvalue() and "private-secret" not in output.getvalue()


def test_upload_configuration():
    from contextlib import redirect_stderr
    from io import StringIO
    import subprocess
    from unittest.mock import patch

    for value, expected in (("4.5", 4.5), ("3", 3.0), ("bad", 4.0), ("0", 4.0),
                            ("-1", 4.0), ("nan", 4.0), ("inf", 4.0), ("", 4.0)):
        warning = StringIO()
        with patch.dict(os.environ, {"PAPERPILOT_MAX_UPLOAD_MB": value}), redirect_stderr(warning):
            assert configured_upload_mb() == expected
        assert bool(warning.getvalue()) == (value not in ("4.5", "3"))
    # Import in a fresh process to check the actual startup constants without reloading the app under test.
    subprocess.run([sys.executable, "-c", """
import os
import step7_api as api
from fastapi.testclient import TestClient
assert api.DEPLOYED_UPLOAD_BYTES == int(4.5 * 1024 * 1024)
os.environ['PAPERPILOT_MAX_UPLOAD_MB'] = 'invalid-after-import'
with TestClient(api.app) as client:
    assert client.get('/session').json()['max_upload_mb'] == 4.5
api.deploy.DEPLOYED = False
assert api.max_upload_mb() == 20
"""], env={**os.environ, "PAPERPILOT_MAX_UPLOAD_MB": "4.5", "PAPERPILOT_DEPLOYED": "1"}, check=True)


def test_midnight_refund():
    from datetime import datetime, timezone
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch
    import httpx

    before = datetime(2026, 10, 27, 23, 59, 59, tzinfo=timezone.utc)
    after = datetime(2026, 10, 28, 0, 0, 1, tzinfo=timezone.utc)
    with TemporaryDirectory() as directory, patch.object(notes, "DB_PATH", Path(directory) / "midnight.db"), \
         patch.object(deploy, "DEPLOYED", True), patch.object(deploy, "datetime") as clock, \
         patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        clock.now.return_value = before

        def fail_after_midnight(prompt):
            clock.now.return_value = after
            assert deploy.take_llm_call("midnight", path=notes.DB_PATH) == ("2026-10-28", None)
            raise RuntimeError("connection failed") from APIConnectionError(request=httpx.Request("POST", "https://test"))

        with patch(__name__ + ".ask_claude", side_effect=fail_after_midnight):
            try:
                get_answerer(SimpleNamespace(state=SimpleNamespace(session_id="midnight")))("question")
            except RuntimeError:
                pass
            else:
                raise AssertionError("API failure must propagate")
        assert deploy.llm_calls_today(path=notes.DB_PATH) == 1
        assert deploy.session_calls_today("midnight", path=notes.DB_PATH) == 1
        clock.now.return_value = before
        assert deploy.llm_calls_today(path=notes.DB_PATH) == 0
        assert deploy.session_calls_today("midnight", path=notes.DB_PATH) == 0


def test_web_deletion():
    import subprocess

    subprocess.run(["node", "selftest.mjs"], cwd=Path(__file__).parent / "web", check=True)


def test_owner_library():
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    from tempfile import TemporaryDirectory
    from threading import Event
    from unittest.mock import patch
    from fastapi.testclient import TestClient

    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), "An uploaded paper")
        data = pdf.tobytes()
        for _ in range(60):
            pdf.new_page()
        long_pdf = pdf.tobytes()
    collection = get_collection()  # selftest has redirected this to a temporary copy.
    sample = collection.get(limit=1, include=["embeddings"])["embeddings"][0].tolist()
    entered, finish = Event(), Event()

    def fake_ingest(path):
        if path.stem == "slow-upload":
            entered.set()
            assert finish.wait(10), "test did not release the fake ingest"
        collection.add(ids=[f"{path.stem}-0"], documents=["Uploaded text"], embeddings=[sample],
                       metadatas=[{"paper_id": path.stem, "title": "Uploaded paper", "page": 1,
                                   "page_end": 1, "section": "Abstract", "chunk_index": 0}])
        return 1

    with TemporaryDirectory() as directory, patch.object(notes, "DB_PATH", Path(directory) / "notes.db"), \
         patch(__name__ + ".PAPERS_DIR", Path(directory) / "papers"), \
         patch(__name__ + ".ingest", side_effect=fake_ingest) as ingest_mock, \
         patch.object(deploy, "DEPLOYED", True), patch.object(auth, "failed_logins", deque()), \
         patch.dict(os.environ, {"APP_PASSWORD": "visitor-password", "OWNER_PASSWORD": "owner-password",
                                 "SESSION_SECRET": "test-secret", "PAPERPILOT_MAX_UPLOAD_MB": "4"}), \
         TestClient(app, base_url="https://testserver") as client:
        def upload(name, content=data):
            return client.post("/papers", files={"file": (name + ".pdf", content, "application/pdf")})

        # Roles: anonymous, visitor and owner permissions.
        assert client.get("/papers/owner-upload/pdf").status_code == 401
        assert client.post("/login", json={"password": "visitor-password"}).status_code == 200
        assert client.get("/session").json()["role"] == "visitor"
        # FastAPI reads multipart through Request.stream. A visitor must never reach it.
        async def unread_stream(request):
            raise AssertionError("read visitor body")
            yield b""  # Keep this an async generator; fail on iteration, not creation.

        with patch("starlette.requests.Request.stream", unread_stream):
            response = client.post("/papers", content=b"never read")
            assert response.status_code == 403
            assert response.json()["detail"] == "Only the owner can add or delete papers."
        assert client.delete("/papers/attention").status_code == 403
        with patch.object(auth.hmac, "compare_digest", wraps=auth.hmac.compare_digest) as compare:
            assert auth.check_login("owner-password") == ("owner", None)
            assert compare.call_count == 2
            compare.reset_mock()
            assert auth.check_login("visitor-password") == ("visitor", None)
            assert compare.call_count == 2
        assert client.post("/login", json={"password": "owner-password"}).status_code == 200
        assert client.get("/session").json() == {"deployed": True, "authenticated": True,
            "role": "owner", "can_upload": True, "max_upload_mb": 4}
        # Upload limits: decimals, bytes, page count, library size and lock contention.
        with patch(__name__ + ".DEPLOYED_UPLOAD_MB", 4.5), \
             patch(__name__ + ".DEPLOYED_UPLOAD_BYTES", int(4.5 * 1024 * 1024)):
            assert client.get("/session").json()["max_upload_mb"] == 4.5
            response = upload("too-large", b"%PDF" + b"x" * int(4.5 * 1024 * 1024))
            assert response.status_code == 422 and response.json()["detail"] == "PDF must be at most 4.5 MB"
            with patch(__name__ + ".save_paper", return_value={"paper_id": "decimal", "title": "Decimal",
                    "chunks": 1, "pages": 1, "builtin": False}):
                assert upload("decimal", b"%PDF" + b"x" * (4 * 1024 * 1024)).status_code == 200
        response = upload("too-large", b"%PDF" + b"x" * (4 * 1024 * 1024))
        assert response.status_code == 422 and response.json()["detail"] == "PDF must be at most 4 MB"
        response = upload("too-many-pages", long_pdf)
        assert response.status_code == 422 and response.json()["detail"] == "PDFs over 60 pages aren't supported here."
        assert not PAPERS_DIR.exists() and not ingest_mock.called
        assert not collection.get(where={"paper_id": "too-many-pages"})["ids"]
        with patch(__name__ + ".list_papers", return_value=[{"paper_id": f"p{i}"} for i in range(20)]):
            response = upload("full")
            assert response.status_code == 409
            assert response.json()["detail"] == "The library is full (20 papers). Delete one first."
        assert not PAPERS_DIR.exists() and not ingest_mock.called
        with paper_lock:
            for response in (upload("busy"), client.delete("/papers/missing")):
                assert response.status_code == 409 and response.json()["detail"] == BUSY_LIBRARY
        # Storage: recover an orphan, preserve indexed duplicates and serve authenticated PDFs.
        PAPERS_DIR.mkdir(parents=True)
        (PAPERS_DIR / "owner-upload.pdf").write_bytes(b"stale partial upload")
        response = upload("owner-upload")
        assert response.status_code == 200 and response.json()["builtin"] is False
        assert all(p["builtin"] == (p["paper_id"] in deploy.PAPERS) for p in client.get("/papers").json())
        assert upload("owner-upload").status_code == 409
        assert (PAPERS_DIR / "owner-upload.pdf").read_bytes() == data
        assert client.delete("/papers/attention").json()["detail"] == "Built-in papers can't be deleted."
        assert client.delete("/papers/attention").status_code == 403
        assert client.delete("/papers/missing").status_code == 404
        for path in ("/papers/missing/pdf", "/papers/..%2Fattention/pdf", "/papers/%2E%2E/pdf",
                     "/papers/OWNER-UPLOAD/pdf"):
            assert client.get(path).status_code == 404
        response = client.get("/papers/owner-upload/pdf")
        assert response.status_code == 200 and response.content == data
        assert response.headers["content-type"] == "application/pdf"
        assert response.headers["content-disposition"].startswith("inline;")
        assert response.headers["x-content-type-options"] == "nosniff"
        client.post("/login", json={"password": "visitor-password"})
        assert client.get("/papers/owner-upload/pdf").content == data, "visitors can read uploaded sources"
        assert client.delete("/papers/owner-upload").status_code == 403
        client.post("/login", json={"password": "owner-password"})
        # Deletion: remove chunks, PDF and notes, then allow the id to be reused.
        notes.add_note("owner-upload", "Delete this note", path=notes.DB_PATH)
        keep = notes.add_note("attention", "Keep this note", path=notes.DB_PATH)
        fingerprint = index_fingerprint()
        assert client.delete("/papers/owner-upload").status_code == 204
        assert not collection.get(where={"paper_id": "owner-upload"})["ids"]
        assert not (PAPERS_DIR / "owner-upload.pdf").exists()
        assert notes.list_notes(path=notes.DB_PATH) == [keep]
        assert index_fingerprint() != fingerprint
        assert client.get("/papers/owner-upload/pdf").status_code == 404
        assert upload("owner-upload").status_code == 200, "deleting frees the id and filename"
        assert client.delete("/papers/owner-upload").status_code == 204
        # A blocked embedding job must not block the event loop or allow a second writer.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(upload, "slow-upload")
            try:
                assert entered.wait(10)
                assert client.get("/health").status_code == 200
                assert upload("second-upload").status_code == 409
                assert client.delete("/papers/attention").status_code == 409
            finally:
                finish.set()
            assert pending.result(timeout=10).status_code == 200
        assert client.delete("/papers/slow-upload").status_code == 204


def test_seed_data():
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    with TemporaryDirectory() as directory:
        root = Path(directory)
        image = root / "image"
        for name in ("chroma_db", "papers"):
            (image / name).mkdir(parents=True)
            (image / name / "built-in").write_bytes(b"image data")
        index, papers = root / "disk" / "chroma_db", root / "disk" / "papers"
        with patch.object(deploy, "IMAGE_DIR", image), patch.object(deploy, "INDEX_DIR", index), \
             patch.object(deploy, "PAPERS_DIR", papers):
            for target in (index, papers):
                stale = target.parent / f".{target.name}-seed-crashed"
                stale.mkdir(parents=True)
                (stale / "partial").write_bytes(b"incomplete copy")
            deploy.seed_data()
            assert not list(index.parent.glob(".*-seed-*")), "crash leftovers are removed"
            assert (index / "built-in").read_bytes() == (papers / "built-in").read_bytes() == b"image data"
            (index / "built-in").write_bytes(b"disk index")
            (papers / "upload.pdf").write_bytes(b"keep uploads")
            deploy.seed_data()
            assert (index / "built-in").read_bytes() == b"disk index"
            assert (papers / "upload.pdf").read_bytes() == b"keep uploads"
            assert sorted(p.name for p in index.parent.iterdir()) == ["chroma_db", "papers"]
            # A failed copy leaves no target, and the next startup can seed it again.
            missing = root / "failed-index"
            with patch.object(deploy, "INDEX_DIR", missing), \
                 patch.object(deploy.shutil, "copytree", side_effect=OSError("interrupted")):
                try:
                    deploy.seed_data()
                except OSError:
                    pass
                else:
                    raise AssertionError("copy failure should propagate")
                assert not missing.exists()
            with patch.object(deploy, "INDEX_DIR", image / "chroma_db"), \
                 patch.object(deploy, "PAPERS_DIR", image / "papers"), \
                 patch.object(deploy.shutil, "copytree", side_effect=AssertionError("copied onto itself")):
                deploy.seed_data()


def test_refunds(payload):
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    import httpx
    import anthropic
    from step4_answer import ask_claude as real_ask_claude

    previous = app.dependency_overrides.pop(get_answerer, None)
    try:
        with TemporaryDirectory() as directory, patch.object(notes, "DB_PATH", Path(directory) / "refund.db"), \
             patch.object(deploy, "DEPLOYED", True), \
             patch.dict(os.environ, {"APP_PASSWORD": "test-password", "ANTHROPIC_API_KEY": "test-key"}), \
             TestClient(app, base_url="https://testserver") as client:
            client.cookies.set(auth.COOKIE_NAME, auth.sign("refund-tests", "visitor", int(time.time()) + 3600))
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}), \
                 patch.object(deploy, "take_llm_call", wraps=deploy.take_llm_call) as reserve:
                response = client.post("/ask", json=payload)
                assert response.status_code == 503 and response.json() == {"detail": "This demo is not configured."}
                assert not reserve.called
                assert deploy.llm_calls_today(path=notes.DB_PATH) == 0
                assert deploy.session_calls_today("refund-tests", path=notes.DB_PATH) == 0
            request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            connection_error = anthropic.APIConnectionError(request=request)
            status_error = anthropic.APIStatusError("unavailable", response=httpx.Response(503, request=request),
                                                   body={"error": {"message": "unavailable"}})
            # Exercise ask_claude's real exception wrapping, with only the SDK network call replaced.
            with patch.object(anthropic, "Anthropic") as sdk:
                create = sdk.return_value.beta.messages.create
                with patch(__name__ + ".ask_claude", wraps=real_ask_claude):
                    for error in (connection_error, status_error):
                        create.side_effect = error
                        assert client.post("/ask", json=payload).status_code == 502
                        assert deploy.llm_calls_today(path=notes.DB_PATH) == 0
                        assert deploy.session_calls_today("refund-tests", path=notes.DB_PATH) == 0
                    create.side_effect = anthropic.APITimeoutError(request=request)
                    assert client.post("/ask", json=payload).status_code == 502
                    assert deploy.llm_calls_today(path=notes.DB_PATH) == 1
                    assert deploy.session_calls_today("refund-tests", path=notes.DB_PATH) == 1
                    create.side_effect = None
                    create.return_value.stop_reason = "end_turn"
                    create.return_value.model = MODEL
                    create.return_value.content = [SimpleNamespace(type="text", text="Eight [1].")]
                    assert client.post("/ask", json=payload).status_code == 200
                    assert deploy.llm_calls_today(path=notes.DB_PATH) == 2
                    cache.clear(path=notes.DB_PATH)
                    for count, reason in enumerate(("refusal", "max_tokens"), start=3):
                        create.return_value.stop_reason = reason
                        assert client.post("/ask", json=payload).status_code == 502
                        assert deploy.llm_calls_today(path=notes.DB_PATH) == count
                        assert deploy.session_calls_today("refund-tests", path=notes.DB_PATH) == count
            day = deploy.datetime.now(deploy.timezone.utc).date().isoformat()
            for _ in range(6):
                deploy.release_llm_call("refund-tests", day, path=notes.DB_PATH)
            assert deploy.llm_calls_today(path=notes.DB_PATH) == 0
            assert deploy.session_calls_today("refund-tests", path=notes.DB_PATH) == 0
    finally:
        if previous is not None:
            app.dependency_overrides[get_answerer] = previous


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
