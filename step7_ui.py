"""Step 7: an HTTP chat client. Run with streamlit run step7_ui.py."""
from datetime import datetime
import os

import httpx
import streamlit as st

API = os.environ.get("PAPERPILOT_API", "http://127.0.0.1:8000").rstrip("/")


def request(method, path, **kwargs):
    try:
        response = httpx.request(method, API + path, timeout=180, **kwargs)
        response.raise_for_status()
        if response.status_code == 204:
            return None, None
        if response.headers.get("content-type", "").startswith("text/markdown"):
            return response.text, None
        return response.json(), None
    except httpx.HTTPStatusError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except ValueError:
            detail = f"Server returned HTTP {e.response.status_code}."
        return None, str(detail)
    except httpx.RequestError:
        return None, f"Cannot reach the server at {API}. Start it with python step7_api.py serve."


def show_excerpts(excerpts):
    for excerpt in excerpts:
        st.write(f"[{excerpt['n']}] {excerpt['title']} · {excerpt['section']} · {excerpt['pages']} · distance {excerpt['distance']:.3f}")
        st.text(excerpt["text"])


def show_result(result):
    if "answer" not in result:
        show_excerpts(result["retrieved"])
        return
    st.write(result["answer"])
    if result.get("cached"):
        st.caption("(cached)")
    st.write("Sources")
    for pid in dict.fromkeys(source["paper_id"] for source in result["sources"]):
        sources = [source for source in result["sources"] if source["paper_id"] == pid]
        st.write(sources[0]["title"])
        for source in sources:
            st.text(f"[{source['n']}] {source['label']}")
    if result["invalid"]:
        numbers = ", ".join(f"[{n}]" for n in result["invalid"])
        st.warning(f"These citation numbers do not refer to retrieved excerpts: {numbers}.")
    if result["refused"]:
        st.warning("The excerpts don't cover this question.")
    if result["uncited"]:
        st.warning("No citations: this may come from outside knowledge.")
    if result["fallback"]:
        st.warning(f"A fallback model answered: {result['model']}.")
    with st.expander("Retrieved excerpts"):
        show_excerpts(result["retrieved"])


def save_to_notes(result, paper_ids, key):
    question = result.get("question", f"Summarize this paper: {result.get('title', '')}")
    text = question
    if len(paper_ids) > 1:
        text += "\nAlso includes: " + ", ".join(paper_ids[1:])
    if st.button("Save to notes", key=key, disabled=not paper_ids):
        _, error = request("POST", "/notes", json={
            "paper_id": paper_ids[0], "text": text, "question": question,
            "answer": result.get("answer", result.get("summary")),
            "sources": [{"n": source["n"], "label": source["label"]} for source in result["sources"]],
        })
        if error:
            st.error(error)
        else:
            st.success("Saved to notes.")


def show_notes(paper_id):
    if not paper_id:
        st.info("Choose a paper to see its notes.")
        return
    if "note_notice" in st.session_state:
        st.success(st.session_state.pop("note_notice"))
    records, error = request("GET", "/notes", params={"paper_id": paper_id})
    if error:
        st.error(error)
        return
    for note in records:
        created_at = datetime.fromisoformat(note["created_at"]).astimezone()
        hour = created_at.hour % 12 or 12
        st.caption(f"{created_at:%b} {created_at.day}, {created_at:%Y}, {hour}:{created_at:%M} {created_at:%p}")
        st.text(note["text"])
        if note["question"]:
            st.text("Question: " + note["question"])
        if note["answer"]:
            st.text(note["answer"])
        for source in note["sources"]:
            st.text(f"[{source['n']}] {source['label']}")
        if st.button("Delete", key=f"delete-{note['id']}"):
            _, error = request("DELETE", f"/notes/{note['id']}")
            if error:
                st.error(error)
            else:
                st.session_state.note_notice = "Note deleted."
                st.rerun()
    with st.form(f"note-{paper_id}", clear_on_submit=True):
        text = st.text_area("New note", max_chars=5000)
        if st.form_submit_button("Add note"):
            _, error = request("POST", "/notes", json={"paper_id": paper_id, "text": text})
            if error:
                st.error(error)
            else:
                st.session_state.note_notice = "Note added."
                st.rerun()
    markdown, error = request("GET", "/notes/export", params={"paper_id": paper_id})
    if error:
        st.error(error)
    else:
        st.download_button("Download Markdown", markdown, file_name=f"notes-{paper_id}.md", mime="text/markdown")


def main():
    st.title("PaperPilot")
    with st.sidebar:
        mode = st.radio("Mode", ["Ask", "Search only (free)", "Compare", "Summarize", "Notes"])
        indexed, error = request("GET", "/papers")
        if error:
            st.warning(error)
        titles = {p["paper_id"]: p["title"] for p in indexed or []}
        if mode in ("Summarize", "Notes"):
            paper_id = st.selectbox("Paper", list(titles), format_func=lambda pid: titles[pid])
        else:
            paper_ids = st.multiselect("Papers", list(titles), format_func=lambda pid: titles[pid],
                                       help="Compare: choose 2-4 papers. Ask/Search: leave empty for all papers.")
            if mode == "Compare":
                k = st.slider("Excerpts per paper", 1, 5, 3)
            else:
                k = st.slider("Excerpts", 1, 10, 5)
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        if st.button("Add paper", disabled=uploaded is None):
            record, error = request("POST", "/papers",
                                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")})
            if error:
                st.error(error)
            else:
                st.session_state.upload_notice = f"Added {record['title']} ({record['chunks']} chunks)."
                st.rerun()
        if "upload_notice" in st.session_state:
            st.success(st.session_state.pop("upload_notice"))
        health, error = request("GET", "/health")
        if error:
            st.warning(error)
        else:
            st.write(f"{health['chunks']} chunks")
            st.write(f"Chunk size: {health['chunk_size']}; overlap: {health['chunk_overlap']}")
    if mode == "Notes":
        show_notes(paper_id)
        return
    if mode == "Summarize":
        if "summaries" not in st.session_state:
            st.session_state.summaries = {}
        if st.button("Summarize", disabled=not titles):
            with st.spinner("Summarizing…"):
                result, error = request("POST", "/summarize", json={"paper_id": paper_id})
            if error:
                st.error(error)
            else:
                st.session_state.summaries[paper_id] = result
        result = st.session_state.summaries.get(paper_id)
        if result:
            st.text(result["summary"])
            if result.get("cached"):
                st.caption("(cached)")
            st.write("Sources")
            for source in result["sources"]:
                st.text(f"[{source['n']}] {source['label']}")
            if result["invalid"]:
                st.warning(f"Invalid citation numbers: {result['invalid']}.")
            if result["uncited"]:
                st.warning("No citations: this may come from outside knowledge.")
            save_to_notes(result, [paper_id], f"save-summary-{paper_id}")
        return
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if "result" in message:
                show_result(message["result"])
                if "answer" in message["result"]:
                    save_to_notes(message["result"], message["paper_ids"], f"save-answer-{index}")
            elif "error" in message:
                st.error(message["error"])
            else:
                st.write(message["text"])
    valid_selection = mode != "Compare" or 2 <= len(paper_ids) <= 4
    if not valid_selection:
        st.info("Choose 2-4 papers to compare.")
    question = st.chat_input("Ask about the papers", disabled=not valid_selection)
    if question:
        st.session_state.messages.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            endpoint = {"Ask": "/ask", "Search only (free)": "/search", "Compare": "/compare"}[mode]
            payload = {"question": question, "paper_ids": paper_ids or None}
            payload["k_per_paper" if mode == "Compare" else "k"] = k
            with st.spinner("Searching…" if mode == "Search only (free)" else "Answering…"):
                result, error = request("POST", endpoint, json=payload)
            if error:
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "error": error})
            else:
                show_result(result)
                # Keep the original selection so later sidebar changes cannot move a saved answer.
                saved_papers = paper_ids or list(dict.fromkeys(c["paper_id"] for c in result["retrieved"]))
                if "answer" in result:
                    save_to_notes(result, saved_papers, f"save-answer-{len(st.session_state.messages)}")
                st.session_state.messages.append({"role": "assistant", "result": result, "paper_ids": saved_papers})


if __name__ == "__main__":
    main()
