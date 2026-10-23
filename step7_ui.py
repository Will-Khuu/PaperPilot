"""Step 7: the chat page. Run with streamlit run step7_ui.py."""
from datetime import datetime
import hmac
import html
import os
import re
import sys

import httpx
import streamlit as st

import deploy

API = os.environ.get("PAPERPILOT_API", "http://127.0.0.1:8000").rstrip("/")


@st.cache_resource
def api_client():
    from fastapi.testclient import TestClient
    from step7_api import app

    return TestClient(app, raise_server_exceptions=False)


@st.cache_resource
def build_index():
    return deploy.ensure_index()


def password_gate():
    password = os.environ.get("APP_PASSWORD", "")
    if not password:
        st.error("This demo is not configured.")
        st.stop()
    if st.session_state.get("authenticated"):
        return
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            if hmac.compare_digest(entered.encode("utf-8"), password.encode("utf-8")):
                st.session_state.authenticated = True
                st.rerun()
            st.error("Incorrect password.")
    st.stop()


def request(method, path, **kwargs):
    try:
        if deploy.DEPLOYED:
            response = api_client().request(method, path, **kwargs)
        else:
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
        if deploy.DEPLOYED:
            return None, "The demo could not complete the request."
        return None, f"Cannot reach the server at {API}. Start it with python step7_api.py serve."


def html_text(text):
    # Escape before adding line breaks so paper and model text cannot inject HTML.
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return html.escape(text, quote=True).replace("\n", "<br>")


def plain_label(text):
    text = str(text)
    for char in "\\`*_[]()#+-.!$:<>|~{}":
        text = text.replace(char, "\\" + char)
    return text


ICON = (
    '<svg viewBox="0 0 24 24" aria-hidden="true">'
    '<path d="m12 2 9 5v10l-9 5-9-5V7Z" fill="currentColor"/>'
    '<path d="m3 7 9 5 9-5M12 12v10M7.5 4.5l9 5v5" fill="none" '
    'stroke="white" stroke-width="1.5" stroke-linejoin="round"/></svg>'
)


def bubble(text, side="assistant"):
    if side == "user":
        return f'<div class="pp-user"><div>{html_text(text)}</div></div>'
    css_class = "pp-error" if side == "error" else "pp-thinking"
    return (f'<div class="pp-row"><span class="pp-icon">{ICON}</span>'
            f'<div class="pp-card {css_class}">{html_text(text)}</div></div>')


def page_style():
    st.markdown("""
<style>
:root {
    --pp-font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Helvetica Neue", system-ui, sans-serif;
}
.pp-brand { display: flex; align-items: center; gap: 10px; font-size: 1.3rem; margin-bottom: 24px; }
.pp-icon { display: inline-flex; color: #3B5BDB; width: 32px; height: 32px;
    background: #EDF1FF; border-radius: 8px; padding: 5px; flex-shrink: 0; }
.pp-icon svg { width: 100%; height: 100%; }
.pp-brand .pp-icon { background: transparent; padding: 2px; }
.pp-user { display: flex; justify-content: flex-end; margin: 16px 0; }
.pp-user > div { background: #EDF1FF; color: #1F2333; border-radius: 14px; padding: 14px 18px;
    max-width: 80%; overflow-wrap: anywhere; line-height: 1.6; }
.pp-row { display: flex; align-items: flex-start; gap: 12px; margin: 16px 0; }
.pp-card { flex: 1; background: white; border: 1px solid #E6E8F0; border-radius: 14px; padding: 16px; }
.pp-answer-text { position: relative; line-height: 1.65; overflow-wrap: anywhere; }
.pp-answer-text > .pp-icon { position: absolute; left: -60px; top: -16px; }
.pp-sources { margin-top: 18px; margin-bottom: 10px; font-size: 0.875rem; line-height: 1.8; overflow-wrap: anywhere; }
.pp-sources a { color: #3B5BDB; text-decoration: none; text-underline-offset: 3px; }
.pp-sources a:hover { text-decoration: underline; }
.pp-meta, .pp-thinking { color: #62697B; font-size: 0.875rem; }
.pp-warning { color: #805400; font-size: 0.875rem; margin-top: 8px; overflow-wrap: anywhere; }
.pp-error { color: #B42318; overflow-wrap: anywhere; }
.pp-cached { color: #62697B; font-size: 0.75rem; background: #F7F8FC;
    border-radius: 6px; padding: 3px 7px; display: inline-block; margin-top: 8px; }
.pp-excerpt { padding: 12px 0; overflow-wrap: anywhere; line-height: 1.6; }
.pp-note { background: white; border: 1px solid #E6E8F0; padding: 16px;
    border-radius: 12px; overflow-wrap: anywhere; line-height: 1.6; }
.pp-chip { display: inline-flex; align-items: center; gap: 12px; border: 1px solid #E6E8F0;
    border-radius: 12px; padding: 12px 16px; margin: 0 8px 8px 0; max-width: 100%; }
.pp-chip-text { min-width: 0; overflow-wrap: anywhere; }
.pp-pdf { display: inline-flex; align-items: center; justify-content: center; flex-shrink: 0;
    background: #D92D20; color: white; border-radius: 5px; width: 30px; height: 38px;
    font-size: 0.65rem; font-weight: 700; }
.pp-paper { display: flex; gap: 14px; align-items: center; overflow-wrap: anywhere; }
.pp-paper-title { font-weight: 600; margin-bottom: 4px; }
.st-key-pp-new-chat button { background: #EDF1FF; color: #3B5BDB; border: 0; }
.st-key-pp-nav button { border: 0; background: transparent; justify-content: flex-start;
    text-align: left; border-radius: 8px; padding: 10px 12px; }
.st-key-pp-nav button:hover, .st-key-pp-new-chat button:hover { background: #E4EAFF; color: #3B5BDB; }
.st-key-pp-sidebar-footer { margin-top: auto; padding-top: 24px; }
[class*="st-key-pp-answer-"] { width: calc(100% - 44px); margin: 12px 0 12px 44px; padding: 16px;
    border: 1px solid #E6E8F0; border-radius: 14px; gap: 8px; }
[class*="st-key-pp-paper-"] { padding: 18px; border: 1px solid #E6E8F0; border-radius: 12px; }
.pp-user ::selection, .pp-answer-text::selection, .pp-note::selection { background: #DCE4FF; }
/* These selectors depend on Streamlit 1.50's markup. */
.st-key-pp-nav button[kind="primary"] { background: #EDF1FF; color: #3B5BDB; }
.stApp, .stApp input, .stApp textarea, .stApp button, .stApp h1, .stApp h2, .stApp h3,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] { font-family: var(--pp-font); }
[data-testid="stMainBlockContainer"] { padding-top: 3rem; max-width: 1100px; }
[data-testid="stSidebar"] { background: #F7F8FC; border-right: 1px solid #E6E8F0; }
[data-testid="stSidebarUserContent"] > [data-testid="stVerticalBlock"] { min-height: calc(100dvh - 7rem); }
[data-testid="stChatInput"] { border-radius: 14px; }
[data-testid="stChatInputSubmitButton"] { background: #3B5BDB; color: white; border-radius: 10px; }
[data-testid="stChatInputSubmitButton"]:hover { background: #2F4AC0; color: white; }
[data-testid="stChatInputSubmitButton"]:disabled,
[data-testid="stChatInputSubmitButton"][disabled] { background: rgba(59, 91, 219, 0.45); color: white; opacity: 1; }
[data-testid="stChatInputSubmitButton"] svg { color: white; }
@media (max-width: 640px) {
    .pp-user > div { max-width: 90%; }
    .pp-chip { padding: 10px; }
}
</style>
""", unsafe_allow_html=True)


def show_excerpts(excerpts):
    for excerpt in excerpts:
        label = (f"[{excerpt['n']}] {excerpt['title']} · {excerpt['section']} · "
                 f"{excerpt['pages']} · distance {excerpt['distance']:.3f}")
        st.markdown(f'<div class="pp-meta">{html_text(label)}</div>'
                    f'<div class="pp-excerpt">{html_text(excerpt["text"])}</div>', unsafe_allow_html=True)


def source_html(result):
    lines = []
    for source in result["sources"]:
        n = source["n"]
        if "retrieved" in result:
            excerpt = result["retrieved"][n - 1]
            paper_id, page = excerpt["paper_id"], excerpt["page"]
        else:
            # Summaries expose source labels, not retrieved chunks; the final label field is the page.
            paper_id = result["paper_id"]
            match = re.search(r", pp?\. (\d+)(?:-\d+)?$", source["label"])
            page = int(match[1]) if match else None
        label = html_text(f"[{n}] {source['label']}")
        url = deploy.PAPERS.get(paper_id)
        if url and page:
            label = (f'<a href="{html_text(f"{url}#page={page}")}" target="_blank" '
                     f'rel="noopener noreferrer">{label}</a>')
        lines.append(label)
    return '<div class="pp-sources"><strong>Sources:</strong><br>' + '<br>'.join(lines) + '</div>'


def show_result(result, paper_ids, key):
    with st.container(key=f"pp-answer-{key}"):
        text = result.get("answer", result.get("summary", f"Top {len(result.get('retrieved', []))} excerpts"))
        st.markdown(f'<div class="pp-answer-text"><span class="pp-icon">{ICON}</span>'
                    f'{html_text(text)}</div>', unsafe_allow_html=True)
        is_answer = "answer" in result or "summary" in result
        if is_answer:
            st.markdown(source_html(result), unsafe_allow_html=True)
        warnings = []
        if result.get("invalid"):
            numbers = ", ".join(f"[{n}]" for n in result["invalid"])
            warnings.append(f"These citation numbers do not refer to retrieved excerpts: {numbers}.")
        if result.get("refused"):
            warnings.append("The excerpts don't cover this question.")
        if result.get("uncited"):
            warnings.append("No citations: this may come from outside knowledge.")
        if result.get("fallback"):
            warnings.append(f"A fallback model answered: {result['model']}.")
        for warning in warnings:
            st.markdown(f'<div class="pp-warning">{html_text(warning)}</div>', unsafe_allow_html=True)
        if result.get("cached"):
            st.markdown('<span class="pp-cached">Cached</span>', unsafe_allow_html=True)
        if "retrieved" in result:
            with st.expander(f"Show {len(result['retrieved'])} excerpts", expanded=not is_answer):
                show_excerpts(result["retrieved"])
        if is_answer:
            save_to_notes(result, paper_ids, f"save-{key}")


def save_to_notes(result, paper_ids, key):
    question = result.get("question", f"Summarize this paper: {result.get('title', '')}")
    text = question
    if len(paper_ids) > 1:
        text += "\nAlso includes: " + ", ".join(paper_ids[1:])
    if st.button("Save to notes", key=key, disabled=not paper_ids, type="tertiary"):
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
    if deploy.DEPLOYED:
        st.caption("Demo notes are temporary: they are deleted when the app restarts.")
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
        lines = [note["text"]]
        if note["question"]:
            lines.append("Question: " + note["question"])
        if note["answer"]:
            lines.append(note["answer"])
        lines.extend(f"[{source['n']}] {source['label']}" for source in note["sources"])
        text = "\n".join(lines)
        st.markdown(f'<div class="pp-note">{html_text(text)}</div>', unsafe_allow_html=True)
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


def open_page(page):
    st.session_state.page = page


def new_chat():
    st.session_state.messages = []
    open_page("Chat")


def select_papers(paper_ids):
    st.session_state.paper_ids = paper_ids
    open_page("Chat")


def save_widget_value(key):
    st.session_state[key] = st.session_state[f"_{key}"]


def show_sidebar():
    with st.sidebar:
        st.markdown(f'<div class="pp-brand"><span class="pp-icon">{ICON}</span>'
                    '<strong>PaperPilot</strong></div>', unsafe_allow_html=True)
        with st.container(key="pp-new-chat"):
            st.button("New Chat", icon=":material/add:", use_container_width=True, on_click=new_chat)
        with st.container(key="pp-nav"):
            for page, icon in [("My Papers", "description"), ("Chat", "chat_bubble"),
                               ("Compare", "compare_arrows"), ("Summaries", "summarize"),
                               ("Notes", "edit_note"), ("Settings", "settings")]:
                st.button(page, icon=f":material/{icon}:", use_container_width=True,
                          type="primary" if st.session_state.page == page else "secondary",
                          on_click=open_page, args=(page,))
        with st.container(key="pp-sidebar-footer"):
            if deploy.DEPLOYED:
                remaining = max(0, deploy.DAILY_LLM_LIMIT - deploy.llm_calls_today())
                st.caption(f"{remaining} of {deploy.DAILY_LLM_LIMIT} paid answers left")


def show_papers(indexed):
    st.header("My Papers")
    st.button("All papers", on_click=select_papers, args=(None,))
    for paper in indexed:
        with st.container(key=f"pp-paper-{paper['paper_id']}"):
            st.markdown('<div class="pp-paper"><span class="pp-pdf">PDF</span><div>'
                        f'<div class="pp-paper-title">{html_text(paper["title"])}</div>'
                        f'<div class="pp-meta">{html_text(paper["pages"])} pages · '
                        f'{html_text(paper["chunks"])} excerpts</div></div></div>', unsafe_allow_html=True)
            st.button("Chat about this paper", key=f"chat-{paper['paper_id']}",
                      on_click=select_papers, args=([paper["paper_id"]],))
    if not indexed:
        st.info("No papers are indexed yet.")
    if not deploy.DEPLOYED:
        st.subheader("Add paper")
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
        st.success(plain_label(st.session_state.pop("upload_notice")))


def show_chips(indexed):
    attachments, change, search = st.columns([5, 1, 3], vertical_alignment="center")
    with attachments:
        if st.session_state.paper_ids is None:
            st.markdown(f'<div class="pp-chip">All papers · {html_text(len(indexed))} papers</div>',
                        unsafe_allow_html=True)
        else:
            for paper in indexed:
                if paper["paper_id"] in st.session_state.paper_ids:
                    st.markdown('<div class="pp-chip"><span class="pp-pdf">PDF</span><div class="pp-chip-text">'
                                f'{html_text(paper["title"] + ".pdf")}<div class="pp-meta">'
                                f'{html_text(paper["pages"])} pages</div></div></div>', unsafe_allow_html=True)
    with change:
        st.button("Change", type="tertiary", on_click=open_page, args=("My Papers",))
    with search:
        st.toggle("Search only (free)", value=st.session_state.search_only, key="_search_only",
                  on_change=save_widget_value, args=("search_only",))


def show_conversation(paper_ids, compare=False):
    messages = st.session_state.compare_messages if compare else st.session_state.messages
    prefix = "compare" if compare else "chat"
    for index, message in enumerate(messages):
        if "result" in message:
            show_result(message["result"], message["paper_ids"], f"{prefix}-{index}")
        elif "error" in message:
            st.markdown(bubble(message["error"], "error"), unsafe_allow_html=True)
        else:
            st.markdown(bubble(message["text"], "user"), unsafe_allow_html=True)
    valid_selection = not compare or 2 <= len(paper_ids) <= 4
    if not valid_selection:
        st.info("Choose 2-4 papers to compare.")
    question = st.chat_input("Ask a question about your papers...", disabled=not valid_selection)
    if question:
        messages.append({"role": "user", "text": question})
        st.markdown(bubble(question, "user"), unsafe_allow_html=True)
        endpoint = "/compare" if compare else "/search" if st.session_state.search_only else "/ask"
        payload = {"question": question, "paper_ids": paper_ids}
        payload["k_per_paper" if compare else "k"] = st.session_state.compare_k if compare else st.session_state.k
        typing = st.empty()
        typing.markdown(bubble("Thinking…"), unsafe_allow_html=True)
        result, error = request("POST", endpoint, json=payload)
        typing.empty()
        if error:
            messages.append({"role": "assistant", "error": error})
        else:
            # Keep the original selection so later navigation cannot move a saved answer.
            saved_papers = paper_ids or list(dict.fromkeys(c["paper_id"] for c in result["retrieved"]))
            messages.append({"role": "assistant", "result": result, "paper_ids": saved_papers})
        st.rerun()


def main():
    st.set_page_config(page_title="PaperPilot", layout="wide")
    page_style()
    if deploy.DEPLOYED:
        password_gate()
        with st.spinner("Building the index (first visit after a restart takes about a minute)…"):
            try:
                build_index()
            except Exception as exc:
                print(f"Could not build the paper index: {exc}", file=sys.stderr)
                st.error("Could not build the paper index. Try again in a few minutes.")
                st.stop()
    for key, value in {"page": "Chat", "messages": [], "compare_messages": [], "paper_ids": None,
                       "summaries": {}, "k": 5, "compare_k": 3, "search_only": False,
                       "compare_ids": [], "summaries_paper": None, "notes_paper": None}.items():
        st.session_state.setdefault(key, value)
    indexed, error = request("GET", "/papers")
    if error:
        st.sidebar.warning(error)
    indexed = indexed or []
    titles = {p["paper_id"]: p["title"] for p in indexed}
    st.session_state.paper_ids = [pid for pid in st.session_state.paper_ids or [] if pid in titles] or None
    st.session_state.compare_ids = [pid for pid in st.session_state.compare_ids if pid in titles]
    for key in ("summaries_paper", "notes_paper"):
        if st.session_state[key] not in titles:
            st.session_state[key] = next(iter(titles), None)
    # Existing widgets must also drop selections removed from the paper list.
    for key in ("compare_ids", "summaries_paper", "notes_paper"):
        if f"_{key}" in st.session_state and st.session_state[f"_{key}"] != st.session_state[key]:
            del st.session_state[f"_{key}"]
    show_sidebar()
    page = st.session_state.page
    if page == "My Papers":
        show_papers(indexed)
    elif page == "Chat":
        show_chips(indexed)
        show_conversation(st.session_state.paper_ids)
    elif page == "Compare":
        st.header("Compare")
        paper_ids = st.multiselect("Choose 2-4 papers", list(titles), max_selections=4,
                                   default=st.session_state.compare_ids,
                                   format_func=lambda pid: plain_label(titles[pid]), key="_compare_ids",
                                   on_change=save_widget_value, args=("compare_ids",))
        show_conversation(paper_ids, compare=True)
    elif page in ("Summaries", "Notes"):
        st.header(page)
        if not indexed:
            if page == "Notes":
                show_notes(None)
            else:
                st.info("Choose a paper to summarize.")
            return
        key = f"{page.lower()}_paper"
        paper_id = st.selectbox("Paper", list(titles), index=list(titles).index(st.session_state[key]),
                                format_func=lambda pid: plain_label(titles[pid]), key=f"_{key}",
                                on_change=save_widget_value, args=(key,))
        if page == "Notes":
            show_notes(paper_id)
            return
        if st.button("Summarize"):
            typing = st.empty()
            typing.markdown(bubble("Thinking…"), unsafe_allow_html=True)
            result, error = request("POST", "/summarize", json={"paper_id": paper_id})
            typing.empty()
            if error:
                st.markdown(bubble(error, "error"), unsafe_allow_html=True)
            else:
                st.session_state.summaries[paper_id] = result
        result = st.session_state.summaries.get(paper_id)
        if result:
            show_result(result, [paper_id], f"summary-{paper_id}")
    elif page == "Settings":
        st.header("Settings")
        # Streamlit removes off-page widget keys; separate values keep these settings across navigation.
        st.slider("Excerpts per answer", 1, 10, st.session_state.k, key="settings_k",
                  on_change=lambda: st.session_state.update(k=st.session_state.settings_k))
        st.slider("Excerpts per paper in compare", 1, 5, st.session_state.compare_k, key="settings_compare_k",
                  on_change=lambda: st.session_state.update(compare_k=st.session_state.settings_compare_k))
        health, error = request("GET", "/health")
        if error:
            st.warning(error)
        else:
            st.info(f"{health['chunks']} excerpts · chunks {health['chunk_size']}/{health['chunk_overlap']}")


if __name__ == "__main__":
    main()
