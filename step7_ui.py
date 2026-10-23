"""Step 7: the chat page. Run with streamlit run step7_ui.py."""
from datetime import datetime
import hmac
import html
import os
import sys

import httpx
import streamlit as st

import deploy

API = os.environ.get("PAPERPILOT_API", "http://127.0.0.1:8000").rstrip("/")
ALL = ":all"
COMPARE = ":compare"


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


def bubble(text, side="assistant"):
    css_class = {"user": "pp-user", "assistant": "pp-assistant", "error": "pp-error"}[side]
    return f'<div class="pp-row {css_class}"><div class="pp-bubble">{html_text(text)}</div></div>'


def page_style():
    dark = st.context.theme.type == "dark"
    colors = ("#3B3B3D", "#FFFFFF", "#B4B4BA", "#FFB340", "#512C30") if dark else (
        "#E9E9EB", "#1C1C1E", "#626269", "#805400", "#FCE8E8")
    background, foreground, muted, warning, error = colors
    st.markdown(f"""
<style>
:root {{
    --pp-font: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", system-ui, sans-serif;
    --pp-gray: {background}; --pp-text: {foreground}; --pp-muted: {muted};
    --pp-warning: {warning}; --pp-error: {error};
}}
.pp-row {{ display: flex; margin: 12px 0 4px; }}
.pp-user {{ justify-content: flex-end; }}
.pp-bubble {{ background: var(--pp-gray); color: var(--pp-text); border-radius: 18px;
    padding: 12px 16px; max-width: 75%; overflow-wrap: anywhere; line-height: 1.5; }}
.pp-user .pp-bubble {{ background: #0A84FF; color: white; max-width: 70%; }}
.pp-error .pp-bubble {{ background: var(--pp-error); }}
.pp-meta, .pp-warning {{ font-size: 0.8rem; line-height: 1.5; margin: 5px 0; overflow-wrap: anywhere; }}
.pp-meta {{ color: var(--pp-muted); }}
.pp-warning {{ color: var(--pp-warning); }}
.pp-excerpt, .pp-note {{ background: var(--pp-gray); color: var(--pp-text);
    padding: 12px 16px; border-radius: 18px; margin-bottom: 8px; overflow-wrap: anywhere; }}
.pp-thread-title {{ font-size: 1.15rem; font-weight: 600; margin: 0; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap; }}
.pp-bubble::selection, .pp-note::selection {{ background: #0A84FF; color: white; }}
/* These selectors depend on Streamlit 1.50's markup. */
.stApp, .stApp input, .stApp textarea, .stApp button, .stApp h1, .stApp h2, .stApp h3,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {{
    font-family: var(--pp-font);
}}
[data-testid="stMainBlockContainer"] {{ padding-top: 3.5rem; max-width: 1100px; }}
[data-testid="stSidebar"] .st-key-pp-threads button {{ width: 100%; border: 0; border-radius: 10px;
    text-align: left; justify-content: flex-start; padding: 10px 12px; }}
[data-testid="stSidebar"] .st-key-pp-threads button [data-testid="stMarkdownContainer"] {{
    width: 100%; min-width: 0;
}}
[data-testid="stSidebar"] .st-key-pp-threads button p {{ overflow: hidden; text-overflow: ellipsis;
    white-space: nowrap; margin: 0; text-align: left; }}
[data-testid="stSidebar"] .st-key-pp-threads button p:first-child {{ font-weight: 600; }}
[data-testid="stSidebar"] .st-key-pp-threads button p + p {{ font-size: 0.8rem; margin-top: 4px; }}
[data-testid="stSidebar"] .st-key-pp-threads button[kind="secondary"] p + p {{ color: var(--pp-muted); }}
[data-testid="stSidebar"] .st-key-pp-threads button[kind="primary"] {{ background: #0A84FF; color: white; }}
[data-testid="stSidebarUserContent"] > [data-testid="stVerticalBlock"] {{ min-height: calc(100dvh - 7rem); }}
.st-key-pp-sidebar-footer {{ margin-top: auto; padding-top: 1.5rem; }}
.stApp input, .stApp textarea {{ caret-color: #0A84FF; }}
input[type="checkbox"], input[type="range"] {{ accent-color: #0A84FF; }}
/* These selectors depend on Streamlit 1.50's markup. */
button[kind="segmented_controlActive"], button[kind="segmented_controlActive"]:hover {{ background: #0A84FF; color: white; border-color: #0A84FF; }}
[data-testid="stCheckbox"] label:has(input:checked) > span:first-child {{ background: #0A84FF; border-color: #0A84FF; }}
[data-testid="stSlider"] [role="slider"] {{ background: #0A84FF; }}
[data-testid="stChatInput"] div:focus-within {{ border-color: #0A84FF; }}
@media (max-width: 640px) {{
    .pp-bubble {{ max-width: 92%; }}
    .pp-user .pp-bubble {{ max-width: 88%; }}
}}
</style>
""", unsafe_allow_html=True)


def show_excerpts(excerpts):
    for excerpt in excerpts:
        label = (f"[{excerpt['n']}] {excerpt['title']} · {excerpt['section']} · "
                 f"{excerpt['pages']} · distance {excerpt['distance']:.3f}")
        st.markdown(f'<div class="pp-meta">{html_text(label)}</div>'
                    f'<div class="pp-excerpt">{html_text(excerpt["text"])}</div>', unsafe_allow_html=True)


def show_result(result):
    if "answer" not in result and "summary" not in result:
        st.markdown(bubble(f"Top {len(result['retrieved'])} excerpts"), unsafe_allow_html=True)
        with st.expander(f"Show {len(result['retrieved'])} excerpts", expanded=True):
            show_excerpts(result["retrieved"])
        return
    st.markdown(bubble(result.get("answer", result.get("summary"))), unsafe_allow_html=True)
    sources = result["sources"]
    lines = ["Sources"]
    paper_ids = list(dict.fromkeys(source.get("paper_id") for source in sources))
    for pid in paper_ids:
        grouped = [source for source in sources if source.get("paper_id") == pid]
        if len(paper_ids) > 1:
            lines.append(grouped[0]["title"])
        lines.extend(f"[{source['n']}] {source['label']}" for source in grouped)
    if result.get("cached"):
        lines.append("Cached")
    text = "\n".join(lines)
    st.markdown(f'<div class="pp-meta">{html_text(text)}</div>', unsafe_allow_html=True)
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
    if "retrieved" in result:
        with st.expander(f"Show {len(result['retrieved'])} excerpts"):
            show_excerpts(result["retrieved"])


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


def select_thread(thread):
    st.session_state.active_thread = thread


def thread_preview(messages):
    if not messages:
        return "No questions yet"
    last = messages[-1]
    result = last.get("result", {})
    text = last.get("text", last.get("error", result.get("answer", result.get("summary", ""))))
    if not text:
        text = f"Top {len(result.get('retrieved', []))} excerpts"
    return " ".join(text.split())[:90]


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
    indexed, error = request("GET", "/papers")
    if error:
        st.sidebar.warning(error)
    titles = {p["paper_id"]: p["title"] for p in indexed or []}
    if "threads" not in st.session_state:
        st.session_state.threads = {ALL: [], COMPARE: []}
    for thread in [ALL, COMPARE, *titles]:
        st.session_state.threads.setdefault(thread, [])
    if st.session_state.get("active_thread", ALL) not in {ALL, *titles}:
        st.session_state.active_thread = ALL
    active_thread = st.session_state.get("active_thread", ALL)
    st.session_state.compare_ids = [pid for pid in st.session_state.get("compare_ids", []) if pid in titles]
    title = titles.get(active_thread, "All papers")
    heading, controls = st.columns([2, 3], vertical_alignment="center")
    with controls:
        previous_mode = st.session_state.get("mode", "Ask")
        selected_mode = st.segmented_control(
            "Mode", ["Ask", "Search", "Compare", "Summary", "Notes"],
            label_visibility="collapsed", default="Ask", key="mode_choice",
        )
        mode = selected_mode or previous_mode
        st.session_state.mode = mode
    with heading:
        heading_text = "Compare papers" if mode == "Compare" else title
        st.markdown(f'<div class="pp-thread-title">{html_text(heading_text)}</div>', unsafe_allow_html=True)
    with st.sidebar:
        st.title("PaperPilot")
        with st.container(key="pp-threads"):
            if mode == "Compare":
                paper_ids = [pid for pid, paper_title in titles.items()
                             if st.checkbox(plain_label(paper_title), key=f"compare-{pid}",
                                            value=pid in st.session_state.compare_ids)]
                st.session_state.compare_ids = paper_ids
            else:
                paper_ids = None if active_thread == ALL else [active_thread]
                for thread, thread_title in {ALL: "All papers", **titles}.items():
                    preview = thread_preview(st.session_state.threads[thread])
                    st.button(f"{plain_label(thread_title)}\n\n{plain_label(preview)}", key=f"thread-{thread}",
                              type="primary" if thread == active_thread else "secondary",
                              use_container_width=True, on_click=select_thread, args=(thread,))
        if mode in ("Ask", "Search", "Compare"):
            with st.expander("Retrieval settings"):
                if mode == "Compare":
                    k = st.slider("Excerpts per paper", 1, 5, 3)
                else:
                    k = st.slider("Excerpts", 1, 10, 5)
        with st.container(key="pp-sidebar-footer"):
            if deploy.DEPLOYED:
                remaining = max(0, deploy.DAILY_LLM_LIMIT - deploy.llm_calls_today())
                st.caption(f"{remaining} of {deploy.DAILY_LLM_LIMIT} paid answers left")
            health, error = request("GET", "/health")
            if error:
                st.warning(error)
            else:
                st.caption(f"{health['chunks']} excerpts · chunks {health['chunk_size']}/{health['chunk_overlap']}")
            if not deploy.DEPLOYED:
                with st.expander("Add a paper"):
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
    paper_id = None if active_thread == ALL else active_thread
    if mode == "Notes":
        show_notes(paper_id)
        return
    if mode == "Summary":
        if not paper_id:
            st.info("Choose a paper to summarize.")
            return
        if "summaries" not in st.session_state:
            st.session_state.summaries = {}
        if st.button("Summarize"):
            typing = st.empty()
            typing.markdown(bubble("•••"), unsafe_allow_html=True)
            result, error = request("POST", "/summarize", json={"paper_id": paper_id})
            typing.empty()
            if error:
                st.markdown(bubble(error, "error"), unsafe_allow_html=True)
            else:
                st.session_state.summaries[paper_id] = result
        result = st.session_state.summaries.get(paper_id)
        if result:
            show_result(result)
            save_to_notes(result, [paper_id], f"save-summary-{paper_id}")
        return
    thread = COMPARE if mode == "Compare" else active_thread
    messages = st.session_state.threads[thread]
    for index, message in enumerate(messages):
        if "result" in message:
            show_result(message["result"])
            if "answer" in message["result"]:
                save_to_notes(message["result"], message["paper_ids"], f"save-{thread}-{index}")
        elif "error" in message:
            st.markdown(bubble(message["error"], "error"), unsafe_allow_html=True)
        else:
            st.markdown(bubble(message["text"], "user"), unsafe_allow_html=True)
    valid_selection = mode != "Compare" or 2 <= len(paper_ids) <= 4
    if not valid_selection:
        st.info("Choose 2-4 papers to compare.")
    input_title = "selected papers" if mode == "Compare" else title
    placeholder = (f"Search {input_title} (free, no AI call)" if mode == "Search"
                   else f"Ask about {input_title}")
    question = st.chat_input(placeholder,
                             disabled=not valid_selection)
    if question:
        messages.append({"role": "user", "text": question, "paper_ids": paper_ids, "mode": mode})
        st.markdown(bubble(question, "user"), unsafe_allow_html=True)
        endpoint = {"Ask": "/ask", "Search": "/search", "Compare": "/compare"}[mode]
        payload = {"question": question, "paper_ids": paper_ids}
        payload["k_per_paper" if mode == "Compare" else "k"] = k
        typing = st.empty()
        typing.markdown(bubble("•••"), unsafe_allow_html=True)
        result, error = request("POST", endpoint, json=payload)
        typing.empty()
        if error:
            messages.append({"role": "assistant", "error": error, "paper_ids": paper_ids, "mode": mode})
        else:
            # Keep the original selection so later sidebar changes cannot move a saved answer.
            saved_papers = paper_ids or list(dict.fromkeys(c["paper_id"] for c in result["retrieved"]))
            messages.append({"role": "assistant", "result": result, "paper_ids": saved_papers, "mode": mode})
        st.rerun()


if __name__ == "__main__":
    main()
