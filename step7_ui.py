"""Step 7: an HTTP chat client. Run with streamlit run step7_ui.py."""
import os

import httpx
import streamlit as st

API = os.environ.get("PAPERPILOT_API", "http://127.0.0.1:8000").rstrip("/")


def request(method, path, **kwargs):
    try:
        response = httpx.request(method, API + path, timeout=180, **kwargs)
        response.raise_for_status()
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
        st.write(f"[{excerpt['n']}] {excerpt['section']} · {excerpt['pages']} · distance {excerpt['distance']:.3f}")
        st.text(excerpt["text"])


def show_result(result):
    if "answer" not in result:
        show_excerpts(result["retrieved"])
        return
    st.write(result["answer"])
    st.write("Sources")
    for source in result["sources"]:
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


def main():
    st.title("PaperPilot")
    with st.sidebar:
        search_only = st.toggle("Search only (free, no LLM)")
        k = st.slider("Excerpts", 1, 10, 5)
        health, error = request("GET", "/health")
        if error:
            st.warning(error)
        else:
            st.write(f"{health['chunks']} chunks")
            st.write(f"Chunk size: {health['chunk_size']}; overlap: {health['chunk_overlap']}")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "result" in message:
                show_result(message["result"])
            elif "error" in message:
                st.error(message["error"])
            else:
                st.write(message["text"])
    question = st.chat_input("Ask about the paper")
    if question:
        st.session_state.messages.append({"role": "user", "text": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching…" if search_only else "Answering…"):
                result, error = request("POST", "/search" if search_only else "/ask",
                                        json={"question": question, "k": k})
            if error:
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "error": error})
            else:
                show_result(result)
                st.session_state.messages.append({"role": "assistant", "result": result})


if __name__ == "__main__":
    main()
