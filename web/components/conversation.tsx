"use client";

import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import { useAppState } from "@/app/state";
import type { Message } from "@/app/state";
import { errorMessage, request } from "@/lib/api";
import type { AnswerResult, SearchResult } from "@/lib/api";
import { AnswerCard, StatusCard } from "./answer-card";
import styles from "./conversation.module.css";

// IDs only need to be unique within this session, including on insecure LAN origins.
let messageId = 0;
function nextMessageId() {
  messageId += 1;
  return String(messageId);
}

function renderMessage(message: Message) {
  if (message.kind === "result") {
    return <AnswerCard key={message.id} result={message.result} paperIds={message.paperIds} />;
  }
  if (message.kind === "error") {
    return <StatusCard key={message.id} text={message.text} error />;
  }
  return <div key={message.id} className={styles.user}><div>{message.text}</div></div>;
}

export function Conversation({ compare = false }: { compare?: boolean }) {
  const state = useAppState();
  const [question, setQuestion] = useState("");
  const bottom = useRef<HTMLDivElement>(null);
  const questionInput = useRef<HTMLInputElement>(null);
  const messages = compare ? state.compareMessages : state.messages;
  const setMessages = compare ? state.setCompareMessages : state.setMessages;
  const pending = compare ? state.comparePending : state.chatPending;
  const setPending = compare ? state.setComparePending : state.setChatPending;
  const paperIds = compare ? state.compareIds : state.paperIds;
  const valid = !compare || (state.compareIds.length >= 2 && state.compareIds.length <= 4);
  const previousPending = useRef(pending);

  useEffect(() => { bottom.current?.scrollIntoView({ block: "nearest" }); }, [messages.length, pending]);
  useEffect(() => {
    // Wait for React to re-enable the input before returning keyboard focus.
    if (previousPending.current !== null && pending === null && valid) {
      questionInput.current?.focus();
    }
    previousPending.current = pending;
  }, [pending, valid]);

  async function send(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = question.trim();
    if (!text || pending || !valid) return;
    const id = nextMessageId();
    const selected = paperIds ? [...paperIds] : null;
    const endpoint = compare ? "/compare" : state.searchOnly ? "/search" : "/ask";
    setMessages(previous => [...previous, { id, kind: "user", text }]);
    setQuestion("");
    setPending(id);
    let message: Message;
    try {
      const result = await request<AnswerResult | SearchResult>(endpoint, {
        method: "POST", body: JSON.stringify({
          question: text, paper_ids: selected,
          ...(compare ? { k_per_paper: state.compareK } : { k: state.k }),
        }),
      });
      // Keep the original papers so changing the selection cannot move a saved answer.
      message = { id: nextMessageId(), kind: "result", result,
        paperIds: selected?.length ? selected : [...new Set(result.retrieved.map(c => c.paper_id))] };
    } catch (error) {
      message = { id: nextMessageId(), kind: "error", text: errorMessage(error) };
    }
    // New Chat can clear a thread while its old request is still in flight.
    setMessages(previous => previous.some(item => item.id === id) ? [...previous, message] : previous);
    setPending(previous => previous === id ? null : previous);
    void state.refreshHealth();
  }

  return <section className={styles.conversation} aria-label={compare ? "Comparison conversation" : "Chat conversation"}>
    <div className={styles.messages}>
      {messages.map(renderMessage)}
      {pending && <StatusCard text="Thinking…" />}
      <div ref={bottom} />
    </div>
    <form className={styles.composer} onSubmit={send}>
      <label className={styles.srOnly} htmlFor={compare ? "compare-question" : "chat-question"}>Ask a question about your papers</label>
      <input ref={questionInput} id={compare ? "compare-question" : "chat-question"} value={question} onChange={e => setQuestion(e.target.value)}
        maxLength={500} placeholder="Ask a question about your papers..." disabled={!!pending || !valid} autoComplete="off" />
      <button type="submit" disabled={!!pending || !valid || !question.trim()}>Send</button>
    </form>
  </section>;
}
