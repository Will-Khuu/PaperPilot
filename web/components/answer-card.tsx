"use client";

import { useState } from "react";
import { errorMessage, request } from "@/lib/api";
import type { Note, Result } from "@/lib/api";
import { Icon } from "./icon";
import styles from "./answer-card.module.css";

export function StatusCard({ text, error = false }: { text: string; error?: boolean }) {
  return <div className={styles.row}><Icon /><div className={styles.card}
    role={error ? "alert" : "status"}><span className={error ? styles.error : styles.meta}>{text}</span></div></div>;
}

export function AnswerCard({ result, paperIds }: { result: Result; paperIds: string[] }) {
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState("");
  const [saveError, setSaveError] = useState("");
  const cited = "answer" in result || "summary" in result ? result : null;
  let text: string;
  if ("answer" in result) {
    text = result.answer;
  } else if ("summary" in result) {
    text = result.summary;
  } else {
    text = `Top ${result.retrieved.length} excerpts`;
  }
  const warnings: string[] = [];
  if (cited?.invalid.length) {
    const numbers = cited.invalid.map(n => `[${n}]`).join(", ");
    warnings.push(`These citation numbers do not refer to retrieved excerpts: ${numbers}.`);
  }
  if ("refused" in result && result.refused) warnings.push("The excerpts don't cover this question.");
  if (cited?.uncited) warnings.push("No citations: this may come from outside knowledge.");
  if (cited?.fallback) warnings.push(`A fallback model answered: ${cited.model}.`);

  async function save() {
    if (!cited || !paperIds.length) return;
    const question = "question" in cited ? cited.question : `Summarize this paper: ${cited.title}`;
    setSaving(true);
    setNotice("");
    setSaveError("");
    try {
      await request<Note>("/notes", { method: "POST", body: JSON.stringify({
        paper_id: paperIds[0],
        text: question + (paperIds.length > 1 ? "\nAlso includes: " + paperIds.slice(1).join(", ") : ""),
        question,
        answer: "answer" in cited ? cited.answer : cited.summary,
        sources: cited.sources.map(({ n, label }) => ({ n, label })),
      }) });
      setNotice("Saved to notes.");
    } catch (error) { setSaveError(errorMessage(error)); }
    finally { setSaving(false); }
  }

  return <div className={styles.row}><Icon /><article className={styles.card}>
    <div className={styles.text}>{text}</div>
    {cited && <div className={styles.sources}><strong>Sources:</strong>
      <ul>{cited.sources.map(source => <li key={source.n}>
        {source.url ? <a href={source.url} target="_blank" rel="noopener noreferrer">[{source.n}] {source.label}</a>
          : <>[{source.n}] {source.label}</>}
      </li>)}</ul>
    </div>}
    {warnings.map(warning => <p key={warning} className={styles.warning}>{warning}</p>)}
    {cited?.cached && <span className={styles.cached}>Cached</span>}
    {"retrieved" in result && <details className={styles.details} open={!cited}>
      <summary>Show {result.retrieved.length} excerpts</summary>
      {result.retrieved.map(excerpt => <div key={excerpt.n} className={styles.excerpt}>
        <div className={styles.meta}>[{excerpt.n}] {excerpt.title} · {excerpt.section} · {excerpt.pages} · distance {excerpt.distance == null ? "-" : excerpt.distance.toFixed(3)}</div>
        <div className={styles.text}>{excerpt.text}</div>
      </div>)}
    </details>}
    {cited && <button className={styles.save} disabled={saving || !paperIds.length} onClick={save}>{saving ? "Saving…" : "Save to notes"}</button>}
    {notice && <p role="status" className={styles.meta}>{notice}</p>}
    {saveError && <p role="alert" className={styles.error}>{saveError}</p>}
  </article></div>;
}
