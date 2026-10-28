"use client";

import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { useAppState } from "../state";
import { errorMessage, request } from "@/lib/api";
import type { Note } from "@/lib/api";
import styles from "./page.module.css";

function noteText(note: Note) {
  const lines: string[] = [];
  if (note.text) lines.push(note.text);
  if (note.question) lines.push(`Question: ${note.question}`);
  if (note.answer) lines.push(note.answer);
  for (const source of note.sources) {
    lines.push(`[${source.n}] ${source.label}`);
  }
  return lines.join("\n");
}

export default function NotesPage() {
  const state = useAppState();
  const paperId = state.notesPaper;
  const [records, setRecords] = useState<{ paperId: string; notes: Note[] } | null>(null);
  const [loadError, setLoadError] = useState<{ paperId: string; text: string } | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    if (!paperId) return;
    let active = true;
    request<Note[]>(`/notes?paper_id=${encodeURIComponent(paperId)}`).then(notes => {
      if (active) { setRecords({ paperId, notes }); setLoadError(null); }
    }).catch(error => { if (active) setLoadError({ paperId, text: errorMessage(error) }); });
    return () => { active = false; };
  }, [paperId, revision]);

  async function add(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!text.trim() || busy) return;
    setBusy(true); setError(""); setNotice("");
    try {
      await request<Note>("/notes", { method: "POST", body: JSON.stringify({ paper_id: paperId, text }) });
      setText(""); setNotice("Note added."); setRevision(n => n + 1);
    } catch (error) { setError(errorMessage(error)); }
    finally { setBusy(false); }
  }
  async function remove(id: number) {
    setBusy(true); setError(""); setNotice("");
    try {
      await request<void>(`/notes/${id}`, { method: "DELETE" });
      setNotice("Note deleted."); setRevision(n => n + 1);
    } catch (error) { setError(errorMessage(error)); }
    finally { setBusy(false); }
  }
  return <>
    <h1>Notes</h1>
    {state.session?.deployed && <p className={styles.meta}>Everyone with the demo password shares these notes.</p>}
    {state.papersError ? <p className={styles.error} role="alert">{state.papersError}</p>
      : state.papersLoading ? <p role="status">Loading papers…</p> : !paperId ? <p>Choose a paper to see its notes.</p> : <>
      <label className={styles.label} htmlFor="notes-paper">Paper</label>
      <select id="notes-paper" value={paperId} disabled={busy} onChange={e => {
        state.setNotesPaper(e.target.value); setNotice(""); setError("");
      }}>{state.papers.map(p => <option key={p.paper_id} value={p.paper_id}>{p.title}</option>)}</select>
      {notice && <p role="status">{notice}</p>}
      {error && <p className={styles.error} role="alert">{error}</p>}
      {loadError?.paperId === paperId && <p className={styles.error} role="alert">{loadError.text}</p>}
      {records?.paperId !== paperId && loadError?.paperId !== paperId && <p role="status">Loading notes…</p>}
      {records?.paperId === paperId && <div className={styles.notes}>{records.notes.map(note => <article key={note.id}>
        <p className={styles.meta}><time dateTime={note.created_at}>{new Date(note.created_at).toLocaleString("en-US", {
          month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit",
        })}</time></p>
        <div className={styles.note}>{noteText(note)}</div>
        <button onClick={() => remove(note.id)} disabled={busy}>Delete</button>
      </article>)}</div>}
      <form onSubmit={add} className={styles.form}>
        <label htmlFor="new-note">New note</label>
        <textarea id="new-note" rows={5} maxLength={5000} value={text} disabled={busy} onChange={e => setText(e.target.value)} />
        <button disabled={busy || !text.trim()}>Add note</button>
      </form>
      {/* Content-Disposition triggers the download; an API error opens as a plain browser page, which is accepted. */}
      <a className={styles.download} href={`/api/notes/export?paper_id=${encodeURIComponent(paperId)}`}>Download Markdown</a>
    </>}
  </>;
}
