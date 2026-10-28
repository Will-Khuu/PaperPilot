"use client";

import { useRef, useState } from "react";
import type { FormEvent } from "react";
import { useRouter } from "next/navigation";
import { useAppState } from "../state";
import { errorMessage, request } from "@/lib/api";
import type { Paper } from "@/lib/api";
import { PdfBadge } from "@/components/icon";
import styles from "./page.module.css";

export default function PapersPage() {
  const state = useAppState();
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  function select(ids: string[] | null) { state.setPaperIds(ids); router.push("/chat"); }
  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || busy || deleting) return;
    setBusy(true); setNotice(""); setError("");
    const body = new FormData();
    body.append("file", file);
    try {
      const paper = await request<Paper>("/papers", { method: "POST", body });
      setNotice(`Added ${paper.title} (${paper.chunks} chunks).`);
      setFile(null);
      if (fileInput.current) fileInput.current.value = "";
      await state.refreshPapers();
      await state.refreshHealth();
    } catch (error) { setError(errorMessage(error)); }
    finally { setBusy(false); }
  }
  async function remove(paper: Paper) {
    if (busy || deleting || !window.confirm(`Delete "${paper.title}" and its notes?`)) return;
    setDeleting(paper.paper_id); setNotice(""); setError("");
    try {
      await request(`/papers/${paper.paper_id}`, { method: "DELETE" });
      state.setSummaries(previous => {
        const remaining = { ...previous };
        delete remaining[paper.paper_id];
        return remaining;
      });
      state.setSummaryErrors(previous => {
        const remaining = { ...previous };
        delete remaining[paper.paper_id];
        return remaining;
      });
      setNotice(`Deleted ${paper.title}.`);
      await state.refreshPapers();
      await state.refreshHealth();
    } catch (error) { setError(errorMessage(error)); }
    finally { setDeleting(null); }
  }
  return <>
    <h1>My Papers</h1>
    <button onClick={() => select(null)}>All papers</button>
    {state.papersLoading && <p role="status">Loading papers…</p>}
    {state.papersError && <p className={styles.error} role="alert">{state.papersError}</p>}
    {!state.papersLoading && !state.papersError && !state.papers.length && <p>No papers are indexed yet.</p>}
    <div className={styles.papers}>{state.papers.map(paper => <article className={styles.card} key={paper.paper_id}>
      <div className={styles.paper}><PdfBadge /><div><div className={styles.title}>{paper.title}</div>
        <div className={styles.meta}>{paper.pages} pages · {paper.chunks} excerpts</div></div></div>
      <button onClick={() => select([paper.paper_id])}>Chat about this paper</button>
      {state.session?.can_upload && !paper.builtin && <button disabled={busy || deleting !== null}
        onClick={() => remove(paper)}>{deleting === paper.paper_id ? "Deleting…" : "Delete"}</button>}
    </article>)}</div>
    {state.session?.can_upload && <><h2>Add paper</h2>
    <form className={styles.upload} onSubmit={upload}>
      <label htmlFor="pdf-upload">Upload a PDF</label>
      <p className={styles.meta}>PDF, up to {state.session.max_upload_mb} MB, 60 pages</p>
      <input ref={fileInput} id="pdf-upload" type="file" accept=".pdf,application/pdf" disabled={busy || deleting !== null}
        onChange={e => setFile(e.target.files?.[0] ?? null)} />
      <button disabled={!file || busy || deleting !== null}>{busy ? "Adding…" : "Add paper"}</button>
    </form>
    {notice && <p role="status">{notice}</p>}
    {error && <p className={styles.error} role="alert">{error}</p>}</>}
  </>;
}
