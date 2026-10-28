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
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  function select(ids: string[] | null) { state.setPaperIds(ids); router.push("/chat"); }
  async function upload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || busy) return;
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
    </article>)}</div>
    {!state.session?.deployed && <><h2>Add paper</h2>
    <form className={styles.upload} onSubmit={upload}>
      <label htmlFor="pdf-upload">Upload a PDF</label>
      <input ref={fileInput} id="pdf-upload" type="file" accept=".pdf,application/pdf" disabled={busy}
        onChange={e => setFile(e.target.files?.[0] ?? null)} />
      <button disabled={!file || busy}>{busy ? "Adding…" : "Add paper"}</button>
    </form>
    {notice && <p role="status">{notice}</p>}
    {error && <p className={styles.error} role="alert">{error}</p>}</>}
  </>;
}
