"use client";

import { useAppState } from "../state";
import { errorMessage, request } from "@/lib/api";
import type { SummaryResult } from "@/lib/api";
import { AnswerCard, StatusCard } from "@/components/answer-card";
import styles from "./page.module.css";

export default function SummariesPage() {
  const state = useAppState();
  const pending = state.summaryPending !== null;
  const paperId = state.summariesPaper;
  const error = state.summaryErrors[paperId];
  async function summarize() {
    if (!paperId || pending) return;
    state.setSummaryPending(paperId);
    state.setSummaryErrors(previous => ({ ...previous, [paperId]: "" }));
    try {
      const result = await request<SummaryResult>("/summarize", { method: "POST", body: JSON.stringify({ paper_id: paperId }) });
      state.setSummaries(previous => ({ ...previous, [paperId]: result }));
    } catch (error) {
      state.setSummaryErrors(previous => ({ ...previous, [paperId]: errorMessage(error) }));
    } finally {
      state.setSummaryPending(null);
      void state.refreshHealth();
    }
  }
  return <>
    <h1>Summaries</h1>
    {state.papersError ? <StatusCard text={state.papersError} error />
      : state.papersLoading ? <p role="status">Loading papers…</p> : !state.papers.length
      ? <p>Choose a paper to summarize.</p>
      : <div className={styles.controls}><label htmlFor="summary-paper">Paper</label>
        <select id="summary-paper" value={paperId} disabled={pending} onChange={e => state.setSummariesPaper(e.target.value)}>
          {state.papers.map(p => <option key={p.paper_id} value={p.paper_id}>{p.title}</option>)}
        </select>
        <button disabled={pending || !paperId} onClick={summarize}>Summarize</button>
      </div>}
    {state.summaryPending === paperId && <StatusCard text="Thinking…" />}
    {error && <StatusCard text={error} error />}
    {state.summaries[paperId] && <AnswerCard key={paperId} result={state.summaries[paperId]} paperIds={[paperId]} />}
  </>;
}
