"use client";

import { useAppState } from "../state";
import { Conversation } from "@/components/conversation";
import { StatusCard } from "@/components/answer-card";
import styles from "./page.module.css";

export default function ComparePage() {
  const state = useAppState();
  return <>
    <h1>Compare</h1>
    {state.papersError ? <StatusCard text={state.papersError} error /> : <>
    <fieldset className={styles.picker}><legend>Choose 2-4 papers</legend>
      {state.papers.map(paper => <label key={paper.paper_id}>
        <input type="checkbox" checked={state.compareIds.includes(paper.paper_id)}
          disabled={!state.compareIds.includes(paper.paper_id) && state.compareIds.length >= 4}
          onChange={e => state.setCompareIds(previous => e.target.checked
            ? [...previous, paper.paper_id] : previous.filter(id => id !== paper.paper_id))} />
        {paper.title}
      </label>)}
    </fieldset>
    {(state.compareIds.length < 2 || state.compareIds.length > 4) && <p>Choose 2-4 papers to compare.</p>}
    </>}
    <Conversation compare />
  </>;
}
