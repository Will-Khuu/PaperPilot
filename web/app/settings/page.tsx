"use client";

import { useAppState } from "../state";
import styles from "./page.module.css";

export default function SettingsPage() {
  const state = useAppState();
  return <>
    <h1>Settings</h1>
    <div className={styles.settings}>
      <label htmlFor="answer-k">Excerpts per answer <output htmlFor="answer-k">{state.k}</output></label>
      <input id="answer-k" type="range" min={1} max={10} value={state.k} onChange={e => state.setK(Number(e.target.value))} />
      <label htmlFor="compare-k">Excerpts per paper in compare <output htmlFor="compare-k">{state.compareK}</output></label>
      <input id="compare-k" type="range" min={1} max={5} value={state.compareK} onChange={e => state.setCompareK(Number(e.target.value))} />
    </div>
    {state.health && <p className={styles.meta}>{state.health.chunks} excerpts · chunks {state.health.chunk_size}/{state.health.chunk_overlap}</p>}
    {state.healthError && <p className={styles.error} role="alert">{state.healthError}</p>}
  </>;
}
