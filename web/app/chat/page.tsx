"use client";

import Link from "next/link";
import { useAppState } from "../state";
import { PdfBadge } from "@/components/icon";
import { Conversation } from "@/components/conversation";
import styles from "./page.module.css";

export default function ChatPage() {
  const state = useAppState();
  return <>
    <h1 className={styles.srOnly}>Chat</h1>
    <div className={styles.toolbar}>
      <div className={styles.chips}>
        {state.papersError ? <p role="alert">{state.papersError}</p>
          : state.paperIds === null ? <div className={styles.chip}>All papers · {state.papers.length} papers</div>
          : state.papers.filter(p => state.paperIds?.includes(p.paper_id)).map(p => <div className={styles.chip} key={p.paper_id}>
            <PdfBadge /><div>{p.title}.pdf<div className={styles.meta}>{p.pages} pages</div></div>
          </div>)}
      </div>
      <Link href="/papers">Change</Link>
      <label className={styles.toggle}><input type="checkbox" checked={state.searchOnly} onChange={e => state.setSearchOnly(e.target.checked)} />Search only (free)</label>
    </div>
    <Conversation />
  </>;
}
