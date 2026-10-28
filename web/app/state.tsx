"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { Dispatch, ReactNode, SetStateAction } from "react";
import { useRouter } from "next/navigation";
import { errorMessage, request } from "@/lib/api";
import type { Health, Paper, Result, Session, SummaryResult } from "@/lib/api";

export type Message = { id: string } & (
  | { kind: "user"; text: string }
  | { kind: "result"; result: Result; paperIds: string[] }
  | { kind: "error"; text: string }
);
type Setter<T> = Dispatch<SetStateAction<T>>;
type AppState = {
  session: Session | null;
  papers: Paper[];
  papersLoading: boolean;
  papersError: string;
  refreshPapers: () => Promise<void>;
  health: Health | null;
  healthError: string;
  refreshHealth: () => Promise<void>;
  messages: Message[];
  setMessages: Setter<Message[]>;
  compareMessages: Message[];
  setCompareMessages: Setter<Message[]>;
  paperIds: string[] | null;
  setPaperIds: Setter<string[] | null>;
  compareIds: string[];
  setCompareIds: Setter<string[]>;
  summaries: Record<string, SummaryResult>;
  setSummaries: Setter<Record<string, SummaryResult>>;
  summaryErrors: Record<string, string>;
  setSummaryErrors: Setter<Record<string, string>>;
  summariesPaper: string;
  setSummariesPaper: Setter<string>;
  notesPaper: string;
  setNotesPaper: Setter<string>;
  k: number;
  setK: Setter<number>;
  compareK: number;
  setCompareK: Setter<number>;
  searchOnly: boolean;
  setSearchOnly: Setter<boolean>;
  chatPending: string | null;
  setChatPending: Setter<string | null>;
  comparePending: string | null;
  setComparePending: Setter<string | null>;
  summaryPending: string | null;
  setSummaryPending: Setter<string | null>;
};
const StateContext = createContext<AppState | null>(null);

// The layout keeps this context mounted so navigation preserves threads and in-flight requests.
export function StateProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [session, setSession] = useState<Session | null>(null);
  const [sessionError, setSessionError] = useState("");
  const [papers, setPapers] = useState<Paper[]>([]);
  const [papersLoading, setPapersLoading] = useState(true);
  const [papersError, setPapersError] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [compareMessages, setCompareMessages] = useState<Message[]>([]);
  const [paperIds, setPaperIds] = useState<string[] | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [summaries, setSummaries] = useState<Record<string, SummaryResult>>({});
  const [summaryErrors, setSummaryErrors] = useState<Record<string, string>>({});
  const [summariesPaper, setSummariesPaper] = useState("");
  const [notesPaper, setNotesPaper] = useState("");
  const [k, setK] = useState(5);
  const [compareK, setCompareK] = useState(3);
  const [searchOnly, setSearchOnly] = useState(false);
  const [chatPending, setChatPending] = useState<string | null>(null);
  const [comparePending, setComparePending] = useState<string | null>(null);
  const [summaryPending, setSummaryPending] = useState<string | null>(null);

  const refreshPapers = useCallback(() => {
    return request<Paper[]>("/papers").then(indexed => {
      setPapers(indexed);
      setPapersError("");
      const known = new Set(indexed.map(p => p.paper_id));
      setPaperIds(previous => {
        const remaining = previous?.filter(id => known.has(id));
        return remaining?.length ? remaining : null;
      });
      setCompareIds(previous => previous.filter(id => known.has(id)));
      setSummariesPaper(previous => known.has(previous) ? previous : indexed[0]?.paper_id ?? "");
      setNotesPaper(previous => known.has(previous) ? previous : indexed[0]?.paper_id ?? "");
    }).catch(error => { setPapersError(errorMessage(error)); })
      .finally(() => { setPapersLoading(false); });
  }, []);
  const refreshHealth = useCallback(() => {
    return request<Health>("/health").then(value => { setHealth(value); setHealthError(""); })
      .catch(error => { setHealth(null); setHealthError(errorMessage(error)); });
  }, []);
  useEffect(() => {
    let active = true;
    request<Session>("/session").then(value => {
      if (!active) return;
      setSession(value);
      if (value.deployed && !value.authenticated) {
        router.replace("/login");
        return;
      }
      void refreshPapers();
      void refreshHealth();
    }).catch(error => { if (active) setSessionError(errorMessage(error)); });
    return () => { active = false; };
  }, [refreshPapers, refreshHealth, router]);

  if (sessionError) return <main><p role="alert">{sessionError}</p>
    <button onClick={() => window.location.reload()}>Try again</button></main>;
  if (!session) return <main><p role="status">Loading PaperPilot…</p></main>;

  return <StateContext.Provider value={{
    session,
    // Server data, including summaries and errors kept per paper.
    papers, papersLoading, papersError, refreshPapers, health, healthError, refreshHealth,
    summaries, setSummaries, summaryErrors, setSummaryErrors,
    // Threads.
    messages, setMessages, compareMessages, setCompareMessages,
    // Selections.
    paperIds, setPaperIds, compareIds, setCompareIds,
    summariesPaper, setSummariesPaper, notesPaper, setNotesPaper,
    // Settings.
    k, setK, compareK, setCompareK, searchOnly, setSearchOnly,
    // Pending requests.
    chatPending, setChatPending, comparePending, setComparePending, summaryPending, setSummaryPending,
  }}>{children}</StateContext.Provider>;
}

export function useAppState() {
  const state = useContext(StateContext);
  if (!state) throw new Error("PaperPilot screens need StateProvider.");
  return state;
}
