"use client";

import { useState } from "react";
import type { ReactNode } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAppState } from "@/app/state";
import { errorMessage, request } from "@/lib/api";
import { Icon } from "./icon";
import styles from "./shell.module.css";

const pages = [["/papers", "My Papers"], ["/chat", "Chat"], ["/compare", "Compare"],
  ["/summaries", "Summaries"], ["/notes", "Notes"], ["/settings", "Settings"]];

export function Shell({ children }: { children: ReactNode }) {
  const state = useAppState();
  const pathname = usePathname();
  const router = useRouter();
  const [signingOut, setSigningOut] = useState(false);
  const [error, setError] = useState("");
  async function signOut() {
    setSigningOut(true); setError("");
    try {
      await request("/logout", { method: "POST" });
      // Reload the provider too, so signing out clears threads and loaded notes.
      window.location.replace("/login");
    } catch (error) { setError(errorMessage(error)); setSigningOut(false); }
  }
  if (pathname === "/login") return <main id="main">{children}</main>;
  if (state.session?.deployed && !state.session.authenticated) return <p role="status">Opening sign in…</p>;
  return <div className={styles.layout}>
    <a className={styles.skip} href="#main">Skip to content</a>
    <aside className={styles.sidebar}>
      <div className={styles.brand}><Icon /><strong>PaperPilot</strong></div>
      <button className={styles.newChat} onClick={() => {
        state.setMessages([]);
        state.setChatPending(null);
        router.push("/chat");
      }}>+ New Chat</button>
      <nav className={styles.nav} aria-label="Main navigation">
        {pages.map(([href, label]) => <Link key={href} href={href}
          aria-current={pathname === href ? "page" : undefined}
          className={pathname === href ? styles.active : undefined}>{label}</Link>)}
      </nav>
      <footer className={styles.footer}>
        {state.health?.answers_left != null && <p>{state.health.answers_left} paid answers left today</p>}
        {state.session?.deployed && <button onClick={signOut} disabled={signingOut}>
          {signingOut ? "Signing out…" : "Sign out"}</button>}
        {error && <p role="alert">{error}</p>}
      </footer>
    </aside>
    <main id="main" className={styles.main}>{children}</main>
  </div>;
}
