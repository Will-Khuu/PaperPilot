"use client";

import { useState } from "react";
import type { FormEvent } from "react";
import { errorMessage, request } from "@/lib/api";
import styles from "./page.module.css";

export default function LoginPage() {
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function signIn(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy) return;
    setBusy(true); setError("");
    try {
      await request("/login", { method: "POST", body: JSON.stringify({ password }) });
      // A fresh page reloads session data and papers after the cookie is set.
      window.location.replace("/chat");
    } catch (error) { setError(errorMessage(error)); setBusy(false); }
  }

  return <div className={styles.page}>
    <form className={styles.form} onSubmit={signIn} aria-busy={busy}>
      <h1>Sign in to PaperPilot</h1>
      <p>Enter the demo password to explore the papers.</p>
      <label htmlFor="password">Password</label>
      <input id="password" name="password" type="password" autoComplete="current-password" required
        value={password} disabled={busy} aria-describedby={error ? "login-error" : undefined}
        onChange={event => setPassword(event.target.value)} />
      {error && <p id="login-error" className={styles.error} role="alert">{error}</p>}
      <button disabled={busy || !password}>{busy ? "Signing in…" : "Sign in"}</button>
    </form>
  </div>;
}
