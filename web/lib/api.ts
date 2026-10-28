// These fields follow step7_api.py; optional Pydantic fields arrive as null.
export type Paper = { paper_id: string; title: string; chunks: number; pages: number };
export type Health = {
  chunks: number;
  chunk_size: number;
  chunk_overlap: number;
  answers_left: number | null;
  daily_limit: number | null;
  session_limit: number | null;
};
export type Session = { deployed: boolean; authenticated: boolean };
export type Excerpt = {
  paper_id: string; n: number; id: string; section: string; title: string;
  page: number; page_end: number; pages: string; distance: number | null; text: string;
};
export type SummarySource = { n: number; id: string; label: string; paper_id: string | null; page: number | null; url: string | null };
export type Source = Omit<SummarySource, "paper_id"> & { paper_id: string; title: string };
export type SearchResult = { question: string; retrieved: Excerpt[] };
export type AnswerResult = SearchResult & {
  answer: string; model: string; fallback: boolean; sources: Source[];
  invalid: number[]; refused: boolean; uncited: boolean; cached: boolean;
};
export type SummaryResult = {
  paper_id: string; title: string; summary: string; model: string; fallback: boolean;
  sources: SummarySource[]; invalid: number[]; uncited: boolean; chunks_used: number; cached: boolean;
};
export type NoteSource = Pick<Source, "n" | "label">;
export type Note = {
  id: number; paper_id: string; text: string; question: string | null;
  answer: string | null; sources: NoteSource[]; created_at: string;
};
export type Result = SearchResult | AnswerResult | SummaryResult;

function detailMessage(data: unknown): string | null {
  if (!data || typeof data !== "object" || !("detail" in data)) return null;
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    const messages = data.detail.flatMap((item: unknown) =>
      item && typeof item === "object" && "msg" in item && typeof item.msg === "string" ? [item.msg] : []);
    return messages.join("; ") || null;
  }
  return null;
}

export function request<T = void>(path: string, options?: RequestInit): Promise<T>;
export async function request(path: string, options: RequestInit = {}): Promise<unknown> {
  let response: Response;
  try {
    // The same-origin /api proxy avoids browser CORS and keeps the backend address on the server.
    response = await fetch(`/api${path}`, {
      ...options,
      headers: options.body instanceof FormData ? options.headers : { "Content-Type": "application/json", ...options.headers },
      cache: "no-store",
    });
  } catch {
    throw new Error("API is not running. Start it with python step7_api.py serve.");
  }
  if (!response.ok) {
    // Handle expired sessions in one place; stay on the login form to show password errors.
    if (response.status === 401 && window.location.pathname !== "/login") {
      window.location.replace("/login");
    }
    let detail: string | null = null;
    let hasJson = false;
    try {
      const data: unknown = await response.json();
      hasJson = true;
      detail = detailMessage(data);
    } catch { /* Proxy failures can return plain text instead of FastAPI's JSON. */ }
    if (detail) throw new Error(detail);
    if (response.status === 502 || response.status === 504 || (response.status === 500 && !hasJson)) {
      throw new Error("The API did not answer (it may be down or still working).");
    }
    throw new Error(`Server returned HTTP ${response.status}.`);
  }
  // DELETE returns no body, so callers use request<void>.
  return response.status === 204 ? undefined : response.json();
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "The request could not be completed.";
}
