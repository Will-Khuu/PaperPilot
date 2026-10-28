# PaperPilot

Ask questions about a research paper and get answers that cite the exact section and page.

PaperPilot is a retrieval-augmented generation (RAG) pipeline built one stage at a time, so each stage can be inspected,
measured and swapped on its own. It supports several papers, with filters, PDF uploads and a compare mode.

```
$ python step4_answer.py ask "How many attention heads does the Transformer use?"

RETRIEVED
  [1] attention-019  d=0.411  p. 5  3.2.2 Multi-Head Attention
  ...

ANSWER
The Transformer uses h = 8 parallel attention layers, or heads [1]. For each head,
dk = dv = dmodel/h = 64 [1]. ...

SOURCES
  [1] Attention Is All You Need, 3.2.2 Multi-Head Attention, p. 5  (attention-019)
```

## How it works

| Stage | Script | What it does |
| --- | --- | --- |
| 1. Parse | `step1_parse.py` | Extracts text page by page with PyMuPDF, then applies narrow cleaning rules: drops page numbers and figure noise, rejoins split words and section headings. |
| 2. Chunk | `step2_chunk.py` | Splits at section headings, then into overlapping windows of at most 220 tokens (80 overlap). Each chunk keeps its section and page range. |
| 3. Embed and retrieve | `step3_embed.py` | Embeds chunks with all-MiniLM-L6-v2 into a local Chroma collection (cosine distance) and returns the closest chunks to a question. |
| 4. Answer | `step4_answer.py` | Sends the top 5 chunks to Claude as numbered excerpts. The answer cites `[n]`, and each `[n]` is mapped back to its section and page. |
| 5. Evaluate | `step5_eval.py` | Scores retrieval on 20 labeled questions: hit rate at 1, 5 and 10, by page and by answer phrase, per question type. |
| 6. Tune | `step6_sweep.py` | Re-scores 12 chunk size and overlap settings in throwaway indexes and applies a decision rule fixed before the run. |
| 7. Serve | `step7_api.py`, `step7_ui.py` | FastAPI server (/health, /search, /ask) and a Streamlit chat page that shows answers with their sources and the retrieved excerpts. |
| 8. Several papers | `step3_embed.py`, `step7_api.py`, `step7_ui.py` | Filter retrieval by paper, compare 2-4 papers, and upload PDFs into the shared index. |
| 9. Summaries | `step9_summary.py`, `step7_api.py`, `step7_ui.py` | Summarize one paper from its chunks in reading order, with citations to sections and pages. |
| 10. Notes | `step10_notes.py`, `step7_api.py`, `step7_ui.py` | Save answers and free-text notes per paper in SQLite, delete notes, and export Markdown. |
| 11. Caching | `step11_cache.py`, `step7_api.py`, `step7_ui.py` | Cache exact-match answers, comparisons and summaries in SQLite. Entries become stale when indexed paper chunk IDs, chunk settings, the prompt or model change. |

## Design decisions

- **Chunks never cross a section heading.** A chunk about one topic gives a focused embedding and an accurate section
  label for citations.
- **Chunk size is set by the embedding model's limit.** all-MiniLM-L6-v2 silently truncates input after 256 word
  pieces. Counting with the model's own tokenizer (a word-count estimate was off by up to 2x) keeps every chunk under it.
- **Citations map by position, not by the model's memory.** `[1]` is the first excerpt in the prompt, so the page comes
  from metadata attached at ingest, never from the LLM.
- **The paper's own citations are rewritten before generation.** The source text contains bracketed references like
  `[2]`; left as-is, the model could copy one and it would point to the wrong excerpt. They become `(ref 2)`.
- **The model must admit what the excerpts don't cover.** It answers with one fixed sentence when the excerpts are
  insufficient, and any answer without citations is flagged as likely outside knowledge.
- **Retrieval is tested before generation.** A bad answer can then be traced to either the search or the LLM.

## Results so far

- **Retrieval** on 20 labeled questions (4 each: fact, paraphrase, table number, concept, who/relation), at
  220-token chunks:

  | | Hit rate at 1 | Hit rate at 5 | Hit rate at 10 |
  | --- | --- | --- | --- |
  | Right page retrieved (40 overlap → 80) | 60% → 70% | 95% → 90% | 100% → 100% |
  | Retrieved chunk contains the answer (40 overlap → 80) | 45% → 55% | 90% → 90% | 100% → 100% |

  Scoring the answer phrase as well as the page matters: for one table question, a chunk from the right page ranked
  first while the chunk holding the number ranked ninth with overlap 40. The questions were written with the paper in hand, so these
  numbers are optimistic.
- **Chunk tuning:** a sweep of chunk sizes 100–256 and overlaps 0–80 (12 settings) moved rank-1 answer hits only
  between 6 and 11 of 20, with no consistent trend. Two settings, 220/80 and 100/80, met the decision rule set before the sweep (+2 answers
  at rank 1, none lost from the top 5). After the run, 100/80 was set aside because its 80-token overlap on
  roughly 90-token windows made 577 near-duplicate chunks and lost a question from the top 10. 220/80 was adopted,
  but the gain is two questions, within noise. The finding is
  that chunking is not the bottleneck: the remaining misses come from paraphrased questions and ranking order, which
  point to a stronger embedding model or adding keyword search.
- **Parser comparison:** Unstructured (`fast` and `hi_res`) against PyMuPDF on the same 20 questions, with cleaning,
  chunking and embeddings held fixed (`step13_parsers.py`). Answer at rank 1: PyMuPDF 12, fast 11, hi_res 9 of 20, so
  PyMuPDF stays. All three kept every answer in their text; Unstructured lost on structure, mixing footnotes into the
  body and merging a run-in heading with its paragraph. hi_res kept tables as rows and columns, but took 28–41 s per
  paper against 0.1–0.2 s for PyMuPDF.
- **Step 14, LangChain comparison:** `step14_langchain.py` measures LangChain's loader, splitter and Chroma
  wrapper against the hand-built pipeline, same MiniLM vectors. Attention phrase hit@1/5/10: ours 12/19/20, LangChain
  defaults 9/12/18 (most 4000-character chunks run past the model's 256-token limit), LangChain sized to match
  9/18/20, our cleaned pages with LangChain's splitter 9/19/20. Nothing adopted. Setup: `/opt/homebrew/bin/python3.12 -m venv .venv-langchain`, then
  `.venv-langchain/bin/python -m pip install -r requirements.txt -r requirements-langchain.txt`.
- **Answers:** all 8 answered; all 17 cited claims were checked against the cited text and matched
  (measured on the earlier 220/40 index). For "What is BERT?"
  (not in the paper) the model declined instead of answering from training data.
- **Known limits:** numbered footnotes can land in the wrong section's chunk, and table text loses its row and column
  structure.

## Run it

Requires Python 3.12 and an Anthropic API key for stage 4.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p papers
curl -L -o papers/attention.pdf https://arxiv.org/pdf/1706.03762v7

python step1_parse.py papers/attention.pdf           # parsed and cleaned text in out/
python step2_chunk.py papers/attention.pdf           # chunks in out/attention.chunks.jsonl
python step3_embed.py ingest papers/*.pdf           # index every PDF
python step3_embed.py ask "What is scaled dot-product attention?"

export ANTHROPIC_API_KEY=...                         # or load it from your OS keychain
python step4_answer.py ask "What is scaled dot-product attention?"
python step4_answer.py ask "..." --dry-run           # print the exact prompt, no API call
python step4_answer.py selftest                      # offline checks, no API key needed

python step5_eval.py run --label current             # retrieval scores, saved to out/eval/current.json
python step6_sweep.py run                            # 12 chunk settings, saved to out/eval/sweep.json
```

Start the API with the key in the environment, then start the chat page in a second terminal
with the virtual environment activated:

```bash
python step7_api.py serve
# Second terminal:
streamlit run step7_ui.py
```

Both servers run on this machine only (127.0.0.1). Search only needs no API key.
The chat page's address comes from `.streamlit/config.toml`.
The chat page uses `PAPERPILOT_API` if set, otherwise `http://127.0.0.1:8000`.

## Deploy

The Next.js app runs on Vercel and the FastAPI API runs on Fly.io. The API requires a shared password and sets a
signed, HttpOnly, Secure session cookie for seven days. Use HTTPS for the deployed frontend. The browser calls
only its own Vercel origin; Next.js rewrites `/api/*` to Fly, so the API needs no CORS middleware.

### Fly.io API

Create a Fly account and install its CLI. From the repository root, keep the settings in the supplied `fly.toml`:

```sh
fly launch --no-deploy
fly volumes create paperpilot_data --size 1 --region lax
fly secrets set ANTHROPIC_API_KEY=... APP_PASSWORD=... SESSION_SECRET=...
fly deploy --ha=false
```

Choose a unique app name if `paperpilot-api` is taken, and keep `primary_region = "lax"`. `--ha=false` creates
one machine instead of a redundant pair. The config uses one shared-cpu-1x machine with 1 GB RAM and 512 MB swap,
stops it when idle, and starts it on a request. The 1 GB volume stores notes, cached answers and budgets at
`/data/paperpilot.db`, selected by `PAPERPILOT_DB`. The index lives at `/data/chroma_db` and PDFs at `/data/papers`,
selected by `PAPERPILOT_INDEX_DIR` and `PAPERPILOT_PAPERS_DIR`. These files survive machine restarts and deploys.

Set a long shared `APP_PASSWORD`. Generate a separate `SESSION_SECRET` with:

```sh
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Owner uploads and deletion require both `SESSION_SECRET` and `OWNER_PASSWORD`. Both must differ from
`APP_PASSWORD`, which gives visitors read access. Matching passwords disable owner access.
In bash, enter the owner secret without echoing it or putting its value in shell history:

```bash
read -rs -p 'Owner password: ' owner_password
echo
fly secrets set OWNER_PASSWORD="$owner_password"
unset owner_password
```

Without either `SESSION_SECRET` or `OWNER_PASSWORD`, the deployed library stays read-only. Changing
`OWNER_PASSWORD` signs owners out. Existing sessions from before owner roles were added must sign in again.

Use a separate demo Anthropic API key and set a monthly spending cap in the Anthropic Console before sharing
the link. Enter real secret values only in the host's secret settings or CLI, never in a tracked file.
`python deploy.py check` reports which secret names are set and the database path without printing values.

The Docker build downloads the four pinned arXiv papers, builds their index and warms the ONNX embedding
model. It runs the build and API as the same non-root user so the model cache under
`/home/paperpilot/.cache/chroma/onnx_models` is accessible. On first startup, `python deploy.py seed` copies the
image's built-in index and PDFs onto the volume. It copies each directory into a temporary sibling, then renames
the complete copy into place. Existing directories are left alone, so uploads survive new image builds.
At startup, the entrypoint briefly runs as root to give that user ownership of Fly's mounted `/data` directory,
then drops privileges before seeding and starting the API. Embedding uses batches of 32 to limit peak memory.
Do not run `python deploy.py build` against a shared local index; it is intended for the image build.

A parsing-code change does not update the disk index until it is reset. To rebuild it from the deployed image,
run `fly ssh console`, then remove both directories inside that console:

```sh
rm -rf /data/chroma_db /data/papers
exit
```

Then run `fly machine restart <machine-id>` locally. This deletes all uploaded PDFs and their indexed chunks;
the next startup seeds only the four built-in papers. Notes and budget records remain in `/data/paperpilot.db`.

### Vercel frontend

Import the repository in Vercel, select `web` as the root directory, and set
`API_URL=https://<app>.fly.dev`, using the Fly app name. Deploy, then sign in at `/login`.
Fly secrets stay on Fly; Vercel needs only `API_URL`.

The demo allows 40 paid answers per UTC day across all visitors and 10 per login session. Signing in again
creates a new session, so the global limit and provider spending cap remain the spending backstop.
Search, cached answers and notes work after either limit. An API or connection failure before an answer gives
the reserved call back for its original UTC day. Timeouts, refusals and cut-off answers still count because
the server may have generated billed tokens.
Everyone with the demo password shares the papers and notes.

Only the owner can add or delete papers on the deployed web app. Uploads are limited to one at a time, 60 pages
per PDF and 20 papers in the library, including the four built-in papers. Deployed uploads default to 4 MB,
controlled by `PAPERPILOT_MAX_UPLOAD_MB`, which accepts decimals such as `3.5`, to stay below Vercel's roughly
4.5 MB request-body limit. Local uploads stay at 20 MB regardless of that variable. Uploaded sources open the
authenticated PDF at its cited page.
Deleting an uploaded paper removes its chunks, PDF and notes. The four built-in papers cannot be deleted.

### Existing Streamlit Community Cloud demo

Keep `step7_ui.py` as the main file and Python 3.12 in Advanced settings. Its root-level TOML secrets are:

```toml
PAPERPILOT_DEPLOYED = "1"
APP_PASSWORD = "..."
ANTHROPIC_API_KEY = "..."
SESSION_SECRET = "..."
```

Add a generated `SESSION_SECRET` for explicit signing-key configuration. Existing Cloud apps that have only
`APP_PASSWORD` continue to work: the API derives a signing key from the password when `SESSION_SECRET` is absent.
Changing that password invalidates cookies signed with the derived key. A missing `APP_PASSWORD` fails closed.

The page keeps its own password gate and calls the API in-process with a signed `streamlit` session cookie.
That cookie has the visitor role; deployed Streamlit stays upload-free, including when `OWNER_PASSWORD` is set.
All Streamlit visitors share one allowance of 10 paid answers per UTC day, within the global 40.
A Cloud cold start still downloads and indexes the papers. Its notes, cache and budget counters reset when
the disk resets; the separate demo API key's monthly spending cap is the backstop.

### Known limits

- Anyone can trigger the global failed-login limit and block new sign-ins for a minute. Existing sessions keep working.
- Sign out only deletes the browser cookie. A copied cookie works until it expires; rotating `SESSION_SECRET` signs everyone out.
- Each login gets a new allowance of 10 paid answers per day. The global 40/day limit and Anthropic spending cap are the real limits.
- Streamlit Cloud and Fly keep separate budget counters. Use separate API keys or rely on the Anthropic spending cap.
- Vercel waits at most 120 seconds for the API. Answers take well under that.

## Roadmap

- [x] Parse, chunk, embed, retrieve
- [x] Answers with page-level citations
- [x] Evaluation set of 20 questions with known answer pages; hit rate at 1, 5 and 10
- [x] Tune chunk size and overlap against that baseline (marginal gain; chunking is not the bottleneck)
- [x] FastAPI backend and Streamlit chat UI
- [x] Multiple papers and a compare mode
- [x] Paper summaries with citations
- [x] Notes per paper with Markdown export
- [x] Exact-match caching for answers, comparisons and summaries
- [x] Compare the Unstructured parser against PyMuPDF (PyMuPDF kept)

## License

MIT
