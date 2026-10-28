# Deployment

The Next.js app runs on Vercel and the FastAPI API runs on Fly.io. The API requires a shared password and sets a
signed, HttpOnly, Secure session cookie for seven days. Use HTTPS for the deployed frontend. The browser calls
only its own Vercel origin; Next.js rewrites `/api/*` to Fly, so the API needs no CORS middleware.

## Fly.io API

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

## Vercel frontend

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
controlled by `PAPERPILOT_MAX_UPLOAD_MB`, which accepts decimals such as `3.5`. Vercel documents a 4.5 MB
body limit for its functions, but a live test through this app's `/api` rewrite forwarded 15 MB and failed at 30 MB. Local uploads stay at 20 MB regardless of that variable. Uploaded sources open the
authenticated PDF at its cited page.
Deleting an uploaded paper removes its chunks, PDF and notes. The four built-in papers cannot be deleted.

## Streamlit Community Cloud (alternative)

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

## Known limits

- Anyone can trigger the global failed-login limit and block new sign-ins for a minute. Existing sessions keep working.
- Sign out only deletes the browser cookie. A copied cookie works until it expires; rotating `SESSION_SECRET` signs everyone out.
- Each login gets a new allowance of 10 paid answers per day. The global 40/day limit and Anthropic spending cap are the real limits.
- Streamlit Cloud and Fly keep separate budget counters. Use separate API keys or rely on the Anthropic spending cap.
- Vercel waits at most 120 seconds for the API. Answers take well under that.
