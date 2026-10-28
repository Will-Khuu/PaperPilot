# PaperPilot web

Run the FastAPI backend from the repository root:

```sh
.venv/bin/python step7_api.py serve
```

In another terminal:

```sh
cd web
npm run dev
```

Open http://localhost:3000. Next.js forwards /api requests to http://127.0.0.1:8000.
Set API_URL before starting Next.js to use another backend.

Chat, compare selections, summaries and settings live in React state across navigation.
Reloading clears them. Notes are stored by the API.

Check the frontend with:

```sh
npx tsc --noEmit
npm run lint
npm run build
```

## Deploy on Vercel

Import the repository, set the root directory to `web`, and add the environment variable
`API_URL=https://<app>.fly.dev`. Deploy the Fly API first using the repository's deployment instructions.
The frontend sends `/api/*` through a same-origin rewrite, including the API's session cookie.
Open `/login` on the HTTPS Vercel deployment and use the shared demo password.
