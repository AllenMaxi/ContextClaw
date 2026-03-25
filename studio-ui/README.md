# ContextClaw Studio UI

React frontend for the local `ContextClaw Studio` control plane.

Demo artifact:

- [../demo-artifacts/contextclaw-studio-demo.mp4](../demo-artifacts/contextclaw-studio-demo.mp4)

## Development

1. Start the Python daemon:

```bash
cclaw studio serve --root /path/to/project
```

2. Start the frontend dev server:

```bash
cd contextclaw/studio-ui
npm install
npm run dev
```

The Vite dev server proxies API requests to `http://127.0.0.1:8765`.

## Production Build

```bash
cd contextclaw/studio-ui
npm install
npm run build
```

When `dist/` exists, the FastAPI daemon serves the built frontend at `/studio`.

The packaging smoke test builds a wheel and verifies that the built frontend
lands inside `contextclaw/studio/_frontend/`.

## Native Shell

This workspace now includes a Tauri shell scaffold in `src-tauri/`.

The shell now starts and stops a bundled Python Studio daemon sidecar
automatically, choosing a local port at runtime and passing the resolved API
base to the React app through the native shell.

Before building the native shell, install the desktop build tooling:

```bash
cd contextclaw
python3 -m pip install -e .[studio,desktop-build]
```

Then launch the native shell:

```bash
cd contextclaw/studio-ui
npm run tauri:dev
```

`npm run tauri:build` bundles the React app together with the compiled sidecar.

If you need a non-default Python interpreter for the sidecar build, set
`CONTEXTCLAW_PYTHON_BIN=/path/to/python`.

If you need to force a specific sidecar port while debugging the desktop app,
set `CONTEXTCLAW_DESKTOP_PORT=<port>`.

The native build has been verified on macOS arm64 and produces:

- `src-tauri/target/debug/bundle/macos/ContextClaw Studio.app`

## CI And Release

The repo now includes:

- a CI frontend job for `npm run test` and `npm run build`
- a packaging smoke job that verifies the built React app lands inside the
  wheel artifact
- a macOS desktop smoke job that builds the Tauri shell and verifies the
  bundled sidecar responds to `/status` and shuts down gracefully

The `Studio Release` workflow builds draft macOS release artifacts for both
Apple Silicon and Intel targets. If Apple signing secrets are available, it
imports the certificate and lets Tauri sign the app; if notarization secrets
(`APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID`) are also present, Tauri can
submit the build for notarization during release packaging.
