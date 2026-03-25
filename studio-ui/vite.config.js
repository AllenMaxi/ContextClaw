import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const isTauriShell = process.env.VITE_STUDIO_SHELL === "tauri";

export default defineConfig({
  base: isTauriShell ? "./" : "/studio/",
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 4173,
    proxy: {
      "/projects": "http://127.0.0.1:8765",
      "/workflow": "http://127.0.0.1:8765",
      "/agents": "http://127.0.0.1:8765",
      "/runs": "http://127.0.0.1:8765",
      "/approvals": "http://127.0.0.1:8765",
      "/memory": "http://127.0.0.1:8765",
      "/memory-files": "http://127.0.0.1:8765",
      "/context": "http://127.0.0.1:8765",
      "/compact": "http://127.0.0.1:8765",
      "/docs": "http://127.0.0.1:8765",
      "/events": "http://127.0.0.1:8765",
      "/connectors": "http://127.0.0.1:8765",
      "/skills": "http://127.0.0.1:8765",
      "/contextgraph": "http://127.0.0.1:8765"
    }
  }
});
