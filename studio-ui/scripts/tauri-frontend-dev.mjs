import { spawnSync } from "node:child_process";

const npmBin = process.platform === "win32" ? "npm.cmd" : "npm";
const env = {
  ...process.env,
  VITE_STUDIO_SHELL: "tauri",
  VITE_STUDIO_API_BASE: "http://127.0.0.1:8765",
};

const buildSidecar = spawnSync(npmBin, ["run", "tauri:sidecar"], {
  stdio: "inherit",
  env,
});
if (buildSidecar.status !== 0) {
  process.exit(buildSidecar.status ?? 1);
}

const viteDev = spawnSync(npmBin, ["run", "dev"], {
  stdio: "inherit",
  env,
});
process.exit(viteDev.status ?? 0);
