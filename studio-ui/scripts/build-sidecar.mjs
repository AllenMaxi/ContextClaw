import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");
const pythonBin = process.env.CONTEXTCLAW_PYTHON_BIN || "python3";
const scriptPath = path.resolve(projectRoot, "../scripts/build_studio_sidecar.py");

const result = spawnSync(
  pythonBin,
  [
    scriptPath,
    ...(process.env.CONTEXTCLAW_TAURI_TARGET_TRIPLE
      ? ["--target-triple", process.env.CONTEXTCLAW_TAURI_TARGET_TRIPLE]
      : []),
  ],
  {
    cwd: path.resolve(projectRoot, ".."),
    stdio: "inherit",
    env: process.env,
  },
);

if (result.status !== 0) {
  process.exit(result.status ?? 1);
}
