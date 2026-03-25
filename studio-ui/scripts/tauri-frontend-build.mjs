import { spawnSync } from "node:child_process";

const npmBin = process.platform === "win32" ? "npm.cmd" : "npm";
const env = {
  ...process.env,
  VITE_STUDIO_SHELL: "tauri",
};

const buildSidecar = spawnSync(npmBin, ["run", "tauri:sidecar"], {
  stdio: "inherit",
  env,
});
if (buildSidecar.status !== 0) {
  process.exit(buildSidecar.status ?? 1);
}

const viteBuild = spawnSync(npmBin, ["run", "build"], {
  stdio: "inherit",
  env,
});
process.exit(viteBuild.status ?? 0);
