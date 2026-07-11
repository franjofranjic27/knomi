#!/usr/bin/env node
"use strict";

// Postinstall hook. Best-effort install of the `knomi` PyPI package so that
// the shim in bin/knomi.js has something to delegate to.
//
// IMPORTANT: this must never abort `npm install`. If Python is missing or the
// install fails, we print guidance and exit 0 with a warning. The shim will
// still print actionable help at runtime if the CLI is unavailable.

const { spawnSync } = require("node:child_process");

// npm sets this during `npm ci --ignore-scripts` etc.; also honour a manual
// opt-out so CI can skip the Python install.
if (process.env.KNOMI_SKIP_POSTINSTALL === "1") {
  console.log("knomi: KNOMI_SKIP_POSTINSTALL=1 set, skipping Python install.");
  process.exit(0);
}

const PY_INSTALL_HELP = [
  "",
  "knomi (npm) is a thin wrapper around the `knomi` PyPI package.",
  "Python >= 3.12 is required but was not found (or is too old).",
  "",
  "Install Python 3.12+ from https://www.python.org/downloads/ (or via your",
  "package manager), then install the CLI yourself with one of:",
  "  uv tool install knomi",
  "  pipx install knomi",
  "  python3 -m pip install --user knomi",
  "",
].join("\n");

function run(cmd, args) {
  return spawnSync(cmd, args, { stdio: "pipe", encoding: "utf8" });
}

// Returns [major, minor] of the given python executable, or null if it is not
// usable / not present.
function pythonVersion(exe) {
  const res = run(exe, ["--version"]);
  if (res.error || res.status !== 0) {
    return null;
  }
  // `python --version` prints e.g. "Python 3.12.4" (to stdout on 3.4+).
  const out = `${res.stdout || ""}${res.stderr || ""}`;
  const m = out.match(/Python\s+(\d+)\.(\d+)/);
  if (!m) {
    return null;
  }
  return [Number(m[1]), Number(m[2])];
}

function isAtLeast312(version) {
  if (!version) return false;
  const [major, minor] = version;
  return major > 3 || (major === 3 && minor >= 12);
}

// Find a usable python interpreter (>= 3.12).
function findPython() {
  for (const exe of ["python3", "python"]) {
    const v = pythonVersion(exe);
    if (isAtLeast312(v)) {
      return { exe, version: v };
    }
  }
  return null;
}

// Check whether an executable exists on PATH.
function have(cmd) {
  const probe = process.platform === "win32" ? "where" : "which";
  const res = run(probe, [cmd]);
  return res.status === 0;
}

function tryInstall(python) {
  // Ordered strategies: uv (fastest, isolated) -> pipx (isolated) -> pip --user.
  const strategies = [];
  if (have("uv")) {
    strategies.push({ label: "uv tool install", cmd: "uv", args: ["tool", "install", "knomi"] });
  }
  if (have("pipx")) {
    strategies.push({ label: "pipx install", cmd: "pipx", args: ["install", "knomi"] });
  }
  strategies.push({
    label: "pip install --user",
    cmd: python.exe,
    args: ["-m", "pip", "install", "--user", "knomi"],
  });

  for (const s of strategies) {
    console.log(`knomi: attempting install via ${s.label} ...`);
    const res = spawnSync(s.cmd, s.args, { stdio: "inherit" });
    if (!res.error && res.status === 0) {
      console.log(`knomi: installed successfully via ${s.label}.`);
      return true;
    }
    console.warn(`knomi: ${s.label} failed, trying next strategy ...`);
  }
  return false;
}

function main() {
  const python = findPython();
  if (!python) {
    console.warn(PY_INSTALL_HELP);
    // Do not fail npm install.
    process.exit(0);
  }

  console.log(`knomi: found Python ${python.version.join(".")} at "${python.exe}".`);

  // If the CLI is already installed (e.g. re-install), we are done.
  if (have("knomi")) {
    console.log("knomi: `knomi` CLI already on PATH, nothing to do.");
    process.exit(0);
  }

  const ok = tryInstall(python);
  if (!ok) {
    console.warn(
      [
        "",
        "knomi: automatic install of the PyPI package did not succeed.",
        "You can finish setup manually with one of:",
        "  uv tool install knomi",
        "  pipx install knomi",
        "  python3 -m pip install --user knomi",
        "",
      ].join("\n"),
    );
  }
  // Always succeed so npm install completes.
  process.exit(0);
}

main();
