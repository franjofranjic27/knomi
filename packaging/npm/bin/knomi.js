#!/usr/bin/env node
"use strict";

// Thin shim: locate the real `knomi` console script (installed from PyPI via
// the postinstall step) and forward all argv + stdio to it. This file never
// reimplements any CLI logic — it only delegates.

const { spawnSync } = require("node:child_process");

// Candidate commands, in order of preference. The real entry point is the
// `knomi` console script created by the PyPI package. As fallbacks we try the
// module invocations, which work as long as the package is importable.
const CANDIDATES = [
  { cmd: "knomi", args: [] },
  { cmd: "python3", args: ["-m", "knomi"] },
  { cmd: "python", args: ["-m", "knomi"] },
];

const forwardedArgs = process.argv.slice(2);

function tryRun(candidate) {
  const result = spawnSync(candidate.cmd, [...candidate.args, ...forwardedArgs], {
    stdio: "inherit",
    // On Windows, resolve commands like `knomi.exe` / `knomi.cmd`.
    shell: false,
  });
  // ENOENT => command not found; move on to the next candidate.
  if (result.error && result.error.code === "ENOENT") {
    return null;
  }
  return result;
}

for (const candidate of CANDIDATES) {
  const result = tryRun(candidate);
  if (result !== null) {
    // `status` may be null if the process was killed by a signal.
    process.exit(result.status === null ? 1 : result.status);
  }
}

// Nothing worked: the underlying PyPI package is not installed / not on PATH.
process.stderr.write(
  [
    "",
    "knomi: could not find the underlying Python CLI.",
    "",
    "This npm package is a thin wrapper around the `knomi` PyPI package.",
    "It requires Python >= 3.12 and the `knomi` package to be installed.",
    "",
    "Install it with one of:",
    "  uv tool install knomi",
    "  pipx install knomi",
    "  python3 -m pip install --user knomi",
    "",
    "Then make sure the install location is on your PATH and retry.",
    "",
  ].join("\n"),
);
process.exit(127);
