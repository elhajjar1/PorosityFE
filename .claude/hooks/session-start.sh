#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Installs PorosityFE in editable mode with the [all] extras (web +
# dev + docs) so a fresh container can immediately run the project's
# real gate: `pytest tests/`, `ruff check .`, `mypy porosity_fe
# porosity_fe_analysis.py`, and `streamlit run app.py`.
#
# Mirrors the CI lint job's `numpy<2` pin so local mypy runs reproduce
# CI's signal (numpy 2.x stubs surface unrelated errors -- see the
# pinning comment in .github/workflows/tests.yml).
set -euo pipefail

# Only run inside the remote (Claude Code on the web) container. Local
# dev environments manage their own venv.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

# Skip the `pip install --upgrade pip` that CI runs: the web-session
# container uses a Debian-managed pip that can't uninstall itself
# ("Cannot uninstall pip ..., RECORD file not found"). The shipped pip
# is new enough to resolve our deps.
pip install -e ".[all]"
# Lint-env pin: matches .github/workflows/tests.yml so `mypy` produces
# the same diagnostics locally as in CI.
pip install "numpy<2"
