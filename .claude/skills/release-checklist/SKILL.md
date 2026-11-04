---
name: release-checklist
description: Cut a new PorosityFE release — bump the version, update the changelog, run the full lint/type/test gate, and rebuild the standalone validate_porosity executable. Use when the user says things like "cut a 1.3.0 release", "prepare a patch release", "ship v1.2.1", or "tag the next version".
---

# Release checklist

PorosityFE has two version sources of truth and one bundled binary:

- `pyproject.toml` → `[project] version` — the canonical version.
- `porosity_fe/__init__.py:__version__` — the fallback used in source
  checkouts that aren't pip-installed. **Keep these two in lockstep.**
- `dist/validate_porosity/validate_porosity` — the PyInstaller-built CLI
  with the bundled validation database. CI builds this automatically per
  push (see `.github/workflows/build-executables.yml`), but a local
  rebuild verifies the spec still works before tagging.

Between releases, master carries the next version with a `.dev0` suffix
(e.g. `1.3.0.dev0`), its changes sit under `## [Unreleased]`, and
`CITATION.cff` and the README BibTeX name the last release. The release
commit drops the suffix and dates the changelog and the citations.

The CI matrix (`.github/workflows/tests.yml`) runs on Ubuntu/macOS/Windows
× Python 3.11/3.12/3.13/3.14 plus the FE regression gate (`fe_fingerprint`),
a decoupled Streamlit job (the app import and the app tests) and a lint job
(`ruff` + `mypy` with `numpy<2`). Mirror it locally before tagging.

## Get from the user first

- **Target version** (semver: e.g. `1.3.0`).
- **Release kind** (patch / minor / major) and a one-line headline for
  the changelog.

## Steps

1. **Confirm the working tree is clean** and you're on the release
   branch the user expects:
   ```bash
   git status
   git log --oneline -10
   ```

2. **Bump the version in both files**, and in the citations.
   - `pyproject.toml`: `version = "<new>"` under `[project]` (from
     `<new>.dev0` when master already carries it).
   - `porosity_fe/__init__.py`: the `__version__ = "<new>"` fallback
     (the comment there explicitly says "keep in sync with
     pyproject.toml on each release").
   - `CITATION.cff` (`version`, `date-released`) and the BibTeX block in
     `README.md` (`version = {<new>}`).

3. **Update `CHANGELOG.md`.** Add a dated section for the new version
   following the existing style. Pull the entries from `git log
   <previous-tag>..HEAD --oneline` and group them into Added / Changed /
   Fixed / Removed. Don't invent entries — every line should map to a
   landed commit. Move everything under `## [Unreleased]` into the new
   section and leave `[Unreleased]` empty: the publish workflow refuses
   to release otherwise. For a minor or major release, update or add the
   docs page `docs/whatsnew.md` ("What's new in X.Y"), with the results
   that change on upgrade.

4. **Run the local gate** (must mirror `.github/workflows/tests.yml`):
   ```bash
   # Lint job (mypy uses numpy<2 to match CI stubs)
   ruff check .
   pip install "numpy<2"
   mypy porosity_fe porosity_fe_analysis.py

   # Restore the runtime numpy and run tests
   pip install "numpy>=1.20"
   python -m pytest tests/ -v

   # Streamlit smoke (decoupled in CI to protect the lib matrix from
   # streamlit wheel issues on 3.13 — see issue #157)
   python -c "import app; print('app imports OK')"

   # The same release check publish.yml runs on the tag
   python .github/scripts/check_release.py v<new>
   ```

5. **Rebuild the standalone CLI** to verify the PyInstaller spec still
   works and the validation datasets bundle cleanly:
   ```bash
   pip install pyinstaller
   python -m PyInstaller ValidatePorosity.spec --noconfirm --clean
   ./dist/validate_porosity/validate_porosity --help
   ./dist/validate_porosity/validate_porosity --output-dir /tmp/rel-check
   ```
   Confirm `validation_master_report.png` + `validation_detail_report.md`
   are produced and the aggregate MAEs match what the README claims (if
   they drift past a tenth of a percent, update the README before
   tagging).

6. **Stop here and hand back to the user for the actual tag + push.**
   Do *not* run `git tag` or `git push --tags` from the skill. Print
   the suggested commands instead:
   ```
   git add pyproject.toml porosity_fe/__init__.py CHANGELOG.md CITATION.cff README.md
   git commit -m "release: v<new>"
   git tag -a v<new> -m "v<new>"
   git push origin <branch> --follow-tags
   ```
   Confirm the version, changelog headline, and gate results in your
   reply so the user can sanity-check before pushing.

## What this skill does NOT cover

- **Pushing tags or creating GitHub releases.** That's a deliberate
  manual step — the user should review the staged commit + changelog
  entry before any tag goes public.
- **Publishing to PyPI.** Pushing the `v<new>` tag triggers
  `.github/workflows/publish.yml`. It reruns the release check, builds
  and checks the sdist and wheel, and publishes through PyPI trusted
  publishing. A published version cannot be re-uploaded, so the tag push
  is the point of no return. Make that clear when handing back. One-time
  setup, done by the maintainer on PyPI: a trusted publisher for
  `elhajjar1/PorosityFE`, workflow `publish.yml`, environment `pypi`.
- **Editing `CITATION.cff`.** Update it manually if the citation year
  rolls over; the existing block already pins `year = {2026}`.
