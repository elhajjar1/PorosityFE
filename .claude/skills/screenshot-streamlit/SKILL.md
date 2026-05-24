---
name: screenshot-streamlit
description: Run the PorosityFE Streamlit app headlessly and capture screenshots for the README. Use when the user says things like "regenerate the screenshots", "update the knockdown_curves.png", "grab a fresh screenshot of the FE stress tab", or "the README screenshot is out of date".
---

# Screenshot the Streamlit app

The README embeds three PNGs from `screenshots/`:
`knockdown_curves.png`, `porosity_distributions.png`, and `void_scf.png`.
They're hand-curated, not auto-generated, so regenerating them is a
deliberate "I changed the UI / a default / a curve and want the README
to match" step.

`app.py` calls `matplotlib.use("Agg")` and adjusts `sys.path` before
importing `pyplot`, so it runs cleanly in a headless container. The web
extra (`streamlit`) is optional — install it explicitly before driving
the UI.

## Required first

- **Which screenshot(s)** to regenerate (or "all three").
- **Whether the user wants the same defaults** as the existing PNGs
  (material `T800_epoxy`, layup QI, the canonical sweep), or different
  inputs. The existing screenshots aren't pinned to a config in
  source — confirm before clobbering them.

## Steps

1. **Install the web extra and start the app headlessly:**
   ```bash
   pip install -e ".[web]"
   streamlit run app.py \
     --server.headless true \
     --server.port 8501 \
     --browser.gatherUsageStats false &
   # Wait for "You can now view your Streamlit app in your browser."
   curl -sf http://localhost:8501/_stcore/health && echo "up"
   ```

2. **Capture screenshots.** Two viable paths depending on the host:
   - **Playwright** (preferred — installable in the web container, no
     display server needed):
     ```bash
     pip install playwright && playwright install chromium
     python - <<'PY'
     from playwright.sync_api import sync_playwright
     with sync_playwright() as p:
         browser = p.chromium.launch()
         page = browser.new_page(viewport={"width": 1440, "height": 900})
         page.goto("http://localhost:8501")
         page.wait_for_selector("button:has-text('Run analysis')", timeout=15000)
         # Drive the sidebar here: select material, set Vp, click Run.
         page.click("button:has-text('Run analysis')")
         page.wait_for_selector("[data-testid='stTabs']", timeout=30000)
         # Switch tabs and screenshot each one into screenshots/.
         page.screenshot(path="screenshots/knockdown_curves.png", full_page=False)
         browser.close()
     PY
     ```
   - **Selenium + headless Chrome**: fall back to this only if
     Playwright is blocked by the environment. Same flow.

3. **Match the existing PNG dimensions/aspect** so the README layout
   doesn't shift. The current files were taken at roughly 1440×900;
   honor that unless the user explicitly wants a different viewport.

4. **Kill the streamlit process** when done:
   ```bash
   pkill -f "streamlit run app.py" || true
   ```

5. **Verify the new PNGs render** by checking file size and dimensions —
   a 0-byte or 100-byte PNG means the selector waited on the wrong
   element:
   ```bash
   for f in screenshots/*.png; do
     printf '%s  ' "$f"; identify "$f" 2>/dev/null || stat -c '%s bytes' "$f"
   done
   ```

6. **Show the user the new PNGs** (the `SendUserFile` tool surfaces them
   in chat) before committing — these are user-facing assets and the
   user should eyeball them.

## Gotchas

- **Don't reorder `app.py` imports.** It deliberately calls
  `matplotlib.use("Agg")` and sets `sys.path` before importing `pyplot`
  and sibling modules; `pyproject.toml` adds `E402` / `I001` to
  ruff's per-file ignores for exactly that reason. Reordering will
  silently break the headless run on hosts without a display.
- **The Streamlit FE stress tab can take 30–60s to render** at the
  production mesh resolution (`_DEFAULT_MESH_RES = (30, 10, 12)`).
  Use a longer `page.wait_for_selector` timeout for the FE tabs than
  for the Profile / Mesh tabs.
- **Streamlit Community Cloud's tier is 1 GB RAM.** If you're trying to
  capture an expert-mode mesh and the page never finishes, that's an
  OOM — reduce `nx*ny*nz` in the sidebar before screenshotting.
