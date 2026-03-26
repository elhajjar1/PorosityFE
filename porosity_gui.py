#!/usr/bin/env python3
"""
Porosity FE Analysis - Web-Based GUI
======================================
Launches a local web server and opens a browser-based configuration GUI.
User defines the problem, clicks Run, and results are generated.
No tkinter/Qt dependencies — uses Python stdlib http.server + browser.
"""

import sys
import os
import json
import threading
import webbrowser
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from porosity_fe_analysis import (
    MATERIALS, POROSITY_CONFIGS, VOID_SHAPES, PorosityField, CompositeMesh,
    EmpiricalSolver, MoriTanakaSolver, FEVisualizer,
    compare_configurations, save_results_to_json
)

# Global state for progress reporting
progress_log = []
analysis_running = False
analysis_done = False

# Default output location — sibling to wherever the app lives
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(os.path.dirname(_SCRIPT_DIR), 'Porosity_Results')


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Porosity FE Analysis</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f5f7; color: #1d1d1f; padding: 20px; }
  .container { max-width: 800px; margin: 0 auto; }
  h1 { font-size: 28px; font-weight: 600; margin-bottom: 5px; }
  .subtitle { color: #86868b; font-size: 14px; margin-bottom: 25px; }
  .card { background: white; border-radius: 12px; padding: 24px;
          margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .card h2 { font-size: 18px; font-weight: 600; margin-bottom: 16px;
             padding-bottom: 8px; border-bottom: 1px solid #e5e5e7; }
  .form-row { display: flex; align-items: center; margin-bottom: 14px; gap: 12px; }
  .form-row label.field-label { width: 160px; font-weight: 500; font-size: 14px; flex-shrink: 0; }
  select, input[type="text"] {
    flex: 1; padding: 8px 12px; border: 1px solid #d2d2d7; border-radius: 8px;
    font-size: 14px; background: #fafafa; outline: none; }
  select:focus, input:focus { border-color: #0071e3; box-shadow: 0 0 0 3px rgba(0,113,227,0.15); }
  .checkbox-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; flex: 1; }
  .checkbox-grid label { width: auto; font-weight: 400; font-size: 13px;
                          display: flex; align-items: center; gap: 6px; cursor: pointer; }
  .checkbox-grid input[type="checkbox"] { width: 16px; height: 16px; accent-color: #0071e3; }
  .btn { display: inline-block; padding: 12px 32px; border-radius: 10px;
         font-size: 16px; font-weight: 600; cursor: pointer; border: none;
         transition: all 0.2s; }
  .btn-primary { background: #0071e3; color: white; }
  .btn-primary:hover { background: #0077ED; transform: scale(1.02); }
  .btn-primary:disabled { background: #86868b; cursor: not-allowed; transform: none; }
  .btn-row { text-align: center; margin-top: 20px; }
  #log-card { display: none; }
  #log { background: #1d1d1f; color: #33ff33; border-radius: 8px;
         padding: 16px; font-family: 'Menlo', monospace; font-size: 12px;
         height: 260px; overflow-y: auto; white-space: pre-wrap; }
  #status { text-align: center; margin: 16px 0; font-size: 15px; font-weight: 500; }
  .status-ready { color: #34c759; }
  .status-running { color: #ff9f0a; }
  .status-done { color: #30d158; }
  .status-error { color: #ff3b30; }
  .info { font-size: 12px; color: #86868b; margin-top: 4px; }
</style>
</head>
<body>
<div class="container">
  <h1>Porosity FE Analysis</h1>
  <p class="subtitle">Composite laminate porosity defect assessment tool</p>

  <div class="card">
    <h2>Material</h2>
    <div class="form-row">
      <label class="field-label">Material system:</label>
      <select id="material">
        <option value="T800_epoxy" selected>T800/Epoxy (CYCOM X850)</option>
        <option value="T700_epoxy">T700/Epoxy (Toray 2510)</option>
        <option value="glass_epoxy">E-Glass/Epoxy</option>
      </select>
    </div>
  </div>

  <div class="card">
    <h2>Porosity Parameters</h2>
    <div class="form-row">
      <label class="field-label">Porosity levels (%):</label>
      <input type="text" id="porosity_levels" value="1, 3, 5">
    </div>
    <p class="info" style="margin-left:172px;">Comma-separated values between 0.5 and 15</p>

    <div class="form-row" style="margin-top:16px; align-items: flex-start;">
      <label class="field-label">Configurations:</label>
      <div class="checkbox-grid">
        <label><input type="checkbox" name="config" value="uniform_spherical" checked> Uniform Spherical</label>
        <label><input type="checkbox" name="config" value="uniform_cylindrical" checked> Uniform Cylindrical</label>
        <label><input type="checkbox" name="config" value="clustered_midplane" checked> Clustered Midplane</label>
        <label><input type="checkbox" name="config" value="clustered_surface"> Clustered Surface</label>
        <label><input type="checkbox" name="config" value="interface_penny" checked> Interface Penny</label>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Mesh &amp; Output</h2>
    <div class="form-row">
      <label class="field-label">Mesh resolution:</label>
      <select id="mesh">
        <option value="20,8,8">Coarse (20 x 8 x 8) &mdash; fast</option>
        <option value="30,10,12" selected>Medium (30 x 10 x 12)</option>
        <option value="50,20,24">Fine (50 x 20 x 24) &mdash; slow</option>
      </select>
    </div>
    <div class="form-row">
      <label class="field-label">Output folder:</label>
      <input type="text" id="output_dir" value="__OUTPUT__">
    </div>
  </div>

  <div class="btn-row">
    <button class="btn btn-primary" id="runBtn" onclick="runAnalysis()">Run Analysis</button>
  </div>

  <div id="status" class="status-ready">Configure your analysis above, then click Run.</div>

  <div class="card" id="log-card">
    <h2>Progress</h2>
    <div id="log"></div>
  </div>
</div>

<script>
let pollInterval = null;

function runAnalysis() {
  const btn = document.getElementById('runBtn');
  const logCard = document.getElementById('log-card');
  const log = document.getElementById('log');
  const status = document.getElementById('status');

  const configs = Array.from(document.querySelectorAll('input[name="config"]:checked'))
                       .map(c => c.value);
  if (configs.length === 0) {
    status.textContent = 'Select at least one configuration.';
    status.className = 'status-error';
    return;
  }

  btn.disabled = true;
  logCard.style.display = 'block';
  log.textContent = '';
  status.textContent = 'Running analysis...';
  status.className = 'status-running';

  const data = {
    material: document.getElementById('material').value,
    porosity_levels: document.getElementById('porosity_levels').value,
    configs: configs.join(','),
    mesh: document.getElementById('mesh').value,
    output_dir: document.getElementById('output_dir').value
  };

  fetch('/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  });

  pollInterval = setInterval(async () => {
    try {
      const resp = await fetch('/progress');
      const result = await resp.json();
      log.textContent = result.log.join('\n');
      log.scrollTop = log.scrollHeight;
      if (result.done) {
        clearInterval(pollInterval);
        status.textContent = 'Analysis complete! Results folder opened.';
        status.className = 'status-done';
        btn.disabled = false;
      }
    } catch(e) {}
  }, 500);
}
</script>
</body>
</html>"""


class GUIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            page = HTML_PAGE.replace('__OUTPUT__', DEFAULT_OUTPUT)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(page.encode())
        elif self.path == '/progress':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'log': progress_log[-300:],
                'done': analysis_done,
                'running': analysis_running,
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global analysis_running, analysis_done
        if self.path == '/run' and not analysis_running:
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"started"}')
            threading.Thread(target=run_analysis_task, args=(body,), daemon=True).start()
        else:
            self.send_response(409)
            self.end_headers()


def log_msg(msg):
    progress_log.append(msg)
    print(msg)


def run_analysis_task(params):
    global analysis_running, analysis_done
    analysis_running = True
    analysis_done = False
    progress_log.clear()

    try:
        material_name = params['material']
        porosity_levels = [float(x.strip()) / 100.0
                           for x in params['porosity_levels'].split(',')]
        config_names = params['configs'].split(',')
        mesh_parts = [int(x) for x in params['mesh'].split(',')]
        nx, ny, nz = mesh_parts
        output_dir = params['output_dir']

        selected_configs = {k: POROSITY_CONFIGS[k] for k in config_names
                            if k in POROSITY_CONFIGS}
        os.makedirs(output_dir, exist_ok=True)

        log_msg("=" * 55)
        log_msg("POROSITY FE ANALYSIS")
        log_msg("=" * 55)
        log_msg(f"Material:       {material_name}")
        log_msg(f"Porosity:       {[f'{v*100:.0f}%' for v in porosity_levels]}")
        log_msg(f"Configurations: {list(selected_configs.keys())}")
        log_msg(f"Mesh:           {nx} x {ny} x {nz}")
        log_msg(f"Output:         {output_dir}")
        log_msg("")

        all_results = {}
        total = len(porosity_levels)

        for idx, Vp in enumerate(porosity_levels):
            Vp_label = f"{int(Vp * 100)}pct"
            log_msg(f"[{idx+1}/{total}] Vp = {Vp*100:.0f}%")

            material = MATERIALS[material_name]
            results = {}

            for name, config in selected_configs.items():
                log_msg(f"  {name}")
                pf = PorosityField(material, Vp, **config)
                mesh = CompositeMesh(pf, material, nx=nx, ny=ny, nz=nz)
                emp = EmpiricalSolver(mesh, material)
                mt = MoriTanakaSolver(mesh, material)

                results[name] = {
                    'config': config,
                    'mesh': mesh,
                    'porosity_field': pf,
                    'empirical_solver': emp,
                    'mori_tanaka_solver': mt,
                    'empirical': emp.get_all_failure_loads(),
                    'mori_tanaka': mt.get_all_failure_loads(),
                }

                ckd = results[name]['empirical']['compression']['judd_wright']['knockdown']
                ikd = results[name]['empirical']['ilss']['judd_wright']['knockdown']
                log_msg(f"    Comp KD: {ckd:.3f}  ILSS KD: {ikd:.3f}")

            all_results[Vp_label] = results

            log_msg(f"  Saving plots...")
            for name in results:
                FEVisualizer.plot_porosity_field(
                    results[name]['porosity_field'],
                    save_path=os.path.join(output_dir, f"porosity_profile_{name}_{Vp_label}.png"))
                plt.close('all')
                FEVisualizer.plot_mesh_detail(
                    results[name]['mesh'],
                    save_path=os.path.join(output_dir, f"porosity_mesh_detail_{name}_{Vp_label}.png"))
                plt.close('all')
                FEVisualizer.plot_damage_contour(
                    results[name]['mesh'],
                    results[name]['empirical_solver'],
                    save_path=os.path.join(output_dir, f"porosity_damage_{name}_{Vp_label}.png"))
                plt.close('all')

            FEVisualizer.plot_model_comparison(
                results,
                save_path=os.path.join(output_dir, f"porosity_comparison_{Vp_label}.png"))
            plt.close('all')

            save_results_to_json(
                results,
                os.path.join(output_dir, f"porosity_analysis_results_{Vp_label}.json"))

        if len(porosity_levels) > 1:
            FEVisualizer.plot_knockdown_curves(
                all_results,
                save_path=os.path.join(output_dir, "porosity_knockdown_curves.png"))
            plt.close('all')

        pngs = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
        jsons = len([f for f in os.listdir(output_dir) if f.endswith('.json')])
        log_msg("")
        log_msg("=" * 55)
        log_msg(f"COMPLETE: {pngs} plots, {jsons} JSON files")
        log_msg(f"Results: {output_dir}")
        log_msg("=" * 55)

        os.system(f'open "{output_dir}"')

    except Exception as e:
        log_msg(f"\nERROR: {e}")
        import traceback
        log_msg(traceback.format_exc())
    finally:
        analysis_running = False
        analysis_done = True


def main():
    port = get_free_port()
    server = HTTPServer(('127.0.0.1', port), GUIHandler)
    url = f'http://127.0.0.1:{port}'
    print(f"Porosity FE Analysis GUI: {url}")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
