#!/usr/bin/env python3
"""
Porosity FE Analysis - Mac GUI Application
============================================
Tkinter-based GUI for running porosity defect analysis on composite laminates.
"""

import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Ensure matplotlib uses a non-interactive backend for thread safety
import matplotlib
matplotlib.use('Agg')

import numpy as np

# Import the analysis engine
from porosity_fe_analysis import (
    MATERIALS, POROSITY_CONFIGS, PorosityField, CompositeMesh,
    EmpiricalSolver, MoriTanakaSolver, FEVisualizer,
    compare_configurations, save_results_to_json
)


class PorosityApp:
    """Main GUI application for Porosity FE Analysis."""

    def __init__(self, root):
        self.root = root
        self.root.title("Porosity FE Analysis Tool")
        self.root.geometry("820x700")
        self.root.minsize(700, 600)

        self.output_dir = tk.StringVar(value=os.path.expanduser("~/Desktop/Porosity_Results"))
        self.running = False

        self._build_ui()

    def _build_ui(self):
        # Main frame
        main = ttk.Frame(self.root, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(main, text="Porosity Defect Analysis for Composite Laminates",
                          font=("Helvetica Neue", 16, "bold"))
        title.pack(pady=(0, 10))

        # --- Parameters Frame ---
        params = ttk.LabelFrame(main, text="Analysis Parameters", padding=10)
        params.pack(fill=tk.X, pady=5)

        # Material selection
        row1 = ttk.Frame(params)
        row1.pack(fill=tk.X, pady=3)
        ttk.Label(row1, text="Material:", width=16).pack(side=tk.LEFT)
        self.material_var = tk.StringVar(value="T800_epoxy")
        mat_combo = ttk.Combobox(row1, textvariable=self.material_var,
                                  values=list(MATERIALS.keys()), state="readonly", width=25)
        mat_combo.pack(side=tk.LEFT, padx=5)

        # Porosity levels
        row2 = ttk.Frame(params)
        row2.pack(fill=tk.X, pady=3)
        ttk.Label(row2, text="Porosity levels (%):", width=16).pack(side=tk.LEFT)
        self.porosity_var = tk.StringVar(value="1, 2, 3, 5, 8")
        ttk.Entry(row2, textvariable=self.porosity_var, width=28).pack(side=tk.LEFT, padx=5)

        # Configuration selection
        row3 = ttk.Frame(params)
        row3.pack(fill=tk.X, pady=3)
        ttk.Label(row3, text="Configurations:", width=16).pack(side=tk.LEFT, anchor=tk.N)

        config_frame = ttk.Frame(row3)
        config_frame.pack(side=tk.LEFT, padx=5)
        self.config_vars = {}
        for name in POROSITY_CONFIGS:
            var = tk.BooleanVar(value=True)
            self.config_vars[name] = var
            ttk.Checkbutton(config_frame, text=name.replace('_', ' ').title(),
                           variable=var).pack(anchor=tk.W)

        # Mesh resolution
        row4 = ttk.Frame(params)
        row4.pack(fill=tk.X, pady=3)
        ttk.Label(row4, text="Mesh (nx, ny, nz):", width=16).pack(side=tk.LEFT)
        self.mesh_var = tk.StringVar(value="30, 10, 12")
        ttk.Entry(row4, textvariable=self.mesh_var, width=28).pack(side=tk.LEFT, padx=5)

        # --- Output Frame ---
        out_frame = ttk.LabelFrame(main, text="Output", padding=10)
        out_frame.pack(fill=tk.X, pady=5)

        out_row = ttk.Frame(out_frame)
        out_row.pack(fill=tk.X)
        ttk.Label(out_row, text="Output folder:", width=16).pack(side=tk.LEFT)
        ttk.Entry(out_row, textvariable=self.output_dir, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(out_row, text="Browse...", command=self._browse_output).pack(side=tk.LEFT)

        # --- Run Button ---
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=10)
        self.run_btn = ttk.Button(btn_frame, text="Run Analysis", command=self._start_analysis)
        self.run_btn.pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(btn_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.LEFT, padx=10)
        self.status_label = ttk.Label(btn_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)

        # --- Log output ---
        log_frame = ttk.LabelFrame(main, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.log = scrolledtext.ScrolledText(log_frame, height=12, font=("Menlo", 11),
                                              state=tk.DISABLED, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_dir.set(path)

    def _log(self, msg):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _start_analysis(self):
        if self.running:
            return
        self.running = True
        self.run_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        self.status_label.config(text="Running...")

        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()

    def _run_analysis(self):
        try:
            # Parse parameters
            material_name = self.material_var.get()
            porosity_levels = [float(x.strip()) / 100.0
                               for x in self.porosity_var.get().split(",")]
            selected_configs = {k: v for k, v in POROSITY_CONFIGS.items()
                                if self.config_vars[k].get()}
            mesh_parts = [int(x.strip()) for x in self.mesh_var.get().split(",")]
            nx, ny, nz = mesh_parts[0], mesh_parts[1], mesh_parts[2]

            out_dir = self.output_dir.get()
            os.makedirs(out_dir, exist_ok=True)

            if not selected_configs:
                self.root.after(0, lambda: messagebox.showwarning(
                    "No configs", "Select at least one configuration."))
                return

            self.root.after(0, lambda: self._log(
                f"Starting analysis: {material_name}, "
                f"{len(porosity_levels)} levels, {len(selected_configs)} configs"))

            all_results = {}
            total_steps = len(porosity_levels)

            for step_i, Vp in enumerate(porosity_levels):
                Vp_label = f"{int(Vp * 100)}pct"
                self.root.after(0, lambda v=Vp: self._log(
                    f"\n--- Porosity level: {v*100:.0f}% ---"))

                # Run analysis
                material = MATERIALS[material_name]
                results = {}
                for name, config in selected_configs.items():
                    self.root.after(0, lambda n=name: self._log(f"  Config: {n}"))
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
                    comp_kd = results[name]['empirical']['compression']['judd_wright']['knockdown']
                    self.root.after(0, lambda kd=comp_kd: self._log(
                        f"    Compression KD: {kd:.3f}"))

                all_results[Vp_label] = results

                # Generate plots
                self.root.after(0, lambda: self._log("  Generating plots..."))
                for name in results:
                    FEVisualizer.plot_porosity_field(
                        results[name]['porosity_field'],
                        save_path=os.path.join(out_dir, f"porosity_profile_{name}_{Vp_label}.png"))
                    import matplotlib.pyplot as plt
                    plt.close('all')

                    FEVisualizer.plot_mesh_3d(
                        results[name]['mesh'],
                        save_path=os.path.join(out_dir, f"porosity_mesh_3d_{name}_{Vp_label}.png"))
                    plt.close('all')

                    FEVisualizer.plot_mesh_detail(
                        results[name]['mesh'],
                        save_path=os.path.join(out_dir, f"porosity_mesh_detail_{name}_{Vp_label}.png"))
                    plt.close('all')

                    FEVisualizer.plot_damage_contour(
                        results[name]['mesh'],
                        results[name]['empirical_solver'],
                        save_path=os.path.join(out_dir, f"porosity_damage_{name}_{Vp_label}.png"))
                    plt.close('all')

                FEVisualizer.plot_model_comparison(
                    results,
                    save_path=os.path.join(out_dir, f"porosity_comparison_{Vp_label}.png"))
                plt.close('all')

                save_results_to_json(results,
                                     os.path.join(out_dir, f"porosity_analysis_results_{Vp_label}.json"))

                self.root.after(0, lambda s=step_i, t=total_steps: self._log(
                    f"  Level complete ({s+1}/{t})"))

            # Cross-severity knockdown curves
            if len(porosity_levels) > 1:
                FEVisualizer.plot_knockdown_curves(
                    all_results,
                    save_path=os.path.join(out_dir, "porosity_knockdown_curves.png"))
                import matplotlib.pyplot as plt
                plt.close('all')

            self.root.after(0, lambda: self._log(
                f"\nAnalysis complete! Results saved to:\n  {out_dir}"))
            self.root.after(0, lambda: self.status_label.config(text="Complete"))

            # Open output folder in Finder
            self.root.after(0, lambda: os.system(f'open "{out_dir}"'))

        except Exception as e:
            self.root.after(0, lambda err=str(e): self._log(f"\nERROR: {err}"))
            self.root.after(0, lambda: self.status_label.config(text="Error"))
            import traceback
            tb = traceback.format_exc()
            self.root.after(0, lambda t=tb: self._log(t))
        finally:
            self.running = False
            self.root.after(0, lambda: self.run_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.progress.stop())


def main():
    root = tk.Tk()
    app = PorosityApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
