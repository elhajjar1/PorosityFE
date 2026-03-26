#!/usr/bin/env python3
"""
Porosity FE Analysis - Mac Application
========================================
Runs porosity analysis and opens results in Finder.
Uses matplotlib Agg backend for plot generation (no GUI toolkit required).
"""

import sys
import os

# Use non-interactive backend — avoids all Tk/Qt issues in bundled apps
import matplotlib
matplotlib.use('Agg')

import numpy as np

from porosity_fe_analysis import (
    MATERIALS, POROSITY_CONFIGS, PorosityField, CompositeMesh,
    EmpiricalSolver, MoriTanakaSolver, FEVisualizer,
    compare_configurations, save_results_to_json
)


def run_analysis():
    """Run the full porosity analysis and save results."""
    output_dir = os.path.expanduser("~/Desktop/Porosity_Results")
    os.makedirs(output_dir, exist_ok=True)

    material_name = 'T800_epoxy'
    porosity_levels = [0.01, 0.02, 0.03, 0.05, 0.08]

    print("=" * 70)
    print("POROSITY FE ANALYSIS TOOL")
    print("=" * 70)
    print(f"Material: {material_name}")
    print(f"Porosity levels: {[f'{v*100:.0f}%' for v in porosity_levels]}")
    print(f"Configurations: {list(POROSITY_CONFIGS.keys())}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    import matplotlib.pyplot as plt

    all_results = {}
    total = len(porosity_levels)

    for idx, Vp in enumerate(porosity_levels):
        Vp_label = f"{int(Vp * 100)}pct"
        print(f"\n[{idx+1}/{total}] Analyzing Vp = {Vp*100:.0f}%...")

        results = compare_configurations(Vp, material_name=material_name)
        all_results[Vp_label] = results

        # Generate plots
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

        save_results_to_json(results,
                             os.path.join(output_dir, f"porosity_analysis_results_{Vp_label}.json"))

    # Cross-severity knockdown curves
    FEVisualizer.plot_knockdown_curves(
        all_results,
        save_path=os.path.join(output_dir, "porosity_knockdown_curves.png"))
    plt.close('all')

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_dir}")

    # Count outputs
    pngs = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
    jsons = len([f for f in os.listdir(output_dir) if f.endswith('.json')])
    print(f"Generated: {pngs} PNG plots, {jsons} JSON result files")

    # Open in Finder
    os.system(f'open "{output_dir}"')

    return 0


def main():
    try:
        return run_analysis()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to close...")
        return 1


if __name__ == "__main__":
    sys.exit(main())
