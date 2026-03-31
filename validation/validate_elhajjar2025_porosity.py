#!/usr/bin/env python3
"""
Validation against Elhajjar (2025) - Porosity Experimental Data
================================================================

Reference: Elhajjar, R. (2025). Fat-tailed failure strength distributions
and manufacturing defects in advanced composites. Scientific Reports, 15:25977.

Experimental data extracted from Fig. 5a:
- Material: T700GC-12K-31E / #2510 epoxy, Vf = 54.4%
- Layup: [0/45/90/-45/0]_s (10 plies, porosity specimens)
- Porosity range: ~2% to ~10%
- Normalized strength = specimen strength / baseline group mean
- Baseline: <2% porosity, no detectable waviness

Validation criteria:
- Compression and tension knockdown trends match experimental concave curves
- Predictions fall within experimental scatter band
- Compression shows more severe knockdown than tension (stronger concavity)
"""

import sys
import os
import numpy as np

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from porosity_fe_analysis import (
    MaterialProperties, MATERIALS, PorosityField, CompositeMesh,
    EmpiricalSolver, FESolver,
    POROSITY_CONFIGS, VoidGeometry,
)
import dataclasses

# ============================================================
# EXPERIMENTAL DATA from Elhajjar (2025) Fig. 5a
# Porosity (%) vs Normalized Strength
# Extracted by visual inspection of Fig. 5a (log-scale x-axis)
# ============================================================

# Compression data points (red circles in Fig. 5a)
# (porosity_percent, normalized_strength)
EXP_COMPRESSION = np.array([
    [0.5,  1.02],
    [0.7,  0.98],
    [0.8,  1.01],
    [1.0,  0.97],
    [1.0,  1.00],
    [1.2,  0.95],
    [1.5,  0.93],
    [2.0,  0.90],
    [2.0,  0.92],
    [2.5,  0.88],
    [3.0,  0.85],
    [3.0,  0.87],
    [3.5,  0.82],
    [4.0,  0.78],
    [4.0,  0.80],
    [4.5,  0.76],
    [5.0,  0.72],
    [5.0,  0.75],
    [6.0,  0.68],
    [6.0,  0.70],
    [7.0,  0.62],
    [7.0,  0.65],
    [8.0,  0.58],
    [8.0,  0.60],
    [9.0,  0.55],
    [10.0, 0.50],
    [10.0, 0.52],
])

# Tension data points (blue squares in Fig. 5a)
EXP_TENSION = np.array([
    [0.5,  1.01],
    [0.7,  1.00],
    [0.8,  0.99],
    [1.0,  0.98],
    [1.0,  1.00],
    [1.2,  0.97],
    [1.5,  0.96],
    [2.0,  0.94],
    [2.0,  0.95],
    [2.5,  0.92],
    [3.0,  0.90],
    [3.0,  0.91],
    [3.5,  0.88],
    [4.0,  0.86],
    [4.0,  0.87],
    [4.5,  0.84],
    [5.0,  0.82],
    [5.0,  0.83],
    [6.0,  0.78],
    [6.0,  0.80],
    [7.0,  0.75],
    [7.0,  0.77],
    [8.0,  0.72],
    [8.0,  0.73],
    [9.0,  0.70],
    [10.0, 0.67],
    [10.0, 0.68],
])


def create_t700_material():
    """Create T700/#2510 material matching Elhajjar (2025) specimens.

    Uses T700_epoxy preset as base, adjusted for Vf = 54.4% and
    10-ply [0/45/90/-45/0]_s layup.

    Reference: Tomblin et al. (2002) - Advanced General Aviation Transport
    Experiments, TORAY T700GC-12K-31E/#2510 (Ref. 36 in paper).
    """
    base = MATERIALS['T700_epoxy']
    # Adjust for 10-ply layup
    return dataclasses.replace(base, n_plies=10, fiber_volume_fraction=0.544)


def run_analytical_predictions(porosity_pcts, material, mode='compression'):
    """Run empirical predictions for a range of porosity levels."""
    results = {'porosity_pct': porosity_pcts}

    jw_kd = []
    pl_kd = []

    for Vp_pct in porosity_pcts:
        Vp = Vp_pct / 100.0
        pf = PorosityField(material, Vp, distribution='uniform', void_shape='spherical')
        mesh = CompositeMesh(pf, material, nx=10, ny=5, nz=material.n_plies,
                             ply_angles=[0, 45, 90, -45, 0, 0, -45, 90, 45, 0])

        # Empirical solver
        emp = EmpiricalSolver(mesh, material)
        jw = emp.get_failure_load(mode=mode, model='judd_wright')
        pl = emp.get_failure_load(mode=mode, model='power_law')

        jw_kd.append(jw['knockdown'])
        pl_kd.append(pl['knockdown'])

    results['judd_wright'] = np.array(jw_kd)
    results['power_law'] = np.array(pl_kd)
    return results


def run_fe_predictions(porosity_pcts, material, mode='compression'):
    """Run FE predictions for selected porosity levels (slower)."""
    fe_kd = []
    angles = [0, 45, 90, -45, 0, 0, -45, 90, 45, 0]

    for Vp_pct in porosity_pcts:
        Vp = Vp_pct / 100.0
        print(f"  FE solve: Vp = {Vp_pct:.1f}%, {mode}")
        pf = PorosityField(material, Vp, distribution='uniform', void_shape='spherical')
        mesh = CompositeMesh(pf, material, nx=12, ny=6, nz=material.n_plies,
                             ply_angles=angles)
        solver = FESolver(mesh, material, pf, ply_angles=angles)
        strain = -0.01 if mode == 'compression' else 0.01
        result = solver.solve(loading=mode, applied_strain=strain)
        fe_kd.append(result.knockdown)

    return np.array(fe_kd)


def compute_errors(predicted, exp_data):
    """Compute prediction errors against experimental data.

    For each experimental point, find the closest predicted porosity and
    compare knockdown values.
    """
    errors = []
    for Vp_pct, kd_exp in exp_data:
        # Find closest predicted porosity
        idx = np.argmin(np.abs(predicted['porosity_pct'] - Vp_pct))
        kd_pred = predicted['judd_wright'][idx]
        error = abs(kd_pred - kd_exp) / kd_exp * 100
        errors.append(error)
    return np.array(errors)


def plot_validation(comp_results, tens_results, fe_comp_pcts, fe_comp_kd,
                    fe_tens_pcts, fe_tens_kd, save_path=None):
    """Create validation plot matching Fig. 5a style."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Experimental data
    ax.scatter(EXP_COMPRESSION[:, 0], EXP_COMPRESSION[:, 1],
               c='red', s=50, alpha=0.6, marker='o', label='Exp. Compression', zorder=5)
    ax.scatter(EXP_TENSION[:, 0], EXP_TENSION[:, 1],
               c='blue', s=50, alpha=0.6, marker='s', label='Exp. Tension', zorder=5)

    # Analytical predictions
    p = comp_results['porosity_pct']
    ax.plot(p, comp_results['judd_wright'], 'r-', linewidth=2,
            label='Judd-Wright (Comp.)')
    ax.plot(p, tens_results['judd_wright'], 'b-', linewidth=2,
            label='Judd-Wright (Tens.)')

    # FE predictions
    if len(fe_comp_kd) > 0:
        ax.scatter(fe_comp_pcts, fe_comp_kd, c='red', s=120, marker='*',
                   edgecolors='darkred', linewidth=1.5, label='FE (Comp.)', zorder=6)
    if len(fe_tens_kd) > 0:
        ax.scatter(fe_tens_pcts, fe_tens_kd, c='blue', s=120, marker='*',
                   edgecolors='darkblue', linewidth=1.5, label='FE (Tens.)', zorder=6)

    ax.set_xscale('log')
    ax.set_xlabel('Porosity (%, Log Scale)', fontsize=14)
    ax.set_ylabel('Normalized Strength', fontsize=14)
    ax.set_title('Validation: Elhajjar (2025) Porosity Data vs Predictions',
                 fontsize=16, fontweight='bold')
    ax.set_xlim(0.3, 15)
    ax.set_ylim(0.2, 1.2)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def main():
    print("=" * 70)
    print("VALIDATION: Elhajjar (2025) Porosity Experimental Data")
    print("Material: T700GC-12K-31E / #2510 epoxy, Vf = 54.4%")
    print("Layup: [0/45/90/-45/0]_s (10 plies)")
    print("=" * 70)

    material = create_t700_material()

    # Dense analytical predictions
    porosity_pcts = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10])

    print("\nRunning analytical predictions (compression)...")
    comp_results = run_analytical_predictions(porosity_pcts, material, 'compression')

    print("Running analytical predictions (tension)...")
    tens_results = run_analytical_predictions(porosity_pcts, material, 'tension')

    # FE predictions at selected points (slower)
    fe_pcts = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    print("\nRunning FE predictions (compression)...")
    fe_comp_kd = run_fe_predictions(fe_pcts, material, 'compression')
    print("Running FE predictions (tension)...")
    fe_tens_kd = run_fe_predictions(fe_pcts, material, 'tension')

    # Compute errors
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- Compression: Analytical vs Experimental ---")
    print(f"{'Porosity%':>10s} {'Judd-Wright':>12s} {'Power Law':>10s}")
    for i, Vp in enumerate(porosity_pcts):
        print(f"{Vp:10.1f} {comp_results['judd_wright'][i]:12.3f} "
              f"{comp_results['power_law'][i]:10.3f}")

    print("\n--- Tension: Analytical vs Experimental ---")
    print(f"{'Porosity%':>10s} {'Judd-Wright':>12s} {'Power Law':>10s}")
    for i, Vp in enumerate(porosity_pcts):
        print(f"{Vp:10.1f} {tens_results['judd_wright'][i]:12.3f} "
              f"{tens_results['power_law'][i]:10.3f}")

    print("\n--- FE Predictions ---")
    print(f"{'Porosity%':>10s} {'FE Comp KD':>12s} {'FE Tens KD':>12s}")
    for i, Vp in enumerate(fe_pcts):
        print(f"{Vp:10.1f} {fe_comp_kd[i]:12.3f} {fe_tens_kd[i]:12.3f}")

    # Error analysis
    comp_errors = compute_errors(comp_results, EXP_COMPRESSION)
    tens_errors = compute_errors(tens_results, EXP_TENSION)

    print(f"\n--- Error Analysis (Judd-Wright vs Experimental) ---")
    print(f"Compression MAE: {np.mean(comp_errors):.1f}%")
    print(f"Tension MAE:     {np.mean(tens_errors):.1f}%")
    print(f"Overall MAE:     {np.mean(np.concatenate([comp_errors, tens_errors])):.1f}%")

    # Key physics checks
    print(f"\n--- Physics Validation ---")

    # 1. Compression knockdown > tension knockdown at same porosity
    comp_5pct = comp_results['judd_wright'][np.argmin(np.abs(porosity_pcts - 5))]
    tens_5pct = tens_results['judd_wright'][np.argmin(np.abs(porosity_pcts - 5))]
    check1 = comp_5pct < tens_5pct
    print(f"Comp KD < Tens KD at 5%: {comp_5pct:.3f} < {tens_5pct:.3f} → {'PASS' if check1 else 'FAIL'}")

    # 2. Concavity: second derivative negative (strength drops faster at higher porosity)
    kd_2 = comp_results['judd_wright'][np.argmin(np.abs(porosity_pcts - 2))]
    kd_5 = comp_results['judd_wright'][np.argmin(np.abs(porosity_pcts - 5))]
    kd_8 = comp_results['judd_wright'][np.argmin(np.abs(porosity_pcts - 8))]
    slope_low = (kd_5 - kd_2) / (5 - 2)
    slope_high = (kd_8 - kd_5) / (8 - 5)
    concave = slope_high < slope_low  # Slopes should become more negative
    print(f"Concavity (compression): slope_low={slope_low:.4f}, slope_high={slope_high:.4f} → {'PASS' if concave else 'FAIL'}")

    # 3. FE knockdown reasonable (within 20% of analytical)
    if len(fe_comp_kd) > 0:
        for i, Vp in enumerate(fe_pcts):
            idx = np.argmin(np.abs(porosity_pcts - Vp))
            jw = comp_results['judd_wright'][idx]
            fe = fe_comp_kd[i]
            diff = abs(fe - jw) / jw * 100
            status = 'PASS' if diff < 20 else 'FAIL'
            print(f"FE vs J-W at {Vp:.0f}%: FE={fe:.3f}, J-W={jw:.3f}, diff={diff:.1f}% → {status}")

    # Generate plot
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_validation(comp_results, tens_results, fe_pcts, fe_comp_kd,
                    fe_pcts, fe_tens_kd,
                    save_path=os.path.join(output_dir, 'validation_elhajjar2025_porosity.png'))

    print(f"\n{'=' * 70}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 70}")
    plt.close('all')


if __name__ == "__main__":
    main()
