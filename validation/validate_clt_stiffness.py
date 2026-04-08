#!/usr/bin/env python3
"""
Validation: Degraded CLT Stiffness Knockdown vs Experimental Data
==================================================================

Tests the new compute_degraded_clt_moduli() function against experimental
modulus data from Liu (2006) and Stamopoulos (2016).

This replaces the old FE stress-ratio approach which was fiber-dominated
and predicted <1% knockdown. The CLT approach degrades each ply via
Mori-Tanaka, then integrates through the layup to get laminate moduli.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from porosity_fe_analysis import (
    MATERIALS, compute_degraded_clt_moduli,
)
import dataclasses


# ============================================================
# EXPERIMENTAL DATA
# ============================================================

# Liu (2006): T700/TDE85, [0/90]3s, normalized to 0.6% baseline
LIU_VP = np.array([0.6, 0.9, 1.0, 1.2, 1.5, 2.0, 2.2, 3.2])
LIU_FLEX_MOD = np.array([1.00, 0.98, 0.97, 0.96, 0.94, 0.90, 0.88, 0.82])
LIU_TENS_MOD = np.array([1.00, 0.99, 0.99, 0.99, 0.98, 0.97, 0.97, 0.96])
LIU_TENS_STR = np.array([1.00, 0.98, 0.98, 0.96, 0.96, 0.93, 0.92, 0.86])

# Stamopoulos (2016): HTA/EHkF420, UD, normalized to Reference (0.82%)
STAM_VP = np.array([0.82, 1.56, 1.62, 3.43])
STAM_TRANS_MOD = np.array([1.000, 0.998, 0.995, 0.991])   # E22 on (90)16
STAM_SHEAR_MOD = np.array([1.000, 0.833, 0.822, 0.792])   # G12 on (0)16
STAM_FLEX_MOD = np.array([1.000, 0.984, 0.967, 0.959])     # Ef on (0)16

# Elhajjar (2025): T700/#2510, [0/45/90/-45/0]_s — strength only (no modulus data)
ELH_COMP_VP = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
ELH_COMP_KD = np.array([0.88, 0.85, 0.82, 0.78, 0.76, 0.72, 0.68, 0.62, 0.58, 0.55, 0.50])
ELH_TENS_KD = np.array([0.94, 0.92, 0.90, 0.88, 0.86, 0.82, 0.78, 0.75, 0.72, 0.70, 0.67])


def run_clt_predictions():
    """Run degraded CLT predictions for all datasets."""
    results = {}

    # --- Liu: [0/90]3s ---
    mat_liu = dataclasses.replace(MATERIALS['T700_epoxy'],
                                  n_plies=12, fiber_volume_fraction=0.60)
    angles_liu = [0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0]

    pcts_liu = np.linspace(0.5, 4.0, 30)
    baseline_liu = compute_degraded_clt_moduli(mat_liu, angles_liu, 0.006)

    liu_ex, liu_ey, liu_gxy = [], [], []
    for vp_pct in pcts_liu:
        kd = compute_degraded_clt_moduli(mat_liu, angles_liu, vp_pct / 100.0)
        # Renormalize to 0.6% baseline
        liu_ex.append(kd['Ex'] / baseline_liu['Ex'])
        liu_ey.append(kd['Ey'] / baseline_liu['Ey'])
        liu_gxy.append(kd['Gxy'] / baseline_liu['Gxy'])

    results['liu'] = {
        'pcts': pcts_liu,
        'kd_Ex': np.array(liu_ex),
        'kd_Ey': np.array(liu_ey),
        'kd_Gxy': np.array(liu_gxy),
    }

    # --- Stamopoulos: UD (90)16 for transverse, (0)16 for shear/flex ---
    mat_stam = dataclasses.replace(MATERIALS['T700_epoxy'],
                                    n_plies=16, fiber_volume_fraction=0.60)
    angles_90 = [90] * 16
    angles_0 = [0] * 16

    pcts_stam = np.linspace(0.5, 4.5, 30)
    baseline_trans = compute_degraded_clt_moduli(mat_stam, angles_90, 0.0082)
    baseline_shear = compute_degraded_clt_moduli(mat_stam, angles_0, 0.0082)

    stam_trans, stam_shear, stam_flex_ex = [], [], []
    for vp_pct in pcts_stam:
        kd_90 = compute_degraded_clt_moduli(mat_stam, angles_90, vp_pct / 100.0)
        kd_0 = compute_degraded_clt_moduli(mat_stam, angles_0, vp_pct / 100.0)
        # Trans modulus = Ex of (90)16 layup (loading in fiber transverse dir)
        stam_trans.append(kd_90['Ex'] / baseline_trans['Ex'])
        # Shear modulus = Gxy of (0)16 layup
        stam_shear.append(kd_0['Gxy'] / baseline_shear['Gxy'])
        # Flexural ~ Ex of (0)16 (bending in fiber direction)
        stam_flex_ex.append(kd_0['Ex'] / baseline_shear['Ex'])

    results['stam'] = {
        'pcts': pcts_stam,
        'kd_trans': np.array(stam_trans),
        'kd_shear': np.array(stam_shear),
        'kd_flex': np.array(stam_flex_ex),
    }

    return results


def plot_validation(results, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    fig.suptitle("Degraded CLT Stiffness Knockdown vs Experimental Data\n"
                 "(Mori-Tanaka Degraded Plies + CLT A-Matrix)",
                 fontsize=13, fontweight='bold')

    # --- Panel 1: Liu [0/90]3s modulus ---
    ax = axes[0]
    ax.set_title("Liu (2006)\nT700/TDE85, [0/90]$_{3s}$", fontsize=10, fontweight='bold')
    ax.scatter(LIU_VP, LIU_FLEX_MOD, c='red', s=60, alpha=0.7,
               marker='o', label='Exp. Flex. Modulus', zorder=5)
    ax.scatter(LIU_VP, LIU_TENS_MOD, c='blue', s=60, alpha=0.7,
               marker='s', label='Exp. Tens. Modulus', zorder=5)

    p = results['liu']['pcts']
    ax.plot(p, results['liu']['kd_Ex'], 'b-', linewidth=2, label='CLT E$_x$ (Tens.)')
    ax.plot(p, results['liu']['kd_Gxy'], 'g--', linewidth=2, label='CLT G$_{xy}$ (Shear)')

    ax.set_xlabel('Void Content (%)', fontsize=10)
    ax.set_ylabel('Normalized Modulus', fontsize=10)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0.75, 1.02)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Stamopoulos UD stiffness ---
    ax = axes[1]
    ax.set_title("Stamopoulos (2016)\nHTA/EHkF420, UD", fontsize=10, fontweight='bold')
    ax.scatter(STAM_VP, STAM_TRANS_MOD, c='blue', s=70, alpha=0.7,
               marker='s', label='Exp. Trans. Modulus', zorder=5)
    ax.scatter(STAM_VP, STAM_SHEAR_MOD, c='purple', s=70, alpha=0.7,
               marker='D', label='Exp. Shear Modulus', zorder=5)
    ax.scatter(STAM_VP, STAM_FLEX_MOD, c='red', s=70, alpha=0.7,
               marker='o', label='Exp. Flex. Modulus', zorder=5)

    p = results['stam']['pcts']
    ax.plot(p, results['stam']['kd_trans'], 'b-', linewidth=2, label='CLT Trans.')
    ax.plot(p, results['stam']['kd_shear'], color='purple', linestyle='-',
            linewidth=2, label='CLT Shear (G$_{xy}$)')
    ax.plot(p, results['stam']['kd_flex'], 'r-', linewidth=2, label='CLT Flex. (E$_x$)')

    ax.set_xlabel('Void Content (%)', fontsize=10)
    ax.set_ylabel('Normalized Modulus', fontsize=10)
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0.70, 1.05)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Summary table ---
    ax = axes[2]
    ax.set_title("CLT Knockdown Summary\nat 3% Porosity", fontsize=10, fontweight='bold')
    ax.axis('off')

    mat = MATERIALS['T700_epoxy']
    configs = [
        ('QI [0/45/90/-45/0]s', [0, 45, 90, -45, 0, 0, -45, 90, 45, 0]),
        ('[0/90]3s', [0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0]),
        ('UD (0)16', [0]*16),
        ('UD (90)16', [90]*16),
    ]
    table_data = [['Layup', 'kd_Ex', 'kd_Ey', 'kd_Gxy']]
    for label, angles in configs:
        m = dataclasses.replace(mat, n_plies=len(angles))
        kd = compute_degraded_clt_moduli(m, angles, 0.03)
        table_data.append([
            label,
            f"{kd['knockdown_Ex']:.4f}",
            f"{kd['knockdown_Ey']:.4f}",
            f"{kd['knockdown_Gxy']:.4f}",
        ])

    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def main():
    print("=" * 70)
    print("VALIDATION: Degraded CLT Stiffness Knockdown")
    print("=" * 70)

    print("\nRunning CLT predictions...")
    results = run_clt_predictions()

    # Error analysis: Liu
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print("=" * 70)

    print("\n--- Liu (2006): CLT vs Experimental Modulus ---")
    print(f"{'Vp%':>6s} {'Flex Exp':>10s} {'CLT Gxy':>10s} {'Tens Exp':>10s} {'CLT Ex':>10s}")
    for i, vp in enumerate(LIU_VP):
        idx = np.argmin(np.abs(results['liu']['pcts'] - vp))
        print(f"{vp:6.1f} {LIU_FLEX_MOD[i]:10.3f} {results['liu']['kd_Gxy'][idx]:10.3f} "
              f"{LIU_TENS_MOD[i]:10.3f} {results['liu']['kd_Ex'][idx]:10.3f}")

    print("\n--- Stamopoulos (2016): CLT vs Experimental Modulus ---")
    print(f"{'Vp%':>6s} {'Trans Exp':>10s} {'CLT Trans':>10s} "
          f"{'Shear Exp':>10s} {'CLT Shear':>10s} "
          f"{'Flex Exp':>10s} {'CLT Flex':>10s}")
    for i, vp in enumerate(STAM_VP):
        idx = np.argmin(np.abs(results['stam']['pcts'] - vp))
        print(f"{vp:6.2f} {STAM_TRANS_MOD[i]:10.3f} {results['stam']['kd_trans'][idx]:10.3f} "
              f"{STAM_SHEAR_MOD[i]:10.3f} {results['stam']['kd_shear'][idx]:10.3f} "
              f"{STAM_FLEX_MOD[i]:10.3f} {results['stam']['kd_flex'][idx]:10.3f}")

    # MAE
    liu_flex_err = []
    liu_tens_err = []
    for i, vp in enumerate(LIU_VP):
        idx = np.argmin(np.abs(results['liu']['pcts'] - vp))
        liu_flex_err.append(abs(results['liu']['kd_Gxy'][idx] - LIU_FLEX_MOD[i]) / LIU_FLEX_MOD[i] * 100)
        liu_tens_err.append(abs(results['liu']['kd_Ex'][idx] - LIU_TENS_MOD[i]) / LIU_TENS_MOD[i] * 100)

    stam_trans_err = []
    stam_shear_err = []
    for i, vp in enumerate(STAM_VP):
        idx = np.argmin(np.abs(results['stam']['pcts'] - vp))
        stam_trans_err.append(abs(results['stam']['kd_trans'][idx] - STAM_TRANS_MOD[i]) / STAM_TRANS_MOD[i] * 100)
        stam_shear_err.append(abs(results['stam']['kd_shear'][idx] - STAM_SHEAR_MOD[i]) / STAM_SHEAR_MOD[i] * 100)

    print(f"\n--- Error Analysis ---")
    print(f"Liu Flex. Modulus MAE:        {np.mean(liu_flex_err):.1f}%")
    print(f"Liu Tens. Modulus MAE:        {np.mean(liu_tens_err):.1f}%")
    print(f"Stam. Trans. Modulus MAE:     {np.mean(stam_trans_err):.1f}%")
    print(f"Stam. Shear Modulus MAE:      {np.mean(stam_shear_err):.1f}%")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_validation(results, save_path=os.path.join(
        output_dir, 'validation_clt_stiffness.png'))

    print(f"\n{'=' * 70}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 70}")
    plt.close('all')


if __name__ == "__main__":
    main()
