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
    compute_degraded_clt_flexural_modulus,
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
    """Run degraded CLT predictions for all datasets.

    Computes both A-matrix (Ex, Ey, Gxy) and D-matrix (Ef_x) predictions
    to compare membrane vs bending stiffness responses to porosity.
    """
    results = {}

    # --- Liu: [0/90]3s ---
    mat_liu = dataclasses.replace(MATERIALS['T700_epoxy'],
                                  n_plies=12, fiber_volume_fraction=0.60)
    angles_liu = [0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0]

    pcts_liu = np.linspace(0.5, 4.0, 30)
    base_a_liu = compute_degraded_clt_moduli(mat_liu, angles_liu, 0.006)
    base_d_liu = compute_degraded_clt_flexural_modulus(mat_liu, angles_liu, 0.006)

    liu_ex, liu_gxy, liu_efx = [], [], []
    for vp_pct in pcts_liu:
        a = compute_degraded_clt_moduli(mat_liu, angles_liu, vp_pct / 100.0)
        d = compute_degraded_clt_flexural_modulus(mat_liu, angles_liu, vp_pct / 100.0)
        liu_ex.append(a['Ex'] / base_a_liu['Ex'])
        liu_gxy.append(a['Gxy'] / base_a_liu['Gxy'])
        liu_efx.append(d['Ef_x'] / base_d_liu['Ef_x'])

    results['liu'] = {
        'pcts': pcts_liu,
        'kd_Ex_membrane': np.array(liu_ex),
        'kd_Gxy': np.array(liu_gxy),
        'kd_Ef_bending': np.array(liu_efx),
    }

    # --- Stamopoulos: UD (90)16 for transverse, (0)16 for shear/flex ---
    mat_stam = dataclasses.replace(MATERIALS['T700_epoxy'],
                                    n_plies=16, fiber_volume_fraction=0.60)
    angles_90 = [90] * 16
    angles_0 = [0] * 16

    pcts_stam = np.linspace(0.5, 4.5, 30)
    base_a_trans = compute_degraded_clt_moduli(mat_stam, angles_90, 0.0082)
    base_a_shear = compute_degraded_clt_moduli(mat_stam, angles_0, 0.0082)
    base_d_flex = compute_degraded_clt_flexural_modulus(mat_stam, angles_0, 0.0082)

    stam_trans, stam_shear, stam_flex_bend = [], [], []
    for vp_pct in pcts_stam:
        a_90 = compute_degraded_clt_moduli(mat_stam, angles_90, vp_pct / 100.0)
        a_0 = compute_degraded_clt_moduli(mat_stam, angles_0, vp_pct / 100.0)
        d_0 = compute_degraded_clt_flexural_modulus(mat_stam, angles_0, vp_pct / 100.0)
        # Trans modulus = Ex of (90)16 layup (A-matrix, fiber transverse dir)
        stam_trans.append(a_90['Ex'] / base_a_trans['Ex'])
        # Shear modulus = Gxy of (0)16 layup (A-matrix, in-plane shear)
        stam_shear.append(a_0['Gxy'] / base_a_shear['Gxy'])
        # Flexural modulus = Ef_x from D-matrix of (0)16 layup (bending)
        stam_flex_bend.append(d_0['Ef_x'] / base_d_flex['Ef_x'])

    results['stam'] = {
        'pcts': pcts_stam,
        'kd_trans': np.array(stam_trans),
        'kd_shear': np.array(stam_shear),
        'kd_flex_bending': np.array(stam_flex_bend),
    }

    return results


def plot_validation(results, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    fig.suptitle("Degraded CLT Stiffness Knockdown vs Experimental Data\n"
                 "(Mori-Tanaka Degraded Plies + CLT A/D-Matrix)",
                 fontsize=13, fontweight='bold')

    # --- Panel 1: Liu [0/90]3s modulus ---
    ax = axes[0]
    ax.set_title("Liu (2006)\nT700/TDE85, [0/90]$_{3s}$", fontsize=11, fontweight='bold')
    ax.scatter(LIU_VP, LIU_FLEX_MOD, c='red', s=60, alpha=0.7,
               marker='o', label='Exp. Flex. Modulus (3-pt bend)', zorder=5)
    ax.scatter(LIU_VP, LIU_TENS_MOD, c='blue', s=60, alpha=0.7,
               marker='s', label='Exp. Tens. Modulus', zorder=5)

    p = results['liu']['pcts']
    ax.plot(p, results['liu']['kd_Ex_membrane'], 'b-', linewidth=2,
            label='CLT E$_x$ (A-matrix, tension)')
    ax.plot(p, results['liu']['kd_Ef_bending'], 'r-', linewidth=2,
            label='CLT E$_f$ (D-matrix, bending)')

    ax.set_xlabel('Void Content (%)', fontsize=11)
    ax.set_ylabel('Normalized Modulus', fontsize=11)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0.75, 1.02)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Annotation: CLT cannot capture flexural drop
    ax.annotate('CLT flex. prediction (D-matrix)\n'
                '~flat — M-T cannot explain\nflex. modulus drop',
                xy=(2.5, 0.85), fontsize=7, fontstyle='italic',
                color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    # --- Panel 2: Stamopoulos UD stiffness ---
    ax = axes[1]
    ax.set_title("Stamopoulos (2016)\nHTA/EHkF420, UD", fontsize=11, fontweight='bold')
    ax.scatter(STAM_VP, STAM_TRANS_MOD, c='blue', s=70, alpha=0.7,
               marker='s', label='Exp. Trans. Modulus E$_{22}$', zorder=5)
    ax.scatter(STAM_VP, STAM_SHEAR_MOD, c='purple', s=70, alpha=0.7,
               marker='D', label='Exp. Shear Modulus G$_{12}$', zorder=5)
    ax.scatter(STAM_VP, STAM_FLEX_MOD, c='red', s=70, alpha=0.7,
               marker='o', label='Exp. Flex. Modulus (3-pt bend)', zorder=5)

    p = results['stam']['pcts']
    ax.plot(p, results['stam']['kd_trans'], 'b-', linewidth=2,
            label='CLT E$_x$ of (90°)$_{16}$')
    ax.plot(p, results['stam']['kd_shear'], color='purple', linestyle='-',
            linewidth=2, label='CLT G$_{xy}$ of (0°)$_{16}$')
    ax.plot(p, results['stam']['kd_flex_bending'], 'r-', linewidth=2,
            label='CLT E$_f$ of (0°)$_{16}$ (D-mat)')

    ax.set_xlabel('Void Content (%)', fontsize=11)
    ax.set_ylabel('Normalized Modulus', fontsize=11)
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0.70, 1.05)
    ax.legend(fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Annotation: CLT cannot capture shear drop
    ax.annotate('M-T underpredicts\nG$_{12}$ drop by ~7×',
                xy=(2.8, 0.85), fontsize=7, fontstyle='italic',
                color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

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
    print(f"{'Vp%':>6s} {'Flex Exp':>10s} {'CLT Ef(D)':>10s} "
          f"{'Tens Exp':>10s} {'CLT Ex(A)':>10s}")
    for i, vp in enumerate(LIU_VP):
        idx = np.argmin(np.abs(results['liu']['pcts'] - vp))
        print(f"{vp:6.1f} {LIU_FLEX_MOD[i]:10.3f} {results['liu']['kd_Ef_bending'][idx]:10.3f} "
              f"{LIU_TENS_MOD[i]:10.3f} {results['liu']['kd_Ex_membrane'][idx]:10.3f}")

    print("\n--- Stamopoulos (2016): CLT vs Experimental Modulus ---")
    print(f"{'Vp%':>6s} {'Trans Exp':>10s} {'CLT Trans':>10s} "
          f"{'Shear Exp':>10s} {'CLT Shear':>10s} "
          f"{'Flex Exp':>10s} {'CLT Flex':>10s}")
    for i, vp in enumerate(STAM_VP):
        idx = np.argmin(np.abs(results['stam']['pcts'] - vp))
        print(f"{vp:6.2f} {STAM_TRANS_MOD[i]:10.3f} {results['stam']['kd_trans'][idx]:10.3f} "
              f"{STAM_SHEAR_MOD[i]:10.3f} {results['stam']['kd_shear'][idx]:10.3f} "
              f"{STAM_FLEX_MOD[i]:10.3f} {results['stam']['kd_flex_bending'][idx]:10.3f}")

    # MAE
    liu_flex_err = []
    liu_tens_err = []
    for i, vp in enumerate(LIU_VP):
        idx = np.argmin(np.abs(results['liu']['pcts'] - vp))
        liu_flex_err.append(abs(results['liu']['kd_Ef_bending'][idx] - LIU_FLEX_MOD[i]) / LIU_FLEX_MOD[i] * 100)
        liu_tens_err.append(abs(results['liu']['kd_Ex_membrane'][idx] - LIU_TENS_MOD[i]) / LIU_TENS_MOD[i] * 100)

    stam_trans_err = []
    stam_shear_err = []
    stam_flex_err = []
    for i, vp in enumerate(STAM_VP):
        idx = np.argmin(np.abs(results['stam']['pcts'] - vp))
        stam_trans_err.append(abs(results['stam']['kd_trans'][idx] - STAM_TRANS_MOD[i]) / STAM_TRANS_MOD[i] * 100)
        stam_shear_err.append(abs(results['stam']['kd_shear'][idx] - STAM_SHEAR_MOD[i]) / STAM_SHEAR_MOD[i] * 100)
        stam_flex_err.append(abs(results['stam']['kd_flex_bending'][idx] - STAM_FLEX_MOD[i]) / STAM_FLEX_MOD[i] * 100)

    print(f"\n--- Error Analysis ---")
    print(f"Liu Tens. Modulus (A-matrix)    MAE: {np.mean(liu_tens_err):.1f}%")
    print(f"Liu Flex. Modulus (D-matrix)    MAE: {np.mean(liu_flex_err):.1f}%")
    print(f"Stam. Trans. Modulus (A-matrix) MAE: {np.mean(stam_trans_err):.1f}%")
    print(f"Stam. Shear Modulus (A-matrix)  MAE: {np.mean(stam_shear_err):.1f}%")
    print(f"Stam. Flex. Modulus (D-matrix)  MAE: {np.mean(stam_flex_err):.1f}%")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_validation(results, save_path=os.path.join(
        output_dir, 'validation_clt_stiffness.png'))

    print(f"\n{'=' * 70}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 70}")
    plt.close('all')


if __name__ == "__main__":
    main()
