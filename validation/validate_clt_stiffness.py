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
    """Run degraded CLT predictions comparing MT vs DS homogenization.

    Returns results for both Mori-Tanaka (default) and Differential Scheme
    methods to show the effect of void-void interactions.
    """
    results = {'mt': {}, 'ds': {}}

    for method in ['mori_tanaka', 'differential']:
        key = 'mt' if method == 'mori_tanaka' else 'ds'

        # --- Liu: [0/90]3s ---
        mat_liu = dataclasses.replace(MATERIALS['T700_epoxy'],
                                       n_plies=12, fiber_volume_fraction=0.60)
        angles_liu = [0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0]

        pcts_liu = np.linspace(0.5, 4.0, 30)
        base_a_liu = compute_degraded_clt_moduli(mat_liu, angles_liu, 0.006,
                                                  method=method)

        liu_ex, liu_gxy = [], []
        for vp_pct in pcts_liu:
            a = compute_degraded_clt_moduli(mat_liu, angles_liu, vp_pct/100.0,
                                             method=method)
            liu_ex.append(a['Ex'] / base_a_liu['Ex'])
            liu_gxy.append(a['Gxy'] / base_a_liu['Gxy'])

        results[key]['liu'] = {
            'pcts': pcts_liu,
            'kd_Ex': np.array(liu_ex),
            'kd_Gxy': np.array(liu_gxy),
        }

        # --- Stamopoulos: UD (90)16 for transverse, (0)16 for shear ---
        mat_stam = dataclasses.replace(MATERIALS['T700_epoxy'],
                                        n_plies=16, fiber_volume_fraction=0.60)
        angles_90 = [90] * 16
        angles_0 = [0] * 16

        pcts_stam = np.linspace(0.5, 4.5, 30)
        base_trans = compute_degraded_clt_moduli(mat_stam, angles_90, 0.0082,
                                                  method=method)
        base_shear = compute_degraded_clt_moduli(mat_stam, angles_0, 0.0082,
                                                  method=method)

        stam_trans, stam_shear = [], []
        for vp_pct in pcts_stam:
            a_90 = compute_degraded_clt_moduli(mat_stam, angles_90, vp_pct/100.0,
                                                method=method)
            a_0 = compute_degraded_clt_moduli(mat_stam, angles_0, vp_pct/100.0,
                                               method=method)
            stam_trans.append(a_90['Ex'] / base_trans['Ex'])
            stam_shear.append(a_0['Gxy'] / base_shear['Gxy'])

        results[key]['stam'] = {
            'pcts': pcts_stam,
            'kd_trans': np.array(stam_trans),
            'kd_shear': np.array(stam_shear),
        }

    return results


def plot_validation(results, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    fig.suptitle("CLT Stiffness Knockdown: Mori-Tanaka vs Differential Scheme\n"
                 "vs Experimental Data",
                 fontsize=13, fontweight='bold')

    # --- Panel 1: Liu [0/90]3s modulus ---
    ax = axes[0]
    ax.set_title("Liu (2006)\nT700/TDE85, [0/90]$_{3s}$",
                 fontsize=11, fontweight='bold')
    ax.scatter(LIU_VP, LIU_FLEX_MOD, c='red', s=60, alpha=0.7,
               marker='o', label='Exp. Flex. Modulus', zorder=5)
    ax.scatter(LIU_VP, LIU_TENS_MOD, c='blue', s=60, alpha=0.7,
               marker='s', label='Exp. Tens. Modulus', zorder=5)

    p = results['mt']['liu']['pcts']
    # MT curves (dashed)
    ax.plot(p, results['mt']['liu']['kd_Ex'], 'b--', linewidth=1.5,
            alpha=0.7, label='MT E$_x$')
    ax.plot(p, results['mt']['liu']['kd_Gxy'], 'r--', linewidth=1.5,
            alpha=0.7, label='MT G$_{xy}$')
    # DS curves (solid)
    ax.plot(p, results['ds']['liu']['kd_Ex'], 'b-', linewidth=2,
            label='DS E$_x$')
    ax.plot(p, results['ds']['liu']['kd_Gxy'], 'r-', linewidth=2,
            label='DS G$_{xy}$')

    ax.set_xlabel('Void Content (%)', fontsize=11)
    ax.set_ylabel('Normalized Modulus', fontsize=11)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0.75, 1.02)
    ax.legend(fontsize=8, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)

    ax.annotate('Both MT and DS\nfar below exp. flex drop',
                xy=(2.0, 0.80), fontsize=7, fontstyle='italic',
                color='darkred',
                bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    # --- Panel 2: Stamopoulos UD ---
    ax = axes[1]
    ax.set_title("Stamopoulos (2016)\nHTA/EHkF420, UD",
                 fontsize=11, fontweight='bold')
    ax.scatter(STAM_VP, STAM_TRANS_MOD, c='blue', s=70, alpha=0.7,
               marker='s', label='Exp. Trans. E$_{22}$', zorder=5)
    ax.scatter(STAM_VP, STAM_SHEAR_MOD, c='purple', s=70, alpha=0.7,
               marker='D', label='Exp. Shear G$_{12}$', zorder=5)

    p = results['mt']['stam']['pcts']
    ax.plot(p, results['mt']['stam']['kd_trans'], 'b--', linewidth=1.5,
            alpha=0.7, label='MT Trans. (90°)')
    ax.plot(p, results['mt']['stam']['kd_shear'], color='purple',
            linestyle='--', linewidth=1.5, alpha=0.7, label='MT G$_{xy}$ (0°)')
    ax.plot(p, results['ds']['stam']['kd_trans'], 'b-', linewidth=2,
            label='DS Trans. (90°)')
    ax.plot(p, results['ds']['stam']['kd_shear'], color='purple',
            linestyle='-', linewidth=2, label='DS G$_{xy}$ (0°)')

    ax.set_xlabel('Void Content (%)', fontsize=11)
    ax.set_ylabel('Normalized Modulus', fontsize=11)
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0.70, 1.05)
    ax.legend(fontsize=7, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)

    ax.annotate('Both MT and DS fail\nto capture 20% G$_{12}$ drop',
                xy=(2.5, 0.82), fontsize=7, fontstyle='italic',
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

    def _mae(pcts, pred, exp_vp, exp_kd):
        errs = []
        for i, vp in enumerate(exp_vp):
            idx = np.argmin(np.abs(pcts - vp))
            errs.append(abs(pred[idx] - exp_kd[i]) / exp_kd[i] * 100)
        return np.mean(errs)

    print("\n--- Liu (2006) [0/90]3s: MT vs DS Error Analysis ---")
    for method_key, method_name in [('mt', 'Mori-Tanaka'), ('ds', 'Differential Scheme')]:
        r = results[method_key]['liu']
        tens_mae = _mae(r['pcts'], r['kd_Ex'], LIU_VP, LIU_TENS_MOD)
        gxy_mae = _mae(r['pcts'], r['kd_Gxy'], LIU_VP, LIU_FLEX_MOD)
        print(f"  {method_name:>20s}: Tens MAE={tens_mae:5.2f}%, Gxy-vs-Flex MAE={gxy_mae:5.2f}%")

    print("\n--- Stamopoulos (2016) UD: MT vs DS Error Analysis ---")
    for method_key, method_name in [('mt', 'Mori-Tanaka'), ('ds', 'Differential Scheme')]:
        r = results[method_key]['stam']
        trans_mae = _mae(r['pcts'], r['kd_trans'], STAM_VP, STAM_TRANS_MOD)
        shear_mae = _mae(r['pcts'], r['kd_shear'], STAM_VP, STAM_SHEAR_MOD)
        print(f"  {method_name:>20s}: Trans MAE={trans_mae:5.2f}%, Shear MAE={shear_mae:5.2f}%")

    # At 3% Vp comparison
    print("\n--- MT vs DS Comparison at Vp=3% (Liu layup) ---")
    p = results['mt']['liu']['pcts']
    idx = np.argmin(np.abs(p - 3.0))
    mt_gxy = results['mt']['liu']['kd_Gxy'][idx]
    ds_gxy = results['ds']['liu']['kd_Gxy'][idx]
    print(f"  MT Gxy knockdown: {mt_gxy:.4f} ({(1-mt_gxy)*100:.1f}% drop)")
    print(f"  DS Gxy knockdown: {ds_gxy:.4f} ({(1-ds_gxy)*100:.1f}% drop)")
    print(f"  Experimental:     0.8200 ({(1-0.82)*100:.1f}% drop)")
    print(f"  DS improvement over MT: {(mt_gxy - ds_gxy)*100:.2f} percentage points")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_validation(results, save_path=os.path.join(
        output_dir, 'validation_clt_stiffness.png'))

    print(f"\n{'=' * 70}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 70}")
    plt.close('all')


if __name__ == "__main__":
    main()
