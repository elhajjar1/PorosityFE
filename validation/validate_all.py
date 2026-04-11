#!/usr/bin/env python3
"""
Comprehensive Validation: All Papers, All Methods
===================================================

Generates three validation plots showing the current state of the porosity
effects model against experimental data from three published papers:

1. Elhajjar (2025) - T700/#2510, [0/45/90/-45/0]_s - compression + tension
2. Liu (2006)      - T700/TDE85,  [0/90]3s         - tension + modulus
3. Stamopoulos (2016) - HTA/EHkF420, UD            - transverse + shear

Output plots:
  validation_strength_all.png   - Judd-Wright strength predictions
  validation_stiffness_all.png  - CLT stiffness (MT vs DS)
  validation_mae_summary.png    - MAE bar chart showing model performance
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from porosity_fe_analysis import (
    MATERIALS, PorosityField, CompositeMesh, EmpiricalSolver,
    compute_degraded_clt_moduli,
)
import dataclasses

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# EXPERIMENTAL DATA (from digitized paper figures)
# ============================================================

# Elhajjar (2025) - normalized to baseline <2% group mean
ELH_COMP_VP = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
ELH_COMP_KD = np.array([0.88, 0.85, 0.82, 0.78, 0.76, 0.72, 0.68, 0.62, 0.58, 0.55, 0.50])
ELH_TENS_KD = np.array([0.94, 0.92, 0.90, 0.88, 0.86, 0.82, 0.78, 0.75, 0.72, 0.70, 0.67])

# Liu (2006) - normalized to 0.6% baseline
LIU_VP = np.array([0.6, 0.9, 1.0, 1.2, 1.5, 2.0, 2.2, 3.2])
LIU_TENS_STR = np.array([1.00, 0.98, 0.98, 0.96, 0.96, 0.93, 0.92, 0.86])
LIU_TENS_MOD = np.array([1.00, 0.99, 0.99, 0.99, 0.98, 0.97, 0.97, 0.96])
LIU_FLEX_MOD = np.array([1.00, 0.98, 0.97, 0.96, 0.94, 0.90, 0.88, 0.82])

# Stamopoulos (2016) - normalized to Reference (0.82%)
STAM_VP = np.array([0.82, 1.56, 1.62, 3.43])
STAM_TRANS_STR = np.array([1.000, 0.962, 0.957, 0.863])
STAM_TRANS_MOD = np.array([1.000, 0.998, 0.995, 0.991])
STAM_SHEAR_MOD = np.array([1.000, 0.833, 0.822, 0.792])


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def _get_jw(material, Vp_pct, mode, ply_angles):
    Vp = Vp_pct / 100.0
    pf = PorosityField(material, Vp, distribution='uniform',
                       void_shape='spherical')
    mesh = CompositeMesh(pf, material, nx=10, ny=5,
                         nz=material.n_plies, ply_angles=ply_angles)
    emp = EmpiricalSolver(mesh, material, ply_angles=ply_angles)
    return emp.get_failure_load(mode=mode, model='judd_wright')['knockdown']


def run_strength_predictions():
    """Judd-Wright strength predictions for all three papers."""
    out = {}

    # --- Elhajjar: QI layup, compression + tension ---
    mat_e = dataclasses.replace(MATERIALS['T700_epoxy'],
                                n_plies=10, fiber_volume_fraction=0.544)
    angles_e = [0, 45, 90, -45, 0, 0, -45, 90, 45, 0]
    pcts_e = np.logspace(np.log10(0.3), np.log10(12), 40)
    out['elh'] = {
        'pcts': pcts_e,
        'comp': np.array([_get_jw(mat_e, v, 'compression', angles_e) for v in pcts_e]),
        'tens': np.array([_get_jw(mat_e, v, 'tension', angles_e) for v in pcts_e]),
    }

    # --- Liu: [0/90]3s, tension only (renormalized to 0.6% baseline) ---
    mat_l = dataclasses.replace(MATERIALS['T700_epoxy'],
                                n_plies=12, fiber_volume_fraction=0.60)
    angles_l = [0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0]
    pcts_l = np.linspace(0.5, 4.0, 30)
    base_l = _get_jw(mat_l, 0.6, 'tension', angles_l)
    out['liu'] = {
        'pcts': pcts_l,
        'tens': np.array([_get_jw(mat_l, v, 'tension', angles_l) / base_l
                          for v in pcts_l]),
    }

    # --- Stamopoulos: UD (90)16 transverse tension ---
    mat_s = dataclasses.replace(MATERIALS['T700_epoxy'],
                                n_plies=16, fiber_volume_fraction=0.60)
    angles_s = [90] * 16
    pcts_s = np.linspace(0.5, 4.5, 30)
    base_s = _get_jw(mat_s, 0.82, 'tension', angles_s)
    out['stam'] = {
        'pcts': pcts_s,
        'trans': np.array([_get_jw(mat_s, v, 'tension', angles_s) / base_s
                           for v in pcts_s]),
    }

    return out


def run_stiffness_predictions():
    """CLT stiffness predictions (MT vs DS) for Liu and Stamopoulos."""
    out = {}

    # --- Liu: [0/90]3s ---
    mat_l = dataclasses.replace(MATERIALS['T700_epoxy'],
                                n_plies=12, fiber_volume_fraction=0.60)
    angles_l = [0, 90, 0, 90, 0, 90, 90, 0, 90, 0, 90, 0]
    pcts_l = np.linspace(0.5, 4.0, 30)

    out['liu'] = {'pcts': pcts_l}
    for method_key, method in [('mt', 'mori_tanaka'), ('ds', 'differential')]:
        base = compute_degraded_clt_moduli(mat_l, angles_l, 0.006, method=method)
        ex = [compute_degraded_clt_moduli(mat_l, angles_l, v/100.0,
                                          method=method)['Ex'] / base['Ex']
              for v in pcts_l]
        gxy = [compute_degraded_clt_moduli(mat_l, angles_l, v/100.0,
                                           method=method)['Gxy'] / base['Gxy']
               for v in pcts_l]
        out['liu'][f'{method_key}_ex'] = np.array(ex)
        out['liu'][f'{method_key}_gxy'] = np.array(gxy)

    # --- Stamopoulos: UD (90)16 transverse, (0)16 shear ---
    mat_s = dataclasses.replace(MATERIALS['T700_epoxy'],
                                n_plies=16, fiber_volume_fraction=0.60)
    angles_90 = [90] * 16
    angles_0 = [0] * 16
    pcts_s = np.linspace(0.5, 4.5, 30)

    out['stam'] = {'pcts': pcts_s}
    for method_key, method in [('mt', 'mori_tanaka'), ('ds', 'differential')]:
        base_90 = compute_degraded_clt_moduli(mat_s, angles_90, 0.0082, method=method)
        base_0 = compute_degraded_clt_moduli(mat_s, angles_0, 0.0082, method=method)
        trans = [compute_degraded_clt_moduli(mat_s, angles_90, v/100.0,
                                             method=method)['Ex'] / base_90['Ex']
                 for v in pcts_s]
        shear = [compute_degraded_clt_moduli(mat_s, angles_0, v/100.0,
                                             method=method)['Gxy'] / base_0['Gxy']
                 for v in pcts_s]
        out['stam'][f'{method_key}_trans'] = np.array(trans)
        out['stam'][f'{method_key}_shear'] = np.array(shear)

    return out


# ============================================================
# PLOT 1: STRENGTH VALIDATION
# ============================================================

def plot_strength(predictions):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Strength Validation: Judd-Wright Empirical Model",
                 fontsize=14, fontweight='bold')

    # Elhajjar
    ax = axes[0]
    ax.set_title("Elhajjar (2025)\nT700/#2510, [0/45/90/-45/0]$_s$",
                 fontsize=11, fontweight='bold')
    ax.scatter(ELH_COMP_VP, ELH_COMP_KD, c='red', s=45, alpha=0.7,
               marker='o', label='Exp. Compression', zorder=5)
    ax.scatter(ELH_COMP_VP, ELH_TENS_KD, c='blue', s=45, alpha=0.7,
               marker='s', label='Exp. Tension', zorder=5)
    p = predictions['elh']['pcts']
    ax.plot(p, predictions['elh']['comp'], 'r-', linewidth=2,
            label='J-W (Comp.)')
    ax.plot(p, predictions['elh']['tens'], 'b-', linewidth=2,
            label='J-W (Tens.)')
    ax.set_xscale('log')
    ax.set_xlim(0.3, 15)
    ax.set_ylim(0.3, 1.1)
    ax.set_xlabel('Porosity (%, log scale)', fontsize=11)
    ax.set_ylabel('Normalized Strength', fontsize=11)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3, which='both')

    # Liu
    ax = axes[1]
    ax.set_title("Liu (2006)\nT700/TDE85, [0/90]$_{3s}$",
                 fontsize=11, fontweight='bold')
    ax.scatter(LIU_VP, LIU_TENS_STR, c='blue', s=60, alpha=0.7,
               marker='s', label='Exp. Tensile Strength', zorder=5)
    ax.plot(predictions['liu']['pcts'], predictions['liu']['tens'],
            'b-', linewidth=2, label='J-W (Tension)')
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0.80, 1.05)
    ax.set_xlabel('Void Content (%)', fontsize=11)
    ax.set_ylabel('Normalized Strength', fontsize=11)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Stamopoulos
    ax = axes[2]
    ax.set_title("Stamopoulos (2016)\nHTA/EHkF420, (90$^\\circ$)$_{16}$ UD",
                 fontsize=11, fontweight='bold')
    ax.scatter(STAM_VP, STAM_TRANS_STR, c='blue', s=70, alpha=0.7,
               marker='s', label='Exp. Trans. Tensile Str.', zorder=5)
    ax.plot(predictions['stam']['pcts'], predictions['stam']['trans'],
            'b-', linewidth=2, label='J-W (Tension)')
    ax.set_xlim(0, 4.5)
    ax.set_ylim(0.75, 1.08)
    ax.set_xlabel('Void Content (%)', fontsize=11)
    ax.set_ylabel('Normalized Strength', fontsize=11)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'validation_strength_all.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# ============================================================
# PLOT 2: STIFFNESS VALIDATION (MT vs DS)
# ============================================================

def plot_stiffness(predictions):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Stiffness Validation: Mori-Tanaka vs Differential Scheme",
                 fontsize=14, fontweight='bold')

    # Liu panel
    ax = axes[0]
    ax.set_title("Liu (2006)\nT700/TDE85, [0/90]$_{3s}$",
                 fontsize=11, fontweight='bold')
    ax.scatter(LIU_VP, LIU_FLEX_MOD, c='red', s=60, alpha=0.7,
               marker='o', label='Exp. Flex. Modulus', zorder=5)
    ax.scatter(LIU_VP, LIU_TENS_MOD, c='blue', s=60, alpha=0.7,
               marker='s', label='Exp. Tens. Modulus', zorder=5)

    p = predictions['liu']['pcts']
    ax.plot(p, predictions['liu']['mt_ex'], 'b--', linewidth=1.5, alpha=0.7,
            label='MT E$_x$')
    ax.plot(p, predictions['liu']['ds_ex'], 'b-', linewidth=2,
            label='DS E$_x$')
    ax.plot(p, predictions['liu']['mt_gxy'], 'r--', linewidth=1.5, alpha=0.7,
            label='MT G$_{xy}$')
    ax.plot(p, predictions['liu']['ds_gxy'], 'r-', linewidth=2,
            label='DS G$_{xy}$')

    ax.set_xlim(0, 4.0)
    ax.set_ylim(0.75, 1.02)
    ax.set_xlabel('Void Content (%)', fontsize=11)
    ax.set_ylabel('Normalized Modulus', fontsize=11)
    ax.legend(fontsize=8, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)

    # Stamopoulos panel
    ax = axes[1]
    ax.set_title("Stamopoulos (2016)\nHTA/EHkF420, UD",
                 fontsize=11, fontweight='bold')
    ax.scatter(STAM_VP, STAM_TRANS_MOD, c='blue', s=70, alpha=0.7,
               marker='s', label='Exp. Trans. Modulus', zorder=5)
    ax.scatter(STAM_VP, STAM_SHEAR_MOD, c='purple', s=70, alpha=0.7,
               marker='D', label='Exp. Shear Modulus', zorder=5)

    p = predictions['stam']['pcts']
    ax.plot(p, predictions['stam']['mt_trans'], 'b--', linewidth=1.5,
            alpha=0.7, label='MT Trans. (90°)')
    ax.plot(p, predictions['stam']['ds_trans'], 'b-', linewidth=2,
            label='DS Trans. (90°)')
    ax.plot(p, predictions['stam']['mt_shear'], color='purple',
            linestyle='--', linewidth=1.5, alpha=0.7, label='MT G$_{xy}$')
    ax.plot(p, predictions['stam']['ds_shear'], color='purple',
            linestyle='-', linewidth=2, label='DS G$_{xy}$')

    ax.set_xlim(0, 4.5)
    ax.set_ylim(0.70, 1.05)
    ax.set_xlabel('Void Content (%)', fontsize=11)
    ax.set_ylabel('Normalized Modulus', fontsize=11)
    ax.legend(fontsize=8, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'validation_stiffness_all.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()


# ============================================================
# PLOT 3: MAE SUMMARY BAR CHART
# ============================================================

def _mae(pcts, pred, exp_vp, exp_kd):
    errs = []
    for i, vp in enumerate(exp_vp):
        idx = np.argmin(np.abs(pcts - vp))
        errs.append(abs(pred[idx] - exp_kd[i]) / exp_kd[i] * 100)
    return np.mean(errs)


def plot_mae_summary(str_pred, stf_pred):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Compute MAEs
    elh_comp_mae = _mae(str_pred['elh']['pcts'], str_pred['elh']['comp'],
                        ELH_COMP_VP, ELH_COMP_KD)
    elh_tens_mae = _mae(str_pred['elh']['pcts'], str_pred['elh']['tens'],
                        ELH_COMP_VP, ELH_TENS_KD)
    liu_tens_str_mae = _mae(str_pred['liu']['pcts'], str_pred['liu']['tens'],
                            LIU_VP, LIU_TENS_STR)
    stam_trans_str_mae = _mae(str_pred['stam']['pcts'], str_pred['stam']['trans'],
                              STAM_VP, STAM_TRANS_STR)

    liu_tens_mod_ds = _mae(stf_pred['liu']['pcts'], stf_pred['liu']['ds_ex'],
                           LIU_VP, LIU_TENS_MOD)
    liu_flex_mod_ds = _mae(stf_pred['liu']['pcts'], stf_pred['liu']['ds_gxy'],
                           LIU_VP, LIU_FLEX_MOD)
    stam_trans_mod_ds = _mae(stf_pred['stam']['pcts'], stf_pred['stam']['ds_trans'],
                             STAM_VP, STAM_TRANS_MOD)
    stam_shear_mod_ds = _mae(stf_pred['stam']['pcts'], stf_pred['stam']['ds_shear'],
                             STAM_VP, STAM_SHEAR_MOD)

    labels = [
        'Elhajjar\nComp. Strength',
        'Elhajjar\nTens. Strength',
        'Liu\nTens. Strength',
        'Stam.\nTrans. Strength',
        'Liu\nTens. Modulus',
        'Liu\nFlex. Modulus*',
        'Stam.\nTrans. Modulus',
        'Stam.\nShear Modulus*',
    ]
    values = [
        elh_comp_mae, elh_tens_mae, liu_tens_str_mae, stam_trans_str_mae,
        liu_tens_mod_ds, liu_flex_mod_ds, stam_trans_mod_ds, stam_shear_mod_ds,
    ]
    # Color code: green < 5%, yellow 5-10%, red > 10%
    colors = []
    for v in values:
        if v < 5:
            colors.append('#5cb85c')
        elif v < 10:
            colors.append('#f0ad4e')
        else:
            colors.append('#d9534f')

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor='black', linewidth=0.6)

    # Separator between strength and stiffness
    ax.axvline(x=3.5, color='gray', linestyle=':', linewidth=1)
    ax.text(1.5, ax.get_ylim()[1]*0.92 if ax.get_ylim()[1] > 0 else 15,
            'STRENGTH\n(Judd-Wright)', ha='center', fontsize=10,
            fontweight='bold', color='#333')
    ax.text(5.5, ax.get_ylim()[1]*0.92 if ax.get_ylim()[1] > 0 else 15,
            'STIFFNESS\n(CLT + DS)', ha='center', fontsize=10,
            fontweight='bold', color='#333')

    for bar, v in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.2, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('MAE (%)', fontsize=12)
    ax.set_title('Model Validation MAE Across All Properties and Papers',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2 + 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#5cb85c', edgecolor='black', label='MAE < 5% (excellent)'),
        Patch(facecolor='#f0ad4e', edgecolor='black', label='MAE 5-10% (acceptable)'),
        Patch(facecolor='#d9534f', edgecolor='black', label='MAE > 10% (poor)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Footnote
    ax.text(0.5, -0.18,
            '* Flex. Modulus and Shear Modulus gaps are fundamental limits of '
            'continuum homogenization.\nThey require mesoscale damage modeling '
            '(interface delamination, void clustering) not captured by MT/DS.',
            transform=ax.transAxes, ha='center', fontsize=8, style='italic',
            color='#555')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'validation_mae_summary.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close()

    # Return MAEs for printing
    return {
        'elh_comp': elh_comp_mae, 'elh_tens': elh_tens_mae,
        'liu_tens_str': liu_tens_str_mae, 'stam_trans_str': stam_trans_str_mae,
        'liu_tens_mod': liu_tens_mod_ds, 'liu_flex_mod': liu_flex_mod_ds,
        'stam_trans_mod': stam_trans_mod_ds, 'stam_shear_mod': stam_shear_mod_ds,
    }


def main():
    print("=" * 70)
    print("COMPREHENSIVE VALIDATION: All Papers, All Methods")
    print("=" * 70)

    print("\n[1/3] Running strength predictions (Judd-Wright)...")
    str_pred = run_strength_predictions()

    print("[2/3] Running stiffness predictions (CLT MT vs DS)...")
    stf_pred = run_stiffness_predictions()

    print("\n[3/3] Generating plots...")
    plot_strength(str_pred)
    plot_stiffness(stf_pred)
    maes = plot_mae_summary(str_pred, stf_pred)

    print(f"\n{'=' * 70}")
    print("MAE SUMMARY")
    print("=" * 70)
    print("\nStrength (Judd-Wright):")
    print(f"  Elhajjar Compression:    {maes['elh_comp']:5.2f}%")
    print(f"  Elhajjar Tension:        {maes['elh_tens']:5.2f}%")
    print(f"  Liu Tensile Strength:    {maes['liu_tens_str']:5.2f}%")
    print(f"  Stam. Trans. Tens. Str.: {maes['stam_trans_str']:5.2f}%")

    print("\nStiffness (CLT + Differential Scheme):")
    print(f"  Liu Tensile Modulus:     {maes['liu_tens_mod']:5.2f}%")
    print(f"  Liu Flexural Modulus:    {maes['liu_flex_mod']:5.2f}% (mesoscale limit)")
    print(f"  Stam. Trans. Modulus:    {maes['stam_trans_mod']:5.2f}%")
    print(f"  Stam. Shear Modulus:     {maes['stam_shear_mod']:5.2f}% (mesoscale limit)")

    print(f"\n{'=' * 70}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
