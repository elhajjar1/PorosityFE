"""The figures: section geometry, colour scales, profile, mesh, results, stress, UQ, fronts.

Split out of ``app.py`` (#224); ``app`` re-exports every name.
"""

from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from porosity_fe import (
    LABEL_KNOCKDOWN,
    LABEL_POROSITY_PCT,
    LABEL_STIFFNESS_RETENTION,
    LABEL_X_MM,
    LABEL_Z_MM,
)

from .config import _FLEXURE_MODES, _FRACTURE_MODES
from .summaries import _crack_front, _distribution_text, front_order


def plot_fracture_uq(result: dict, compare: float | None = None):
    """Histogram of the λ_c draws with the p5-p95 band, the median, the
    nominal value and, with ``compare``, the entered factor."""
    band = result.get("fe_fracture_uq")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    finite = (np.asarray(band.samples)[np.isfinite(band.samples)]
              if band is not None else np.empty(0))
    if finite.size == 0:
        ax.text(0.5, 0.5, "No finite fracture load factor in the draws.",
                transform=ax.transAxes, ha="center", va="center", color="0.5")
        ax.set_axis_off()
        return fig
    pct = band.percentiles
    ax.hist(finite, bins=40, color="0.7", edgecolor="0.4")
    if math.isfinite(pct["p5"]) and math.isfinite(pct["p95"]):
        ax.axvspan(pct["p5"], pct["p95"], color="tab:blue", alpha=0.15,
                   label=f"p5-p95: {pct['p5']:.3g}-{pct['p95']:.3g}")
    if math.isfinite(pct["p50"]):
        ax.axvline(pct["p50"], color="tab:blue", linewidth=2,
                   label=f"p50: {pct['p50']:.3g}")
    if math.isfinite(band.nominal):
        ax.axvline(band.nominal, color="k", linestyle="--",
                   label=f"nominal: {band.nominal:.3g}")
    if compare is not None:
        ax.axvline(compare, color="tab:red", linewidth=1.5,
                   label=f"{compare:g}: {100.0 * band.probability_below(compare):.1f} % below")
    ax.set_xlabel("Fracture load factor λ_c")
    ax.set_ylabel("Draws")
    _set_titles(ax, "Fracture load factor: toughness scatter",
                f"{band.n_samples} draws · CoV "
                + ", ".join(f"{k} {v[1]:g}" for k, v in band.spec.items()))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              frameon=False)
    fig.tight_layout()
    return fig


#: Line colours of the front plot: G_T, ψ.
_FRONT_COLOURS = ("#1f77b4", "#d62728")


def plot_front(result: dict):
    """``G_T`` and ``ψ`` along the delamination front.

    ``G_T`` is at the analysis load, so it is usually far below the
    toughness; the table holds ``G_c`` and ``λ_c`` at every node, and the
    governing node is starred.
    """
    front = _crack_front(result)
    fig, (ax_g, ax_p) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    if front is None:
        ax_g.text(0.5, 0.5, "No delamination front.", transform=ax_g.transAxes,
                  ha="center", va="center", color="0.5")
        return fig
    segments, s, label = front_order(front)
    for n, seg in enumerate(segments):
        tag = "" if len(segments) == 1 else f" (x = {front.x[seg[0]]:.1f} mm)"
        style = "-o" if n == 0 else "-s"
        ax_g.plot(s[seg], front.G_T[seg], style, color=_FRONT_COLOURS[0],
                  markersize=3, linewidth=1.2, label=f"G_T at the analysis load{tag}")
        ax_p.plot(s[seg], front.psi[seg], style, color=_FRONT_COLOURS[1],
                  markersize=3, linewidth=1.2)
    row = front.governing
    if row is not None:
        ax_g.plot([s[row]], [front.G_T[row]], "*", color="black", markersize=10,
                  label=f"governing, λ_c = {float(front.lambda_c[row]):.3g}")
    ax_g.set_ylabel("G_T (N/mm)")
    ax_g.set_ylim(bottom=0.0)
    ax_g.legend(loc="best", frameon=False, fontsize=8)
    ax_p.set_ylabel("Mode mixity ψ")
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.set_xlabel(label)
    _set_titles(ax_g, "Delamination front: energy release rate and mode mixity",
                f"VCCT, {len(front)} front nodes · ψ = (G_II + G_III) / G_T")
    fig.tight_layout()
    return fig


# ======================================================================
# Plot routines (return a matplotlib Figure for st.pyplot)
# ======================================================================

#: Drawn height over drawn length that the cross-section plots aim for. A
#: coupon is 10 to 40 times longer than it is thick, so at true scale its
#: section is a sliver: the z axis is stretched by the smallest factor of
#: :func:`_exaggeration_factors` that reaches this ratio, and the z axis
#: label says so.
_SECTION_TARGET_ASPECT = 0.3


#: Width of the cross-section figures and the cap on the section's drawn
#: height (inches).
_SECTION_FIG_WIDTH_IN = 9.0


_SECTION_MAX_AXES_HEIGHT_IN = 3.6


#: Margins around the section axes (inches): the title and subtitle above,
#: the x tick labels and axis label between the section and the horizontal
#: colorbar, the colorbar's ticks and label below.
_SECTION_MARGINS_IN = {"left": 1.3, "right": 0.4, "top": 0.95,
                       "gap": 0.75, "cbar": 0.18, "bottom": 0.75}


#: Spread, relative to the magnitude, below which a field is drawn as
#: constant (its colour scale would otherwise span round-off).
_CONSTANT_FIELD_RTOL = 1e-6


#: Half-range of a single-signed stress scale, relative to the mean, below
#: which the colorbar label calls the field nearly uniform.
_NEARLY_UNIFORM_SPREAD = 0.05


def _exaggeration_factors():
    """1, 2, 3, 4, 5, 6, 8, then 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, ..."""
    yield from (1, 2, 3, 4, 5, 6, 8)
    scale = 10
    while True:
        for m in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
            yield int(round(m * scale))
        scale *= 10


def section_exaggeration(length: float, thickness: float,
                         target: float = _SECTION_TARGET_ASPECT) -> int:
    """Vertical exaggeration of a ``length`` x ``thickness`` section plot.

    The smallest factor of :func:`_exaggeration_factors` that draws the
    section at least ``target`` times as high as it is long; 1 for a
    section that deep already, or when a length is not positive and finite.
    """
    if not (math.isfinite(length) and math.isfinite(thickness)) \
            or length <= 0 or thickness <= 0:
        return 1
    ratio = thickness / length
    for factor in _exaggeration_factors():
        if ratio * factor >= target or factor >= 1_000_000:
            return factor
    return 1   # not reached: the factors never run out


def _z_label(factor: int) -> str:
    """z axis label, naming the vertical exaggeration when there is one."""
    return LABEL_Z_MM if factor == 1 else f"{LABEL_Z_MM}\nz exaggerated ×{factor}"


def _section_figure(x_range, z_range):
    """Figure for a cross-section spanning ``x_range`` x ``z_range`` (mm).

    Returns ``(fig, ax, cax, factor)``: the section axes, stretched along z
    by ``factor`` (:func:`section_exaggeration`) through the data aspect so
    the tick labels stay in mm, and a horizontal colorbar axes of the same
    width under it. The axes are placed in inches, so the colorbar always
    matches the section and the title sits clear of both.
    """
    x0, x1 = (float(v) for v in x_range)
    z0, z1 = (float(v) for v in z_range)
    length, thickness = x1 - x0, z1 - z0
    factor = section_exaggeration(length, thickness)
    drawn = (thickness * factor / length if length > 0 and thickness > 0
             else _SECTION_TARGET_ASPECT)
    m = _SECTION_MARGINS_IN
    fig_w = _SECTION_FIG_WIDTH_IN
    ax_w = fig_w - m["left"] - m["right"]
    ax_h = ax_w * drawn
    if ax_h > _SECTION_MAX_AXES_HEIGHT_IN:
        ax_h = _SECTION_MAX_AXES_HEIGHT_IN
        ax_w = ax_h / drawn
    fig_h = m["top"] + ax_h + m["gap"] + m["cbar"] + m["bottom"]
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes((m["left"] / fig_w, (m["bottom"] + m["cbar"] + m["gap"]) / fig_h,
                       ax_w / fig_w, ax_h / fig_h))
    cax = fig.add_axes((m["left"] / fig_w, m["bottom"] / fig_h,
                        ax_w / fig_w, m["cbar"] / fig_h))
    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)
    if length > 0 and thickness > 0:
        ax.set_aspect(factor)     # 1 mm of z is drawn as long as `factor` mm of x
    ax.grid(False)
    ax.set_xlabel(LABEL_X_MM)
    ax.set_ylabel(_z_label(factor))
    return fig, ax, cax, factor


def _set_titles(ax, title: str, subtitle: str | None = None) -> None:
    """Left-aligned bold ``title`` above ``ax``, with a smaller grey ``subtitle``
    line between it and the axes."""
    ax.set_title(title, loc="left", pad=22 if subtitle else 8)
    if subtitle:
        ax.annotate(subtitle, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(0, 6), textcoords="offset points",
                    ha="left", va="bottom", fontsize=10, color="0.35")


def _colour_limits(values, *, floor: float, robust: bool = False
                   ) -> tuple[float, float, bool]:
    """Colour-scale limits ``(vmin, vmax, constant)`` of the finite ``values``.

    A field whose spread is at most ``_CONSTANT_FIELD_RTOL`` of its
    magnitude (or 1e-12 absolute) is constant: a scale over its round-off
    would show an offset and meaningless ticks, so the limits are the value
    plus or minus the larger of 1 % of it and ``floor``. Otherwise they are
    the data range or, with ``robust``, its 2nd and 98th percentiles, so a
    few singular peaks do not wash out the rest (the colorbar marks the
    clipped ends). ``(0, 1, False)`` when no value is finite.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0, False
    lo, hi = float(v.min()), float(v.max())
    if hi - lo <= max(_CONSTANT_FIELD_RTOL * max(abs(lo), abs(hi)), 1e-12):
        mid = 0.5 * (lo + hi)
        half = max(0.01 * abs(mid), floor)
        return mid - half, mid + half, True
    if robust:
        p_lo, p_hi = (float(p) for p in np.percentile(v, [2.0, 98.0]))
        if p_hi > p_lo:
            lo, hi = p_lo, p_hi
    return lo, hi, False


def _colorbar_extend(values, vmin: float, vmax: float) -> str:
    """Colorbar ``extend`` marking the values outside ``[vmin, vmax]``."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "neither"
    tol = 1e-9 * max(abs(vmin), abs(vmax), 1e-300)
    below, above = bool(v.min() < vmin - tol), bool(v.max() > vmax + tol)
    return {(True, True): "both", (True, False): "min",
            (False, True): "max"}.get((below, above), "neither")


def _finish_colorbar(cb, label: str) -> None:
    """Plain tick labels on ``cb`` (no offset, no scientific notation) and its label."""
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    cb.formatter = formatter
    cb.locator = MaxNLocator(nbins=7)
    cb.update_ticks()
    cb.ax.grid(False)
    cb.set_label(label)


def _mark_uniform_value(cb, value: float) -> None:
    """Mark a constant field's ``value`` on its (horizontal) colorbar."""
    cb.ax.axvline(value, color="k", linewidth=2.0)


def _sequential_cmap(name: str, *, reverse: bool = False) -> ListedColormap:
    """``name`` without its palest fifth, so its low end stays visible on white."""
    colours = matplotlib.colormaps[name](np.linspace(0.2, 1.0, 256))
    return ListedColormap(colours[::-1] if reverse else colours,
                          name=f"{name}{'_r' if reverse else ''}_from_20pct")


#: Stress colormaps. Signed fields crossing zero use RdBu_r on a range
#: symmetric about zero, so its white midpoint is true zero (#51); a field of
#: one sign uses the matching half (blue: compression, darkest = most
#: negative; red: tension), sequential over the field's own range; a
#: magnitude (von Mises) uses the project's cividis (#51).
_STRESS_CMAP_DIVERGING = "RdBu_r"


_STRESS_CMAP_NEGATIVE = _sequential_cmap("Blues", reverse=True)


_STRESS_CMAP_POSITIVE = _sequential_cmap("Reds")


_STRESS_CMAP_MAGNITUDE = "cividis"


def _stress_colour_scale(values, *, magnitude: bool = False):
    """``(cmap, norm, extend, constant)`` for a stress field (MPa).

    The diverging map only when the field crosses zero, otherwise a
    sequential one over the field's range (see ``_STRESS_CMAP_*``), so a
    field of one sign shows its variation instead of one flat colour. The
    range is the 2nd to 98th percentile; a constant field gets its value
    ±1 % (at least ±1 MPa).
    """
    vmin, vmax, constant = _colour_limits(values, floor=1.0, robust=True)
    if magnitude:
        cmap = _STRESS_CMAP_MAGNITUDE
    elif vmin < 0.0 < vmax:
        v = max(-vmin, vmax)
        vmin, vmax = -v, v
        cmap = _STRESS_CMAP_DIVERGING
    elif vmax <= 0.0:
        cmap = _STRESS_CMAP_NEGATIVE
    else:
        cmap = _STRESS_CMAP_POSITIVE
    return (cmap, Normalize(vmin=vmin, vmax=vmax),
            _colorbar_extend(values, vmin, vmax), constant)


def plot_profile(result: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    pf = result["porosity_field"]
    z, Vp = pf.effective_porosity_profile(nz=200)
    ax.plot(Vp * 100, z, "b-", linewidth=2)
    ax.set_xlabel(LABEL_POROSITY_PCT)
    ax.set_ylabel(LABEL_Z_MM)
    ax.set_title("Through-Thickness Porosity Profile")
    ax.set_xlim(left=0)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    fig.tight_layout()
    return fig


def plot_mesh(result: dict):
    """Mid-y cross-section coloured by stiffness retention with void overlays."""
    mesh = result["mesh"]
    section = mesh.mid_y_section_indices()   # (nz + 1, nx + 1)
    X = mesh.nodes[section, 0]
    Z = mesh.nodes[section, 2]
    Sr = mesh.stiffness_reduction[section] * 100.0
    fig, ax, cax, factor = _section_figure((X.min(), X.max()), (Z.min(), Z.max()))

    vmin, vmax, constant = _colour_limits(Sr, floor=1.0)
    if constant:
        vmax = min(vmax, 100.0)      # retention never exceeds 100 %
    else:
        # 100 % (pristine) anchors the top of the scale.
        vmin, vmax = max(0.0, vmin - 1.0), max(100.0, vmax)
    im = ax.pcolormesh(X, Z, Sr, shading="gouraud", cmap="cividis",
                       vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    label = LABEL_STIFFNESS_RETENTION
    if constant:
        value = float(np.mean(Sr))
        label += f" · uniform: {value:.1f} %"
        _mark_uniform_value(cb, value)
    _finish_colorbar(cb, label)

    step_x = max(1, mesh.nx // 20)
    step_z = max(1, mesh.nz // 24)
    for k in range(0, mesh.nz + 1, step_z):
        ax.plot(X[k], Z[k], "k-", linewidth=0.3, alpha=0.4)
    for i in range(0, mesh.nx + 1, step_x):
        ax.plot(X[:, i], Z[:, i], "k-", linewidth=0.3, alpha=0.4)

    n_voids = 0
    void_elems = mesh.void_elements
    if len(void_elems) > 0:
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        void_patches = []
        for e_idx in void_elems:
            j_e = (e_idx // mesh.nx) % mesh.ny
            if j_e != mesh.ny // 2:
                continue
            node_coords = mesh.nodes[mesh.elements[e_idx]]
            xz = node_coords[:, [0, 2]]
            unique_xz = np.unique(xz, axis=0)
            if len(unique_xz) < 3:
                continue
            cx_p, cz_p = unique_xz.mean(axis=0)
            angles = np.arctan2(
                unique_xz[:, 1] - cz_p, unique_xz[:, 0] - cx_p,
            )
            order = np.argsort(angles)
            void_patches.append(Polygon(unique_xz[order], closed=True))

        if void_patches:
            pc = PatchCollection(
                void_patches, facecolor="white", edgecolor="red",
                linewidth=1.0, zorder=5, alpha=1.0,
            )
            ax.add_collection(pc)
            n_voids = len(void_patches)

    subtitle = (f"mid-y section · {len(mesh.nodes):,} nodes, "
                f"{len(mesh.elements):,} elements")
    if n_voids:
        subtitle += f" · {n_voids} void elements (white, red outline)"
    if _draw_delaminations(ax, mesh, section, X, Z):
        subtitle += " · delamination in red"
    _set_titles(ax, "FE mesh: stiffness retention", subtitle)
    return fig


def _draw_delaminations(ax, mesh, section, X, Z) -> bool:
    """Draw the delaminated part of each duplicated interface of ``mesh`` on
    its mid-y section (node ids ``section``, coordinates ``X``, ``Z``):
    every element edge of the interface with a delaminated node. Returns
    whether anything was drawn."""
    if not getattr(mesh, "delaminations", ()):
        return False
    originals, _ = mesh.interface_node_pairs()
    opened = originals[~mesh.bonded_pairs()]
    n_plies = int(mesh.material.n_plies)
    drawn = False
    for k in mesh.delaminated_interfaces:
        layer = (k + 1) * mesh.nz // n_plies
        is_open = np.isin(section[layer], opened)
        for i in np.flatnonzero(is_open[:-1] | is_open[1:]):
            ax.plot(X[layer, i:i + 2], Z[layer, i:i + 2], color="red",
                    linewidth=2.0, solid_capstyle="butt")
            drawn = True
    return drawn


def plot_results(result: dict, layup_str: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    emp = result["empirical"]
    fe_field = result.get("fe_field")
    fe_loading = result.get("fe_loading", "compression")
    cfg = result["config"]
    layup_scale = result.get("layup_scale", {})

    modes = ["compression", "tension", "shear", "ilss"]
    group_labels = [m.upper() for m in modes]
    # (key, legend label, colour, hatch)
    series: list[tuple[str, str, str, str | None]] = [
        ("judd_wright", "Judd-Wright", "#1f77b4", None),
        ("power_law", "Power Law", "#ff7f0e", None),
        ("linear", "Linear", "#2ca02c", None),
    ]

    # A thermal-only solve has no stiffness measure (knockdown NaN): no bar.
    # A fracture specimen's knockdown is a compliance ratio to the bonded
    # specimen, not a stiffness loss of the laminate: no bar either.
    fracture = fe_loading in _FRACTURE_MODES
    has_fe = fe_field is not None and bool(np.isfinite(fe_field.knockdown)) \
        and not fracture
    flexure = has_fe and fe_loading in _FLEXURE_MODES
    if has_fe:
        series.append(("fe", f"FE Stiffness ({fe_loading})", "#d62728", "//"))
    if flexure:
        # FE only: its own group, after the empirical modes.
        modes.append(fe_loading)
        group_labels.append("FLEXURE (FE)")

    def knockdown(key: str, mode: str) -> float:
        if key == "fe":
            return float(fe_field.knockdown) if mode == fe_loading else float("nan")
        return float(emp[mode][key]["knockdown"]) if mode in emp else float("nan")

    x = np.arange(len(modes))
    width = 0.8 / len(series)
    labelled: set[str] = set()
    for j, mode in enumerate(modes):
        bars = [(s, knockdown(s[0], mode)) for s in series]
        bars = [(s, v) for s, v in bars if np.isfinite(v)]
        # Centre the bars each group has (the FE bar is in one group only).
        offsets = (np.arange(len(bars)) - (len(bars) - 1) / 2.0) * width
        for ((key, label, colour, hatch), value), off in zip(bars, offsets, strict=True):
            ax.bar(x[j] + off, value, width, color=colour, hatch=hatch,
                   edgecolor="white" if hatch is None else "0.3",
                   label="_nolegend_" if key in labelled else label)
            labelled.add(key)
    if flexure:
        ax.axvline(x[-1] - 0.5, color="0.6", linewidth=0.8, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(LABEL_KNOCKDOWN)
    _set_titles(
        ax, "Knockdown factor by loading mode",
        f"Vp = {cfg['Vp']:.1f} %, {cfg['void_shape']} voids, "
        f"{_distribution_text(cfg)} distribution, {layup_str}")
    ax.set_ylim(0, 1.1)
    # Outside the axes, right of the bars, so it never covers one.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              frameon=False)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    note = ("Solid bars = strength knockdown (at mean Vp); "
            "hatched bar = stiffness knockdown (FE)")
    if fe_field is not None and not has_fe:
        note = ("Solid bars = strength knockdown (at mean Vp); the FE "
                "thermal-only solve has no stiffness knockdown (n/a)")
        if fracture:
            note = ("Solid bars = strength knockdown (at mean Vp); the FE "
                    f"{fe_loading.upper()} specimen's results are under Defects")
    if flexure:
        note += ("\nFLEXURE: FE beam-stiffness knockdown only; there is no "
                 "empirical flexure model")
    amplified = {m: s for m, s in layup_scale.items() if s > 1.0}
    if amplified:
        factors = ", ".join(f"{m} x{s:.2f}" for m, s in sorted(amplified.items()))
        note += (f"\nLayup amplification ({factors}): layup more "
                 "matrix-dominated than the QI calibration; not validated")
    # Footnote intentionally smaller than rcParams.font.size (annotation, not
    # primary data); explicit override kept here on purpose. It sits below
    # the tick labels (not inside the axes, where it ran under the legend);
    # tight_layout makes room for it.
    ax.annotate(note, xy=(0.0, 0.0), xycoords="axes fraction",
                xytext=(0, -24), textcoords="offset points",
                fontsize=7, color="0.4", ha="left", va="top")

    fig.tight_layout()
    return fig


_STRESS_COMPONENTS = {
    "σ₁₁ (fiber)": (0, r"$\sigma_{11}$ local (MPa)"),
    "σ₂₂ (transverse)": (1, r"$\sigma_{22}$ local (MPa)"),
    "σ₃₃ (through-thickness)": (2, r"$\sigma_{33}$ local (MPa)"),
    "τ₂₃ (interlaminar)": (3, r"$\tau_{23}$ local (MPa)"),
    "τ₁₃ (interlaminar)": (4, r"$\tau_{13}$ local (MPa)"),
    "τ₁₂ (in-plane shear)": (5, r"$\tau_{12}$ local (MPa)"),
    "Von Mises": (-1, "Von Mises Stress (MPa)"),
}


#: File-name slug of each stress component (by component index).
_STRESS_SLUGS = {0: "s11", 1: "s22", 2: "s33", 3: "t23", 4: "t13", 5: "t12",
                 -1: "von_mises"}


def default_stress_component(result: dict) -> str:
    """The stress component the Stress tab opens on for ``result``.

    The one that governs the load case: τ₁₃ for ILSS (interlaminar shear)
    and the ENF and MMB specimens, σ₃₃ for the DCB (crack opening), σ₂₂ for
    a thermal-only solve (transverse stress, where residual cracking
    starts), τ₁₂ for in-plane shear, σ₁₁ otherwise (fibre stress, also in
    flexure).
    """
    fe_loading = result.get("fe_loading")
    if fe_loading in ("ilss", "enf", "mmb"):
        return "τ₁₃ (interlaminar)"
    if fe_loading == "dcb":
        return "σ₃₃ (through-thickness)"
    if fe_loading == "thermal":
        return "σ₂₂ (transverse)"
    if fe_loading == "shear":
        return "τ₁₂ (in-plane shear)"
    return "σ₁₁ (fiber)"


def plot_uq(uq: dict):
    """Histogram of sampled knockdowns with the p5 / p50 / p95 band."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    kd = np.asarray(uq["samples"]["knockdown"])
    pct = uq["knockdown"]["percentiles"]
    ax.hist(kd, bins=40, color="0.7", edgecolor="0.4")
    ax.axvspan(pct["p5"], pct["p95"], color="tab:blue", alpha=0.15,
               label=f"p5-p95: {pct['p5']:.3f}-{pct['p95']:.3f}")
    ax.axvline(pct["p50"], color="tab:blue", linewidth=2,
               label=f"p50: {pct['p50']:.3f}")
    ax.axvline(uq["nominal"]["knockdown"], color="k", linestyle="--",
               label=f"nominal: {uq['nominal']['knockdown']:.3f}")
    ax.set_xlabel(LABEL_KNOCKDOWN)
    ax.set_ylabel("Draws")
    ax.set_title(f"Knockdown uncertainty: {uq['mode']}, {uq['n_samples']} draws")
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    # Outside the axes, so it never covers the histogram's tails.
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
              frameon=False)
    fig.tight_layout()
    return fig


def plot_stress(result: dict, comp_name: str):
    fe_field = result.get("fe_field")
    if fe_field is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No FE results available.",
                transform=ax.transAxes, ha="center", va="center",
                color="0.5")
        ax.set_axis_off()
        return fig

    mesh = result["mesh"]
    stress_local = fe_field.stress_local
    comp_idx, label = _STRESS_COMPONENTS.get(comp_name, (0, comp_name + " (MPa)"))

    if comp_idx == -1:
        s = stress_local.mean(axis=1)
        s1, s2, s3 = s[:, 0], s[:, 1], s[:, 2]
        s4, s5, s6 = s[:, 3], s[:, 4], s[:, 5]
        elem_stress = np.sqrt(0.5 * (
            (s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2
            + 6.0 * (s4 ** 2 + s5 ** 2 + s6 ** 2)
        ))
    else:
        elem_stress = stress_local.mean(axis=1)[:, comp_idx]

    # Mid-y element row (nz, nx) and its nodes (nz + 1, nx + 1); the
    # structured mesh makes the nodes the corners of the element cells.
    elems = mesh.mid_y_element_indices()
    section = mesh.mid_y_section_indices()
    X = mesh.nodes[section, 0]
    Z = mesh.nodes[section, 2]
    cx = mesh.nodes[mesh.elements[elems[0]]][:, :, 0].mean(axis=1)
    x_lo, x_hi = float(X.min()), float(X.max())
    # Leave out the 10 % at each end (grips, supports, free-edge peaks).
    cols = np.flatnonzero((cx > x_lo + 0.10 * (x_hi - x_lo))
                          & (cx < x_lo + 0.90 * (x_hi - x_lo)))
    sv = elem_stress[elems][:, cols] if cols.size else np.empty((0, 0))
    if cols.size == 0 or np.isfinite(sv).sum() < 3:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Insufficient interior data for a stress plot.",
                transform=ax.transAxes, ha="center", va="center",
                color="0.5")
        ax.set_axis_off()
        return fig
    i0, i1 = int(cols[0]), int(cols[-1])
    Xc, Zc = X[:, i0:i1 + 2], Z[:, i0:i1 + 2]

    fig, ax, cax, factor = _section_figure((Xc.min(), Xc.max()), (Zc.min(), Zc.max()))
    cmap, norm, extend, constant = _stress_colour_scale(sv, magnitude=comp_idx == -1)
    # One flat cell per element: the element-averaged stress, which jumps
    # between plies (a contour would smear it across the interfaces).
    qm = ax.pcolormesh(Xc, Zc, np.ma.masked_invalid(sv), cmap=cmap, norm=norm,
                       shading="flat")
    cb = fig.colorbar(qm, cax=cax, orientation="horizontal", extend=extend)
    if constant:
        value = float(np.nanmean(sv))
        label += f" · uniform: {value:.4g} MPa"
        _mark_uniform_value(cb, value)
    elif norm.vmin * norm.vmax > 0.0:
        # A sequential scale over a narrow range makes small differences
        # look large: say how small they are.
        mid = 0.5 * (norm.vmin + norm.vmax)
        spread = 0.5 * (norm.vmax - norm.vmin) / abs(mid)
        if spread < _NEARLY_UNIFORM_SPREAD:
            label += f" · nearly uniform: {mid:.4g} MPa ± {100 * spread:.2g} %"
    _finish_colorbar(cb, label)
    _set_titles(ax, f"FE stress: {comp_name}",
                "ply (material) axes · mid-y section, middle 80 % of the "
                "length · element averages")
    return fig
