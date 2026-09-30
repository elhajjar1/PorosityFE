"""Dedicated unit tests for :mod:`porosity_fe.viz` (IMPROVEMENT_PLAN 6.6).

tests/test_integration.py (``TestFEVisualizer``) already pins that every
``FEVisualizer`` method returns a figure, that ``save_path`` writes a file
and closes the figure, and that an unsaved figure is left open. This
module checks what is drawn: the plotted data against independently
computed values, labels / titles, the void-highlight cap, and the
``os.PathLike`` / extension handling of ``save_path``. Inputs are tiny
meshes or hand-built result dicts so the whole file runs in a few seconds.
"""

import types

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from porosity_fe import (
    LABEL_KNOCKDOWN,
    LABEL_POROSITY_PCT,
    LABEL_SCF,
    LABEL_STIFFNESS_RETENTION_FRAC,
    LABEL_X_MM,
    LABEL_Y_MM,
    LABEL_Z_MM,
    MATERIALS,
    CompositeMesh,
    FEVisualizer,
    PorosityField,
    VoidGeometry,
)

MODELS = ('judd_wright', 'power_law', 'linear')
MODES = ('compression', 'tension', 'shear', 'ilss')


@pytest.fixture(autouse=True)
def _close_figures():
    plt.close('all')
    yield
    plt.close('all')


@pytest.fixture(scope="module")
def material():
    return MATERIALS['T800_epoxy']


@pytest.fixture(scope="module")
def clustered(material):
    pf = PorosityField(material, 0.03, distribution='clustered', cluster_location='midplane')
    mesh = CompositeMesh(pf, material, nx=4, ny=2, nz=4)
    return pf, mesh


def _fake_result(kd_by_mode_model):
    """Minimal ``{'empirical': {mode: {model: {'knockdown': kd}}}}`` entry."""
    return {'empirical': {mode: {model: {'knockdown': kd_by_mode_model(mode, model)}
                                 for model in MODELS} for mode in MODES}}


def _is_red(line):
    return to_rgba(line.get_color()) == to_rgba('red')


class TestSavePath:
    """``save_path`` accepts ``os.PathLike`` and honours the file extension."""

    def test_pathlib_path_accepted(self, clustered, tmp_path):
        """A ``pathlib.Path`` works like a string and the figure is closed."""
        pf, _ = clustered
        out = tmp_path / "profile.png"
        fig = FEVisualizer.plot_porosity_field(pf, save_path=out)
        assert out.read_bytes().startswith(b"\x89PNG")
        assert plt.get_fignums() == []
        assert fig.axes  # still a usable Figure

    @pytest.mark.parametrize("ext, magic", [(".pdf", b"%PDF"), (".svg", b"<?xml")])
    def test_format_follows_extension(self, clustered, tmp_path, ext, magic):
        """The output format is chosen from the file extension."""
        _, mesh = clustered
        out = tmp_path / f"detail{ext}"
        FEVisualizer.plot_mesh_detail(mesh, save_path=str(out))
        assert out.read_bytes().startswith(magic)

    def test_empty_string_does_not_save(self, clustered, tmp_path, monkeypatch):
        """A falsy ``save_path`` ('' ) is treated as "don't save" and leaves the figure open."""
        pf, _ = clustered
        monkeypatch.chdir(tmp_path)
        fig = FEVisualizer.plot_porosity_field(pf, save_path='')
        assert plt.get_fignums() == [fig.number]
        assert list(tmp_path.iterdir()) == []


class TestPorosityFieldPlot:
    """Through-thickness profile plot."""

    def test_line_is_profile_in_percent(self, clustered):
        """The single line plots Vp*100 (x) against z (y) from a 200-point profile."""
        pf, _ = clustered
        fig = FEVisualizer.plot_porosity_field(pf)
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        z, Vp = pf.effective_porosity_profile(nz=200)
        np.testing.assert_allclose(ax.lines[0].get_xdata(), Vp * 100)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), z)
        assert len(z) == 200

    def test_labels_and_limits(self, clustered):
        """Axis labels come from the shared label constants; x starts at 0 %."""
        pf, _ = clustered
        ax = FEVisualizer.plot_porosity_field(pf).axes[0]
        assert ax.get_xlabel() == LABEL_POROSITY_PCT
        assert ax.get_ylabel() == LABEL_Z_MM
        assert ax.get_xlim()[0] == 0
        assert ax.get_title() == 'Through-Thickness Porosity Profile'

    def test_uniform_profile_mean_matches_Vp(self, material):
        """For a uniform field the plotted percentages average to 100*Vp."""
        pf = PorosityField(material, 0.02, distribution='uniform')
        x = FEVisualizer.plot_porosity_field(pf).axes[0].lines[0].get_xdata()
        assert np.mean(x) == pytest.approx(2.0, rel=1e-6)


class TestMesh3DPlot:
    """3D wireframe with void elements outlined in red."""

    def test_no_voids_no_red_edges(self, clustered):
        """Without discrete voids only the two gray surface grids are drawn."""
        _, mesh = clustered
        ax = FEVisualizer.plot_mesh_3d(mesh).axes[0]
        assert not any(_is_red(line) for line in ax.lines)
        assert len(ax.collections) == 2  # bottom + top wireframe

    def test_twelve_edges_per_void_element(self, material):
        """Each highlighted void element contributes its 12 hex edges."""
        probe = CompositeMesh(PorosityField(material, 0.02), material, nx=4, ny=2, nz=4)
        lo, hi = probe.nodes.min(axis=0), probe.nodes.max(axis=0)
        void = VoidGeometry(center=tuple((lo + hi) / 2), radii=tuple((hi - lo) / 2))
        mesh = CompositeMesh(PorosityField(material, 0.02, discrete_voids=[void]),
                             material, nx=4, ny=2, nz=4)
        n_void = len(mesh.void_elements)
        assert 0 < n_void <= 50
        ax = FEVisualizer.plot_mesh_3d(mesh).axes[0]
        assert sum(_is_red(line) for line in ax.lines) == 12 * n_void

    def test_highlight_capped_at_fifty_elements(self, material):
        """At most 50 void elements are outlined, however many there are."""
        probe = CompositeMesh(PorosityField(material, 0.02), material, nx=8, ny=4, nz=2)
        lo, hi = probe.nodes.min(axis=0), probe.nodes.max(axis=0)
        # Semi-axes equal to the full extents enclose every element centroid.
        void = VoidGeometry(center=tuple((lo + hi) / 2), radii=tuple(hi - lo))
        mesh = CompositeMesh(PorosityField(material, 0.02, discrete_voids=[void]),
                             material, nx=8, ny=4, nz=2)
        assert len(mesh.void_elements) == 64
        ax = FEVisualizer.plot_mesh_3d(mesh).axes[0]
        assert sum(_is_red(line) for line in ax.lines) == 12 * 50

    def test_axis_labels(self, clustered):
        """x / y / z axes carry mm labels and the plot is 3D."""
        _, mesh = clustered
        ax = FEVisualizer.plot_mesh_3d(mesh).axes[0]
        assert ax.name == '3d'
        assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == (LABEL_X_MM, LABEL_Y_MM, LABEL_Z_MM)


class TestMeshDetailPlot:
    """Cross-section contour plus the reference hex-element diagram."""

    def test_panels_and_labels(self, clustered):
        """Two titled panels share mm units; a colorbar reports porosity in %."""
        _, mesh = clustered
        fig = FEVisualizer.plot_mesh_detail(mesh)
        left, right, cbar = fig.axes
        assert left.get_title() == 'Cross-Section Porosity'
        assert right.get_title() == '8-Node Hexahedral Element'
        for ax in (left, right):
            assert (ax.get_xlabel(), ax.get_ylabel()) == (LABEL_X_MM, LABEL_Z_MM)
        assert cbar.get_ylabel() == LABEL_POROSITY_PCT

    def test_hex_diagram_numbers_all_eight_nodes(self, clustered):
        """The element diagram annotates nodes 0..7 and draws 12 edges + 8 markers."""
        _, mesh = clustered
        right = FEVisualizer.plot_mesh_detail(mesh).axes[1]
        assert sorted(t.get_text() for t in right.texts) == [str(i) for i in range(8)]
        assert len(right.lines) == 12 + 8

    def test_contour_spans_mid_y_porosity(self, clustered):
        """Contour levels bracket the porosity (in %) found on the mid-y node plane."""
        _, mesh = clustered
        cs = FEVisualizer.plot_mesh_detail(mesh).axes[0].collections[0]
        y_mid = np.unique(mesh.nodes[:, 1])[mesh.ny // 2]
        on_plane = np.isclose(mesh.nodes[:, 1], y_mid)
        P = mesh.porosity[on_plane] * 100
        assert P.max() > P.min()  # clustered field: something to contour
        assert cs.levels[0] <= P.min() + 1e-12
        assert cs.levels[-1] >= P.max() - 1e-12


class TestDamageContourPlot:
    """Midplane stiffness map reads the midplane node layer only."""

    @staticmethod
    def _midplane_mask(mesh):
        z_layers = np.unique(mesh.nodes[:, 2])
        return np.isclose(mesh.nodes[:, 2], z_layers[mesh.nz // 2])

    def _ramp_field(self, mesh):
        """0.2..0.8 ramp in x on the midplane, 5.0 on every other layer."""
        field = np.full(len(mesh.nodes), 5.0)
        mid = self._midplane_mask(mesh)
        x = mesh.nodes[mid, 0]
        field[mid] = 0.2 + 0.6 * (x - x.min()) / (x.max() - x.min())
        return field

    def test_uses_solver_nodal_knockdown_midplane(self, clustered):
        """With ``nodal_knockdown`` set, only its midplane values are contoured."""
        _, mesh = clustered
        solver = types.SimpleNamespace(nodal_knockdown=self._ramp_field(mesh))
        cs = FEVisualizer.plot_damage_contour(mesh, solver).axes[0].collections[0]
        assert cs.levels[0] <= 0.2 + 1e-12
        assert 0.8 - 1e-12 <= cs.levels[-1] < 5.0

    def test_falls_back_to_mesh_stiffness_reduction(self, material):
        """With ``nodal_knockdown=None`` the mesh's own retention field is used."""
        mesh = CompositeMesh(PorosityField(material, 0.03), material, nx=4, ny=2, nz=4)
        mesh.stiffness_reduction = self._ramp_field(mesh)
        solver = types.SimpleNamespace(nodal_knockdown=None)
        cs = FEVisualizer.plot_damage_contour(mesh, solver).axes[0].collections[0]
        assert cs.levels[0] <= 0.2 + 1e-12
        assert 0.8 - 1e-12 <= cs.levels[-1] < 5.0

    def test_labels(self, clustered):
        """x-y axes in mm, colorbar reports a dimensionless retention fraction."""
        _, mesh = clustered
        solver = types.SimpleNamespace(nodal_knockdown=self._ramp_field(mesh))
        ax, cbar = FEVisualizer.plot_damage_contour(mesh, solver).axes
        assert (ax.get_xlabel(), ax.get_ylabel()) == (LABEL_X_MM, LABEL_Y_MM)
        assert ax.get_title() == 'Stiffness Reduction at Midplane'
        assert cbar.get_ylabel() == LABEL_STIFFNESS_RETENTION_FRAC


class TestVoidSCFPlot:
    """Stress-concentration field around a single void."""

    def test_extent_is_three_times_largest_radius(self):
        """The plotted window is +/- 3 * max(radii) in both x and y."""
        void = VoidGeometry(center=(0, 0, 0), radii=(2.0, 1.0, 0.5))
        ax = FEVisualizer.plot_void_scf(void).axes[0]
        assert ax.get_xlim() == pytest.approx((-6.0, 6.0))
        assert ax.get_ylim() == pytest.approx((-6.0, 6.0))

    def test_title_reports_aspect_ratio(self):
        """The title shows the void aspect ratio to one decimal place."""
        void = VoidGeometry(center=(0, 0, 0), radii=(2.0, 1.0, 0.5))
        ax = FEVisualizer.plot_void_scf(void).axes[0]
        assert ax.get_title() == f'SCF Field (aspect ratio={void.aspect_ratio:.1f})'

    def test_levels_span_zero_to_compression_scf(self):
        """Inside the void the field is 0; its peak approaches the compression SCF."""
        void = VoidGeometry(center=(0, 0, 0), radii=(2.0, 1.0, 0.5))
        fig = FEVisualizer.plot_void_scf(void)
        levels = fig.axes[0].collections[0].levels
        step = levels[1] - levels[0]
        scf_max = void.stress_concentration_factor()['compression']
        assert scf_max > 1.0
        assert levels[0] <= 0.0
        assert scf_max - step <= levels[-1] <= scf_max + step
        assert fig.axes[1].get_ylabel() == LABEL_SCF


class TestKnockdownCurvesPlot:
    """Strength-vs-porosity panels built from a ``{label: {config: result}}`` map."""

    # Knockdown = 1 - 0.1 * Vp% for judd_wright, 1 - 0.2 * Vp% for the others,
    # with a per-mode offset so the panels are distinguishable.
    @staticmethod
    def _kd(vp_pct, mode, model):
        slope = 0.1 if model == 'judd_wright' else 0.2
        return 1.0 - slope * vp_pct - 0.01 * MODES.index(mode)

    def _results(self, configs=('uniform_spherical', 'clustered_midplane')):
        return {
            f'{vp}pct': {cfg: _fake_result(lambda mode, model, vp=vp: self._kd(vp, mode, model))
                         for cfg in configs}
            for vp in (1, 3, 5)
        }

    def test_panels_titles_and_limits(self):
        """Four panels (one per mode) with knockdown y-axis clamped to [0, 1.1]."""
        fig = FEVisualizer.plot_knockdown_curves(self._results())
        axes = fig.axes
        assert [ax.get_title() for ax in axes] == [m.upper() for m in MODES]
        for ax in axes:
            assert ax.get_xlabel() == LABEL_POROSITY_PCT
            assert ax.get_ylabel() == LABEL_KNOCKDOWN
            assert ax.get_ylim() == (0.0, 1.1)
        assert fig._suptitle.get_text() == 'Porosity Knockdown Curves'

    def test_one_line_per_config_and_model(self):
        """Each panel has configs x 3 models lines."""
        fig = FEVisualizer.plot_knockdown_curves(self._results())
        for ax in fig.axes:
            assert len(ax.lines) == 2 * len(MODELS)

    def test_line_data_matches_inputs(self):
        """x is Vp in percent (from the label) and y the matching knockdowns."""
        fig = FEVisualizer.plot_knockdown_curves(self._results(configs=('uniform_spherical',)))
        for mode, ax in zip(MODES, fig.axes, strict=True):
            for model, line in zip(MODELS, ax.lines, strict=True):
                np.testing.assert_allclose(line.get_xdata(), [1.0, 3.0, 5.0])
                np.testing.assert_allclose(line.get_ydata(),
                                           [self._kd(v, mode, model) for v in (1, 3, 5)])

    def test_linestyle_and_colour_encoding(self):
        """Solid for 'uniform*' configs, dashed otherwise; one colour per model."""
        fig = FEVisualizer.plot_knockdown_curves(self._results())
        lines = fig.axes[0].lines
        uniform, clustered = lines[:3], lines[3:]
        assert all(line.get_linestyle() == '-' for line in uniform)
        assert all(line.get_linestyle() == '--' for line in clustered)
        expected = [to_rgba(c) for c in ('blue', 'red', 'green')]
        assert [to_rgba(line.get_color()) for line in uniform] == expected
        assert [to_rgba(line.get_color()) for line in clustered] == expected

    def test_accepts_config_result_like_mapping(self):
        """Only ``['empirical'][mode][model]['knockdown']`` access is required."""
        class _Shim:
            def __init__(self, emp):
                self._emp = emp

            def __getitem__(self, key):
                if key != 'empirical':
                    raise KeyError(key)
                return self._emp

        results = {'2pct': {'uniform_x': _Shim(_fake_result(lambda m, mo: 0.9)['empirical'])}}
        fig = FEVisualizer.plot_knockdown_curves(results)
        np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [0.9])


class TestModelComparisonPlot:
    """Grouped bar chart of compression and ILSS knockdowns."""

    @staticmethod
    def _results():
        configs = ('uniform_spherical', 'clustered_midplane', 'interface_penny')

        def kd(ci):
            return lambda mode, model: 0.5 + 0.1 * ci + 0.01 * MODELS.index(model) + (0.001 if mode == 'ilss' else 0)

        return {cfg: _fake_result(kd(ci)) for ci, cfg in enumerate(configs)}

    def test_titles_labels_and_legend(self):
        """Compression and ILSS panels, knockdown y-label, title-cased model legend."""
        fig = FEVisualizer.plot_model_comparison(self._results())
        left, right = fig.axes
        assert (left.get_title(), right.get_title()) == ('Compression', 'ILSS')
        for ax in (left, right):
            assert ax.get_ylabel() == LABEL_KNOCKDOWN
            assert [t.get_text() for t in ax.get_legend().get_texts()] == ['Judd Wright', 'Power Law', 'Linear']
        assert fig._suptitle.get_text() == 'Model Comparison'

    def test_bar_heights_match_knockdowns(self):
        """One bar per (model, config); heights equal the input knockdowns."""
        results = self._results()
        configs = list(results)
        fig = FEVisualizer.plot_model_comparison(results)
        for ax, mode in zip(fig.axes, ('compression', 'ilss'), strict=True):
            bars = ax.patches
            assert len(bars) == len(MODELS) * len(configs)
            expected = [results[c]['empirical'][mode][m]['knockdown'] for m in MODELS for c in configs]
            np.testing.assert_allclose([b.get_height() for b in bars], expected)

    def test_bar_groups_are_offset_by_width(self):
        """Bars for successive models are shifted by the bar width (0.2) within a group."""
        fig = FEVisualizer.plot_model_comparison(self._results())
        bars = fig.axes[0].patches
        n_cfg = 3
        centres = [b.get_x() + b.get_width() / 2 for b in bars]
        for i in range(n_cfg):
            group = [centres[m * n_cfg + i] for m in range(len(MODELS))]
            np.testing.assert_allclose(np.diff(group), [0.2, 0.2])
            assert group[0] == pytest.approx(i)

    def test_tick_labels_wrap_underscores(self):
        """Config names are shown with underscores replaced by line breaks."""
        fig = FEVisualizer.plot_model_comparison(self._results())
        labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
        assert labels == ['uniform\nspherical', 'clustered\nmidplane', 'interface\npenny']
