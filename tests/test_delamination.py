"""Explicit delaminations: the defect, the crack mesh and the bonded ties.

Delamination plan, phase 2.1-2.2. A :class:`Delamination` duplicates the node
plane of its interface (:class:`CompositeMesh`). Checked here:

- the defect's validation and its open-region / plate-edge semantics;
- the mesh requirements, node and element counts, unchanged structured ids,
  the appended duplicates and the rewiring;
- the bonded pairs, ties and crack front;
- the structured helpers on a delaminated mesh.
"""

from __future__ import annotations

import dataclasses
import logging
import math

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from porosity_fe import (  # noqa: E402
    MATERIALS,
    CompositeMesh,
    Delamination,
    EmpiricalSolver,
    FEVisualizer,
    PorosityField,
    VoidGeometry,
    check_mesh_quality,
    extrapolate_to_nodes,
)

#: 8-ply AS4/3501-6 (it has CTEs): a 1 mm laminate, interfaces 0 to 6.
MAT8 = dataclasses.replace(MATERIALS['AS4_3501_6_epoxy'], n_plies=8)
MID = 3   # mid-plane interface of MAT8


@pytest.fixture(autouse=True)
def _quiet():
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)
    plt.close('all')


def _pf(vp: float = 0.03, **kw) -> PorosityField:
    return PorosityField(MAT8, vp, distribution='clustered', **kw)


def _coupon(pf: PorosityField, delaminations=None, *, nx=8, ny=4, nz=8,
            L=(8.0, 4.0), ply_angles='QI') -> CompositeMesh:
    """An ``L[0] x L[1]`` mm coupon (1 mm elements in plane by default)."""
    mesh = CompositeMesh(pf, pf.material, nx=nx, ny=ny, nz=nz,
                         ply_angles=ply_angles, delaminations=delaminations)
    mesh.L_x, mesh.L_y = L
    mesh.generate_mesh()
    return mesh






# --------------------------------------------------------------------------
# The defect
# --------------------------------------------------------------------------

class TestDelaminationDefect:
    def test_strip_fields_are_normalized(self):
        d = Delamination.strip(np.int64(2), 1, 5)
        assert d == Delamination(2, 'strip', x_start=1.0, x_end=5.0)
        assert type(d.interface) is int and type(d.x_start) is float
        assert d.bounding_box == (1.0, 5.0, -math.inf, math.inf)
        assert hash(d) == hash(Delamination(2, x_start=1.0, x_end=5.0))

    def test_planform_fields_are_normalized(self):
        d = Delamination.ellipse(0, [10, 5], np.array([2.0, 1.0]))
        assert d.center == (10.0, 5.0) and d.half_size == (2.0, 1.0)
        assert d.bounding_box == (8.0, 12.0, 4.0, 6.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.interface = 1  # type: ignore[misc]

    @pytest.mark.parametrize('kwargs, match', [
        (dict(interface=-1, x_start=0.0, x_end=1.0), 'integer >= 0'),
        (dict(interface=1.0, x_start=0.0, x_end=1.0), 'integer >= 0'),
        (dict(interface=True, x_start=0.0, x_end=1.0), 'integer >= 0'),
        (dict(interface=0, shape='circle', center=(1, 1), half_size=(1, 1)), 'shape'),
        (dict(interface=0, x_start=0.0), 'x_start and x_end'),
        (dict(interface=0, x_start=2.0, x_end=2.0), 'greater than x_start'),
        (dict(interface=0, x_start=0.0, x_end=math.inf), 'finite'),
        (dict(interface=0, x_start='0', x_end=1.0), 'number'),
        (dict(interface=0, x_start=0.0, x_end=1.0, center=(1, 1)), 'center and half_size are for'),
        (dict(interface=0, shape='rectangle', center=(1, 1)), 'needs center'),
        (dict(interface=0, shape='ellipse', center=(1, 1), half_size=(1, 0)), 'positive'),
        (dict(interface=0, shape='ellipse', center=(1, 1), half_size=(1, -2)), 'positive'),
        (dict(interface=0, shape='rectangle', center=(math.nan, 1), half_size=(1, 1)), 'finite'),
        (dict(interface=0, shape='rectangle', center=(1, 1, 1), half_size=(1, 1)), 'two numbers'),
        (dict(interface=0, shape='rectangle', center=(1, 1), half_size=(1, 1), x_end=2.0),
         'x_start and x_end are for'),
    ])
    def test_invalid_input_raises(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            Delamination(**kwargs)

    def test_region_is_open(self):
        rect = Delamination.rectangle(0, (5.0, 5.0), (2.0, 1.0))
        x = np.array([5.0, 3.0, 7.0, 6.9, 5.0, 5.0])
        y = np.array([5.0, 5.0, 5.0, 5.9, 6.0, 4.1])
        np.testing.assert_array_equal(rect.contains(x, y),
                                      [True, False, False, True, False, True])
        ell = Delamination.ellipse(0, (0.0, 0.0), (5.0, 5.0))
        np.testing.assert_array_equal(ell.contains([3.0, 0.0, 4.9], [4.0, 0.0, 0.0]),
                                      [False, True, True])
        strip = Delamination.strip(0, 1.0, 3.0)
        np.testing.assert_array_equal(strip.contains([1.0, 2.0, 3.0], [1e9, -1e9, 0.0]),
                                      [False, True, False])

    def test_tolerance_keeps_near_boundary_points_out(self):
        strip = Delamination.strip(0, 1.0, 3.0)
        assert strip.contains(3.0 - 1e-12, 0.0)
        assert not strip.contains(3.0 - 1e-12, 0.0, tol=1e-9)

    def test_edges_on_the_plate_boundary_are_open(self):
        plate = (0.0, 10.0, 0.0, 4.0)
        strip = Delamination.strip(0, 0.0, 3.0)
        assert not strip.contains(0.0, 2.0)
        assert strip.contains(0.0, 2.0, plate=plate)
        assert not strip.contains(3.0, 2.0, plate=plate)       # the front
        beyond = Delamination.strip(0, -5.0, 3.0)
        assert beyond.contains(0.0, 2.0) and beyond.contains(0.0, 2.0, plate=plate)
        corner = Delamination.rectangle(0, (0.0, 0.0), (2.0, 1.0))
        assert corner.contains([0.0, 1.0, 0.0], [0.0, 0.0, 0.5], plate=plate).all()
        assert not corner.contains(2.0, 0.0, plate=plate)
        assert not corner.contains(1.0, 1.0, plate=plate)

    def test_overlaps_plate(self):
        plate = (0.0, 50.0, 0.0, 20.0)
        assert Delamination.strip(0, -10.0, 0.5).overlaps_plate(plate)
        assert not Delamination.strip(0, 50.0, 60.0).overlaps_plate(plate)
        assert not Delamination.rectangle(0, (25.0, 21.0), (5.0, 1.0)).overlaps_plate(plate)
        # The bounding box reaches into the plate corner, the ellipse does not.
        off_corner = Delamination.ellipse(0, (-1.0, -1.0), (1.2, 1.2))
        assert not off_corner.overlaps_plate(plate)
        assert Delamination.ellipse(0, (-1.0, -1.0), (1.5, 1.5)).overlaps_plate(plate)


# --------------------------------------------------------------------------
# The mesh
# --------------------------------------------------------------------------

class TestMeshRequirements:
    def test_nz_must_put_every_interface_on_a_node_layer(self):
        with pytest.raises(ValueError, match='multiple of n_plies = 8'):
            CompositeMesh(_pf(), MAT8, 4, 2, 12, delaminations=[Delamination.strip(MID, 0, 10)])

    def test_nz_twice_the_ply_count(self):
        mesh = CompositeMesh(_pf(), MAT8, 4, 2, 16, delaminations=[Delamination.strip(MID, 0, 10)])
        originals, _ = mesh.interface_node_pairs()
        np.testing.assert_allclose(mesh.nodes[originals, 2], 0.5 * MAT8.total_thickness)

    @pytest.mark.parametrize('k', [7, 12])
    def test_interface_must_exist(self, k):
        with pytest.raises(ValueError, match='interfaces 0 to 6'):
            CompositeMesh(_pf(), MAT8, 4, 2, 8, delaminations=[Delamination.strip(k, 0, 10)])

    @pytest.mark.parametrize('d', [
        Delamination.strip(MID, 50.0, 60.0),
        Delamination.rectangle(MID, (25.0, 25.0), (3.0, 3.0)),
        Delamination.ellipse(MID, (-1.0, -1.0), (1.2, 1.2)),
    ])
    def test_region_must_overlap_the_plate(self, d):
        with pytest.raises(ValueError, match='does not overlap'):
            CompositeMesh(_pf(), MAT8, 4, 2, 8, delaminations=[d])

    def test_items_must_be_delaminations(self):
        with pytest.raises(ValueError, match='Delamination'):
            CompositeMesh(_pf(), MAT8, 4, 2, 8, delaminations=[(MID, 0.0, 10.0)])
        with pytest.raises(ValueError, match='sequence'):
            CompositeMesh(_pf(), MAT8, 4, 2, 8, delaminations=3)

    def test_a_single_delamination_is_accepted(self):
        d = Delamination.strip(MID, 0, 10)
        assert CompositeMesh(_pf(), MAT8, 4, 2, 8, delaminations=d).delaminations == (d,)

    def test_cutting_an_interface_through_raises(self):
        with pytest.raises(ValueError, match='separate'):
            CompositeMesh(_pf(), MAT8, 4, 2, 8, delaminations=[Delamination.strip(MID, -1, 60)])

    def test_a_new_interface_needs_generate_mesh(self):
        mesh = CompositeMesh(_pf(), MAT8, 4, 2, 8, delaminations=[Delamination.strip(MID, 0, 10)])
        mesh.delaminations = (Delamination.strip(1, 0, 10),)
        with pytest.raises(ValueError, match='generate_mesh'):
            mesh.bonded_pairs()
        plain = CompositeMesh(_pf(), MAT8, 4, 2, 8)
        plain.delaminations = (Delamination.strip(1, 0, 10),)
        with pytest.raises(ValueError, match='generate_mesh'):
            plain.bonded_ties()
        plain.generate_mesh()
        assert plain.delaminated_interfaces == (1,)

    def test_builders_check_the_final_specimen(self):
        t800 = MATERIALS['T800_epoxy']
        pf = PorosityField(t800, 0.0)
        # Beyond the 50 mm default plate, inside the 169 mm D7264 specimen.
        mesh = CompositeMesh.for_flexure(pf, t800, delaminations=[Delamination.strip(11, 100, 110)])
        assert mesh.delaminated_interfaces == (11,) and mesh.L_x > 110
        # Inside the default 20 mm width, outside the 13 mm specimen.
        with pytest.raises(ValueError, match='does not overlap'):
            CompositeMesh.for_flexure(pf, t800, delaminations=[
                Delamination.rectangle(11, (50.0, 16.0), (5.0, 2.0))])


@pytest.fixture(scope='module')
def pair():
    """An undelaminated coupon with a discrete void, and the same coupon with
    delaminations at interfaces 1 and 3."""
    void = VoidGeometry(center=(4.0, 2.0, 0.5), radii=(1.5, 1.0, 0.2))
    pf = _pf(discrete_voids=[void])
    delams = [Delamination.strip(MID, 2.0, 5.0),
              Delamination.ellipse(1, (4.0, 2.0), (2.5, 1.5)),
              Delamination.rectangle(MID, (6.0, 2.0), (1.0, 1.0))]
    return _coupon(pf), _coupon(pf, delams)


class TestMeshTopology:
    def test_counts(self, pair):
        ref, mesh = pair
        layer = (mesh.nx + 1) * (mesh.ny + 1)
        assert mesh.n_structured_nodes == ref.n_nodes == ref.n_structured_nodes == 9 * 5 * 9
        assert mesh.n_nodes == ref.n_nodes + 2 * layer
        assert mesh.n_dof == 3 * mesh.n_nodes
        assert mesh.n_elements == ref.n_elements
        assert mesh.delaminated_interfaces == (1, MID)
        assert 'delaminations=3' in repr(mesh) and 'delaminations' not in repr(ref)

    def test_structured_ids_unchanged_and_duplicates_appended(self, pair):
        ref, mesh = pair
        n0 = ref.n_nodes
        np.testing.assert_array_equal(mesh.nodes[:n0], ref.nodes)
        originals, duplicates = mesh.interface_node_pairs()
        layer = (mesh.nx + 1) * (mesh.ny + 1)
        np.testing.assert_array_equal(duplicates, n0 + np.arange(2 * layer))
        # Interface 1 at node layer 2, interface 3 at node layer 4.
        np.testing.assert_array_equal(originals, np.r_[2 * layer + np.arange(layer),
                                                       4 * layer + np.arange(layer)])
        for name in ('nodes', 'porosity', 'stiffness_reduction', 'ply_ids'):
            arr = np.asarray(getattr(mesh, name))
            assert arr.shape[0] == mesh.n_nodes, name
            np.testing.assert_array_equal(arr[:n0], np.asarray(getattr(ref, name)), err_msg=name)
            np.testing.assert_array_equal(arr[duplicates], arr[originals], err_msg=name)
        o3, d3 = mesh.interface_node_pairs(MID)
        np.testing.assert_array_equal(o3, originals[layer:])
        np.testing.assert_array_equal(d3, duplicates[layer:])

    def test_rewiring(self, pair):
        ref, mesh = pair
        per_layer = mesh.nx * mesh.ny
        layer = (mesh.nx + 1) * (mesh.ny + 1)
        changed = np.flatnonzero((mesh.elements != ref.elements).any(axis=1))
        np.testing.assert_array_equal(
            changed, np.r_[2 * per_layer + np.arange(per_layer), 4 * per_layer + np.arange(per_layer)])
        np.testing.assert_array_equal(mesh.elements[:, 4:], ref.elements[:, 4:])
        np.testing.assert_array_equal(mesh.elements[changed, :4] - ref.elements[changed, :4],
                                      np.repeat([ref.n_nodes - 2 * layer, ref.n_nodes - 3 * layer],
                                                per_layer)[:, None] * np.ones((1, 4), dtype=int))
        # No element holds both nodes of a pair; coordinates are unchanged.
        np.testing.assert_array_equal(mesh.nodes[mesh.elements], ref.nodes[ref.elements])
        originals, duplicates = mesh.interface_node_pairs()
        pairs = set(zip(originals.tolist(), duplicates.tolist(), strict=True))
        for conn in mesh.elements.tolist():
            assert not any((a, b) in pairs for a in conn for b in conn)

    def test_element_arrays_and_quality_unchanged(self, pair):
        ref, mesh = pair
        assert ref.void_elements.size > 0
        np.testing.assert_array_equal(mesh.void_elements, ref.void_elements)
        np.testing.assert_array_equal(mesh.elem_ply_ids, ref.elem_ply_ids)
        np.testing.assert_array_equal(mesh.ply_angles, ref.ply_angles)
        assert check_mesh_quality(mesh) == check_mesh_quality(ref)

    def test_builders_keep_the_duplication(self):
        pf = _pf()
        flex = CompositeMesh.for_flexure(pf, MAT8, delaminations=[Delamination.strip(MID, 5, 15)])
        ilss = CompositeMesh.for_ilss(pf, MAT8, delaminations=[Delamination.strip(MID, 1, 2)])
        for mesh in (flex, ilss):
            layer = (mesh.nx + 1) * (mesh.ny + 1)
            assert mesh.n_nodes == mesh.n_structured_nodes + layer
            assert mesh.flexure_geometry is not None
            assert not mesh.bonded_pairs().all()
        assert ilss.ilss_geometry is not None


class TestBondedTies:
    """Ties, the front and the plate-edge rule on a 1 mm grid (nodes at
    integer x in [0, 8], y in [0, 4])."""

    def _untied_xy(self, mesh):
        originals, _ = mesh.interface_node_pairs()
        xy = mesh.nodes[originals[~mesh.bonded_pairs()], :2]
        return {tuple(p) for p in np.round(xy, 9).tolist()}

    def test_ties_hold_three_dofs_per_bonded_pair(self):
        mesh = _coupon(_pf(), [Delamination.strip(MID, 2.0, 5.0)])
        originals, duplicates = mesh.interface_node_pairs()
        bonded = mesh.bonded_pairs()
        slaves, masters = mesh.bonded_ties()
        assert slaves.size == masters.size == 3 * bonded.sum()
        np.testing.assert_array_equal(slaves.reshape(-1, 3) // 3,
                                      np.repeat(duplicates[bonded][:, None], 3, axis=1))
        np.testing.assert_array_equal(masters.reshape(-1, 3) % 3, np.tile([0, 1, 2], (bonded.sum(), 1)))
        np.testing.assert_array_equal(masters.reshape(-1, 3)[:, 0] // 3, originals[bonded])
        # Strip 2 < x < 5: x = 3, 4 across the full width; x = 2 and 5 are the front.
        assert self._untied_xy(mesh) == {(x, y) for x in (3.0, 4.0) for y in range(5)}
        fo, fd = mesh.crack_front_pairs()
        assert {tuple(p) for p in mesh.nodes[fo, :2].tolist()} == \
            {(x, y) for x in (2.0, 5.0) for y in range(5)}
        np.testing.assert_array_equal(fd, duplicates[np.isin(originals, fo)])

    def test_rectangle_front_is_its_boundary(self):
        mesh = _coupon(_pf(), [Delamination.rectangle(MID, (4.0, 2.0), (2.0, 1.0))])
        assert self._untied_xy(mesh) == {(3.0, 2.0), (4.0, 2.0), (5.0, 2.0)}
        fo, _ = mesh.crack_front_pairs()
        ring = {(x, y) for x in range(2, 7) for y in (1, 2, 3)} - self._untied_xy(mesh)
        assert {tuple(p) for p in mesh.nodes[fo, :2].tolist()} == ring
        assert len(ring) == 12          # the four corners are front nodes

    def test_ellipse(self):
        mesh = _coupon(_pf(), [Delamination.ellipse(1, (4.0, 2.0), (2.0, 2.0))])
        expected = {(float(x), float(y)) for x in range(9) for y in range(5)
                    if (x - 4) ** 2 + (y - 2) ** 2 < 4}
        assert self._untied_xy(mesh) == expected
        assert (2.0, 2.0) not in expected     # on the circle: bonded, a front node

    def test_plate_edges_are_open(self):
        mesh = _coupon(_pf(), [Delamination.strip(MID, 0.0, 3.0)])
        assert self._untied_xy(mesh) == {(x, y) for x in (0.0, 1.0, 2.0) for y in range(5)}
        corner = _coupon(_pf(), [Delamination.rectangle(MID, (0.0, 0.0), (2.0, 2.0))])
        assert self._untied_xy(corner) == {(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)}

    def test_union_of_delaminations_at_one_interface(self):
        mesh = _coupon(_pf(), [Delamination.strip(MID, 0.0, 2.0), Delamination.strip(MID, 1.5, 3.0)])
        assert self._untied_xy(mesh) == {(x, y) for x in (0.0, 1.0, 2.0) for y in range(5)}

    def test_regrowing_changes_only_the_ties(self):
        mesh = _coupon(_pf(), [Delamination.strip(MID, 0.0, 3.0)])
        nodes, elements = mesh.nodes.copy(), mesh.elements.copy()
        mesh.delaminations = (Delamination.strip(MID, 0.0, 5.0),)
        np.testing.assert_array_equal(mesh.nodes, nodes)
        np.testing.assert_array_equal(mesh.elements, elements)
        assert self._untied_xy(mesh) == {(x, y) for x in range(5) for y in range(5)}
        mesh.delaminations = ()
        assert mesh.bonded_pairs().all() and mesh.crack_front_pairs()[0].size == 0

    def test_no_delamination_no_ties(self):
        mesh = _coupon(_pf())
        assert mesh.delaminations == () and mesh.delaminated_interfaces == ()
        assert all(a.size == 0 for a in (*mesh.interface_node_pairs(), *mesh.bonded_ties(),
                                         *mesh.crack_front_pairs(), mesh.bonded_pairs()))


class TestStructuredMeshHelpers:
    def test_ply_ids_and_sections(self):
        mesh = _coupon(_pf(), [Delamination.strip(MID, 2.0, 5.0)])
        assert len(mesh.ply_ids) == mesh.n_nodes
        assert mesh.mid_y_section_indices().max() < mesh.n_structured_nodes

    def test_viz(self):
        pf = PorosityField(MAT8, 0.03, distribution='uniform')
        mesh = _coupon(pf, [Delamination.strip(MID, 2.0, 5.0)])
        solver = EmpiricalSolver(mesh, MAT8)
        solver.apply_loading('compression', 'judd_wright')
        assert solver.nodal_knockdown.shape == (mesh.n_nodes,)
        for fig in (FEVisualizer.plot_mesh_3d(mesh), FEVisualizer.plot_mesh_detail(mesh),
                    FEVisualizer.plot_damage_contour(mesh, solver)):
            assert fig is not None

    def test_extrapolate_to_nodes_splits_the_crack_faces(self):
        mesh = _coupon(_pf(), [Delamination.strip(MID, 2.0, 5.0)])
        # Element value = its element layer: the interface node layer 4 sits
        # between layer 3 (below) and layer 4 (above).
        layer = np.repeat(np.arange(mesh.nz), mesh.nx * mesh.ny).astype(float)
        field = np.repeat(layer[:, None], 8, axis=1)
        originals, duplicates = mesh.interface_node_pairs()
        bonded = mesh.bonded_pairs()
        nodal, corner = extrapolate_to_nodes(field, mesh, average='ply')
        np.testing.assert_allclose(nodal[originals[bonded]], 4.0)     # the ply above
        np.testing.assert_allclose(nodal[duplicates[bonded]], 4.0)
        np.testing.assert_allclose(nodal[originals[~bonded]], 3.0)    # face below the crack
        np.testing.assert_allclose(nodal[duplicates[~bonded]], 4.0)   # face above
        nodal, _ = extrapolate_to_nodes(field, mesh, average='all')
        np.testing.assert_allclose(nodal[originals[bonded]], 3.5)
        np.testing.assert_allclose(nodal[duplicates[bonded]], 3.5)
        np.testing.assert_allclose(nodal[originals[~bonded]], 3.0)
        np.testing.assert_allclose(nodal[duplicates[~bonded]], 4.0)
        assert corner.shape == (mesh.n_elements, 8)
