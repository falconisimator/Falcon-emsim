r"""Passive sheet-metal enclosures (gmsh-free / web path).

A thin steel enclosure (walls modelled as thin rectangles with ``group=None``)
must (a) solve and carry induced eddy current + loss, and (b) NOT blow up the
node count -- per-region mesh sizing keeps the thin walls fine only locally
instead of pinning the whole domain to the wall thickness.
"""

from __future__ import annotations

from emsim.geometry.model import Conductor
from emsim.geometry.shapes import Placement, Rectangle
from emsim.materials import COPPER, STEEL
from emsim.post.losses import region_current, region_ohmic_loss
from emsim.scene import Scene


def _bars():
    return [
        Conductor("A", Rectangle(0.01, 0.05), Placement(-0.03, 0, 0), COPPER, "A", "bb1"),
        Conductor("B", Rectangle(0.01, 0.05), Placement(0.03, 0, 0), COPPER, "B", "bb2"),
    ]


def _box(half=0.15, t=0.002):
    return [
        Conductor("Wt", Rectangle(2 * half, t), Placement(0, half, 0), STEEL, None, "enc"),
        Conductor("Wb", Rectangle(2 * half, t), Placement(0, -half, 0), STEEL, None, "enc"),
        Conductor("Wl", Rectangle(t, 2 * half), Placement(-half, 0, 0), STEEL, None, "enc"),
        Conductor("Wr", Rectangle(t, 2 * half), Placement(half, 0, 0), STEEL, None, "enc"),
    ]


def _scene(conductors):
    """Configured like the web build (emsim.web.solve_scene): per-region sizing
    with lc_far decoupled from the thinnest feature."""
    sc = Scene(conductors=conductors, frequency=60.0, three_phase=False,
               boundary="dirichlet", mesh_backend="py")
    sc.group_currents = {"A": 1000 + 0j, "B": -1000 + 0j}
    ext0 = max(abs(c.placement.x) + abs(c.placement.y) + c.shape.bounding_radius()
               for c in conductors)
    sc.domain_radius = 2.3 * ext0
    sc.lc_far = 0.18 * ext0
    floor = sc.lc_far / 60.0
    sc.region_sizes = [max(min(c.shape.char_size() / 6.0, sc.lc_far * 0.5), floor)
                       for c in conductors]
    return sc


def test_passive_steel_enclosure_solves_and_is_bounded() -> None:
    walls = _box()
    sc = _scene(walls + _bars())
    sol = sc.solve()
    # mesh stays browser-tractable despite 2 mm walls on a 300 mm box (pre-fix this
    # was millions of nodes); magnetic steel needs the finest resolution of the
    # three enclosure materials, so this is the worst case.
    assert sol.mesh.num_nodes < 20000, f"node count blew up: {sol.mesh.num_nodes}"
    # the passive enclosure carries induced eddy current and dissipates loss
    enc = {w.region_tag for w in walls}
    assert abs(region_current(sol, enc)) > 0.0
    assert region_ohmic_loss(sol, enc) > 0.0


def test_thin_wall_does_not_refine_the_bars() -> None:
    """Adding a thin wall must not materially refine the bar regions far from it
    (the whole point of per-region sizing)."""
    import numpy as np

    def bar_node_count(conductors, bars):
        sol = _scene(conductors).solve()       # region_tag assigned during solve
        tags = {c.region_tag for c in bars}
        m = sol.mesh
        mask = np.array([t in tags for t in m.region_tag])
        return int(np.unique(m.tris[mask]).size)

    bars1 = _bars()
    n0 = bar_node_count(bars1, bars1)
    bars2 = _bars()
    n1 = bar_node_count(_box() + bars2, bars2)
    # the bar mesh density must not collapse to the wall thickness
    assert n1 < 3 * n0 + 200, f"bars over-refined by the wall: {n0} -> {n1}"
