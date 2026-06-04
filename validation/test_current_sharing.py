r"""Milestone 2 validation: multi-bar current sharing.

Three gates, each with an analytic/symmetry anchor:

1. **Symmetric split** -- two identical parallel wires forming one terminal
   carrying total I. Mirror symmetry forces an exact 50/50 split, at any
   frequency.
2. **Composite single terminal** -- a concentric Cu core + Al shell wired as a
   single terminal. At low frequency the current splits by DC conductance,
   I_core : I_shell = sigma_core A_core : sigma_shell A_shell.
3. **Two independent terminals** -- a go/return pair as two parallel groups
   with prescribed +I and -I; each group must recover its prescribed current
   (exercises the multi-unknown bordered system).
"""

from __future__ import annotations

import math

from emsim.config import SimulationConfig
from emsim.materials import ALUMINIUM, AIR, COPPER, MaterialTable
from emsim.mesh.gmsh_backend import AIR_TAG, Disk, mesh_disks
from emsim.fem.constraints import ParallelGroup
from emsim.solve.solver import solve
from emsim.post.losses import group_losses, region_current

SIGMA = COPPER.sigma


def _skin_lc(cfg: SimulationConfig, a: float, sigma: float) -> float:
    d = cfg.skin_depth(sigma)
    return min(d / 4.0, a / 8.0)


def test_symmetric_two_wire_split() -> None:
    a = 0.005
    d = 0.02  # centre-to-origin offset
    cfg = SimulationConfig(2000.0)
    tag_l, tag_r = 10, 11
    disks = [Disk(-d, 0.0, a, tag_l), Disk(d, 0.0, a, tag_r)]
    R = 10.0 * (d + a)
    lc = _skin_lc(cfg, a, SIGMA)
    mesh = mesh_disks(disks, R, lc_surface=lc, lc_far=(d + a), grade_distance=2 * a)
    mats = MaterialTable({tag_l: COPPER, tag_r: COPPER, AIR_TAG: AIR})
    pair = ParallelGroup("pair", (tag_l, tag_r), 1.0 + 0j)
    sol = solve(mesh, mats, [pair], cfg)

    il = region_current(sol, {tag_l})
    ir = region_current(sol, {tag_r})
    print(f"two-wire split: I_left={il:.5f}  I_right={ir:.5f}  total={il+ir:.5f}")
    assert abs(il + ir - 1.0) < 1e-6
    # exact 50/50 by symmetry (allow small mesh-asymmetry tolerance)
    assert abs(il - 0.5) < 2e-2, il
    assert abs(ir - 0.5) < 2e-2, ir


def test_composite_dc_split() -> None:
    a1, a2 = 0.006, 0.01  # core radius, shell outer radius
    cfg = SimulationConfig(1e-2)  # DC-like: negligible eddy effect
    core_tag, shell_tag = 10, 11
    disks = [Disk(0, 0, a1, core_tag), Disk(0, 0, a2, shell_tag)]
    R = 10.0 * a2
    # The DC split equals the *meshed* conductance ratio, so the geometry must
    # be resolved finely enough that the disk areas converge to nominal.
    mesh = mesh_disks(
        disks, R, lc_surface=a2 / 25.0, lc_far=a2 / 6.0, grade_distance=a2
    )
    mats = MaterialTable({core_tag: COPPER, shell_tag: ALUMINIUM, AIR_TAG: AIR})
    bar = ParallelGroup("composite", (core_tag, shell_tag), 1.0 + 0j)
    sol = solve(mesh, mats, [bar], cfg)

    i_core = region_current(sol, {core_tag})
    i_shell = region_current(sol, {shell_tag})
    area_core = math.pi * a1 * a1
    area_shell = math.pi * (a2 * a2 - a1 * a1)
    g_core = COPPER.sigma * area_core
    g_shell = ALUMINIUM.sigma * area_shell
    f_core_expected = g_core / (g_core + g_shell)
    f_core = i_core.real
    print(
        f"composite DC: I_core={i_core:.5f} I_shell={i_shell:.5f} "
        f"frac_core={f_core:.4f} expected={f_core_expected:.4f}"
    )
    assert abs(i_core + i_shell - 1.0) < 1e-6
    assert abs(f_core - f_core_expected) < 1e-2


def test_two_independent_terminals() -> None:
    a = 0.005
    d = 0.02
    cfg = SimulationConfig(2000.0)
    go_tag, ret_tag = 10, 11
    disks = [Disk(-d, 0.0, a, go_tag), Disk(d, 0.0, a, ret_tag)]
    R = 10.0 * (d + a)
    lc = _skin_lc(cfg, a, SIGMA)
    mesh = mesh_disks(disks, R, lc_surface=lc, lc_far=(d + a), grade_distance=2 * a)
    mats = MaterialTable({go_tag: COPPER, ret_tag: COPPER, AIR_TAG: AIR})
    groups = [
        ParallelGroup("go", (go_tag,), 1.0 + 0j),
        ParallelGroup("return", (ret_tag,), -1.0 + 0j),
    ]
    sol = solve(mesh, mats, groups, cfg)
    losses = {gl.name: gl for gl in group_losses(sol)}
    print(
        f"go  I={losses['go'].current_recovered:.5f}  "
        f"return I={losses['return'].current_recovered:.5f}"
    )
    assert abs(losses["go"].current_recovered - 1.0) < 1e-6
    assert abs(losses["return"].current_recovered + 1.0) < 1e-6


def test_terminal_current_sharing_is_em_determined() -> None:
    """A specified terminal current splits *unevenly* among parallel bars (EM),
    sums to the specified value, and yields a positive terminal resistance."""
    from emsim.geometry.model import Conductor
    from emsim.geometry.shapes import Placement, Rectangle
    from emsim.scene import Scene

    sc = Scene(
        conductors=[
            Conductor("near", Rectangle(0.01, 0.04), Placement(0.0, 0, 0), COPPER, "A", "bbA"),
            Conductor("far", Rectangle(0.01, 0.04), Placement(-0.04, 0, 0), COPPER, "A", "bbA"),
            Conductor("ret", Rectangle(0.01, 0.04), Placement(0.06, 0, 0), COPPER, "R", "bbR"),
        ],
        group_currents={"A": 1000 + 0j, "R": -1000 + 0j},
        three_phase=False,
        frequency=2000.0,
        boundary="dirichlet",
    )
    res = sc.analyse(sc.solve())
    a_shares = {cr.name: cr.share for cr in res.conductors if cr.group == "A"}
    print("shares:", a_shares)
    assert abs(sum(a_shares.values()) - 1.0) < 1e-3  # shares sum to the terminal total
    # the bar nearer the return carries clearly more current (proximity), not 50/50
    assert a_shares["near"] > a_shares["far"] + 0.2
    term_a = next(t for t in res.terminals if t.name == "A")
    assert term_a.impedance.real > 0 and term_a.impedance.imag > 0  # R + jX


if __name__ == "__main__":
    test_symmetric_two_wire_split()
    test_composite_dc_split()
    test_two_independent_terminals()
    test_terminal_current_sharing_is_em_determined()
    print("M2 current-sharing gates passed.")
