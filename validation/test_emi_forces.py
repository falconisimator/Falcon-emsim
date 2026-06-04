r"""Milestone 4 validation: enclosure eddy loss, EMI shielding, and forces.

Gates:
1. **Enclosure energy balance** -- with a conducting/magnetic steel enclosure
   carrying induced eddy currents, the real input power 1/2 Re(sum u_g I_g*)
   must equal the total ohmic loss (conductors + enclosure). This is an exact
   identity and validates that the enclosure eddy loss is computed correctly.
2. **Maxwell-stress force** -- the force per length on one of two parallel
   wires matches the analytic mu0 Re(I1 I2*) / (4 pi D).
3. **Shielding effectiveness** -- a steel enclosure around a go/return pair
   reduces the external field, and the shielding effectiveness increases with
   frequency (magnetostatic shielding at low f, eddy shielding at high f).
"""

from __future__ import annotations

import math
from pathlib import Path

from emsim.config import MU0, SimulationConfig
from emsim.materials import AIR, COPPER, STEEL, MaterialTable
from emsim.mesh.gmsh_backend import AIR_TAG, Disk, mesh_disks
from emsim.fem.constraints import ParallelGroup
from emsim.solve.solver import solve
from emsim.post.losses import input_power, total_ohmic_loss
from emsim.post.forces import maxwell_force
from emsim.post.emi import shielding_effectiveness

STEEL_TAG = 99


def _pair_groups():
    return [
        ParallelGroup("go", (10,), 1.0 + 0j),
        ParallelGroup("return", (11,), -1.0 + 0j),
    ]


def test_enclosure_energy_balance() -> None:
    a, off = 0.005, 0.01
    ain, aout = 0.03, 0.04
    disks = [
        Disk(-off, 0, a, 10),
        Disk(off, 0, a, 11),
        Disk(0, 0, ain, AIR_TAG),
        Disk(0, 0, aout, STEEL_TAG),
    ]
    mesh = mesh_disks(disks, 8 * aout, lc_surface=a / 6, lc_far=aout / 6, grade_distance=aout)
    mats = MaterialTable({10: COPPER, 11: COPPER, AIR_TAG: AIR, STEEL_TAG: STEEL})
    sol = solve(mesh, mats, _pair_groups(), SimulationConfig(2000.0))
    pin = input_power(sol)
    ploss = total_ohmic_loss(sol)
    rel = abs(pin - ploss) / ploss
    print(f"energy balance: Pin={pin:.6e}  Ploss={ploss:.6e}  rel={rel*100:.4f}%")
    assert rel < 5e-3


def test_two_wire_force() -> None:
    a, D = 0.005, 0.04
    disks = [Disk(-D / 2, 0, a, 10), Disk(D / 2, 0, a, 11)]
    mesh = mesh_disks(disks, 8 * D, lc_surface=a / 10, lc_far=a * 0.8, grade_distance=D)
    mats = MaterialTable({10: COPPER, 11: COPPER, AIR_TAG: AIR})
    sol = solve(mesh, mats, _pair_groups(), SimulationConfig(50.0))
    fx, fy = maxwell_force(sol, (-D / 2, 0.0), 2.5 * a, n_samples=720)
    f_analytic = -MU0 / (4.0 * math.pi * D)  # repulsive (-x), unit peak currents
    rel = abs(fx - f_analytic) / abs(f_analytic)
    print(f"force: Fx={fx:.4e} analytic={f_analytic:.4e} err={rel*100:.2f}%  Fy={fy:.2e}")
    assert rel < 0.04
    assert abs(fy) < 0.03 * abs(f_analytic)


def test_shielding_increases_with_frequency() -> None:
    a, off = 0.004, 0.012
    ain, aout = 0.025, 0.035
    disks = [
        Disk(-off, 0, a, 10),
        Disk(off, 0, a, 11),
        Disk(0, 0, ain, AIR_TAG),
        Disk(0, 0, aout, STEEL_TAG),
    ]
    mesh = mesh_disks(disks, 6 * aout, lc_surface=a / 6, lc_far=aout / 8, grade_distance=aout)
    mats_steel = MaterialTable({10: COPPER, 11: COPPER, AIR_TAG: AIR, STEEL_TAG: STEEL})
    mats_air = MaterialTable({10: COPPER, 11: COPPER, AIR_TAG: AIR, STEEL_TAG: AIR})
    groups = _pair_groups()
    r_sample = 1.5 * aout
    se = []
    for f in (10.0, 1e3, 1e5):
        cfg = SimulationConfig(f)
        sh = solve(mesh, mats_steel, groups, cfg)
        un = solve(mesh, mats_air, groups, cfg)
        se.append(shielding_effectiveness(sh, un, r_sample))
    print(f"SE(10Hz)={se[0]:.1f}  SE(1kHz)={se[1]:.1f}  SE(100kHz)={se[2]:.1f} dB")
    assert se[0] > 5.0  # magnetostatic shielding from mu_r=200 steel
    assert se[1] > se[0] and se[2] > se[1]  # eddy shielding strengthens with f


def _make_field_figure() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    from emsim.post.fields import nodal_B

    a, off = 0.005, 0.01
    ain, aout = 0.03, 0.04
    disks = [
        Disk(-off, 0, a, 10),
        Disk(off, 0, a, 11),
        Disk(0, 0, ain, AIR_TAG),
        Disk(0, 0, aout, STEEL_TAG),
    ]
    mesh = mesh_disks(disks, 4 * aout, lc_surface=a / 8, lc_far=aout / 10, grade_distance=2 * aout)
    mats = MaterialTable({10: COPPER, 11: COPPER, AIR_TAG: AIR, STEEL_TAG: STEEL})
    sol = solve(mesh, mats, _pair_groups(), SimulationConfig(2000.0))
    nB = nodal_B(sol)
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris)
    bmag = np.sqrt(np.abs(nB[:, 0]) ** 2 + np.abs(nB[:, 1]) ** 2)
    fig, ax = plt.subplots(figsize=(6, 5))
    tcf = ax.tricontourf(tri, bmag, levels=50, cmap="inferno")
    th = np.linspace(0, 2 * np.pi, 200)
    for rr in (ain, aout):
        ax.plot(rr * np.cos(th), rr * np.sin(th), "c-", lw=0.6)
    ax.set_aspect("equal")
    ax.set_title("|B| with steel enclosure (flux containment), 2 kHz")
    fig.colorbar(tcf, ax=ax)
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "enclosure_Bfield.png", dpi=130, bbox_inches="tight")
    print(f"figure -> {out / 'enclosure_Bfield.png'}")


if __name__ == "__main__":
    test_enclosure_energy_balance()
    test_two_wire_force()
    test_shielding_increases_with_frequency()
    _make_field_figure()
    print("M4 gates passed.")
