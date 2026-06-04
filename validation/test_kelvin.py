r"""Milestone 3 validation: Kelvin open boundary.

A balanced go/return wire pair (net-zero current, so the exterior field decays
and the open problem is well posed). The loop inductance is computed from the
total magnetic energy -- which, with the Kelvin disk, includes the *exact*
exterior energy by conformal invariance -- and compared to the analytic
two-wire value.

Gates:
1. **Accuracy** -- with the interface at R = 4 D the Kelvin result is within
   1.5 % of analytic.
2. **Compactness** -- at a moderate R = 2 D the Kelvin error is much smaller
   than a plain Dirichlet-box truncation at the same R (which forces the field
   to zero too close in and underestimates the inductance).
3. **Insensitivity** -- in the converged range the result barely changes with
   the truncation radius R.

Run directly to also write a convergence figure.
"""

from __future__ import annotations

from pathlib import Path

from emsim.config import SimulationConfig
from emsim.materials import AIR, COPPER, MaterialTable
from emsim.mesh.gmsh_backend import AIR_TAG, KELVIN_TAG, Disk, mesh_disks
from emsim.mesh.kelvin import open_mesh
from emsim.fem.constraints import ParallelGroup
from emsim.solve.solver import solve
from emsim.post.energy import loop_inductance
from emsim.analytic.two_wire import dc_loop_inductance

A = 0.005
D = 0.04
DC = SimulationConfig(0.0)
MATS = MaterialTable({10: COPPER, 11: COPPER, AIR_TAG: AIR, KELVIN_TAG: AIR})
GROUPS = [
    ParallelGroup("go", (10,), 1.0 + 0j),
    ParallelGroup("return", (11,), -1.0 + 0j),
]
L_ANALYTIC = dc_loop_inductance(A, D)


def _physical(R: float):
    return mesh_disks(
        [Disk(-D / 2, 0, A, 10), Disk(D / 2, 0, A, 11)],
        R,
        lc_surface=A / 8,
        lc_far=A * 1.5,
        grade_distance=D,
    )


def inductance(Rf: float, kelvin: bool) -> float:
    R = Rf * D
    phys = _physical(R)
    mesh = open_mesh(phys, center_size=R / 12) if kelvin else phys
    return loop_inductance(solve(mesh, MATS, GROUPS, DC), 1.0)


def _err(L: float) -> float:
    return (L - L_ANALYTIC) / L_ANALYTIC


def test_kelvin_accuracy() -> None:
    err = _err(inductance(4.0, kelvin=True))
    print(f"Kelvin @4D: err={err*100:+.2f}%")
    assert abs(err) < 0.015


def test_kelvin_beats_dirichlet_box() -> None:
    err_k = _err(inductance(2.0, kelvin=True))
    err_box = _err(inductance(2.0, kelvin=False))
    print(f"@2D  Kelvin err={err_k*100:+.2f}%  box err={err_box*100:+.2f}%")
    assert abs(err_k) < 0.6 * abs(err_box)


def test_kelvin_insensitive_to_R() -> None:
    l4 = inductance(4.0, kelvin=True)
    l6 = inductance(6.0, kelvin=True)
    rel = abs(l4 - l6) / l4
    print(f"R-insensitivity: L(4D)={l4:.4e} L(6D)={l6:.4e} rel diff={rel*100:.2f}%")
    assert rel < 0.01


def _run_and_plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rfs = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
    box, kel = [], []
    for rf in rfs:
        box.append(_err(inductance(rf, kelvin=False)) * 100)
        kel.append(_err(inductance(rf, kelvin=True)) * 100)
        print(f"R={rf:.1f}D  box={box[-1]:+.2f}%  kelvin={kel[-1]:+.2f}%")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(rfs, box, "s--", label="Dirichlet box")
    ax.plot(rfs, kel, "o-", label="Kelvin open")
    ax.set_xlabel("interface radius R / D")
    ax.set_ylabel("inductance error (%)")
    ax.set_title("Open boundary: Kelvin vs Dirichlet truncation")
    ax.grid(alpha=0.3)
    ax.legend()
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "kelvin_convergence.png", dpi=130, bbox_inches="tight")
    print(f"figure -> {out / 'kelvin_convergence.png'}")


if __name__ == "__main__":
    _run_and_plot()
