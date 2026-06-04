r"""Milestone 6 validation: second-order (P2) elements.

On the round wire (delta/a = 0.3) the P2 elements (assembled through the same
quadrature kernel as P1) must:

1. be markedly more accurate than P1 at the *same* element size;
2. converge to the analytic R_AC/R_DC as the mesh is refined;
3. show a higher convergence order than P1.

(The asymptotic rate is partly limited by the straight-sided approximation of
the circular boundary, but P2 is consistently more accurate per element.)
"""

from __future__ import annotations

import math
from pathlib import Path

from emsim.config import MU0, SimulationConfig
from emsim.materials import AIR, COPPER, MaterialTable
from emsim.mesh.gmsh_backend import AIR_TAG, WIRE_TAG, mesh_round_wire
from emsim.fem.constraints import ParallelGroup
from emsim.solve.solver import solve
from emsim.post.losses import group_losses
from emsim.analytic.round_wire import rac_rdc_ratio

A = 0.01
SIGMA = COPPER.sigma
MATS = MaterialTable({WIRE_TAG: COPPER, AIR_TAG: AIR})
GROUPS = [ParallelGroup("w", (WIRE_TAG,), 1.0 + 0j)]


def _cfg():
    delta = 0.3 * A
    omega = 2.0 / (MU0 * SIGMA * delta**2)
    return SimulationConfig(omega / (2.0 * math.pi))


def _err(order: int, lc: float, cfg, ana: float) -> tuple[float, int]:
    mesh = mesh_round_wire(A, 8 * A, lc_surface=lc, lc_far=lc, grade_distance=8 * A, order=order)
    r = group_losses(solve(mesh, MATS, GROUPS, cfg))[0].rac_rdc
    return abs(r - ana) / ana, mesh.num_nodes


def test_p2_more_accurate_and_higher_order() -> None:
    cfg = _cfg()
    ana = rac_rdc_ratio(A, SIGMA, cfg.omega)
    lcs = [A / 2, A / 4]
    e1 = [_err(1, lc, cfg, ana) for lc in lcs]
    e2 = [_err(2, lc, cfg, ana) for lc in lcs]
    for tag, e in (("P1", e1), ("P2", e2)):
        for lc, (err, dof) in zip(lcs, e):
            print(f"{tag} lc=a/{int(round(A/lc))} dof={dof:6d} err={err*100:+.3f}%")
    # (1) P2 more accurate than P1 at the same element size
    assert e2[0][0] < e1[0][0]
    assert e2[1][0] < e1[1][0]
    # (2) P2 converges
    assert e2[1][0] < e2[0][0]
    assert e2[1][0] < 0.01
    # (3) higher convergence order
    order_p1 = math.log2(e1[0][0] / e1[1][0])
    order_p2 = math.log2(e2[0][0] / e2[1][0])
    print(f"convergence order: P1={order_p1:.2f}  P2={order_p2:.2f}")
    assert order_p2 > order_p1


def _make_figure() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = _cfg()
    ana = rac_rdc_ratio(A, SIGMA, cfg.omega)
    lcs = [A / 2, A / 4, A / 8]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for order, marker in ((1, "o-"), (2, "s-")):
        dofs, errs = [], []
        for lc in lcs:
            err, dof = _err(order, lc, cfg, ana)
            dofs.append(dof)
            errs.append(err * 100)
        ax.loglog(dofs, errs, marker, label=f"P{order}")
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("R_AC/R_DC error (%)")
    ax.set_title("P1 vs P2 convergence (round wire, delta/a=0.3)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "p2_convergence.png", dpi=130, bbox_inches="tight")
    print(f"figure -> {out / 'p2_convergence.png'}")


if __name__ == "__main__":
    _make_figure()
    print("M6 done.")
