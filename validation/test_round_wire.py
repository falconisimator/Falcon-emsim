r"""Milestone 1 validation: single round wire vs analytic skin effect.

Sweeps delta/a from the DC-like regime down into strong skin effect and
compares the FEM R_ac/R_dc against the closed-form Bessel-function solution.

Gate (per the project plan):
* R_ac/R_dc within < 3 % of analytic for delta/a >= 0.3 on a mesh graded to
  ~ delta/3 at the conductor surface;
* the delta/a = 0.1 case is *reported* (not asserted) to demonstrate the
  documented under-resolution of P1 elements in a thin skin layer.

Run directly (``python -m validation.test_round_wire``) to also write the
comparison figure to ``validation/figures/round_wire_sweep.png``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from emsim.config import MU0, SimulationConfig
from emsim.materials import AIR, COPPER, MaterialTable
from emsim.mesh.gmsh_backend import AIR_TAG, WIRE_TAG, mesh_round_wire
from emsim.fem.constraints import ParallelGroup
from emsim.solve.solver import solve
from emsim.post.losses import group_losses
from emsim.analytic.round_wire import rac_rdc_ratio

A = 0.01  # wire radius (m)
SIGMA = COPPER.sigma
MU = MU0  # non-magnetic
R_OUTER = 10.0 * A

# delta/a target -> assertion tolerance (None => report only)
SWEEP = {
    10.0: 0.03,
    3.0: 0.03,
    1.0: 0.03,
    0.5: 0.03,
    0.3: 0.03,
    0.1: None,  # demonstrate degradation; do not assert
}


def frequency_for_ratio(delta_over_a: float) -> float:
    """Frequency giving a target delta/a:  delta = sqrt(2/(w mu sigma))."""
    delta = delta_over_a * A
    omega = 2.0 / (MU * SIGMA * delta * delta)
    return omega / (2.0 * math.pi)


def fem_ratio(delta_over_a: float) -> tuple[float, float]:
    """Return (fem_ratio, analytic_ratio) for one delta/a point."""
    freq = frequency_for_ratio(delta_over_a)
    cfg = SimulationConfig(freq)
    delta = cfg.skin_depth(SIGMA)
    # Resolve the skin layer: element size ~ delta/4 at the surface, capped so
    # the wire is well resolved even when delta > a (low frequency).
    lc_surface = min(delta / 4.0, A / 8.0)
    grade = max(3.0 * delta, 0.3 * A)
    mesh = mesh_round_wire(
        a=A, R=R_OUTER, lc_surface=lc_surface, lc_far=A / 2.0, grade_distance=grade
    )
    mats = MaterialTable({WIRE_TAG: COPPER, AIR_TAG: AIR})
    group = ParallelGroup("wire", (WIRE_TAG,), 1.0 + 0j)
    sol = solve(mesh, mats, [group], cfg)
    gl = group_losses(sol)[0]
    # sanity: the recovered current must match the prescribed 1 A
    assert abs(gl.current_recovered - 1.0) < 1e-6, gl.current_recovered
    return gl.rac_rdc, rac_rdc_ratio(A, SIGMA, cfg.omega)


@pytest.mark.parametrize("delta_over_a,tol", list(SWEEP.items()))
def test_round_wire_rac(delta_over_a: float, tol: float | None) -> None:
    fem, ana = fem_ratio(delta_over_a)
    rel = abs(fem - ana) / ana
    print(
        f"delta/a={delta_over_a:5.2f}  FEM={fem:9.4f}  analytic={ana:9.4f}  "
        f"err={rel*100:6.2f}%"
    )
    if tol is not None:
        assert rel < tol, f"delta/a={delta_over_a}: rel err {rel*100:.2f}% exceeds {tol*100:.0f}%"


def _run_and_plot() -> None:
    from emsim.plotting import plot_rac_sweep
    import matplotlib.pyplot as plt

    ratios = sorted(SWEEP, reverse=True)
    fem_vals, ana_vals = [], []
    for r in ratios:
        fem, ana = fem_ratio(r)
        fem_vals.append(fem)
        ana_vals.append(ana)
        rel = abs(fem - ana) / ana
        print(f"delta/a={r:5.2f}  FEM={fem:9.4f}  analytic={ana:9.4f}  err={rel*100:6.2f}%")
    ax = plot_rac_sweep(ratios, fem_vals, ana_vals)
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)
    ax.figure.savefig(out / "round_wire_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"figure -> {out / 'round_wire_sweep.png'}")


if __name__ == "__main__":
    _run_and_plot()
