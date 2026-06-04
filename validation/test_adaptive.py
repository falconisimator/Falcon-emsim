r"""Milestone 5 validation: adaptive ZZ refinement.

Starting from a near-uniform *coarse* mesh that badly under-resolves the skin
layer of a strong-skin round wire (delta/a = 0.1), successive ZZ-driven adapt
cycles must:

1. drive the R_AC/R_DC error down to < 2 % (from ~9 % on the coarse mesh);
2. reduce the global energy-norm error estimate monotonically;
3. achieve a large net accuracy gain (final error << initial error),

demonstrating that the estimator steers refinement into the skin layer where
loss accuracy is set.
"""

from __future__ import annotations

import math
from pathlib import Path

from emsim.config import MU0, SimulationConfig
from emsim.materials import AIR, COPPER, MaterialTable
from emsim.mesh.gmsh_backend import AIR_TAG, WIRE_TAG, Disk
from emsim.fem.constraints import ParallelGroup
from emsim.refine import adaptive_solve
from emsim.post.losses import group_losses
from emsim.analytic.round_wire import rac_rdc_ratio

A = 0.01
SIGMA = COPPER.sigma


def _run(n_iters: int):
    delta = 0.1 * A
    omega = 2.0 / (MU0 * SIGMA * delta**2)
    cfg = SimulationConfig(omega / (2.0 * math.pi))
    ana = rac_rdc_ratio(A, SIGMA, cfg.omega)
    hist = adaptive_solve(
        [Disk(0, 0, A, WIRE_TAG)],
        10 * A,
        MaterialTable({WIRE_TAG: COPPER, AIR_TAG: AIR}),
        [ParallelGroup("w", (WIRE_TAG,), 1.0 + 0j)],
        cfg,
        init_lc=A / 3,
        h_min=delta / 4,
        h_max=A / 2,
        n_iters=n_iters,
        theta=0.7,
    )
    errs = [abs(group_losses(s.solution)[0].rac_rdc - ana) / ana for s in hist]
    return hist, errs


def test_adaptive_reduces_loss_error() -> None:
    hist, errs = _run(3)
    etas = [s.eta_global for s in hist]
    for i, (s, e) in enumerate(zip(hist, errs)):
        print(f"iter {i}: nodes={s.num_nodes:6d} err={e*100:+.2f}% eta={s.eta_global:.3e}")
    assert errs[0] > 0.05  # coarse mesh is poor
    assert errs[-1] < 0.02  # adapted mesh is accurate
    assert errs[-1] < errs[0] / 3.0  # large net improvement
    # global error estimate decreases monotonically
    assert all(etas[i + 1] < etas[i] for i in range(len(etas) - 1))


def _make_figure() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    hist, errs = _run(4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([s.num_nodes for s in hist], [e * 100 for e in errs], "o-")
    axes[0].set_xlabel("nodes")
    axes[0].set_ylabel("R_AC/R_DC error (%)")
    axes[0].set_title("Adaptive convergence (delta/a=0.1)")
    axes[0].grid(alpha=0.3)
    m = hist[-1].solution.mesh
    tri = Triangulation(m.nodes[:, 0], m.nodes[:, 1], m.tris)
    axes[1].triplot(tri, lw=0.2, color="C0")
    axes[1].set_aspect("equal")
    axes[1].set_xlim(-1.3 * A, 1.3 * A)
    axes[1].set_ylim(-1.3 * A, 1.3 * A)
    axes[1].set_title("Adapted mesh (skin-layer refinement)")
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "adaptive_refinement.png", dpi=130, bbox_inches="tight")
    print(f"figure -> {out / 'adaptive_refinement.png'}")


if __name__ == "__main__":
    _make_figure()
    print("M5 done.")
