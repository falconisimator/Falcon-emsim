"""Matplotlib result views: field maps and sweep plots.

Uses a non-interactive backend by default so figures can be saved headlessly.
"""

from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

from emsim.post import fields  # noqa: E402
from emsim.results import Solution  # noqa: E402


def _triangulation(solution: Solution) -> Triangulation:
    m = solution.mesh
    return Triangulation(m.nodes[:, 0], m.nodes[:, 1], m.tris)


def plot_nodal_field(
    solution: Solution, values: np.ndarray, title: str, ax=None, cmap="viridis"
):
    """Filled contour of a nodal scalar field (e.g. |A_z|)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    tri = _triangulation(solution)
    tcf = ax.tricontourf(tri, values, levels=40, cmap=cmap)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.figure.colorbar(tcf, ax=ax)
    return ax


def plot_element_field(
    solution: Solution, values: np.ndarray, title: str, ax=None, cmap="magma"
):
    """Flat-shaded per-element field (e.g. |B|, |J|)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    tri = _triangulation(solution)
    tpc = ax.tripcolor(tri, facecolors=values, cmap=cmap, shading="flat")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.figure.colorbar(tpc, ax=ax)
    return ax


def field_overview(solution: Solution):
    """A 1x3 panel of |A_z|, |B|, |J| for a quick look at a solution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_nodal_field(solution, np.abs(solution.a), "|A_z| (Wb/m)", ax=axes[0])

    Bmag = fields.element_B_magnitude(solution)
    plot_element_field(solution, Bmag, "|B| (T)", ax=axes[1])

    Jz, _, _ = fields.current_density_at_quadrature(solution)
    Jmag = np.abs(Jz).mean(axis=1)  # element-average over quadrature points
    plot_element_field(solution, Jmag, "|J_z| (A/m^2)", ax=axes[2])
    fig.tight_layout()
    return fig


def plot_rac_sweep(delta_over_a, fem_ratio, analytic_ratio, ax=None):
    """FEM vs analytic R_ac/R_dc against delta/a (log-x)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    order = np.argsort(delta_over_a)
    x = np.asarray(delta_over_a)[order]
    ax.plot(x, np.asarray(analytic_ratio)[order], "k-", label="analytic (Bessel)")
    ax.plot(x, np.asarray(fem_ratio)[order], "o--", color="C1", label="FEM (P1)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\delta / a$")
    ax.set_ylabel(r"$R_{AC}/R_{DC}$")
    ax.set_title("Round-wire skin effect: FEM vs analytic")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return ax
