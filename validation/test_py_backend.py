r"""The gmsh-free (Pyodide-compatible) mesher reproduces the round-wire result.

This backend uses only numpy + scipy.spatial, both available in Pyodide, so the
web build can mesh entirely client-side. Accuracy is validated against the same
analytic Bessel oracle as milestone 1.
"""

from __future__ import annotations

import math

import pytest

from emsim.analytic.round_wire import rac_rdc_ratio
from emsim.config import MU0, SimulationConfig
from emsim.fem.constraints import ParallelGroup
from emsim.materials import AIR, COPPER, MaterialTable
from emsim.mesh.gmsh_backend import AIR_TAG, WIRE_TAG
from emsim.mesh.py_backend import mesh_round_wire
from emsim.post.losses import group_losses
from emsim.solve.solver import solve

A = 0.01
SIGMA = COPPER.sigma


@pytest.mark.parametrize("ratio,tol", [(3.0, 0.02), (1.0, 0.02), (0.5, 0.02), (0.3, 0.03)])
def test_py_backend_round_wire(ratio: float, tol: float) -> None:
    delta = ratio * A
    omega = 2.0 / (MU0 * SIGMA * delta**2)
    cfg = SimulationConfig(omega / (2.0 * math.pi))
    lc = min(delta / 4.0, A / 8.0)
    mesh = mesh_round_wire(A, 8 * A, lc_surface=lc, lc_far=A / 2.0, grade_distance=3 * delta)
    mats = MaterialTable({WIRE_TAG: COPPER, AIR_TAG: AIR})
    sol = solve(mesh, mats, [ParallelGroup("w", (WIRE_TAG,), 1.0 + 0j)], cfg)
    gl = group_losses(sol)[0]
    ana = rac_rdc_ratio(A, SIGMA, cfg.omega)
    rel = abs(gl.rac_rdc - ana) / ana
    print(f"delta/a={ratio}: FEM={gl.rac_rdc:.4f} analytic={ana:.4f} err={rel*100:.2f}%")
    assert abs(gl.current_recovered - 1.0) < 1e-6
    assert rel < tol
