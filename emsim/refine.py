r"""Adaptive mesh refinement driven by the ZZ error estimator.

Repeatedly: solve -> estimate per-element error -> build a target size field
-> remesh the same geometry following that field. The error concentrates at
conductor surfaces / skin layers, so refinement is steered exactly where loss
accuracy is determined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emsim.config import SimulationConfig
from emsim.fem.constraints import ParallelGroup
from emsim.materials import MaterialTable
from emsim.mesh.gmsh_backend import KELVIN_TAG, Disk, mesh_disks
from emsim.post.estimator import global_error, target_node_sizes, zz_error_indicators
from emsim.results import Solution
from emsim.solve.solver import solve


@dataclass
class AdaptiveStep:
    solution: Solution
    eta_global: float
    num_nodes: int


def adaptive_solve(
    disks: list[Disk],
    R: float,
    materials: MaterialTable,
    groups: list[ParallelGroup],
    config: SimulationConfig,
    *,
    init_lc: float,
    h_min: float,
    h_max: float,
    n_iters: int = 4,
    theta: float = 0.7,
) -> list[AdaptiveStep]:
    """Run ``n_iters`` adapt cycles; return the per-iteration history.

    The initial mesh is near-uniform (``init_lc``); subsequent meshes follow the
    estimator-driven size field clamped to ``[h_min, h_max]``.
    """
    mesh = mesh_disks(disks, R, lc_surface=init_lc, lc_far=init_lc, grade_distance=R)
    history: list[AdaptiveStep] = []
    for it in range(n_iters):
        sol = solve(mesh, materials, groups, config)
        eta = zz_error_indicators(sol)
        history.append(AdaptiveStep(sol, global_error(eta), mesh.num_nodes))
        if it == n_iters - 1:
            break
        node_sizes = target_node_sizes(sol, eta, h_min, h_max, theta)
        phys = mesh.region_tag != KELVIN_TAG
        mesh = mesh_disks(
            disks,
            R,
            lc_surface=init_lc,
            lc_far=init_lc,
            background=(mesh.nodes, mesh.tris[phys], node_sizes),
        )
    return history
