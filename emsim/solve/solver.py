"""High-level solve: mesh + materials + groups + frequency -> Solution."""

from __future__ import annotations

from emsim.config import SimulationConfig
from emsim.fem import assembly
from emsim.fem.constraints import GroupSystem, ParallelGroup
from emsim.materials import MaterialTable
from emsim.mesh.mesh import Mesh
from emsim.results import Solution
from emsim.solve.linear import lu_solve


def solve(
    mesh: Mesh,
    materials: MaterialTable,
    groups: GroupSystem | list[ParallelGroup],
    config: SimulationConfig,
) -> Solution:
    """Assemble and solve the bordered MQS system for one frequency."""
    if isinstance(groups, list):
        gs = GroupSystem()
        for g in groups:
            gs.add(g)
        groups = gs

    system = assembly.assemble(mesh, materials, groups, config.omega)
    x = lu_solve(system.matrix, system.rhs)

    n = system.num_nodes
    a = x[:n]
    u = x[n:]
    return Solution(
        a=a,
        u=u,
        group_conductance=system.group_conductance,
        group_order=system.group_order,
        mesh=mesh,
        materials=materials,
        groups=groups,
        config=config,
    )
