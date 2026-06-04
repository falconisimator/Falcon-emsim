"""The :class:`Solution` object returned by a solve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emsim.config import SimulationConfig
from emsim.fem.constraints import GroupSystem
from emsim.materials import MaterialTable
from emsim.mesh.mesh import Mesh


@dataclass
class Solution:
    """Result of a magnetoquasistatic solve.

    Attributes
    ----------
    a:
        ``(N,)`` complex nodal magnetic vector potential A_z (Wb/m).
    u:
        ``(G,)`` complex per-group voltage gradient V_dot/L (V/m).
    group_conductance:
        ``(G,)`` real per-group DC self-conductance g_g (S/m).
    group_order:
        Group names in the order matching ``u`` and ``group_conductance``.
    mesh, materials, groups, config:
        The inputs, retained so post-processors are pure functions of the
        solution.
    """

    a: np.ndarray
    u: np.ndarray
    group_conductance: np.ndarray
    group_order: list[str]
    mesh: Mesh
    materials: MaterialTable
    groups: GroupSystem
    config: SimulationConfig

    @property
    def omega(self) -> float:
        return self.config.omega
