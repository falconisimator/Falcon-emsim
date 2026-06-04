"""emsim - 2D magnetoquasistatic FEM simulator for busbars.

Complex-phasor formulation, single frequency per run, linear materials.
Solves for the complex magnetic vector potential A_z(x, y) plus one bordered
unknown (per-unit-length voltage gradient V_dot/L) per parallel group.

All quantities are SI and *per unit length* in the out-of-plane (z) direction:
resistance is Ohm/m, loss is W/m, force is N/m, and the prescribed group
current is the total current (A) through the conductor cross-section.
"""

from emsim.config import SimulationConfig
from emsim.materials import Material, MaterialTable
from emsim.fem.constraints import ParallelGroup
from emsim.results import Solution

__all__ = [
    "SimulationConfig",
    "Material",
    "MaterialTable",
    "ParallelGroup",
    "Solution",
]

__version__ = "0.1.0"
