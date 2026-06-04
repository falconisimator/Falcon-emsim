r"""Magnetic energy and loop inductance from the solved field.

Magnetic (co-)energy per unit length:

    W = integral |B|^2 / (2 mu) dA      [J/m]

over the whole domain. When the open boundary is modelled with a Kelvin disk,
summing over the Kelvin elements automatically includes the exterior energy
(conformal invariance), so ``W`` is the true total.  The loop inductance of a
balanced (net-zero-current) system follows from ``W = 1/2 L |I|^2``.
"""

from __future__ import annotations

import numpy as np

from emsim.config import MU0
from emsim.fem.assembly import element_material_arrays
from emsim.post.fields import element_B
from emsim.results import Solution


def magnetic_energy(solution: Solution) -> float:
    """Total magnetic energy per unit length W = sum |B|^2 / (2 mu) * area [J/m]."""
    B = element_B(solution)  # (M,2) complex
    b2 = np.abs(B[:, 0]) ** 2 + np.abs(B[:, 1]) ** 2
    inv_mu, _ = element_material_arrays(solution.mesh, solution.materials)
    area = solution.mesh.areas()
    return float(0.5 * np.sum(b2 * inv_mu * area))


def loop_inductance(solution: Solution, current: float) -> float:
    """Loop inductance per unit length L = 2 W / |I|^2 [H/m] for a balanced run."""
    return 2.0 * magnetic_energy(solution) / (current * current)
