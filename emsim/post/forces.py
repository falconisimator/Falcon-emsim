r"""Maxwell-stress-tensor forces per conductor (fault bracing).

Time-average force per unit length on a conductor, from a closed contour Gamma
in air enclosing it:

    F_i = (1 / 2 mu0) * contour_integral[ Re( B_i (B.n)* ) - 1/2 n_i |B|^2 ] dl

using phasor (peak) amplitudes. For two parallel wires carrying peak currents
I1, I2 with centre spacing D, the time-average force per length is
``mu0 Re(I1 I2*) / (4 pi D)`` (attractive for like currents).
"""

from __future__ import annotations

import numpy as np

from emsim.config import MU0
from emsim.post.fields import nodal_B, sample_B
from emsim.results import Solution


def maxwell_force(
    solution: Solution,
    center: tuple[float, float],
    radius: float,
    n_samples: int = 360,
    nB: np.ndarray | None = None,
) -> tuple[float, float]:
    """Time-average force per unit length (Fx, Fy) on the conductor inside Gamma.

    ``center``/``radius`` define a circular contour that must lie in air and
    enclose exactly the target conductor.
    """
    if nB is None:
        nB = nodal_B(solution)
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    nx, ny = np.cos(theta), np.sin(theta)  # outward normal
    pts = np.column_stack([center[0] + radius * nx, center[1] + radius * ny])
    B = sample_B(solution, pts, nB)  # (n,2) complex
    if np.any(np.isnan(B)):
        raise ValueError("force contour leaves the meshed region; check radius/center")
    Bx, By = B[:, 0], B[:, 1]
    Bn = Bx * nx + By * ny
    b2 = np.abs(Bx) ** 2 + np.abs(By) ** 2
    fx = np.real(Bx * np.conj(Bn)) - 0.5 * nx * b2
    fy = np.real(By * np.conj(Bn)) - 0.5 * ny * b2
    dl = radius * (2.0 * np.pi / n_samples)
    return (
        float(fx.sum() * dl / (2.0 * MU0)),
        float(fy.sum() * dl / (2.0 * MU0)),
    )
