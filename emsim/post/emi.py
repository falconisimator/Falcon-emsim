r"""EMI post-processing: external leakage and shielding effectiveness.

These operate on the recovered nodal B field, sampled at points (typically on
a contour outside the enclosure).
"""

from __future__ import annotations

import numpy as np

from emsim.post.fields import nodal_B, sample_B
from emsim.results import Solution


def b_magnitude_at(solution: Solution, pts: np.ndarray) -> np.ndarray:
    """|B| (phasor amplitude) at sample points, shape ``(P,)``."""
    B = sample_B(solution, pts)
    return np.sqrt(np.abs(B[:, 0]) ** 2 + np.abs(B[:, 1]) ** 2)


def external_contour(radius: float, n: int = 360) -> np.ndarray:
    """A circular sampling contour of given radius about the origin."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def leakage(solution: Solution, radius: float, n: int = 360) -> float:
    """Peak |B| leaking onto a circle of given radius (outside the enclosure)."""
    return float(np.nanmax(b_magnitude_at(solution, external_contour(radius, n))))


def shielding_effectiveness(
    shielded: Solution, unshielded: Solution, radius: float, n: int = 360
) -> float:
    """SE = 20 log10(|B|_unshielded / |B|_shielded) [dB], averaged on the contour.

    Both solutions must share the same geometry/excitation, differing only in
    whether the enclosure material is magnetic/conducting (shielded) or air
    (unshielded).
    """
    pts = external_contour(radius, n)
    b_sh = b_magnitude_at(shielded, pts)
    b_un = b_magnitude_at(unshielded, pts)
    # area/contour-averaged magnitudes
    mean_sh = float(np.nanmean(b_sh))
    mean_un = float(np.nanmean(b_un))
    return 20.0 * np.log10(mean_un / mean_sh)
