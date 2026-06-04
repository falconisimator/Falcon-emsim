r"""Ohmic-loss post-processing: per-conductor loss and R_ac / R_dc.

Time-average ohmic loss per unit length over a region:

    P = integral |J_z|^2 / (2 sigma) dA      [W/m]

For a parallel group carrying prescribed amplitude current ``I``,
``P = 1/2 |I|^2 R_ac``, hence ``R_ac = 2 P / |I|^2``.  The DC resistance per
unit length is ``R_dc = 1 / g_g`` with ``g_g = integral sigma`` over the group.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emsim.post.fields import current_density_at_quadrature
from emsim.results import Solution


@dataclass
class GroupLoss:
    """Loss summary for one parallel group."""

    name: str
    power: float  # time-average ohmic loss, W/m
    current_prescribed: complex
    current_recovered: complex  # int J dA, should match prescribed
    r_ac: float  # Ohm/m
    r_dc: float  # Ohm/m

    @property
    def rac_rdc(self) -> float:
        return self.r_ac / self.r_dc


def _quad_quantities(solution: Solution):
    Jz, weights, sigma = current_density_at_quadrature(solution)
    return Jz, weights, sigma


def _loss_integrand(Jz: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """|J|^2 / (2 sigma) per quadrature point; zero where sigma == 0 (air)."""
    two_sigma = 2.0 * sigma[:, None]
    return np.divide(
        np.abs(Jz) ** 2, two_sigma, out=np.zeros_like(np.abs(Jz)), where=two_sigma > 0
    )


def region_current(solution: Solution, tags: set[int]) -> complex:
    """Total complex current I = integral J_z dA over the given region tags.

    Use this to measure how current splits between the sub-regions of a single
    parallel group (e.g. between two parallel bars sharing one terminal, or
    between the core and shell of a composite bar).
    """
    Jz, weights, _ = _quad_quantities(solution)
    mask = solution.mesh.tris_in_regions(tags)
    return complex((Jz * weights)[mask].sum())


def region_ohmic_loss(solution: Solution, tags: set[int]) -> float:
    """Total time-average ohmic loss (W/m) over the given region tags."""
    Jz, weights, sigma = _quad_quantities(solution)
    mask = solution.mesh.tris_in_regions(tags)
    per_elem = (_loss_integrand(Jz, sigma) * weights).sum(axis=1)
    return float(per_elem[mask].sum())


def total_ohmic_loss(solution: Solution) -> float:
    """Total time-average ohmic loss over all conducting regions (W/m).

    Includes passive eddy regions such as a steel enclosure (which carry no
    terminal current but dissipate induced eddy currents).
    """
    Jz, weights, sigma = _quad_quantities(solution)
    return float((_loss_integrand(Jz, sigma) * weights).sum())


def input_power(solution: Solution) -> float:
    """Real input power per unit length, 1/2 Re(sum_g u_g I_g*) [W/m].

    For a magnetoquasistatic system this must equal the total ohmic loss
    (conductors + enclosure); the check validates the enclosure eddy loss.
    """
    total = 0.0 + 0.0j
    for gi, group in enumerate(solution.groups):
        total += solution.u[gi] * np.conj(group.current)
    return float(0.5 * total.real)


def group_losses(solution: Solution) -> list[GroupLoss]:
    """Per-group loss, recovered current and R_ac / R_dc."""
    Jz, weights, sigma = _quad_quantities(solution)
    mesh = solution.mesh
    results: list[GroupLoss] = []
    integrand = _loss_integrand(Jz, sigma)
    for gi, group in enumerate(solution.groups):
        mask = mesh.tris_in_regions(group.tag_set)
        power = float((integrand * weights)[mask].sum())
        i_rec = complex((Jz * weights)[mask].sum())
        i_pre = group.current
        amp2 = abs(i_pre) ** 2
        r_ac = 2.0 * power / amp2 if amp2 > 0 else float("nan")
        r_dc = 1.0 / solution.group_conductance[gi]
        results.append(
            GroupLoss(
                name=group.name,
                power=power,
                current_prescribed=i_pre,
                current_recovered=i_rec,
                r_ac=r_ac,
                r_dc=r_dc,
            )
        )
    return results
