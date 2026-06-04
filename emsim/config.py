"""Simulation configuration: frequency and solver options."""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Vacuum permeability (H/m).
MU0 = 4.0e-7 * math.pi


@dataclass
class SimulationConfig:
    """Top-level run settings.

    Attributes
    ----------
    frequency:
        Excitation frequency in Hz. A single frequency is solved per run;
        it may be changed between runs.
    """

    frequency: float

    @property
    def omega(self) -> float:
        """Angular frequency omega = 2 pi f (rad/s)."""
        return 2.0 * math.pi * self.frequency

    def skin_depth(self, sigma: float, mu_r: float = 1.0) -> float:
        """Classical skin depth delta = sqrt(2 / (omega mu sigma)) for a material.

        Returns ``inf`` at DC (frequency == 0).
        """
        if self.frequency == 0.0:
            return math.inf
        mu = MU0 * mu_r
        return math.sqrt(2.0 / (self.omega * mu * sigma))
