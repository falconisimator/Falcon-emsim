"""Post-processing: fields, losses, forces, EMI."""

from emsim.post.losses import (
    group_losses,
    input_power,
    region_current,
    region_ohmic_loss,
    total_ohmic_loss,
    GroupLoss,
)
from emsim.post.energy import loop_inductance, magnetic_energy
from emsim.post.forces import maxwell_force
from emsim.post.emi import leakage, shielding_effectiveness

__all__ = [
    "group_losses",
    "input_power",
    "region_current",
    "region_ohmic_loss",
    "total_ohmic_loss",
    "GroupLoss",
    "loop_inductance",
    "magnetic_energy",
    "maxwell_force",
    "leakage",
    "shielding_effectiveness",
]
