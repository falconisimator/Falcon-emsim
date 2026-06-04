"""Closed-form reference solutions used to validate the FEM solver."""

from emsim.analytic.round_wire import (
    rac_rdc_ratio,
    rac_rdc_ratio_kelvin,
    internal_impedance,
    r_dc,
)

__all__ = [
    "rac_rdc_ratio",
    "rac_rdc_ratio_kelvin",
    "internal_impedance",
    "r_dc",
]
