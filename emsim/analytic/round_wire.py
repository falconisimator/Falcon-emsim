r"""Analytic skin-effect solution for an isolated solid round wire.

For a round wire of radius ``a``, conductivity ``sigma`` and permeability
``mu`` carrying a total current at angular frequency ``omega``, the classical
internal impedance per unit length is

    Z_int = k / (2 pi a sigma) * J0(k a) / J1(k a),     k = sqrt(-j omega mu sigma)

with skin depth ``delta = sqrt(2 / (omega mu sigma))``.  The AC/DC resistance
ratio is

    R_ac / R_dc = Re{ (k a / 2) * J0(k a) / J1(k a) }.

This module provides the ratio computed two independent ways -- via the
complex Bessel functions ``J0/J1`` and via the real Kelvin functions
``ber/bei`` -- so the two can be cross-checked.

Phasor convention: e^{+j omega t}.  Under this convention the internal
reactance is positive (inductive) and the resistance ratio is >= 1.
"""

from __future__ import annotations

import cmath
import math

from scipy import special

from emsim.config import MU0


def r_dc(a: float, sigma: float) -> float:
    """DC resistance per unit length, R_dc = 1 / (sigma * pi * a^2)  [Ohm/m]."""
    return 1.0 / (sigma * math.pi * a * a)


def internal_impedance(
    a: float, sigma: float, omega: float, mu_r: float = 1.0
) -> complex:
    """Internal impedance per unit length Z_int [Ohm/m].

    At ``omega == 0`` this is the real DC resistance.
    """
    if omega == 0.0:
        return complex(r_dc(a, sigma), 0.0)
    mu = MU0 * mu_r
    k = cmath.sqrt(-1j * omega * mu * sigma)
    q = k * a
    return (k / (2.0 * math.pi * a * sigma)) * (special.jv(0, q) / special.jv(1, q))


def rac_rdc_ratio(a: float, sigma: float, omega: float, mu_r: float = 1.0) -> float:
    """R_ac / R_dc for a round wire, via complex Bessel functions J0/J1."""
    if omega == 0.0:
        return 1.0
    mu = MU0 * mu_r
    k = cmath.sqrt(-1j * omega * mu * sigma)
    q = k * a
    ratio = 0.5 * q * special.jv(0, q) / special.jv(1, q)
    return float(ratio.real)


def rac_rdc_ratio_kelvin(
    a: float, sigma: float, omega: float, mu_r: float = 1.0
) -> float:
    r"""R_ac / R_dc via the real Kelvin functions ber/bei.

    With xi = sqrt(2) * a / delta = a * sqrt(omega mu sigma),

        R_ac/R_dc = (xi/2) *
            (ber(xi) bei'(xi) - bei(xi) ber'(xi))
            / (ber'(xi)^2 + bei'(xi)^2).

    This is mathematically identical to :func:`rac_rdc_ratio` and is used as
    an independent cross-check (and is numerically more robust at large xi).
    """
    if omega == 0.0:
        return 1.0
    mu = MU0 * mu_r
    xi = a * math.sqrt(omega * mu * sigma)
    ber = special.ber(xi)
    bei = special.bei(xi)
    berp = special.berp(xi)
    beip = special.beip(xi)
    num = ber * beip - bei * berp
    den = berp * berp + beip * beip
    return 0.5 * xi * num / den
