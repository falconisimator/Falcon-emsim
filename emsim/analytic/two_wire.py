r"""Analytic inductance of a two-wire transmission line (per unit length).

Two parallel round wires of radius ``a`` with centre-to-centre spacing ``D``,
carrying equal and opposite currents.  External inductance:

    L_ext = (mu0 / pi) * arccosh(D / (2 a))

(For D >> a this tends to (mu0/pi) ln(D/a).)  At DC the current is uniform and
each wire adds an internal inductance mu0/(8 pi); for the pair:

    L_dc = (mu0 / pi) * ( arccosh(D / (2 a)) + 1/4 )
"""

from __future__ import annotations

import math

from emsim.config import MU0


def external_inductance(a: float, D: float) -> float:
    """External loop inductance per unit length [H/m]."""
    return (MU0 / math.pi) * math.acosh(D / (2.0 * a))


def dc_loop_inductance(a: float, D: float) -> float:
    """DC loop inductance per unit length (external + uniform-current internal)."""
    return (MU0 / math.pi) * (math.acosh(D / (2.0 * a)) + 0.25)
