r"""Reference triangle shape functions, gradients and quadrature.

Supports P1 (3-node) and P2 (6-node) Lagrange triangles, written in
barycentric coordinates ``L = (L0, L1, L2)`` with ``L0 + L1 + L2 = 1``.

Node ordering matches Gmsh's 2nd-order triangle (type 9): vertices 0,1,2 then
edge-midside nodes on edges (0-1), (1-2), (2-0).

Element gradients are obtained from ``dN/dx = sum_k (dN/dL_k) grad L_k`` where
``grad L_k`` are the constant P1 barycentric gradients of the (straight-sided)
element. For P1 the shape-function gradients are constant; for P2 they vary
across the element, hence the quadrature loop.
"""

from __future__ import annotations

import numpy as np

# Triangle quadrature rules in barycentric coords; weights sum to 1.
# Degree-2 (3 points) and degree-4 (6 points) Strang/Dunavant rules.
_QUAD = {
    2: (
        np.array(
            [[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]
        ),
        np.array([1 / 3, 1 / 3, 1 / 3]),
    ),
    4: (
        # Dunavant degree-4, 6-point rule
        np.array(
            [
                [0.108103018168070, 0.445948490915965, 0.445948490915965],
                [0.445948490915965, 0.108103018168070, 0.445948490915965],
                [0.445948490915965, 0.445948490915965, 0.108103018168070],
                [0.816847572980459, 0.091576213509771, 0.091576213509771],
                [0.091576213509771, 0.816847572980459, 0.091576213509771],
                [0.091576213509771, 0.091576213509771, 0.816847572980459],
            ]
        ),
        np.array(
            [
                0.223381589678011,
                0.223381589678011,
                0.223381589678011,
                0.109951743655322,
                0.109951743655322,
                0.109951743655322,
            ]
        ),
    ),
}


def quadrature(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (bary points ``(Q,3)``, weights ``(Q,)`` summing to 1)."""
    return _QUAD[degree]


def shape_values(order: int, bary: np.ndarray) -> np.ndarray:
    """Shape-function values at barycentric points, shape ``(Q, K)``."""
    L0, L1, L2 = bary[:, 0], bary[:, 1], bary[:, 2]
    if order == 1:
        return np.stack([L0, L1, L2], axis=1)
    if order == 2:
        return np.stack(
            [
                L0 * (2 * L0 - 1),
                L1 * (2 * L1 - 1),
                L2 * (2 * L2 - 1),
                4 * L0 * L1,
                4 * L1 * L2,
                4 * L2 * L0,
            ],
            axis=1,
        )
    raise ValueError(f"unsupported order {order}")


def shape_grads_bary(order: int, bary: np.ndarray) -> np.ndarray:
    """dN/dL_k at barycentric points, shape ``(Q, K, 3)``."""
    L0, L1, L2 = bary[:, 0], bary[:, 1], bary[:, 2]
    Q = bary.shape[0]
    if order == 1:
        g = np.zeros((Q, 3, 3))
        g[:, 0, 0] = 1.0
        g[:, 1, 1] = 1.0
        g[:, 2, 2] = 1.0
        return g
    if order == 2:
        g = np.zeros((Q, 6, 3))
        # vertices: d/dL_i [L_i(2L_i-1)] = 4L_i-1 (only wrt own L)
        g[:, 0, 0] = 4 * L0 - 1
        g[:, 1, 1] = 4 * L1 - 1
        g[:, 2, 2] = 4 * L2 - 1
        # edge 0-1: 4 L0 L1
        g[:, 3, 0] = 4 * L1
        g[:, 3, 1] = 4 * L0
        # edge 1-2: 4 L1 L2
        g[:, 4, 1] = 4 * L2
        g[:, 4, 2] = 4 * L1
        # edge 2-0: 4 L2 L0
        g[:, 5, 2] = 4 * L0
        g[:, 5, 0] = 4 * L2
        return g
    raise ValueError(f"unsupported order {order}")


def nodes_per_element(order: int) -> int:
    return 3 if order == 1 else 6
