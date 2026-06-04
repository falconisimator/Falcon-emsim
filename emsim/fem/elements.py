r"""Linear (P1) triangular element kernels and quadrature.

For a P1 triangle with vertices :math:`(x_i, y_i)` and area :math:`\Delta`,
the shape functions :math:`N_i` are linear and their gradients are constant:

    grad N_i = (b_i, c_i) / (2 Delta)

with ``b = [y2-y3, y3-y1, y1-y2]`` and ``c = [x3-x2, x1-x3, x2-x1]``.

Element matrices used by the assembler:

* stiffness  K^e_ij = (1/mu) * (b_i b_j + c_i c_j) / (4 Delta)
* mass       M^e_ij = sigma * (Delta/12) * [[2,1,1],[1,2,1],[1,1,2]]
* load       b^e_i  = sigma * (Delta/3)            (column for the group)

The quadrature rule is exposed separately so the same assembly loop can later
serve the radius-dependent Kelvin coefficient (numerically integrated) and
second-order elements.
"""

from __future__ import annotations

import numpy as np

# 3-point triangle quadrature (barycentric midpoints), exact for degree 2.
# Sufficient for the quadratic integrand |J|^2 on P1 fields and for the
# consistent mass matrix.
_TRI_QUAD_BARY = np.array(
    [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ]
)
_TRI_QUAD_W = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])  # weights sum to 1

# Consistent-mass reference matrix (integral of N_i N_j over a unit-area tri = 1/12 off, 1/6 diag).
_MASS_REF = np.array(
    [
        [2.0, 1.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
    ]
) / 12.0


def shape_gradients(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Constant P1 shape-function gradients and areas for many triangles.

    Parameters
    ----------
    verts:
        ``(M, 3, 2)`` vertex coordinates.

    Returns
    -------
    grads:
        ``(M, 3, 2)`` gradient of each shape function (1/m).
    area:
        ``(M,)`` triangle areas (m^2).
    """
    x = verts[:, :, 0]
    y = verts[:, :, 1]
    # b_i, c_i with cyclic indices (i, i+1, i+2)
    b = np.stack(
        [y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1
    )
    c = np.stack(
        [x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1
    )
    # signed area * 2
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (
        x[:, 2] - x[:, 0]
    ) * (y[:, 1] - y[:, 0])
    area = 0.5 * np.abs(det)
    two_area = np.abs(det)[:, None]
    grads = np.stack([b / two_area, c / two_area], axis=2)  # (M,3,2)
    return grads, area


def stiffness_matrices(grads: np.ndarray, area: np.ndarray, inv_mu: np.ndarray) -> np.ndarray:
    """Element stiffness matrices, shape ``(M, 3, 3)``.

    ``inv_mu`` is the per-element reluctivity ``1/mu`` (shape ``(M,)``).
    """
    # K_ij = inv_mu * (grad N_i . grad N_j) * area
    gg = np.einsum("mik,mjk->mij", grads, grads)  # (M,3,3)
    return (inv_mu * area)[:, None, None] * gg


def mass_matrices(area: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Element (consistent) mass matrices sigma * int N_i N_j, shape ``(M,3,3)``."""
    return (sigma * area)[:, None, None] * _MASS_REF[None, :, :]


def load_vectors(area: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Element load vectors sigma * int N_i = sigma * area / 3, shape ``(M, 3)``."""
    return (sigma * area / 3.0)[:, None] * np.ones((1, 3))


def quadrature_points(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical quadrature points, weights and shape values for triangles.

    Returns
    -------
    pts:
        ``(M, Q, 2)`` quadrature point coordinates.
    weights:
        ``(M, Q)`` integration weights (already including the element area,
        so ``sum_q weights == area``).
    shape_vals:
        ``(Q, 3)`` shape-function values at each quadrature point (same for
        every triangle in reference barycentric coordinates).
    """
    grads_unused = None  # documented: gradients are constant, fetched elsewhere
    del grads_unused
    # area via the same cross product
    x = verts[:, :, 0]
    y = verts[:, :, 1]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (
        x[:, 2] - x[:, 0]
    ) * (y[:, 1] - y[:, 0])
    area = 0.5 * np.abs(det)
    # physical points = sum_i bary_i * vertex_i
    pts = np.einsum("qi,mid->mqd", _TRI_QUAD_BARY, verts)  # (M,Q,2)
    weights = area[:, None] * _TRI_QUAD_W[None, :]  # (M,Q)
    shape_vals = _TRI_QUAD_BARY.copy()  # (Q,3): N_i == bary_i for P1
    return pts, weights, shape_vals
