"""Direct sparse complex solve of the bordered system.

S = K + j omega M is complex *symmetric* (not Hermitian), and the bordered
arrowhead system is unsymmetric, so a general sparse LU is the right tool --
not Cholesky/CG. SciPy's ``splu`` factors the CSC matrix once and can be
reused across multiple right-hand sides.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def lu_solve(matrix: sp.csc_matrix, rhs: np.ndarray) -> np.ndarray:
    """Solve ``matrix @ x = rhs`` via sparse LU. Returns the complex solution."""
    A = matrix.tocsc()
    lu = spla.splu(A)
    return lu.solve(rhs)
