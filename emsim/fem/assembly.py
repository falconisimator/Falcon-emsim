r"""Global assembly of the bordered magnetoquasistatic system.

The field operator is the complex-symmetric matrix

    S = K + j omega M

assembled from the element stiffness ``K`` (reluctivity 1/mu) and mass
``M`` (conductivity sigma).  Each parallel group ``g`` adds one bordered
unknown ``u_g = V_dot_g / L`` (the per-unit-length voltage gradient) coupled
through the load column ``b_g`` and the self-conductance ``g_g``:

    [ S        -B   ] [ a ]   [ 0 ]
    [ -jw/gg Bᵀ  I  ] [ u ] = [ I_g / g_g ]

The bottom (current-constraint) block-row is scaled by ``1/g_g`` so its
diagonal is unity; this keeps the arrowhead border well conditioned against
the large magnitude of the field block (sigma ~ 1e7, 1/mu0 ~ 8e5).

The field block has a constant null space (A_z defined up to a constant) when
the boundary is pure Neumann, so a Dirichlet pin (A_z = 0 on the outer
boundary) is applied to fix the gauge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from emsim.fem import elements, shapes
from emsim.fem.constraints import GroupSystem
from emsim.materials import MaterialTable
from emsim.mesh.mesh import Mesh


def _quad_degree(order: int) -> int:
    """Quadrature degree: exact for the stiffness/mass integrands per order."""
    return 2 if order == 1 else 4


@dataclass
class AssembledSystem:
    """The assembled bordered system and the metadata to interpret its solution."""

    matrix: sp.csc_matrix  # (N+G, N+G) complex
    rhs: np.ndarray  # (N+G,) complex
    num_nodes: int
    group_conductance: np.ndarray  # (G,) real, g_g = int sigma over group
    group_order: list[str]  # group names in column order


def element_material_arrays(
    mesh: Mesh, materials: MaterialTable
) -> tuple[np.ndarray, np.ndarray]:
    """Per-element reluctivity (1/mu) and conductivity (sigma) arrays."""
    tags = np.unique(mesh.region_tag)
    inv_mu = np.empty(mesh.num_tris, dtype=np.float64)
    sigma = np.empty(mesh.num_tris, dtype=np.float64)
    for tag in tags:
        mat = materials.get(int(tag))
        mask = mesh.region_tag == tag
        inv_mu[mask] = 1.0 / mat.mu
        sigma[mask] = mat.sigma
    return inv_mu, sigma


def _element_matrices(
    mesh: Mesh, inv_mu: np.ndarray, sigma: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Element stiffness, mass and load via quadrature (P1 or P2).

    Returns ``(ke, me, load)`` of shapes ``(M,K,K)``, ``(M,K,K)``, ``(M,K)``.
    """
    order = mesh.order
    verts = mesh.triangle_vertices()  # (M,3,2) vertices only
    gradL, area = elements.shape_gradients(verts)  # P1 barycentric grads (M,3,2)
    bary, w = shapes.quadrature(_quad_degree(order))
    N = shapes.shape_values(order, bary)  # (Q,K)
    dNdL = shapes.shape_grads_bary(order, bary)  # (Q,K,3)
    # physical shape-function gradients at each quad point: (M,Q,K,2)
    gradN = np.einsum("qkl,mld->mqkd", dNdL, gradL)
    kk = np.einsum("q,mqka,mqla->mkl", w, gradN, gradN)  # (M,K,K)
    ke = (inv_mu * area)[:, None, None] * kk
    nn = np.einsum("q,qk,ql->kl", w, N, N)  # (K,K) reference
    me = (sigma * area)[:, None, None] * nn[None, :, :]
    ln = np.einsum("q,qk->k", w, N)  # (K,)
    load = (sigma * area)[:, None] * ln[None, :]
    return ke, me, load


def assemble_field_matrix(
    mesh: Mesh, inv_mu: np.ndarray, sigma: np.ndarray, omega: float
) -> sp.csr_matrix:
    """Assemble S = K + j omega M as a complex CSR matrix (P1 or P2)."""
    ke, me, _ = _element_matrices(mesh, inv_mu, sigma)
    se = ke + 1j * omega * me  # (M,K,K) complex
    tris = mesh.tris
    k = tris.shape[1]
    rows = np.repeat(tris, k, axis=1).reshape(-1)
    cols = np.tile(tris, (1, k)).reshape(-1)
    data = se.reshape(-1)
    n = mesh.num_nodes
    return sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def assemble_group_loads(
    mesh: Mesh, sigma: np.ndarray, groups: GroupSystem
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Build the load column matrix B (N x G) and conductances g_g (G,).

    ``b_g,i = int_{group g} sigma N_i``  and  ``g_g = int_{group g} sigma``.
    """
    verts = mesh.triangle_vertices()
    _, area = elements.shape_gradients(verts)
    _, _, load = _element_matrices(mesh, np.ones_like(sigma), sigma)  # (M,K)
    n = mesh.num_nodes
    cols = []
    g_self = np.zeros(len(groups), dtype=np.float64)
    for gi, group in enumerate(groups):
        mask = mesh.tris_in_regions(group.tag_set)
        bvec = np.zeros(n, dtype=np.float64)
        # scatter-add the per-node loads of the selected elements
        sel_tris = mesh.tris[mask]
        sel_load = load[mask]
        np.add.at(bvec, sel_tris.reshape(-1), sel_load.reshape(-1))
        cols.append(bvec)
        g_self[gi] = float((sigma[mask] * area[mask]).sum())
    B = sp.csc_matrix(np.column_stack(cols)) if cols else sp.csc_matrix((n, 0))
    return B, g_self


def apply_dirichlet_pin(S: sp.csr_matrix, pinned: np.ndarray) -> sp.csr_matrix:
    """Zero the rows and columns of pinned nodes and set their diagonal to 1.

    Enforces A_z = 0 on the pinned nodes (the homogeneous Dirichlet gauge
    pin). Because the pinned value is zero, no right-hand-side correction is
    needed.
    """
    n = S.shape[0]
    if pinned.size == 0:
        return S
    keep = np.ones(n, dtype=np.float64)
    keep[pinned] = 0.0
    Dk = sp.diags(keep)
    pin_diag = np.zeros(n, dtype=np.float64)
    pin_diag[pinned] = 1.0
    S_bc = Dk @ S @ Dk + sp.diags(pin_diag)
    return S_bc.tocsr()


def assemble(
    mesh: Mesh,
    materials: MaterialTable,
    groups: GroupSystem,
    omega: float,
) -> AssembledSystem:
    """Assemble the full bordered system ready for a direct solve."""
    inv_mu, sigma = element_material_arrays(mesh, materials)
    S = assemble_field_matrix(mesh, inv_mu, sigma, omega)
    S = apply_dirichlet_pin(S, mesh.boundary_nodes)
    B, g_self = assemble_group_loads(mesh, sigma, groups)

    n = mesh.num_nodes
    ng = len(groups)
    currents = np.array([g.current for g in groups], dtype=np.complex128)

    if ng == 0:
        A = S.tocsc()
        rhs = np.zeros(n, dtype=np.complex128)
        return AssembledSystem(A, rhs, n, g_self, [])

    # Top-right coupling block: -B  (N x G)
    top_right = -B
    # Bottom-left current-constraint block, scaled by 1/g_g:  -(j omega / g_g) Bᵀ
    inv_g = sp.diags(1.0 / g_self)
    bottom_left = (-1j * omega) * (inv_g @ B.T)
    bottom_right = sp.identity(ng, format="csc", dtype=np.complex128)

    A = sp.bmat(
        [[S, top_right], [bottom_left, bottom_right]], format="csc", dtype=np.complex128
    )
    rhs = np.concatenate([np.zeros(n, dtype=np.complex128), currents / g_self])
    group_order = [g.name for g in groups]
    return AssembledSystem(A, rhs, n, g_self, group_order)
