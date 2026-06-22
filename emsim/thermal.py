"""Ballpark steady-state thermal solve, reusing the EM loss field.

Cheap conjugate-heat-transfer model for a 2D busbar cross-section (per unit
length):

* Conduction is solved **only in the solids** (conductors + passive metal);
  the air is replaced by surface convection + radiation boundary conditions
  (meshing air as a conductor would be both wrong and expensive).
* A single constant ambient air temperature (forced axial airflow enters at
  ambient).
* Temperature-corrected resistance: the EM loss is scaled by
  ``1 + alpha*(T - Tref)`` per element so heating raises losses -- WITHOUT
  re-running the EM solve (the distribution change is second order).
* Radiation is linearised about the current temperature and folded into the
  same fixed-point as the resistance scaling (a few iterations).

Phase 1 here radiates/convects every surface to ambient (open exposure). The
inter-conductor view-factor exchange (lines + W/m between busbars) layers on
top of this in a later pass.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

SIGMA_SB = 5.670e-8          # Stefan-Boltzmann, W/m^2/K^4
K_AIR = 0.026               # air conductivity, W/m/K (~300 K)
NU_AIR = 1.57e-5            # air kinematic viscosity, m^2/s
PR_AIR = 0.71              # air Prandtl number
H_NATURAL = 6.0            # natural-convection floor, W/m^2/K
T_REF = 20.0              # temperature the EM conductivities were taken at, deg C

# Typical installed-busbar thermal properties keyed by material name.
THERMAL_PROPS = {
    "copper":    {"k": 400.0, "alpha": 0.0039, "eps": 0.5},
    "aluminium": {"k": 237.0, "alpha": 0.0040, "eps": 0.2},
    "steel":     {"k": 50.0,  "alpha": 0.0050, "eps": 0.7},
    "stainless": {"k": 16.0,  "alpha": 0.0010, "eps": 0.4},
}


def _forced_h(u: float, length: float) -> float:
    """Forced-convection coefficient for parallel (axial) flow over a surface
    (laminar flat-plate average Nusselt). Ballpark."""
    if u <= 0 or length <= 0:
        return 0.0
    re = u * length / NU_AIR
    nu = 0.664 * re ** 0.5 * PR_AIR ** (1.0 / 3.0)
    return nu * K_AIR / length


def solve_thermal(state: dict, u: float, t_amb: float, max_iter: int = 8) -> dict:
    """Solve the steady temperature field. ``state`` is cached by emsim.web from
    the last EM solve; ``u`` is the axial airflow (m/s), ``t_amb`` ambient degC."""
    nodes = state["nodes"]                  # (Nn,2)
    tris = state["tris"]                    # (Ne,3)
    region = state["region"]               # (Ne,)
    areas = state["areas"]                 # (Ne,)
    ploss = state["ploss"]                 # (Ne,) W/m^3 at T_REF
    rprops = state["region_props"]         # tag -> dict(k, alpha, eps, Lchar, name, group)
    cond_tags = state["cond_tags"]         # set of solid region tags

    Nn = nodes.shape[0]
    solid = np.isin(region, list(cond_tags))
    se = tris[solid]                       # (M,3) solid elements
    sreg = region[solid]
    sploss = ploss[solid]
    used = np.unique(se)
    remap = np.full(Nn, -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    Ns = used.size
    eidx = remap[se]                       # (M,3) solid-local node indices

    # P1 gradients + area per solid element
    p = nodes
    x = p[se, 0]; y = p[se, 1]             # (M,3)
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    A = np.abs(det) / 2.0
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1) / det[:, None]
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1) / det[:, None]

    kel = np.array([rprops[t]["k"] for t in sreg])
    alel = np.array([rprops[t]["alpha"] for t in sreg])

    # constant conduction stiffness  Ke_ij = k*A*(bi bj + ci cj)
    Ke = kel[:, None, None] * A[:, None, None] * (b[:, :, None] * b[:, None, :]
                                                  + c[:, :, None] * c[:, None, :])
    R = np.broadcast_to(eidx[:, :, None], (eidx.shape[0], 3, 3))
    C = np.broadcast_to(eidx[:, None, :], (eidx.shape[0], 3, 3))
    Kc = sp.coo_matrix((Ke.ravel(), (R.ravel(), C.ravel())), shape=(Ns, Ns)).tocsr()

    # surface edges = solid edges shared by only one solid element (solid/air)
    edges = np.concatenate([se[:, [0, 1]], se[:, [1, 2]], se[:, [2, 0]]], axis=0)
    owner = np.tile(np.arange(se.shape[0]), 3)
    es = np.sort(edges, axis=1)
    key = es[:, 0].astype(np.int64) * Nn + es[:, 1]
    uniq, first, counts = np.unique(key, return_index=True, return_counts=True)
    surf = first[counts == 1]
    sa, sb = es[surf, 0], es[surf, 1]
    sowner = owner[surf]
    ra, rb = remap[sa], remap[sb]
    Le = np.hypot(p[sa, 0] - p[sb, 0], p[sa, 1] - p[sb, 1])
    eps_e = np.array([rprops[sreg[o]]["eps"] for o in sowner])
    hconv_e = np.array([H_NATURAL + _forced_h(u, rprops[sreg[o]]["Lchar"]) for o in sowner])

    tak = t_amb + 273.15
    T = np.full(Ns, t_amb)
    for _ in range(max_iter):
        # temperature-corrected heat source (W/m per element -> nodal)
        telem = T[eidx].mean(axis=1)
        qe = sploss * A * (1.0 + alel * (telem - T_REF))
        f = np.zeros(Ns)
        np.add.at(f, eidx.ravel(), np.repeat(qe / 3.0, 3))

        # convection + linearised radiation, each surface edge to ambient
        tedge = 0.5 * (T[ra] + T[rb])
        tk = tedge + 273.15
        hrad = eps_e * SIGMA_SB * (tk * tk + tak * tak) * (tk + tak)
        h = hconv_e + hrad
        m = h * Le / 6.0
        Kb = sp.coo_matrix(
            (np.concatenate([2 * m, m, m, 2 * m]),
             (np.concatenate([ra, ra, rb, rb]), np.concatenate([ra, rb, ra, rb]))),
            shape=(Ns, Ns)).tocsr()
        fb = h * t_amb * Le / 2.0
        np.add.at(f, ra, fb)
        np.add.at(f, rb, fb)

        # SPD system (conduction + positive convection) -> CG with a Jacobi
        # preconditioner. Iterative on purpose: SuperLU (splu/spsolve) hangs on
        # this matrix in Pyodide, and CG is bounded-time regardless.
        Amat = (Kc + Kb).tocsr()
        Mj = sp.diags(1.0 / Amat.diagonal())
        Tn, _info = spla.cg(Amat, f, x0=T, rtol=1e-8, atol=1e-10, maxiter=5000, M=Mj)
        if np.max(np.abs(Tn - T)) < 0.05:
            T = Tn
            break
        T = Tn

    # final fluxes for the energy split (W/m)
    tedge = 0.5 * (T[ra] + T[rb])
    tk = tedge + 273.15
    hrad = eps_e * SIGMA_SB * (tk * tk + tak * tak) * (tk + tak)
    dT = tedge - t_amb
    p_conv = float(np.sum(hconv_e * dT * Le))
    p_rad = float(np.sum(hrad * dT * Le))
    telem = T[eidx].mean(axis=1)
    p_total = float(np.sum(sploss * A * (1.0 + alel * (telem - T_REF))))

    conductors = []
    for tag, pr in rprops.items():
        em = sreg == tag
        if not em.any():
            continue
        tn = T[np.unique(eidx[em])]
        conductors.append({"name": pr["name"], "group": pr.get("group"),
                           "Tmean": float(tn.mean()), "Tmax": float(tn.max())})

    T_full = np.full(Nn, t_amb)
    T_full[used] = T
    return {
        "T": T_full.tolist(),
        "Tamb": float(t_amb),
        "Tmax": float(T.max()),
        "Tmean": float(T.mean()),
        "P_total": p_total,
        "P_conv": p_conv,
        "P_rad": p_rad,
        "conductors": conductors,
    }
