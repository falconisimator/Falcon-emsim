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
RHO_AIR = 1.16             # air density, kg/m^3 (~300 K)
CP_AIR = 1005.0            # air specific heat, J/kg/K
MU_AIR = 1.85e-5           # air dynamic viscosity, Pa*s (~300 K)
DUCT_LEN = 1.0             # axial length the air flows over, m (matches the 3D view)
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


def _view_factors(mid, nrm, Le, Pa, Pb):
    """2D diffuse view factors F[i,j] between surface segments (crossed-strings
    differential form) with line-of-sight occlusion. F[i,j] = fraction of i's
    hemisphere subtended by j; rows sum to <=1 (rest = sky). O(N^2) + an
    occluder loop; geometry-only so the caller caches it across re-solves."""
    N = len(mid)
    r = mid[None, :, :] - mid[:, None, :]          # i -> j
    dist = np.hypot(r[:, :, 0], r[:, :, 1]) + 1e-12
    ci = (r[:, :, 0] * nrm[:, None, 0] + r[:, :, 1] * nrm[:, None, 1]) / dist
    cj = (-r[:, :, 0] * nrm[None, :, 0] - r[:, :, 1] * nrm[None, :, 1]) / dist
    F = np.clip(ci, 0, None) * np.clip(cj, 0, None) / (2.0 * dist) * Le[None, :]
    np.fill_diagonal(F, 0.0)
    # occlusion: zero F[i,j] if the ray mid_i->mid_j crosses another segment k
    Px, Py = mid[:, 0], mid[:, 1]
    dX, dY = Px[None, :] - Px[:, None], Py[None, :] - Py[:, None]
    eps = 1e-3                                       # shrink ray off its endpoints
    P1x, P1y = Px[:, None] + eps * dX, Py[:, None] + eps * dY
    P2x, P2y = Px[None, :] - eps * dX, Py[None, :] - eps * dY
    occ = range(N) if N <= 500 else range(0, N, 2)   # subsample occluders if huge
    for k in occ:
        ax, ay, bx, by = Pa[k, 0], Pa[k, 1], Pb[k, 0], Pb[k, 1]
        d1 = (by - ay) * (P1x - ax) - (bx - ax) * (P1y - ay)
        d2 = (by - ay) * (P2x - ax) - (bx - ax) * (P2y - ay)
        d3 = (P2y - P1y) * (ax - P1x) - (P2x - P1x) * (ay - P1y)
        d4 = (P2y - P1y) * (bx - P1x) - (P2x - P1x) * (by - P1y)
        cross = (d1 * d2 < 0) & (d3 * d4 < 0)
        cross[k, :] = False; cross[:, k] = False
        F[cross] = 0.0
    skyfrac = np.clip(1.0 - F.sum(axis=1), 0.0, 1.0)
    return F, skyfrac


def _airflow_unit(nodes, tris, region, cond_tags, solid_nodes):
    """Fully-developed laminar axial duct flow: solve nabla^2 w = -1 on the air
    region with no-slip (w=0) on conductor surfaces and the outer boundary. The
    unit profile w_hat is geometry-only (cached); actual velocity scales to the
    nominal mean. Returns (w_full[Nn], mean_w_hat) ; w_full is 0 outside air."""
    Nn = nodes.shape[0]
    air = ~np.isin(region, list(cond_tags))
    ae = tris[air]
    x, y = nodes[ae, 0], nodes[ae, 1]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    A2 = np.abs(det) / 2.0
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], axis=1) / det[:, None]
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], axis=1) / det[:, None]
    anodes = np.unique(ae)
    ramap = np.full(Nn, -1, dtype=np.int64)
    ramap[anodes] = np.arange(anodes.size)
    na = anodes.size
    ei = ramap[ae]
    ke = A2[:, None, None] * (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :])
    R = np.broadcast_to(ei[:, :, None], (ei.shape[0], 3, 3))
    C = np.broadcast_to(ei[:, None, :], (ei.shape[0], 3, 3))
    K = sp.coo_matrix((ke.ravel(), (R.ravel(), C.ravel())), shape=(na, na)).tocsr()
    f = np.zeros(na)
    np.add.at(f, ei.ravel(), np.repeat(A2 / 3.0, 3))
    # Dirichlet w=0: conductor-interface nodes + outer-boundary nodes
    rad = np.hypot(nodes[:, 0], nodes[:, 1])
    rmax = rad[anodes].max()
    dirich = np.zeros(Nn, dtype=bool)
    dirich[np.intersect1d(anodes, solid_nodes)] = True
    dirich[anodes[rad[anodes] >= 0.999 * rmax]] = True
    fa = ramap[anodes[~dirich[anodes]]]                  # free air-node locals
    w = np.zeros(na)
    if fa.size:
        Kff = K[fa][:, fa]
        Mj = sp.diags(1.0 / Kff.diagonal())
        wf, _ = spla.cg(Kff, f[fa], rtol=1e-8, atol=1e-12, maxiter=5000, M=Mj)
        w[fa] = wf
    welem = w[ei].mean(axis=1)
    mean_w = float((welem * A2).sum() / A2.sum()) if A2.sum() > 0 else 0.0
    w_full = np.zeros(Nn)
    w_full[anodes] = w
    return w_full, mean_w


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
    forced_e = np.array([_forced_h(u, rprops[sreg[o]]["Lchar"]) for o in sowner])
    # per-surface busbar label (group for phases, busbar id for passive metal)
    def _label(tag):
        pr = rprops[tag]
        return pr["group"] if pr.get("group") is not None else pr.get("busbar", "encl")
    lab_e = np.array([_label(sreg[o]) for o in sowner], dtype=object)

    # radiation view factors (geometry only) -> cached across airflow/ambient changes
    vf = state.get("_vf")
    if vf is None or vf[0].shape[0] != len(sa):
        Pa, Pb = nodes[sa], nodes[sb]
        mid = 0.5 * (Pa + Pb)
        dd = Pb - Pa
        nrm = np.stack([dd[:, 1], -dd[:, 0]], axis=1)
        nrm /= (np.hypot(nrm[:, 0], nrm[:, 1])[:, None] + 1e-30)
        third = se[sowner].sum(axis=1) - sa - sb           # outward = away from element
        cen = (Pa + Pb + nodes[third]) / 3.0
        flip = ((mid - cen) * nrm).sum(axis=1) < 0
        nrm[flip] *= -1.0
        F, skyfrac = _view_factors(mid, nrm, Le, Pa, Pb)
        state["_vf"] = vf = (F, skyfrac, mid)
    F, skyfrac, mid = vf

    # confined surfaces (low sky fraction) lose forced airflow -> natural only
    hconv_e = H_NATURAL + skyfrac * forced_e
    tak = t_amb + 273.15

    T = np.full(Ns, t_amb)
    for _ in range(max_iter):
        telem = T[eidx].mean(axis=1)
        qe = sploss * A * (1.0 + alel * (telem - T_REF))
        f = np.zeros(Ns)
        np.add.at(f, eidx.ravel(), np.repeat(qe / 3.0, 3))

        tedge = 0.5 * (T[ra] + T[rb])
        tk = tedge + 273.15
        # radiation to AMBIENT is weighted by the sky fraction (rest is exchange)
        hrad = eps_e * SIGMA_SB * skyfrac * (tk * tk + tak * tak) * (tk + tak)
        h = hconv_e + hrad
        m = h * Le / 6.0
        Kb = sp.coo_matrix(
            (np.concatenate([2 * m, m, m, 2 * m]),
             (np.concatenate([ra, ra, rb, rb]), np.concatenate([ra, rb, ra, rb]))),
            shape=(Ns, Ns)).tocsr()
        fb = h * t_amb * Le / 2.0
        np.add.at(f, ra, fb)
        np.add.at(f, rb, fb)

        # inter-surface IR exchange (lagged): net flux leaving edge i toward the
        # bars it sees. q_i = eps_i*sigma*sum_j eps_j F_ij (Ti^4 - Tj^4) [W/m^2].
        t4 = tk ** 4
        qx = eps_e * SIGMA_SB * (F * eps_e[None, :] * (t4[:, None] - t4[None, :])).sum(axis=1)
        fx = qx * Le / 2.0
        np.add.at(f, ra, -fx)
        np.add.at(f, rb, -fx)

        Amat = (Kc + Kb).tocsr()
        Mj = sp.diags(1.0 / Amat.diagonal())
        Tn, _info = spla.cg(Amat, f, x0=T, rtol=1e-8, atol=1e-10, maxiter=5000, M=Mj)
        if np.max(np.abs(Tn - T)) < 0.05:
            T = Tn
            break
        T = Tn

    # final fluxes / energy split (W/m). Inter-surface exchange nets to ~0
    # globally, so heat leaves only by convection + radiation-to-ambient.
    tedge = 0.5 * (T[ra] + T[rb])
    tk = tedge + 273.15
    t4 = tk ** 4
    hrad = eps_e * SIGMA_SB * skyfrac * (tk * tk + tak * tak) * (tk + tak)
    dT = tedge - t_amb
    p_conv = float(np.sum(hconv_e * dT * Le))
    p_rad = float(np.sum(hrad * dT * Le))
    telem = T[eidx].mean(axis=1)
    p_total = float(np.sum(sploss * A * (1.0 + alel * (telem - T_REF))))

    # axial airflow gradient: the cooling air warms as it flows over the length.
    # mdot = rho*u*A_flow, A_flow ~ the in-plane air gap (bbox of solids minus the
    # conductor cross-section); dT_air = P_conv*L/(mdot*cp). Tight-duct estimate.
    sx, sy = nodes[used, 0], nodes[used, 1]
    a_box = (sx.max() - sx.min()) * (sy.max() - sy.min())
    a_flow = max(3.0 * a_box - float(A.sum()), 1e-5)   # loose duct ~3x the bar bbox
    mdot = RHO_AIR * u * a_flow
    dT_air = (p_conv * DUCT_LEN / (mdot * CP_AIR)) if mdot > 0 else 0.0
    dT_air = min(dT_air, max(0.0, float(T.max()) - t_amb))   # air can't exceed the surface
    bar_max = float(T.max())

    # axial airspeed field (Poisson duct flow); unit profile is geometry-only so
    # cache it and just scale to the nominal mean velocity u. dP = fan/pumping drop.
    air = state.get("_air")
    if air is None:
        air = _airflow_unit(nodes, tris, region, cond_tags, used)
        state["_air"] = air
    w_full, mean_w = air
    if mean_w > 0 and u > 0:
        vair = (u / mean_w) * w_full
        dP = MU_AIR * u * DUCT_LEN / mean_w
    else:
        vair = np.zeros(Nn)
        dP = 0.0
    vmax = float(vair.max())

    # net radiative power transmitted between busbars (W/m). Q_ij = eps_i eps_j
    # sigma Le_i F_ij (Ti^4 - Tj^4); aggregate by surface label.
    Qfull = eps_e[:, None] * eps_e[None, :] * SIGMA_SB * F * (t4[:, None] - t4[None, :]) * Le[:, None]
    labels = sorted({str(x) for x in lab_e})
    lidx = {lb: np.where(lab_e.astype(str) == lb)[0] for lb in labels}
    ir_pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            q = float(Qfull[np.ix_(lidx[labels[i]], lidx[labels[j]])].sum())
            if abs(q) > 0.02:
                # report net flow hot -> cold
                a, b, w = (labels[i], labels[j], q) if q >= 0 else (labels[j], labels[i], -q)
                ir_pairs.append({"from": a, "to": b, "watts": w})
    ir_pairs.sort(key=lambda d: -d["watts"])

    # IR "rays" for the view: strongest cross-busbar surface exchanges
    lab_s = lab_e.astype(str)
    iu, ju = np.triu_indices(len(sa), k=1)
    aq = np.abs(Qfull[iu, ju])
    keep = (aq > 1e-4) & (lab_s[iu] != lab_s[ju])
    ii, jj, ww = iu[keep], ju[keep], aq[keep]
    order = np.argsort(-ww)[:300]
    ir_wmax = float(ww[order[0]]) if order.size else 0.0
    ir_lines = [[float(mid[i, 0]), float(mid[i, 1]), float(mid[j, 0]), float(mid[j, 1]), float(w)]
                for i, j, w in zip(ii[order], jj[order], ww[order])]

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
        "ir_pairs": ir_pairs,
        "ir_lines": ir_lines,
        "ir_wmax": ir_wmax,
        "conductors": conductors,
        "duct_len": DUCT_LEN,
        "dT_air": float(dT_air),                 # air rise inlet->outlet over the length
        "air_out": float(t_amb + dT_air),
        "bar_max_out": float(bar_max + dT_air),  # outlet (hot end) busbar max
        "vair": vair.tolist(),                   # per-node axial airspeed, m/s
        "vmax": vmax,
        "dP": float(dP),                         # pressure drop over the length, Pa
    }
