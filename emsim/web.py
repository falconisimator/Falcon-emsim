"""Adapter for the browser (Pyodide) build.

The JS editor builds the same scene dictionary that :mod:`emsim.io` uses, hands
it here, and gets back a JSON-able payload of mesh + complex fields + evaluation
results. The browser then renders and animates entirely client-side.

Forces the gmsh-free backend and the Dirichlet open boundary (Kelvin / P2 need
gmsh and stay desktop-only).
"""

from __future__ import annotations

import json

import numpy as np

from emsim.io import scene_from_dict
from emsim.mesh.gmsh_backend import KELVIN_TAG
from emsim.post.fields import element_B, element_Jz


def solve_scene(scene_dict) -> str:
    """Solve a scene (dict or JSON string) in the browser; return results JSON."""
    if isinstance(scene_dict, str):
        scene_dict = json.loads(scene_dict)
    sc = scene_from_dict(scene_dict)
    sc.mesh_backend = "py"          # gmsh-free mesher (numpy + scipy.spatial)
    if sc.boundary == "kelvin":
        sc.boundary = "dirichlet"   # Kelvin mirror-disk needs gmsh
    sc.order = 1                    # P2 needs gmsh midside nodes

    # web-tuned mesh: tighter air domain, but a finer far field than the first
    # cut so the field map isn't visibly faceted (still browser-friendly).
    ext0 = max(abs(c.placement.x) + abs(c.placement.y) + c.shape.bounding_radius()
               for c in sc.conductors)
    min_char = min(c.shape.char_size() for c in sc.conductors)
    sc.domain_radius = 2.3 * ext0
    sc.lc_surface = min_char / 8.0
    sc.lc_far = 0.18 * ext0

    sol = sc.solve()
    res = sc.analyse(sol)
    mesh = sol.mesh
    phys = mesh.region_tag != KELVIN_TAG
    B = element_B(sol)[phys]
    J = element_Jz(sol)[phys]
    # per-element A_z (centroid) for the vector-potential view
    from emsim.fem import shapes
    n_c = shapes.shape_values(mesh.order, np.array([[1 / 3, 1 / 3, 1 / 3]]))[0]
    Az = (sol.a[mesh.tris] @ n_c)[phys]
    # complex average current density per terminal, I_group / A_group. The JS uses
    # it to form J(x,t) / (i(t)/A) -- current density relative to the terminal's
    # average AT THAT INSTANT -- and the steady |J|/avg utilization (period sum).
    from collections import defaultdict

    reg, areas = mesh.region_tag, mesh.areas()
    g_area = defaultdict(float)
    for c in sc.conductors:
        if c.group is not None:
            g_area[c.group] += areas[reg == c.region_tag].sum()
    javg = np.zeros(reg.shape[0], dtype=np.complex128)
    for c in sc.conductors:
        if c.group is not None and g_area[c.group] > 0:
            javg[reg == c.region_tag] = sc.current_for_group(c.group) / g_area[c.group]

    # applied current density of the whole system = total terminal current / total
    # conductor area, and the total loss normalized to it (W/m per A/mm^2).
    total_area = sum(g_area.values())  # m^2
    total_current = sum(abs(sc.current_for_group(g)) for g in g_area)  # A
    j_app = (total_current / total_area) / 1e6 if total_area > 0 else 0.0  # A/mm^2
    # current-INDEPENDENT figure of merit (loss ~ J^2): W/m per (A/mm^2)^2.
    loss_coeff = (res.total_loss / j_app**2) if j_app > 0 else 0.0

    ext = max(abs(c.placement.x) + abs(c.placement.y) + 1.6 * c.shape.bounding_radius()
              for c in sc.conductors)

    payload = {
        "nodes": mesh.nodes.ravel().tolist(),
        "tris": mesh.tris[phys][:, :3].ravel().tolist(),
        "region": mesh.region_tag[phys].tolist(),   # per element, for edge outlines
        "a_re": sol.a.real.tolist(), "a_im": sol.a.imag.tolist(),  # nodal A_z, for flux lines
        "Bx_re": B[:, 0].real.tolist(), "Bx_im": B[:, 0].imag.tolist(),
        "By_re": B[:, 1].real.tolist(), "By_im": B[:, 1].imag.tolist(),
        "J_re": J.real.tolist(), "J_im": J.imag.tolist(),
        "Az_re": Az.real.tolist(), "Az_im": Az.imag.tolist(),
        "javg_re": javg[phys].real.tolist(), "javg_im": javg[phys].imag.tolist(),
        "extent": float(ext),
        "num_nodes": int(mesh.num_nodes),
        "total_loss": float(res.total_loss),
        "applied_density": float(j_app),       # A/mm^2 (total I / total area)
        "loss_coeff": float(loss_coeff),        # W/m per (A/mm^2)^2 (current-independent)
        "conductors": [
            {"name": c.name, "group": c.group,
             "I": float(abs(c.current)),
             "phase": float(np.degrees(np.angle(c.current))),
             "loss": float(c.loss),
             "share": float(c.share) if c.share == c.share else None,
             "fx": (float(c.force[0]) if c.force else None),
             "fy": (float(c.force[1]) if c.force else None)}
            for c in res.conductors
        ],
        "terminals": [
            {"name": t.name, "I": float(abs(t.current)),
             "vgrad": float(abs(t.voltage_gradient)),
             "z_re": float(t.impedance.real), "z_im": float(t.impedance.imag)}
            for t in res.terminals
        ],
    }
    return json.dumps(payload)
