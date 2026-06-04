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

    # web-tuned mesh: tighter air domain + coarser far field keeps the WASM solve fast
    ext0 = max(abs(c.placement.x) + abs(c.placement.y) + c.shape.bounding_radius()
               for c in sc.conductors)
    sc.domain_radius = 2.2 * ext0
    sc.lc_far = 0.5 * ext0

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
    ext = max(abs(c.placement.x) + abs(c.placement.y) + 1.6 * c.shape.bounding_radius()
              for c in sc.conductors)

    payload = {
        "nodes": mesh.nodes.ravel().tolist(),
        "tris": mesh.tris[phys][:, :3].ravel().tolist(),
        "Bx_re": B[:, 0].real.tolist(), "Bx_im": B[:, 0].imag.tolist(),
        "By_re": B[:, 1].real.tolist(), "By_im": B[:, 1].imag.tolist(),
        "J_re": J.real.tolist(), "J_im": J.imag.tolist(),
        "Az_re": Az.real.tolist(), "Az_im": Az.imag.tolist(),
        "extent": float(ext),
        "num_nodes": int(mesh.num_nodes),
        "total_loss": float(res.total_loss),
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
