"""Save / load a :class:`~emsim.scene.Scene` as JSON.

The on-disk format is a plain JSON document so configurations are portable and
human-readable. Materials are stored by name + properties and remapped to the
built-in presets on load (so the GUI's material picker keeps working).
"""

from __future__ import annotations

import json
from pathlib import Path

from emsim.geometry.model import Conductor
from emsim.geometry.shapes import Circle, Placement, Polygon, Rectangle
from emsim.materials import AIR, ALUMINIUM, COPPER, STEEL, Material
from emsim.scene import Scene

FORMAT_VERSION = 1
_PRESETS = {m.name: m for m in (COPPER, ALUMINIUM, STEEL, AIR)}


def _shape_to_dict(s) -> dict:
    if isinstance(s, Rectangle):
        return {"type": "rect", "width": s.width, "height": s.height}
    if isinstance(s, Circle):
        return {"type": "circle", "radius": s.radius}
    if isinstance(s, Polygon):
        return {"type": "polygon", "points": [list(p) for p in s.points]}
    raise TypeError(f"cannot serialise shape {s!r}")


def _shape_from_dict(d: dict):
    t = d["type"]
    if t == "rect":
        return Rectangle(d["width"], d["height"])
    if t == "circle":
        return Circle(d["radius"])
    if t == "polygon":
        return Polygon([tuple(p) for p in d["points"]])
    raise ValueError(f"unknown shape type {t!r}")


def _material_to_dict(m: Material) -> dict:
    return {"name": m.name, "sigma": m.sigma, "mu_r": m.mu_r}


def _material_from_dict(d: dict) -> Material:
    # prefer a preset (keeps object identity for the GUI material picker)
    preset = _PRESETS.get(d["name"])
    if preset is not None and preset.sigma == d["sigma"] and preset.mu_r == d["mu_r"]:
        return preset
    return Material(d["name"], d["sigma"], d["mu_r"])


def scene_to_dict(scene: Scene) -> dict:
    return {
        "format": FORMAT_VERSION,
        "frequency": scene.frequency,
        "three_phase": scene.three_phase,
        "line_current": scene.line_current,
        "mesh_backend": scene.mesh_backend,
        "boundary": scene.boundary,
        "order": scene.order,
        "domain_radius": scene.domain_radius,
        "lc_surface": scene.lc_surface,
        "lc_far": scene.lc_far,
        "group_currents": {k: [v.real, v.imag] for k, v in scene.group_currents.items()},
        "conductors": [
            {
                "name": c.name,
                "shape": _shape_to_dict(c.shape),
                "placement": [c.placement.x, c.placement.y, c.placement.rotation],
                "material": _material_to_dict(c.material),
                "group": c.group,
                "busbar": c.busbar,
            }
            for c in scene.conductors
        ],
    }


def scene_from_dict(d: dict) -> Scene:
    conductors = []
    for cd in d["conductors"]:
        px, py, rot = cd["placement"]
        conductors.append(
            Conductor(
                name=cd["name"],
                shape=_shape_from_dict(cd["shape"]),
                placement=Placement(px, py, rot),
                material=_material_from_dict(cd["material"]),
                group=cd.get("group"),
                busbar=cd.get("busbar", ""),
            )
        )
    return Scene(
        conductors=conductors,
        group_currents={k: complex(v[0], v[1]) for k, v in d.get("group_currents", {}).items()},
        frequency=d.get("frequency", 50.0),
        domain_radius=d.get("domain_radius", 0.0),
        boundary=d.get("boundary", "kelvin"),
        order=d.get("order", 1),
        lc_surface=d.get("lc_surface", 0.0),
        lc_far=d.get("lc_far", 0.0),
        three_phase=d.get("three_phase", True),
        line_current=d.get("line_current", 1000.0),
        mesh_backend=d.get("mesh_backend", "gmsh"),
    )


def save_scene(scene: Scene, path: str | Path) -> None:
    Path(path).write_text(json.dumps(scene_to_dict(scene), indent=2), encoding="utf-8")


def load_scene(path: str | Path) -> Scene:
    return scene_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
