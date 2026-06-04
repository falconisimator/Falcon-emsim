"""A conductor (one meshed region) with shape, placement, material and group."""

from __future__ import annotations

from dataclasses import dataclass

from emsim.geometry.shapes import Placement, Shape
from emsim.materials import Material


@dataclass
class Conductor:
    """One conductor region.

    Attributes
    ----------
    name:
        Label.
    shape, placement:
        Cross-section geometry and where it sits in the world.
    material:
        Region material (conductivity, permeability).
    group:
        Parallel-group (terminal) name this region belongs to. ``None`` marks a
        passive region (e.g. a steel enclosure) that carries induced eddy
        currents but no terminal current.
    region_tag:
        Integer mesh tag (assigned by the scene).
    """

    name: str
    shape: Shape
    placement: Placement
    material: Material
    group: str | None = None  # electrical phase / terminal (A/B/C); None = passive
    busbar: str = ""  # geometry-group id; shapes sharing it form one busbar
    region_tag: int = 0
