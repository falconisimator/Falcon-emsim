r"""Conductor cross-section shapes with free placement (x, y, rotation).

Each shape can: (a) report a world-space point-containment test (for mesh
region classification), (b) create itself in a Gmsh OCC model at its placement,
and (c) report a characteristic size and area (for mesh sizing and
innermost-region ordering).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Placement:
    """Rigid placement: translation (x, y) then rotation (degrees, CCW)."""

    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0

    def to_local(self, px: float, py: float) -> tuple[float, float]:
        """Map a world point into the shape's local (un-placed) frame."""
        dx, dy = px - self.x, py - self.y
        c = math.cos(math.radians(self.rotation))
        s = math.sin(math.radians(self.rotation))
        # inverse rotation
        return (c * dx + s * dy, -s * dx + c * dy)


class Shape:
    """Abstract conductor cross-section in its own local frame."""

    def contains_local(self, lx: float, ly: float) -> bool:
        raise NotImplementedError

    def area(self) -> float:
        raise NotImplementedError

    def char_size(self) -> float:
        """A characteristic dimension (used for default mesh sizing)."""
        raise NotImplementedError

    def bounding_radius(self) -> float:
        """Radius of a circle (about the local origin) enclosing the shape."""
        raise NotImplementedError

    def occ_add(self, occ, placement: Placement) -> int:
        """Create the shape in a Gmsh OCC model; return the surface tag."""
        raise NotImplementedError

    def contains(self, px: float, py: float, placement: Placement) -> bool:
        return self.contains_local(*placement.to_local(px, py))


@dataclass
class Circle(Shape):
    radius: float

    def contains_local(self, lx: float, ly: float) -> bool:
        return lx * lx + ly * ly <= self.radius * self.radius * (1.0 + 1e-9)

    def area(self) -> float:
        return math.pi * self.radius * self.radius

    def char_size(self) -> float:
        return 2.0 * self.radius

    def bounding_radius(self) -> float:
        return self.radius

    def occ_add(self, occ, placement: Placement) -> int:
        # rotation of a circle is irrelevant
        return occ.addDisk(placement.x, placement.y, 0.0, self.radius, self.radius)


@dataclass
class Rectangle(Shape):
    width: float
    height: float

    def contains_local(self, lx: float, ly: float) -> bool:
        return abs(lx) <= 0.5 * self.width * (1 + 1e-9) and abs(ly) <= 0.5 * self.height * (
            1 + 1e-9
        )

    def area(self) -> float:
        return self.width * self.height

    def char_size(self) -> float:
        return min(self.width, self.height)

    def bounding_radius(self) -> float:
        return 0.5 * math.hypot(self.width, self.height)

    def occ_add(self, occ, placement: Placement) -> int:
        tag = occ.addRectangle(
            -0.5 * self.width, -0.5 * self.height, 0.0, self.width, self.height
        )
        occ.rotate([(2, tag)], 0, 0, 0, 0, 0, 1, math.radians(placement.rotation))
        occ.translate([(2, tag)], placement.x, placement.y, 0.0)
        return tag


@dataclass
class Polygon(Shape):
    """A simple polygon given by local (x, y) vertices (CCW)."""

    points: list[tuple[float, float]] = field(default_factory=list)

    def contains_local(self, lx: float, ly: float) -> bool:
        pts = self.points
        inside = False
        n = len(pts)
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > ly) != (yj > ly):
                xint = (xj - xi) * (ly - yi) / (yj - yi) + xi
                if lx < xint:
                    inside = not inside
            j = i
        return inside

    def area(self) -> float:
        pts = np.asarray(self.points)
        x, y = pts[:, 0], pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def char_size(self) -> float:
        return float(np.sqrt(self.area()))

    def bounding_radius(self) -> float:
        pts = np.asarray(self.points)
        return float(np.max(np.hypot(pts[:, 0], pts[:, 1])))

    def occ_add(self, occ, placement: Placement) -> int:
        c = math.cos(math.radians(placement.rotation))
        s = math.sin(math.radians(placement.rotation))
        pt_tags = []
        for lx, ly in self.points:
            wx = placement.x + c * lx - s * ly
            wy = placement.y + s * lx + c * ly
            pt_tags.append(occ.addPoint(wx, wy, 0.0))
        n = len(pt_tags)
        lines = [occ.addLine(pt_tags[i], pt_tags[(i + 1) % n]) for i in range(n)]
        loop = occ.addCurveLoop(lines)
        return occ.addPlaneSurface([loop])
