"""The :class:`Mesh` data structure (pure NumPy; no Gmsh dependency).

Once a mesh is generated it is fully described by plain arrays, so the
assembly, solve and post-processing stages never need to import gmsh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    """A 2D triangular mesh with region tags.

    Attributes
    ----------
    nodes:
        ``(N, 2)`` float array of node coordinates (x, y) in metres.
    tris:
        ``(M, K)`` int array of triangle connectivity (node indices, 0-based);
        ``K == 3`` for first-order (P1) or ``K == 6`` for second-order (P2,
        Gmsh type-9 node ordering: vertices then edge midsides). The first
        three columns are always the vertices.
    region_tag:
        ``(M,)`` int array; the region/material tag of each triangle.
    boundary_nodes:
        Optional 1D int array of node indices on the outer (Dirichlet)
        boundary, used to pin the gauge in the first-order milestone.
    """

    nodes: np.ndarray
    tris: np.ndarray
    region_tag: np.ndarray
    boundary_nodes: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )

    def __post_init__(self) -> None:
        self.nodes = np.ascontiguousarray(self.nodes, dtype=np.float64)
        self.tris = np.ascontiguousarray(self.tris, dtype=np.int64)
        self.region_tag = np.ascontiguousarray(self.region_tag, dtype=np.int64)
        self.boundary_nodes = np.ascontiguousarray(
            self.boundary_nodes, dtype=np.int64
        )
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must have shape (N, 2)")
        if self.tris.ndim != 2 or self.tris.shape[1] not in (3, 6):
            raise ValueError("tris must have shape (M, 3) [P1] or (M, 6) [P2]")
        if self.region_tag.shape[0] != self.tris.shape[0]:
            raise ValueError("region_tag must have one entry per triangle")

    @property
    def num_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def num_tris(self) -> int:
        return self.tris.shape[0]

    @property
    def order(self) -> int:
        """Element order: 1 (3-node) or 2 (6-node)."""
        return 1 if self.tris.shape[1] == 3 else 2

    def triangle_vertices(self) -> np.ndarray:
        """Return ``(M, 3, 2)`` array of the three vertex coordinates per tri."""
        return self.nodes[self.tris[:, :3]]

    def areas(self) -> np.ndarray:
        """Signed-then-absolute triangle areas, shape ``(M,)``."""
        v = self.triangle_vertices()
        x1, y1 = v[:, 0, 0], v[:, 0, 1]
        x2, y2 = v[:, 1, 0], v[:, 1, 1]
        x3, y3 = v[:, 2, 0], v[:, 2, 1]
        return 0.5 * np.abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    def centroids(self) -> np.ndarray:
        """Triangle centroids, shape ``(M, 2)``."""
        return self.triangle_vertices().mean(axis=1)

    def tris_in_regions(self, tags: set[int]) -> np.ndarray:
        """Boolean mask of triangles whose region tag is in ``tags``."""
        return np.isin(self.region_tag, np.fromiter(tags, dtype=np.int64))
