"""The editable scene: conductors + frequency + domain, plus solve/analyse.

This is the model the GUI manipulates and the single entry point for running a
simulation from a high-level description.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from emsim.config import SimulationConfig
from emsim.fem.constraints import ParallelGroup
from emsim.geometry.model import Conductor
from emsim.materials import AIR, Material, MaterialTable
from emsim.mesh.gmsh_backend import AIR_TAG, KELVIN_TAG, mesh_model
from emsim.mesh.kelvin import open_mesh
from emsim.mesh.mesh import Mesh
from emsim.post.forces import maxwell_force
from emsim.post.losses import group_losses, region_current, region_ohmic_loss
from emsim.results import Solution


@dataclass
class ConductorResult:
    name: str
    group: str | None
    current: complex
    loss: float  # W/m
    force: tuple[float, float] | None  # (Fx, Fy) N/m, or None if not computable
    share: float = float("nan")  # |I| as a fraction of its terminal's total current


@dataclass
class Terminal:
    """One electrical terminal (phase / parallel group) operating point."""

    name: str
    current: complex  # prescribed total (end) current, A
    voltage_gradient: complex  # solved V_dot/L, V/m (the "voltage source" value)
    impedance: complex  # Z = (V_dot/L) / I, Ohm/m

    @property
    def r_ac(self) -> float:
        return self.impedance.real


@dataclass
class SceneResult:
    solution: Solution
    conductors: list[ConductorResult]
    total_loss: float
    group_losses: dict
    terminals: list[Terminal]


# Balanced three-phase terminal phases (degrees).
THREE_PHASE_DEG = {"A": 0.0, "B": -120.0, "C": 120.0}


@dataclass
class Scene:
    """A complete problem description."""

    conductors: list[Conductor] = field(default_factory=list)
    group_currents: dict[str, complex] = field(default_factory=dict)
    frequency: float = 50.0
    domain_radius: float = 0.0  # 0 => auto from geometry extent
    boundary: str = "kelvin"  # "kelvin" or "dirichlet"
    order: int = 1
    lc_surface: float = 0.0  # 0 => auto from skin depth
    lc_far: float = 0.0  # 0 => auto
    three_phase: bool = True  # groups A/B/C forced to 0/-120/+120 deg
    line_current: float = 1000.0  # phase-current magnitude for 3-phase mode
    mesh_backend: str = "gmsh"  # "gmsh" (desktop) or "py" (gmsh-free, for web/Pyodide)
    region_sizes: list[float] | None = None  # per-conductor mesh size (py backend); None => auto

    def busbar_of(self, c: Conductor) -> str:
        """Stable busbar id for a conductor (falls back to a per-object id)."""
        return c.busbar or f"_bb{id(c)}"

    def members(self, busbar: str) -> list[Conductor]:
        """All conductors belonging to a busbar (geometry group)."""
        return [c for c in self.conductors if self.busbar_of(c) == busbar]

    def busbar_ids(self) -> list[str]:
        seen: list[str] = []
        for c in self.conductors:
            b = self.busbar_of(c)
            if b not in seen:
                seen.append(b)
        return seen

    def next_busbar_id(self) -> str:
        n = 1
        existing = set(self.busbar_ids())
        while f"bb{n}" in existing:
            n += 1
        return f"bb{n}"

    def set_busbar_phase(self, busbar: str, phase: str | None) -> None:
        for c in self.members(busbar):
            c.group = phase

    def current_for_group(self, name: str) -> complex:
        """Complex terminal current for a group name.

        In three-phase mode groups A/B/C are locked to the line-current
        magnitude at 0 / -120 / +120 degrees; any other group falls back to
        ``group_currents``.
        """
        import cmath
        import math

        if self.three_phase and name in THREE_PHASE_DEG:
            return cmath.rect(self.line_current, math.radians(THREE_PHASE_DEG[name]))
        return self.group_currents.get(name, 1.0 + 0j)

    # ------------------------------------------------------------------ setup
    def _assign_tags(self) -> None:
        for i, c in enumerate(self.conductors):
            c.region_tag = 10 + i

    def _auto_radius(self) -> float:
        if self.domain_radius > 0:
            return self.domain_radius
        ext = max(
            (
                (abs(c.placement.x) + abs(c.placement.y) + c.shape.bounding_radius())
                for c in self.conductors
            ),
            default=0.01,
        )
        return 4.0 * max(ext, 1e-3)

    def _auto_sizes(self, cfg: SimulationConfig) -> tuple[float, float]:
        if self.lc_surface > 0 and self.lc_far > 0:
            return self.lc_surface, self.lc_far
        min_char = min(c.shape.char_size() for c in self.conductors)
        # finest skin depth among conductors
        deltas = [
            cfg.skin_depth(c.material.sigma, c.material.mu_r)
            for c in self.conductors
            if c.material.sigma > 0
        ]
        delta = min(deltas) if deltas else min_char
        lc_s = self.lc_surface or min(delta / 3.0, min_char / 6.0)
        lc_f = self.lc_far or min_char
        return lc_s, lc_f

    def _region_sizes(self, cfg: SimulationConfig) -> list[float]:
        """Per-conductor target mesh size: resolve each region's own skin depth
        and a few cells across its thinnest dimension, independently. This is
        what lets a thin steel wall stay local instead of refining the whole
        domain (the "crazy mesh")."""
        sizes = []
        for c in self.conductors:
            char = c.shape.char_size()
            if c.material.sigma > 0:
                d = cfg.skin_depth(c.material.sigma, c.material.mu_r)
                h = min(d / 3.0, char / 4.0)
            else:
                h = char / 4.0
            sizes.append(h)
        return sizes

    def material_table(self) -> MaterialTable:
        mt = MaterialTable({AIR_TAG: AIR, KELVIN_TAG: AIR})
        for c in self.conductors:
            mt.set(c.region_tag, c.material)
        return mt

    def parallel_groups(self) -> list[ParallelGroup]:
        names: dict[str, list[int]] = {}
        for c in self.conductors:
            if c.group is None:
                continue
            names.setdefault(c.group, []).append(c.region_tag)
        groups = []
        for name, tags in names.items():
            groups.append(ParallelGroup(name, tuple(tags), self.current_for_group(name)))
        return groups

    # ------------------------------------------------------------------ solve
    def build_mesh(self) -> Mesh:
        if not self.conductors:
            raise ValueError("scene has no conductors")
        self._assign_tags()
        cfg = SimulationConfig(self.frequency)
        R = self._auto_radius()
        lc_s, lc_f = self._auto_sizes(cfg)
        regions = [(c.shape, c.placement, c.region_tag) for c in self.conductors]
        if self.mesh_backend == "py":
            from emsim.mesh.py_backend import mesh_model as _mesh_model
            # per-region sizing so a thin wall doesn't refine the whole domain
            sizes = self.region_sizes or self._region_sizes(cfg)
            sizes = [min(max(h, lc_f / 60.0), lc_f) for h in sizes]
            mesh = _mesh_model(
                regions, R, lc_surface=lc_s, lc_far=lc_f, region_sizes=sizes,
                grade_distance=R / 3.0, order=self.order,
            )
        else:
            mesh = mesh_model(
                regions, R, lc_surface=lc_s, lc_far=lc_f, grade_distance=R / 3.0, order=self.order
            )
        if self.boundary == "kelvin":
            # Kelvin open boundary uses the gmsh mirror-disk; not available in the
            # gmsh-free backend, so fall back to the Dirichlet box there.
            if self.mesh_backend == "py":
                mesh.boundary_nodes = mesh.boundary_nodes  # outer circle already pinned
            else:
                mesh = open_mesh(mesh)
        return mesh

    def solve(self, mesh: Mesh | None = None) -> Solution:
        if mesh is None:
            mesh = self.build_mesh()
        cfg = SimulationConfig(self.frequency)
        from emsim.solve.solver import solve as _solve

        return _solve(mesh, self.material_table(), self.parallel_groups(), cfg)

    # --------------------------------------------------------------- analysis
    def analyse(self, solution: Solution) -> SceneResult:
        # terminal totals (the prescribed "end" current per phase/group) and the
        # solved voltage gradient V_dot/L -> impedance Z = (V_dot/L)/I
        term_current = {name: self.current_for_group(name)
                        for name in solution.group_order}
        uvals = {name: solution.u[i] for i, name in enumerate(solution.group_order)}
        terminals = []
        for name in solution.group_order:
            itot = term_current[name]
            u = complex(uvals[name])
            z = u / itot if abs(itot) > 0 else 0j
            terminals.append(Terminal(name, itot, u, z))

        cresults: list[ConductorResult] = []
        for c in self.conductors:
            cur = region_current(solution, {c.region_tag})
            loss = region_ohmic_loss(solution, {c.region_tag})
            force: tuple[float, float] | None = None
            try:
                rc = 1.25 * c.shape.bounding_radius()
                force = maxwell_force(solution, (c.placement.x, c.placement.y), rc)
            except Exception:
                force = None
            itot = term_current.get(c.group)
            share = abs(cur) / abs(itot) if itot and abs(itot) > 0 else float("nan")
            cresults.append(ConductorResult(c.name, c.group, cur, loss, force, share))
        gls = {gl.name: gl for gl in group_losses(solution)}
        total = sum(cr.loss for cr in cresults)
        return SceneResult(solution, cresults, total, gls, terminals)
