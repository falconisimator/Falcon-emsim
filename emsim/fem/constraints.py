"""Parallel groups: the electrical terminals that impose a total current.

A parallel group is one electrical terminal. It owns one or more mesh region
tags (a composite/multi-material bar uses several tags but is still a *single*
terminal sharing one per-unit-length voltage gradient V_dot/L). The solver
treats V_dot/L as a bordered unknown and enforces the prescribed total complex
current over all the group's regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParallelGroup:
    """An electrical terminal carrying a prescribed total current.

    Parameters
    ----------
    name:
        Label for reporting.
    region_tags:
        The mesh region tags belonging to this terminal. More than one tag
        means a composite bar (e.g. Al core + Cu cladding) that nonetheless
        shares a single voltage gradient -- the current splits between the
        sub-regions according to the physics.
    current:
        Prescribed total complex current (A) through the group's cross
        section, as a phasor amplitude.
    """

    name: str
    region_tags: tuple[int, ...]
    current: complex

    def __post_init__(self) -> None:
        self.region_tags = tuple(int(t) for t in self.region_tags)
        if len(self.region_tags) == 0:
            raise ValueError(f"parallel group {self.name!r} has no region tags")
        self.current = complex(self.current)

    @property
    def tag_set(self) -> set[int]:
        return set(self.region_tags)


@dataclass
class GroupSystem:
    """All parallel groups for a run, with a stable ordering."""

    groups: list[ParallelGroup] = field(default_factory=list)

    def add(self, group: ParallelGroup) -> ParallelGroup:
        self._check_disjoint(group)
        self.groups.append(group)
        return group

    def _check_disjoint(self, group: ParallelGroup) -> None:
        existing: set[int] = set()
        for g in self.groups:
            existing |= g.tag_set
        clash = existing & group.tag_set
        if clash:
            raise ValueError(
                f"region tag(s) {sorted(clash)} already assigned to another "
                f"parallel group; each conductor region belongs to exactly one terminal"
            )

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    def __iter__(self):
        return iter(self.groups)

    def __len__(self) -> int:
        return len(self.groups)
