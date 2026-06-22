"""Material definitions and the region-tag -> material table."""

from __future__ import annotations

from dataclasses import dataclass

from emsim.config import MU0


@dataclass(frozen=True)
class Material:
    """A linear isotropic material.

    Parameters
    ----------
    name:
        Human-readable label.
    sigma:
        Electrical conductivity (S/m). Use 0.0 for non-conducting regions
        (air / free space).
    mu_r:
        Relative permeability (dimensionless). 1.0 for non-magnetic media.
    """

    name: str
    sigma: float
    mu_r: float = 1.0

    @property
    def mu(self) -> float:
        """Absolute permeability mu = mu0 * mu_r (H/m)."""
        return MU0 * self.mu_r


# Convenience library of common busbar materials (room-temperature values).
COPPER = Material("copper", sigma=5.8e7, mu_r=1.0)
ALUMINIUM = Material("aluminium", sigma=3.5e7, mu_r=1.0)
AIR = Material("air", sigma=0.0, mu_r=1.0)
# Generic linear (low-field) magnetic steel for the enclosure.
STEEL = Material("steel", sigma=1.0e7, mu_r=200.0)
# Austenitic stainless (304/316): conductive but effectively non-magnetic, so
# EM-easy (skin depth ~55 mm @ 60 Hz) -- a light enclosure option.
STAINLESS = Material("stainless", sigma=1.4e6, mu_r=1.0)


class MaterialTable:
    """Maps integer region tags (as produced by the mesher) to materials."""

    def __init__(self, mapping: dict[int, Material] | None = None):
        self._by_tag: dict[int, Material] = dict(mapping or {})

    def set(self, tag: int, material: Material) -> None:
        self._by_tag[int(tag)] = material

    def get(self, tag: int) -> Material:
        try:
            return self._by_tag[int(tag)]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(
                f"No material assigned to region tag {tag}; "
                f"known tags: {sorted(self._by_tag)}"
            ) from exc

    def tags(self) -> list[int]:
        return sorted(self._by_tag)

    def __contains__(self, tag: int) -> bool:
        return int(tag) in self._by_tag

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        items = ", ".join(f"{t}:{m.name}" for t, m in sorted(self._by_tag.items()))
        return f"MaterialTable({{{items}}})"
