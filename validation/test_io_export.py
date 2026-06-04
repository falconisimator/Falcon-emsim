r"""Save/load round-trip and animation/graph export (no GUI needed)."""

from __future__ import annotations

import math

from emsim.geometry.model import Conductor
from emsim.geometry.shapes import Placement, Rectangle
from emsim.io import load_scene, save_scene
from emsim.materials import COPPER
from emsim.post.animate import evaluation_figure, field_gif
from emsim.scene import Scene


def _small_scene() -> Scene:
    return Scene(
        conductors=[
            Conductor("A", Rectangle(0.01, 0.03), Placement(-0.02, 0, 0), COPPER, "A", "bb1"),
            Conductor("B", Rectangle(0.01, 0.03), Placement(0.02, 0, 0), COPPER, "B", "bb2"),
        ],
        frequency=200.0,
        three_phase=True,
        line_current=500.0,
        boundary="dirichlet",
    )


def test_scene_roundtrip(tmp_path) -> None:
    sc = _small_scene()
    p = tmp_path / "cfg.json"
    save_scene(sc, p)
    back = load_scene(p)
    assert [c.group for c in back.conductors] == ["A", "B"]
    assert back.busbar_ids() == ["bb1", "bb2"]
    assert back.three_phase and back.line_current == 500.0
    assert back.conductors[0].material is COPPER  # preset identity preserved
    assert back.conductors[0].shape.width == 0.01


def test_export_gif_and_graphs(tmp_path) -> None:
    sc = _small_scene()
    sol = sc.solve()
    result = sc.analyse(sol)

    for kind in ("B", "J", "A"):  # field, current-distribution, vector-potential
        gif = tmp_path / f"{kind}.gif"
        field_gif(sol, gif, kind=kind, nframes=6, fps=10)
        assert gif.exists() and gif.stat().st_size > 0

    fig = evaluation_figure(sc, result)
    assert len(fig.axes) == 4  # current, loss, force, phasors

    # three-phase currents are balanced at 0 / -120 / +120 degrees
    phases = sorted(round(math.degrees(math.atan2(c.current.imag, c.current.real)))
                    for c in result.conductors)
    assert phases == [-120, 0]
