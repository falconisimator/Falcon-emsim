r"""Time-domain animation and evaluation-criteria graphs.

The solver yields complex phasors; the instantaneous field at phase ``phi`` is
``Re(X e^{j phi})``. :func:`field_gif` sweeps ``phi`` over one full period to
produce an animated GIF of the magnetic field (|B| colour map + instantaneous
flux lines = contours of A_z). For balanced three-phase systems this visualises
the rotating field.

:func:`evaluation_figure` renders the four must-have evaluation criteria
(current sharing, losses / R_AC/R_DC, forces, and the current phasor diagram).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

from emsim.mesh.gmsh_backend import KELVIN_TAG  # noqa: E402
from emsim.post import fields  # noqa: E402
from emsim.results import Solution  # noqa: E402

PHASE_COLORS = {"A": "#d65f5f", "B": "#5f9ed6", "C": "#5fd68a"}


def _phase_color(group):
    return PHASE_COLORS.get(group, "#9a5fd6" if group else "#8a8a8a")


def _physical(solution: Solution):
    mesh = solution.mesh
    phys = mesh.region_tag != KELVIN_TAG
    tri = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.tris[phys][:, :3])
    return tri, phys


# Per-kind display configuration: how to render each animated quantity.
FIELD_KINDS = {
    "B": {"label": "|B(t)|  (T)", "cmap": "inferno", "diverging": False},
    "J": {"label": "J_z(t)  (A/m²)", "cmap": "RdBu_r", "diverging": True},
    "A": {"label": "A_z(t)  (Wb/m)", "cmap": "RdBu_r", "diverging": True},
}


class AnimationData:
    """Cached per-solution fields used to render any instantaneous frame."""

    def __init__(self, solution: Solution, extent: float | None = None):
        mesh = solution.mesh
        self.tri, phys = _physical(solution)
        self.a = solution.a  # nodal complex A_z
        self.B = fields.element_B(solution)[phys]  # (Mphys, 2) complex
        self.J = fields.element_Jz(solution)[phys]  # (Mphys,) complex
        bamp = np.sqrt(np.abs(self.B[:, 0]) ** 2 + np.abs(self.B[:, 1]) ** 2)
        self.scale = {
            "B": float(bamp.max()) or 1.0,
            "J": float(np.abs(self.J).max()) or 1.0,
            "A": float(np.abs(self.a).max()) or 1.0,
        }
        if extent is None:
            r = np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])
            extent = float(np.percentile(r[r > 0], 40))
        self.extent = extent

    def instantaneous(self, kind: str, phi: float):
        """Return (mode, values) for phase phi; mode is 'nodal' or 'elem'."""
        rot = np.exp(1j * phi)
        if kind == "A":
            return "nodal", np.real(self.a * rot)
        if kind == "J":
            return "elem", np.real(self.J * rot)
        bx, by = np.real(self.B[:, 0] * rot), np.real(self.B[:, 1] * rot)
        return "elem", np.sqrt(bx**2 + by**2)


def field_gif(
    solution: Solution,
    path: str | Path,
    *,
    kind: str = "B",
    extent: float | None = None,
    nframes: int = 30,
    fps: int = 15,
) -> Path:
    """Write an animated GIF of the chosen field (``kind`` in {'B','J','A'})."""
    cfg = FIELD_KINDS[kind]
    data = AnimationData(solution, extent)
    vmax = data.scale[kind]
    vmin = -vmax if cfg["diverging"] else 0.0
    norm = matplotlib.colors.Normalize(vmin, vmax)

    fig = Figure(figsize=(5.2, 4.6))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    frames = []
    for k, phi in enumerate(np.linspace(0.0, 2.0 * np.pi, nframes, endpoint=False)):
        mode, vals = data.instantaneous(kind, phi)
        ax.clear()
        if mode == "nodal":
            ax.tripcolor(data.tri, vals, cmap=cfg["cmap"], norm=norm, shading="gouraud")
        else:
            ax.tripcolor(data.tri, facecolors=vals, cmap=cfg["cmap"], norm=norm)
        ax.set_aspect("equal")
        ax.set_xlim(-data.extent, data.extent)
        ax.set_ylim(-data.extent, data.extent)
        ax.set_title(f"{cfg['label']}   (phase {math.degrees(phi):.0f}°)")
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cfg["cmap"])
            fig.colorbar(sm, ax=ax, label=cfg["label"])
            fig.tight_layout()
        canvas.draw()
        frames.append(np.asarray(canvas.buffer_rgba())[..., :3].copy())

    from PIL import Image

    images = [Image.fromarray(f) for f in frames]
    path = Path(path)
    images[0].save(
        path, save_all=True, append_images=images[1:],
        duration=int(1000 / fps), loop=0, disposal=2,
    )
    return path


def evaluation_figure(scene, result) -> Figure:
    """A 2x2 panel of the evaluation criteria: current sharing, loss, force, phasors."""
    conds = result.conductors
    names = [c.name for c in conds]
    colors = [_phase_color(c.group) for c in conds]
    x = np.arange(len(conds))

    fig = Figure(figsize=(9, 6.5), tight_layout=True)

    # current sharing (label = share % of the conductor's terminal)
    ax1 = fig.add_subplot(221)
    mags = [abs(c.current) for c in conds]
    ax1.bar(x, mags, color=colors)
    for i, c in enumerate(conds):
        lbl = f"{c.share * 100:.0f}%" if c.share == c.share else f"{math.degrees(np.angle(c.current)):+.0f}°"
        ax1.text(i, mags[i], lbl, ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("|I| (A)")
    ax1.set_title("Current sharing (label = % of terminal)")

    # losses
    ax2 = fig.add_subplot(222)
    losses = [c.loss for c in conds]
    ax2.bar(x, losses, color=colors)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("loss (W/m)")
    rac = "  ".join(f"{n}: R_AC/R_DC={gl.rac_rdc:.2f}" for n, gl in result.group_losses.items())
    ax2.set_title(f"Ohmic loss (total {result.total_loss:.4g} W/m)\n{rac}", fontsize=8)

    # forces
    ax3 = fig.add_subplot(223)
    fmag = [math.hypot(*c.force) if c.force else 0.0 for c in conds]
    ax3.bar(x, fmag, color=colors)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("|F| (N/m)")
    ax3.set_title("Maxwell-stress force per conductor")

    # current phasor diagram
    ax4 = fig.add_subplot(224, projection="polar")
    for c in conds:
        if abs(c.current) == 0:
            continue
        ax4.annotate(
            "", xy=(np.angle(c.current), abs(c.current)), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=_phase_color(c.group), lw=1.6),
        )
    ax4.set_rlim(0, max(mags) * 1.1 if mags else 1)
    ax4.set_title("Current phasors", pad=18)
    return fig
