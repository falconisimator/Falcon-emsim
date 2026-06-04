"""Result views: embedded matplotlib field maps and a per-conductor table."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

from emsim.mesh.gmsh_backend import KELVIN_TAG  # noqa: E402
from emsim.post import fields  # noqa: E402
from emsim.scene import SceneResult  # noqa: E402


class FieldCanvas(FigureCanvasQTAgg):
    """A matplotlib canvas that renders |A_z|, |B| or |J_z| over the conductors."""

    def __init__(self):
        self.fig = Figure(figsize=(5, 4), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._anim = None

    @staticmethod
    def _conductor_extent(scene) -> float:
        return max(
            (abs(c.placement.x) + abs(c.placement.y) + 1.6 * c.shape.bounding_radius())
            for c in scene.conductors
        )

    def begin_anim(self, scene, result: SceneResult, kind: str) -> None:
        """Set up an inline time-domain animation of the field (kind in B/J/A)."""
        from matplotlib.colors import Normalize

        from emsim.post.animate import AnimationData, FIELD_KINDS

        cfg = FIELD_KINDS[kind]
        data = AnimationData(result.solution, extent=self._conductor_extent(scene))
        vmax = data.scale[kind]
        norm = Normalize(-vmax if cfg["diverging"] else 0.0, vmax)
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        mode, vals = data.instantaneous(kind, 0.0)
        if mode == "nodal":
            coll = self.ax.tripcolor(data.tri, vals, cmap=cfg["cmap"], norm=norm,
                                     shading="gouraud")
        else:
            coll = self.ax.tripcolor(data.tri, facecolors=vals, cmap=cfg["cmap"], norm=norm)
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-data.extent, data.extent)
        self.ax.set_ylim(-data.extent, data.extent)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.colorbar(coll, ax=self.ax, label=cfg["label"])
        self._anim = {"data": data, "kind": kind, "coll": coll, "cfg": cfg}
        self.step_anim(0.0)

    def step_anim(self, phi: float) -> None:
        if not self._anim:
            return
        a = self._anim
        _, vals = a["data"].instantaneous(a["kind"], phi)
        a["coll"].set_array(vals)
        self.ax.set_title(f"{a['cfg']['label']}   (phase {int(round(phi * 180 / 3.14159))}°)")
        self.draw_idle()

    def show_field(self, scene, result: SceneResult, kind: str) -> None:
        self._anim = None
        sol = result.solution
        mesh = sol.mesh
        phys = mesh.region_tag != KELVIN_TAG
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        tri_v = mesh.tris[phys][:, :3]
        triang = Triangulation(mesh.nodes[:, 0], mesh.nodes[:, 1], tri_v)
        if kind == "|A_z|":
            mappable = self.ax.tricontourf(
                triang, np.abs(sol.a), levels=40, cmap="viridis"
            )
            title = "|A_z|  (Wb/m)"
        elif kind == "|B|":
            bmag = fields.element_B_magnitude(sol)[phys]
            mappable = self.ax.tripcolor(triang, facecolors=bmag, cmap="magma")
            title = "|B|  (T)"
        else:  # |J_z|
            Jz, _, _ = fields.current_density_at_quadrature(sol)
            jmag = np.abs(Jz).mean(axis=1)[phys]
            mappable = self.ax.tripcolor(triang, facecolors=jmag, cmap="inferno")
            title = "|J_z|  (A/m^2)"
        self.ax.set_aspect("equal")
        self.ax.set_title(title)
        # zoom to the conductor region
        ext = max(
            (abs(c.placement.x) + abs(c.placement.y) + 1.6 * c.shape.bounding_radius())
            for c in scene.conductors
        )
        self.ax.set_xlim(-ext, ext)
        self.ax.set_ylim(-ext, ext)
        self.fig.colorbar(mappable, ax=self.ax)
        self.draw()


class ResultsPanel(QtWidgets.QWidget):
    """Field-map selector + canvas + per-conductor results table + totals."""

    def __init__(self):
        super().__init__()
        self._scene = None
        self._result: SceneResult | None = None
        layout = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Field:"))
        self.field_combo = QtWidgets.QComboBox()
        self.field_combo.addItems(["|A_z|", "|B|", "|J_z|"])
        self.field_combo.currentTextChanged.connect(self._refresh_field)
        row.addWidget(self.field_combo)
        self.play_btn = QtWidgets.QPushButton("▶ Animate")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._toggle_play)
        row.addWidget(self.play_btn)
        row.addStretch()
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._on_tick)
        self._phase = 0.0
        self._nframes = 30
        self.graphs_btn = QtWidgets.QPushButton("Evaluation graphs")
        self.gif_btn = QtWidgets.QPushButton("Export field GIF…")
        self.graphs_btn.clicked.connect(self._show_graphs)
        self.gif_btn.clicked.connect(self._export_gif)
        row.addWidget(self.graphs_btn)
        row.addWidget(self.gif_btn)
        layout.addLayout(row)

        self.canvas = FieldCanvas()
        layout.addWidget(self.canvas, stretch=3)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Conductor", "Phase", "|I| (A) / phase", "Share %", "Loss (W/m)", "Force (N/m)"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, stretch=2)

        self.summary = QtWidgets.QLabel("No solution yet.")
        layout.addWidget(self.summary)

    def set_result(self, scene, result: SceneResult) -> None:
        self._scene = scene
        self._result = result
        if self.play_btn.isChecked():
            self.play_btn.setChecked(False)  # stops timer, reverts to static
        self._refresh_field()
        self._fill_table()

    _KIND = {"|A_z|": "A", "|B|": "B", "|J_z|": "J"}

    def _refresh_field(self) -> None:
        if self._result is None:
            return
        if self.play_btn.isChecked():  # switching field while animating: restart
            self._phase = 0.0
            self.canvas.begin_anim(self._scene, self._result, self._KIND[self.field_combo.currentText()])
        else:
            self.canvas.show_field(self._scene, self._result, self.field_combo.currentText())

    def _toggle_play(self, on: bool) -> None:
        if self._result is None:
            self.play_btn.setChecked(False)
            return
        if on:
            self.play_btn.setText("⏸ Pause")
            self._phase = 0.0
            self.canvas.begin_anim(self._scene, self._result, self._KIND[self.field_combo.currentText()])
            self._timer.start()
        else:
            self.play_btn.setText("▶ Animate")
            self._timer.stop()
            self.canvas.show_field(self._scene, self._result, self.field_combo.currentText())

    def _on_tick(self) -> None:
        self._phase = (self._phase + 2.0 * 3.141592653589793 / self._nframes) % (
            2.0 * 3.141592653589793
        )
        self.canvas.step_anim(self._phase)

    def _show_graphs(self) -> None:
        if self._result is None:
            QtWidgets.QMessageBox.information(self, "No solution", "Solve first.")
            return
        from emsim.post.animate import evaluation_figure

        fig = evaluation_figure(self._scene, self._result)
        dlg = QtWidgets.QDialog(self, QtCore.Qt.Window)
        dlg.setWindowTitle("Evaluation criteria")
        dlg.resize(900, 680)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(FigureCanvasQTAgg(fig))
        self._dialogs = getattr(self, "_dialogs", [])
        self._dialogs.append(dlg)  # keep a reference
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _export_gif(self) -> None:
        if self._result is None:
            QtWidgets.QMessageBox.information(self, "No solution", "Solve first.")
            return
        kind = self._KIND[self.field_combo.currentText()]
        default = {"A": "Az", "B": "Bfield", "J": "Jcurrent"}[kind] + ".gif"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export field animation", default, "GIF (*.gif)"
        )
        if not path:
            return
        from emsim.post.animate import field_gif

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            field_gif(self._result.solution, path, kind=kind)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QMessageBox.information(self, "Saved", f"Animation written to\n{path}")

    def _fill_table(self) -> None:
        res = self._result
        self.table.setRowCount(len(res.conductors))
        for i, cr in enumerate(res.conductors):
            mag = abs(cr.current)
            phase = np.degrees(np.angle(cr.current))
            force = "-" if cr.force is None else f"({cr.force[0]:.3g}, {cr.force[1]:.3g})"
            share = "-" if cr.share != cr.share else f"{cr.share * 100:.1f}%"  # NaN check
            vals = [
                cr.name,
                cr.group or "(passive)",
                f"{mag:.4g} / {phase:+.1f}°",
                share,
                f"{cr.loss:.4g}",
                force,
            ]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        lines = [f"Total ohmic loss: {res.total_loss:.4g} W/m"]
        for t in res.terminals:
            gl = res.group_losses.get(t.name)
            rr = f", R_AC/R_DC={gl.rac_rdc:.3f}" if gl else ""
            lines.append(
                f"  Terminal {t.name}: I={abs(t.current):.4g} A, "
                f"V̇/L={abs(t.voltage_gradient):.4g} V/m, "
                f"Z=({t.impedance.real:.3e}{t.impedance.imag:+.3e}j) Ω/m{rr}"
            )
        self.summary.setText("\n".join(lines))
