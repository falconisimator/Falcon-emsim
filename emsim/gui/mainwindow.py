"""Main application window: canvas + editors + results, with a solve worker."""

from __future__ import annotations

import copy

from PySide6 import QtCore, QtGui, QtWidgets

from emsim.geometry.model import Conductor
from emsim.geometry.shapes import Circle, Placement, Rectangle
from emsim.gui.canvas import GRID, MM, CanvasView
from emsim.gui.panels import PropertyPanel, SettingsPanel
from emsim.gui.results import ResultsPanel
from emsim.io import load_scene, save_scene
from emsim.materials import AIR, COPPER, STEEL
from emsim.scene import Scene, SceneResult


def default_scene() -> Scene:
    """Three copper phase bars (A/B/C), balanced three-phase at 50 Hz."""
    bars = [
        Conductor(f"Phase {g}", Rectangle(0.01, 0.05), Placement(x, 0, 0), COPPER,
                  group=g, busbar=f"bb{i + 1}")
        for i, (g, x) in enumerate(zip("ABC", (-0.03, 0.0, 0.03)))
    ]
    return Scene(conductors=bars, frequency=50.0, three_phase=True, line_current=1000.0)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, scene: Scene | None = None):
        super().__init__()
        self.setWindowTitle("emsim — Busbar EM Simulator")
        self.resize(1200, 760)
        self.scene_model = scene or default_scene()
        self._clipboard: list[Conductor] = []
        self._edit_busbar: str | None = None
        self._path = None

        self.canvas = CanvasView(self.scene_model)
        self.properties = PropertyPanel()
        self.properties.set_scene(self.scene_model)
        self.settings = SettingsPanel()
        self.settings.load_from(self.scene_model)
        self.results = ResultsPanel()

        edit = QtWidgets.QWidget()
        ev = QtWidgets.QVBoxLayout(edit)
        ev.addWidget(self.settings)
        ev.addWidget(self.properties)
        ev.addStretch()

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(edit, "Model")
        self.tabs.addTab(self.results, "Results")

        split = QtWidgets.QSplitter()
        split.addWidget(self.canvas)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        self.setCentralWidget(split)

        self._build_menu()
        self._build_toolbar()
        self._build_shortcuts()
        self.statusBar().showMessage(
            "Tip: drag edges to resize, R to rotate, Ctrl+C/V to copy. Same phase = composite bar."
        )

        self.canvas.selectionChangedTo.connect(self._on_selection)
        self.canvas.conductorEdited.connect(lambda c: self._on_selection(c))
        self.canvas.enterGroupRequested.connect(self._set_edit_busbar)
        self.properties.changed.connect(self._on_model_changed)
        self.settings.solveRequested.connect(self._solve)
        self.settings.modeChanged.connect(self._on_mode_changed)

    # ------------------------------------------------------------------ menu
    def _build_menu(self) -> None:
        filem = self.menuBar().addMenu("&File")
        for label, seq, fn in [
            ("&Open configuration…", "Ctrl+O", self._open),
            ("&Save", "Ctrl+S", self._save),
            ("Save &As…", "Ctrl+Shift+S", self._save_as),
        ]:
            act = QtGui.QAction(label, self)
            act.setShortcut(QtGui.QKeySequence(seq))
            act.triggered.connect(fn)
            filem.addAction(act)

    def _set_scene(self, scene: Scene) -> None:
        self.scene_model = scene
        self._edit_busbar = None
        self.canvas._isolation = None
        self.canvas.scene_model = scene
        self.properties.set_scene(scene)
        self.settings.load_from(scene)
        self.canvas.rebuild()
        self.properties.set_conductor(None)

    def _open(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open configuration", "", "emsim config (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            self._set_scene(load_scene(path))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open failed", str(exc))
            return
        self._path = path
        self.statusBar().showMessage(f"Loaded {path}", 5000)

    def _save(self) -> None:
        if self._path:
            self.settings.apply_to(self.scene_model)
            save_scene(self.scene_model, self._path)
            self.statusBar().showMessage(f"Saved {self._path}", 4000)
        else:
            self._save_as()

    def _save_as(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save configuration", "busbar.json", "emsim config (*.json)"
        )
        if not path:
            return
        self.settings.apply_to(self.scene_model)
        save_scene(self.scene_model, path)
        self._path = path
        self.statusBar().showMessage(f"Saved {path}", 4000)

    # ----------------------------------------------------------- toolbar/edit
    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        actions = [
            ("Add bar", lambda: self._add(Rectangle(0.04, 0.01), "A")),
            ("Add round", lambda: self._add(Circle(0.01), "A")),
            ("Add enclosure", self._add_enclosure),
            ("Combine", self._combine),
            ("Copy", self._copy),
            ("Paste", self._paste),
            ("Rotate +15°", lambda: self._rotate_selected(15)),
            ("Delete", self._delete_selected),
            ("Reset example", self._reset_example),
        ]
        for label, fn in actions:
            act = QtGui.QAction(label, self)
            act.triggered.connect(fn)
            tb.addAction(act)
        tb.addSeparator()
        snap = QtGui.QAction("Snap to grid", self)
        snap.setCheckable(True)
        snap.setChecked(GRID["snap"])
        snap.toggled.connect(lambda on: GRID.__setitem__("snap", on))
        tb.addAction(snap)

    def _build_shortcuts(self) -> None:
        QtGui.QShortcut(QtGui.QKeySequence.Copy, self, self._copy)
        QtGui.QShortcut(QtGui.QKeySequence.Paste, self, self._paste)
        QtGui.QShortcut(QtGui.QKeySequence.Delete, self, self._delete_selected)
        QtGui.QShortcut(QtGui.QKeySequence("R"), self, lambda: self._rotate_selected(15))
        QtGui.QShortcut(QtGui.QKeySequence("Shift+R"), self, lambda: self._rotate_selected(-15))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+G"), self, self._combine)
        QtGui.QShortcut(QtGui.QKeySequence("Escape"), self, lambda: self._set_edit_busbar(None))

    def _on_selection(self, cond) -> None:
        """A busbar (outside isolation) or a single shape (inside) was selected."""
        if cond is None:
            self.properties.set_conductor(None)
            return
        members = self.scene_model.members(self.scene_model.busbar_of(cond))
        editable_geom = self._edit_busbar is not None or len(members) == 1
        self.properties.set_conductor(cond, editable_geom=editable_geom)

    def _set_edit_busbar(self, busbar: str | None) -> None:
        """Enter/leave busbar isolation mode (double-click a bar / Esc)."""
        self._edit_busbar = busbar
        self.canvas.set_isolation(busbar)
        if busbar:
            phase = next((c.group for c in self.scene_model.members(busbar)), "?")
            self.statusBar().showMessage(
                f"Editing busbar {busbar} (phase {phase}): Add places shapes inside it, "
                f"drag edges to resize, R to rotate. Esc to exit."
            )
        else:
            self.statusBar().showMessage(
                "Double-click a busbar to edit its shapes. Click selects a whole busbar; "
                "Ctrl+C/V copies it; Combine merges a selection into one busbar."
            )

    def _add(self, shape, group: str | None) -> None:
        if self._edit_busbar is not None:  # new shape joins the busbar being edited
            busbar = self._edit_busbar
            members = self.scene_model.members(busbar)
            group = members[0].group if members else group
        else:  # a brand-new busbar
            busbar = self.scene_model.next_busbar_id()
        name = f"C{len(self.scene_model.conductors) + 1}"
        cond = Conductor(name, shape, Placement(0, 0, 0), COPPER, group=group, busbar=busbar)
        self.scene_model.conductors.append(cond)
        self.canvas.rebuild()
        self.canvas.select_conductor(cond)

    def _combine(self) -> None:
        sel = self.canvas.selected_conductors()
        if len(sel) < 2:
            self.statusBar().showMessage("Select 2+ shapes (rubber-band drag) to combine", 4000)
            return
        busbar = self.scene_model.next_busbar_id()
        phase = next((c.group for c in sel if c.group), None) or "A"
        for c in sel:
            c.busbar = busbar
            c.group = phase
        self.canvas.rebuild()
        self.statusBar().showMessage(
            f"Combined {len(sel)} shapes into one busbar (phase {phase})", 5000
        )

    def _add_enclosure(self) -> None:
        ext = max(
            (abs(c.placement.x) + abs(c.placement.y) + 2 * c.shape.bounding_radius())
            for c in self.scene_model.conductors
        ) if self.scene_model.conductors else 0.06
        outer = Conductor("Encl", Rectangle(2 * ext + 0.02, 2 * ext + 0.02),
                          Placement(0, 0, 0), STEEL, None)
        hole = Conductor("EnclHole", Rectangle(2 * ext, 2 * ext),
                         Placement(0, 0, 0), AIR, None)
        self.scene_model.conductors.extend([outer, hole])
        self.canvas.rebuild()

    def _clone(self, c: Conductor) -> Conductor:
        # keep the shared Material instance (identity matters for the UI combo)
        return Conductor(c.name, copy.deepcopy(c.shape), copy.deepcopy(c.placement),
                         c.material, group=c.group, busbar=c.busbar)

    def _busbars_of_selection(self) -> list[Conductor]:
        """All shapes of every busbar touched by the current selection."""
        selected = self.canvas.selected_conductors()
        busbars = {self.scene_model.busbar_of(c) for c in selected}
        return [c for c in self.scene_model.conductors
                if self.scene_model.busbar_of(c) in busbars]

    def _copy(self) -> None:
        if self._edit_busbar is not None:  # copy the whole busbar being edited
            sel = self.scene_model.members(self._edit_busbar)
        else:  # outside isolation: copy the whole busbar(s) the selection touches
            sel = self._busbars_of_selection()
        if sel:
            self._clipboard = [self._clone(c) for c in sel]
            n_bb = len({self.scene_model.busbar_of(c) for c in sel})
            self.statusBar().showMessage(f"Copied {n_bb} busbar(s), {len(sel)} shape(s)", 3000)

    def _paste(self) -> None:
        if not self._clipboard:
            return
        step = 4 * GRID["mm"] / MM
        existing = set(self.scene_model.busbar_ids())

        def fresh() -> str:
            n = 1
            while f"bb{n}" in existing:
                n += 1
            existing.add(f"bb{n}")
            return f"bb{n}"

        remap: dict[str, str] = {}
        new = []
        for c in self._clipboard:
            src = c.busbar or "_"
            remap.setdefault(src, fresh())
            clone = self._clone(c)
            clone.busbar = remap[src]
            clone.placement.x += step
            clone.placement.y += step
            self.scene_model.conductors.append(clone)
            new.append(clone)
        self._set_edit_busbar(None)  # show the pasted busbar(s) in full view
        self.canvas.rebuild()
        if new:
            self.canvas.select_conductor(new[0])
        self.statusBar().showMessage(
            f"Pasted {len(remap)} new busbar(s) — set their phase in the panel", 5000
        )

    def _rotate_selected(self, delta: float) -> None:
        sel = self.canvas.selected_conductors()
        for c in sel:
            c.placement.rotation = ((c.placement.rotation + delta + 180) % 360) - 180
        if sel:
            self.canvas.rebuild()
            self.canvas.select_conductor(sel[0])

    def _delete_selected(self) -> None:
        if self._edit_busbar is not None:
            sel = self.canvas.selected_conductors()  # individual shapes inside busbar
        else:
            sel = self._busbars_of_selection()  # whole busbar(s)
        for cond in sel:
            if cond in self.scene_model.conductors:
                self.scene_model.conductors.remove(cond)
        self.properties.set_conductor(None)
        self.canvas.rebuild()

    def _reset_example(self) -> None:
        self._path = None
        self._set_scene(default_scene())

    def _on_model_changed(self) -> None:
        cond = self.properties._cond
        if cond is not None:
            self.canvas.refresh_item(cond)
            self.canvas.repaint_busbar(self.scene_model.busbar_of(cond))

    def _on_mode_changed(self) -> None:
        self.settings.apply_to(self.scene_model)
        if self.properties._cond is not None:
            self._on_selection(self.properties._cond)

    # ----------------------------------------------------------------- solve
    def _solve(self) -> None:
        # gmsh installs a SIGINT handler on init, which Python only permits on
        # the main thread -- so the mesh+solve runs synchronously here (with a
        # busy cursor) rather than in a worker thread.
        self.settings.apply_to(self.scene_model)
        self.settings.solve_btn.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.statusBar().showMessage("Meshing…")
            QtWidgets.QApplication.processEvents()
            mesh = self.scene_model.build_mesh()
            self.statusBar().showMessage(f"Solving ({mesh.num_nodes} nodes)…")
            QtWidgets.QApplication.processEvents()
            sol = self.scene_model.solve(mesh)
            result = self.scene_model.analyse(sol)
        except Exception as exc:
            import traceback

            self._on_failed(f"{exc}\n\n{traceback.format_exc()}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.settings.solve_btn.setEnabled(True)
        self._on_solved(result)

    def _on_solved(self, result: SceneResult) -> None:
        self.statusBar().showMessage(
            f"Solved: total loss {result.total_loss:.4g} W/m", 8000
        )
        self.results.set_result(self.scene_model, result)
        self.tabs.setCurrentWidget(self.results)
        self.settings.solve_btn.setEnabled(True)

    def _on_failed(self, msg: str) -> None:
        self.settings.solve_btn.setEnabled(True)
        self.statusBar().showMessage("Solve failed", 5000)
        QtWidgets.QMessageBox.critical(self, "Solve failed", msg)
