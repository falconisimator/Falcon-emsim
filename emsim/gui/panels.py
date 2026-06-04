"""Editor panels: per-conductor properties and global simulation settings."""

from __future__ import annotations

import cmath
import math

from PySide6 import QtCore, QtWidgets

from emsim.geometry.model import Conductor
from emsim.geometry.shapes import Circle, Rectangle
from emsim.materials import ALUMINIUM, COPPER, STEEL
from emsim.scene import THREE_PHASE_DEG

MM = 1000.0
MATERIALS = {"Copper": COPPER, "Aluminium": ALUMINIUM, "Steel": STEEL}
PHASE_GROUPS = ["A", "B", "C"]


def _spin(minimum, maximum, value, step, decimals=3, suffix="") -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(minimum, maximum)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    if suffix:
        s.setSuffix(suffix)
    return s


class PropertyPanel(QtWidgets.QGroupBox):
    """Edits the selected conductor (dimensions, placement, material, group)."""

    changed = QtCore.Signal()

    def __init__(self):
        super().__init__("Selected busbar / shape")
        self._scene = None
        self._cond: Conductor | None = None
        self._editable_geom = True
        self._form = QtWidgets.QFormLayout(self)

        self.name = QtWidgets.QLineEdit()
        self.dim1 = _spin(0.1, 1000, 20, 1, suffix=" mm")
        self.dim2 = _spin(0.1, 1000, 5, 1, suffix=" mm")
        self.posx = _spin(-1000, 1000, 0, 1, suffix=" mm")
        self.posy = _spin(-1000, 1000, 0, 1, suffix=" mm")
        self.rot = _spin(-180, 180, 0, 15, suffix=" °")
        self.material = QtWidgets.QComboBox()
        self.material.addItems(list(MATERIALS))
        self.group = QtWidgets.QComboBox()
        self.group.addItems(PHASE_GROUPS)
        self.cur_mag = _spin(0, 1e9, 1000, 100, decimals=2, suffix=" A")
        self.cur_phase = _spin(-180, 180, 0, 5, suffix=" °")
        self.passive = QtWidgets.QCheckBox("Passive (enclosure / no terminal)")

        self._form.addRow("Name", self.name)
        self._form.addRow("Width / Radius", self.dim1)
        self._form.addRow("Height", self.dim2)
        self._form.addRow("x", self.posx)
        self._form.addRow("y", self.posy)
        self._form.addRow("Rotation", self.rot)
        self._form.addRow("Material", self.material)
        self._form.addRow("Phase / group", self.group)
        self._form.addRow("Current |I|", self.cur_mag)
        self._form.addRow("Current phase", self.cur_phase)
        self._form.addRow("", self.passive)

        for w in (self.dim1, self.dim2, self.posx, self.posy, self.rot,
                  self.cur_mag, self.cur_phase):
            w.valueChanged.connect(self._apply)
        self.name.editingFinished.connect(self._apply)
        self.group.currentTextChanged.connect(self._apply)
        self.material.currentTextChanged.connect(self._apply)
        self.passive.toggled.connect(self._on_passive)
        self.setEnabled(False)

    def set_scene(self, scene) -> None:
        self._scene = scene

    def set_conductor(self, cond: Conductor | None, editable_geom: bool = True) -> None:
        self._cond = cond
        self._editable_geom = editable_geom
        self.setEnabled(cond is not None)
        if cond is None:
            return
        widgets = self.findChildren(QtWidgets.QWidget)
        for w in widgets:
            w.blockSignals(True)
        self.name.setText(cond.name)
        is_rect = isinstance(cond.shape, Rectangle)
        self.dim2.setVisible(is_rect)
        self._form.labelForField(self.dim2).setVisible(is_rect)
        if is_rect:
            self.dim1.setValue(cond.shape.width * MM)
            self.dim2.setValue(cond.shape.height * MM)
        elif isinstance(cond.shape, Circle):
            self.dim1.setValue(cond.shape.radius * MM)
        self.posx.setValue(cond.placement.x * MM)
        self.posy.setValue(cond.placement.y * MM)
        self.rot.setValue(cond.placement.rotation)
        self.material.setCurrentText(
            next((k for k, v in MATERIALS.items() if v is cond.material), "Copper")
        )
        self.passive.setChecked(cond.group is None)

        active = cond.group is not None
        self.group.setEnabled(active)
        if active and cond.group in PHASE_GROUPS:
            self.group.setCurrentText(cond.group)
        three_phase = active and self._scene.three_phase and cond.group in THREE_PHASE_DEG
        if active and not self._scene.three_phase:
            cur = self._scene.group_currents.get(cond.group, 1000 + 0j)
            self.cur_mag.setValue(abs(cur))
            self.cur_phase.setValue(math.degrees(cmath.phase(cur)))
        elif three_phase:
            self.cur_mag.setValue(self._scene.line_current)
            self.cur_phase.setValue(THREE_PHASE_DEG[cond.group])
        # in 3-phase mode the current is set globally (Simulation panel)
        self.cur_mag.setEnabled(active and not self._scene.three_phase)
        self.cur_phase.setEnabled(active and not self._scene.three_phase)
        # geometry is edited only inside a busbar (double-click to enter)
        for w in (self.dim1, self.dim2, self.posx, self.posy, self.rot):
            w.setEnabled(editable_geom)
        if not editable_geom:
            self.name.setText(f"Busbar {cond.busbar or '?'} ({len(self._scene.members(self._scene.busbar_of(cond)))} shapes)")
            self.name.setEnabled(False)
        else:
            self.name.setEnabled(True)
        for w in widgets:
            w.blockSignals(False)

    def _members(self):
        return self._scene.members(self._scene.busbar_of(self._cond))

    def _on_passive(self, checked: bool) -> None:
        if self._cond is None:
            return
        phase = None if checked else (self.group.currentText() or "A")
        for m in self._members():  # passive/phase applies to the whole busbar
            m.group = phase
            if checked:
                m.material = STEEL
        self._apply()
        self.set_conductor(self._cond, self._editable_geom)

    def _apply(self) -> None:
        c = self._cond
        if c is None or self._scene is None:
            return
        # busbar-wide: phase and material apply to every shape in the busbar
        mat = MATERIALS[self.material.currentText()]
        phase = None if self.passive.isChecked() else self.group.currentText()
        for m in self._members():
            m.material = mat
            m.group = phase
        if phase is not None and not self._scene.three_phase:
            self._scene.group_currents[phase] = cmath.rect(
                self.cur_mag.value(), math.radians(self.cur_phase.value())
            )
        # per-shape: geometry only when editing inside the busbar
        if self._editable_geom:
            c.name = self.name.text() or c.name
            if isinstance(c.shape, Rectangle):
                c.shape.width = self.dim1.value() / MM
                c.shape.height = self.dim2.value() / MM
            elif isinstance(c.shape, Circle):
                c.shape.radius = self.dim1.value() / MM
            c.placement.x = self.posx.value() / MM
            c.placement.y = self.posy.value() / MM
            c.placement.rotation = self.rot.value()
        self.changed.emit()


class SettingsPanel(QtWidgets.QGroupBox):
    """Global settings: frequency, 3-phase current, boundary, order; Solve."""

    solveRequested = QtCore.Signal()
    modeChanged = QtCore.Signal()

    def __init__(self):
        super().__init__("Simulation")
        form = QtWidgets.QFormLayout(self)
        self.freq = _spin(0, 1e9, 50, 10, decimals=2, suffix=" Hz")
        self.three_phase = QtWidgets.QCheckBox("Balanced 3-phase (A/B/C = 0 / -120 / +120°)")
        self.three_phase.setChecked(True)
        self.line_current = _spin(0, 1e9, 1000, 100, decimals=2, suffix=" A")
        self.line_current.setToolTip(
            "Total current per phase terminal. Parallel bars sharing a phase divide "
            "this current according to the EM solution (uneven); the solver reports each "
            "bar's share and the terminal voltage gradient / impedance."
        )
        self.boundary = QtWidgets.QComboBox()
        self.boundary.addItems(["kelvin", "dirichlet"])
        self.order = QtWidgets.QComboBox()
        self.order.addItems(["P1 (linear)", "P2 (quadratic)"])
        self.solve_btn = QtWidgets.QPushButton("Solve")
        form.addRow("Frequency", self.freq)
        form.addRow(self.three_phase)
        form.addRow("Terminal current |I|", self.line_current)
        form.addRow("Open boundary", self.boundary)
        form.addRow("Elements", self.order)
        form.addRow(self.solve_btn)
        self.solve_btn.clicked.connect(self.solveRequested)
        self.three_phase.toggled.connect(self.modeChanged)
        self.line_current.valueChanged.connect(self.modeChanged)

    def apply_to(self, scene) -> None:
        scene.frequency = self.freq.value()
        scene.three_phase = self.three_phase.isChecked()
        scene.line_current = self.line_current.value()
        scene.boundary = self.boundary.currentText()
        scene.order = 2 if self.order.currentIndex() == 1 else 1

    def load_from(self, scene) -> None:
        self.freq.setValue(scene.frequency)
        self.three_phase.setChecked(scene.three_phase)
        self.line_current.setValue(scene.line_current)
        self.boundary.setCurrentText(scene.boundary)
        self.order.setCurrentIndex(1 if scene.order == 2 else 0)
