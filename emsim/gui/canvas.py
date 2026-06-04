"""Interactive canvas: place, move, resize and rotate conductor profiles.

Scene coordinates are millimetres with y pointing up (the view applies a
vertical flip). Each conductor is one graphics item kept in sync with the
underlying :class:`~emsim.geometry.model.Conductor`. Selected rectangles expose
edge/corner handles (drag to resize) and a rotate handle; circles expose a
radius handle. Positions snap to the grid.
"""

from __future__ import annotations

import math

from PySide6 import QtCore, QtGui, QtWidgets

from emsim.geometry.model import Conductor
from emsim.geometry.shapes import Circle, Polygon, Rectangle

MM = 1000.0  # scene units (mm) per metre

# Grid settings (scene units = mm), shared by the view (drawing) and items
# (snapping). Minor line every ``mm``; a heavier major line every 5th.
GRID = {"mm": 5.0, "snap": True}

_GROUP_COLORS = {
    "A": "#d65f5f", "B": "#5f9ed6", "C": "#5fd68a",
    "D": "#d6b25f", "E": "#9a5fd6", "F": "#5fd6d0",
}


def group_color(group: str | None) -> QtGui.QColor:
    if group is None:
        return QtGui.QColor("#8a8a8a")  # passive (enclosure)
    return QtGui.QColor(_GROUP_COLORS.get(group, "#d65f5f"))


def _snap(v: float) -> float:
    g = GRID["mm"]
    return round(v / g) * g if GRID["snap"] else v


class ConductorItem(QtWidgets.QGraphicsItem):
    """A movable/resizable/rotatable item bound to one Conductor."""

    def __init__(self, conductor: Conductor, view: "CanvasView"):
        super().__init__()
        self.conductor = conductor
        self._view = view
        self._mode: str | None = None
        self._hx = self._hy = 0
        self._resizing = False
        self._editable = False  # individual-shape editing (isolation mode only)
        self.setFlags(
            QtWidgets.QGraphicsItem.ItemIsMovable
            | QtWidgets.QGraphicsItem.ItemIsSelectable
            | QtWidgets.QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setPos(conductor.placement.x * MM, conductor.placement.y * MM)
        self.setRotation(-conductor.placement.rotation)

    # ---- geometry helpers (local mm) ----------------------------------
    def _half(self) -> tuple[float, float]:
        s = self.conductor.shape
        if isinstance(s, Rectangle):
            return s.width * MM / 2, s.height * MM / 2
        if isinstance(s, Circle):
            return s.radius * MM, s.radius * MM
        r = s.bounding_radius() * MM
        return r, r

    def _handle_size(self) -> float:
        ax, ay = self._half()
        return min(max(0.12 * min(ax, ay), 1.2), 4.0)

    def _rot_gap(self) -> float:
        _, ay = self._half()
        return max(6.0, 0.4 * ay)

    def _resize_handles(self) -> list[tuple[float, float, int, int]]:
        """Handle (x, y, hx, hy) in local mm; hx/hy = which edge moves."""
        ax, ay = self._half()
        s = self.conductor.shape
        if isinstance(s, Rectangle):
            out = []
            for hx in (-1, 0, 1):
                for hy in (-1, 0, 1):
                    if hx == 0 and hy == 0:
                        continue
                    out.append((hx * ax, hy * ay, hx, hy))
            return out
        if isinstance(s, Circle):
            return [(ax, 0, 1, 0), (-ax, 0, -1, 0), (0, ay, 0, 1), (0, -ay, 0, -1)]
        return []

    def boundingRect(self) -> QtCore.QRectF:
        ax, ay = self._half()
        hs = self._handle_size() + 2
        top = ay + (self._rot_gap() + hs if isinstance(self.conductor.shape, Rectangle) else hs)
        return QtCore.QRectF(-ax - hs, -ay - hs, 2 * ax + 2 * hs, ay + top + hs)

    # ---- painting -----------------------------------------------------
    def paint(self, painter: QtGui.QPainter, option, widget=None) -> None:
        s = self.conductor.shape
        col = group_color(self.conductor.group)
        painter.setPen(
            QtGui.QPen(QtGui.QColor("#1f6feb"), 2.0)
            if self.isSelected()
            else QtGui.QPen(QtGui.QColor("#222"), 1.2)
        )
        painter.setBrush(QtGui.QBrush(col))
        ax, ay = self._half()
        if isinstance(s, Rectangle):
            painter.drawRect(QtCore.QRectF(-ax, -ay, 2 * ax, 2 * ay))
        elif isinstance(s, Circle):
            painter.drawEllipse(QtCore.QRectF(-ax, -ay, 2 * ax, 2 * ay))
        else:
            painter.drawPolygon(
                QtGui.QPolygonF([QtCore.QPointF(px * MM, py * MM) for px, py in s.points])
            )

        # upright label (counter the view y-flip): short phase letter
        label = self.conductor.group or ""
        if label:
            painter.save()
            painter.scale(1, -1)
            f = painter.font()
            f.setBold(True)
            painter.setFont(f)
            painter.setPen(QtGui.QColor("#fff"))
            painter.drawText(
                QtCore.QRectF(-ax, -ay, 2 * ax, 2 * ay), QtCore.Qt.AlignCenter, label
            )
            painter.restore()

        if self.isSelected() and self._editable:
            self._paint_handles(painter)

    def _paint_handles(self, painter: QtGui.QPainter) -> None:
        hs = self._handle_size()
        painter.setPen(QtGui.QPen(QtGui.QColor("#1f6feb"), 1.0))
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#ffffff")))
        for x, y, _, _ in self._resize_handles():
            painter.drawRect(QtCore.QRectF(x - hs, y - hs, 2 * hs, 2 * hs))
        if isinstance(self.conductor.shape, Rectangle):
            _, ay = self._half()
            ry = ay + self._rot_gap()
            painter.drawLine(QtCore.QLineF(0, ay, 0, ry))
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#1f6feb")))
            painter.drawEllipse(QtCore.QPointF(0, ry), hs, hs)

    # ---- interaction --------------------------------------------------
    def _hit_rotate(self, p: QtCore.QPointF) -> bool:
        if not isinstance(self.conductor.shape, Rectangle):
            return False
        _, ay = self._half()
        ry = ay + self._rot_gap()
        return math.hypot(p.x() - 0, p.y() - ry) <= self._handle_size() * 1.8

    def _hit_resize(self, p: QtCore.QPointF):
        hs = self._handle_size() * 1.6
        for x, y, hx, hy in self._resize_handles():
            if abs(p.x() - x) <= hs and abs(p.y() - y) <= hs:
                return hx, hy
        return None

    def mousePressEvent(self, event) -> None:
        p = event.pos()
        if self._editable and self.isSelected() and self._hit_rotate(p):
            self._mode = "rotate"
            event.accept()
            return
        hit = self._hit_resize(p) if (self._editable and self.isSelected()) else None
        if hit is not None:
            self._mode = "resize"
            self._hx, self._hy = hit
            event.accept()
            return
        self._mode = "move"
        super().mousePressEvent(event)
        if not self._editable:  # whole-busbar drag: select all siblings now
            self._view.select_busbar(self.conductor)

    def mouseMoveEvent(self, event) -> None:
        if self._mode == "resize":
            self._do_resize(event.pos())
        elif self._mode == "rotate":
            self._do_rotate(event.scenePos())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        mode, self._mode = self._mode, None
        super().mouseReleaseEvent(event)
        if mode in ("resize", "rotate", "move"):
            self._view.conductorEdited.emit(self.conductor)

    def _do_resize(self, p: QtCore.QPointF) -> None:
        s = self.conductor.shape
        ax, ay = self._half()
        if isinstance(s, Circle):
            r_mm = max(_snap(math.hypot(p.x(), p.y())), GRID["mm"])
            self.prepareGeometryChange()
            s.radius = r_mm / MM
            self.update()
            return
        left, right, bottom, top = -ax, ax, -ay, ay
        # snap the dragged edge(s) to the grid so dimensions land on grid lines
        if self._hx == 1:
            right = max(_snap(p.x()), left + GRID["mm"])
        elif self._hx == -1:
            left = min(_snap(p.x()), right - GRID["mm"])
        if self._hy == 1:
            top = max(_snap(p.y()), bottom + GRID["mm"])
        elif self._hy == -1:
            bottom = min(_snap(p.y()), top - GRID["mm"])
        new_w, new_h = right - left, top - bottom
        center_local = QtCore.QPointF((left + right) / 2, (bottom + top) / 2)
        center_scene = self.mapToScene(center_local)
        self.prepareGeometryChange()
        s.width, s.height = new_w / MM, new_h / MM
        self._resizing = True
        self.setPos(center_scene)
        self._resizing = False
        self.update()

    def _do_rotate(self, scene_pos: QtCore.QPointF) -> None:
        center = self.pos()  # item origin in scene coords
        ang = math.degrees(math.atan2(scene_pos.y() - center.y(), scene_pos.x() - center.x()))
        rot = ang - 90.0  # handle sits at +y (top) when rotation is 0
        rot = round(rot / 15.0) * 15.0  # snap to 15 deg
        self.conductor.placement.rotation = rot
        self.setRotation(-rot)
        self.update()

    def itemChange(self, change, value):
        if (
            change == QtWidgets.QGraphicsItem.ItemPositionChange
            and GRID["snap"]
            and not self._resizing
        ):
            value.setX(_snap(value.x()))
            value.setY(_snap(value.y()))
            return value
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self.conductor.placement.x = self.pos().x() / MM
            self.conductor.placement.y = self.pos().y() / MM
        return super().itemChange(change, value)


class CanvasView(QtWidgets.QGraphicsView):
    """View showing all conductors; emits selection and edit notifications."""

    selectionChangedTo = QtCore.Signal(object)  # Conductor | None
    conductorEdited = QtCore.Signal(object)  # Conductor
    enterGroupRequested = QtCore.Signal(object)  # group name | None

    def __init__(self, scene_model):
        super().__init__()
        self.scene_model = scene_model
        self._isolation: str | None = None  # busbar id being edited, or None
        self._expanding = False
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setTransform(QtGui.QTransform().scale(1, -1))  # y up
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setBackgroundBrush(QtGui.QColor("#fbfcfd"))
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        self._scene.selectionChanged.connect(self._on_selection)
        self.rebuild()

    def rebuild(self) -> None:
        self._scene.blockSignals(True)
        self._scene.clear()
        for c in self.scene_model.conductors:
            self._scene.addItem(ConductorItem(c, self))
        self._scene.blockSignals(False)
        self._apply_isolation()
        self._fit()

    def set_isolation(self, busbar: str | None) -> None:
        """Enter/leave busbar edit mode: only that busbar's shapes stay active."""
        self._isolation = busbar
        self._apply_isolation()

    def _apply_isolation(self) -> None:
        iso = self._isolation
        for it in self._scene.items():
            if not isinstance(it, ConductorItem):
                continue
            member = iso is None or self.scene_model.busbar_of(it.conductor) == iso
            it.setOpacity(1.0 if member else 0.22)
            it.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, member)
            it.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, member)
            # individual-shape editing (handles, resize, rotate) only in isolation
            it._editable = iso is not None and member

    def mouseDoubleClickEvent(self, event) -> None:
        sp = self.mapToScene(event.position().toPoint())
        item = self._scene.itemAt(sp, self.transform())
        busbar = (
            self.scene_model.busbar_of(item.conductor)
            if isinstance(item, ConductorItem)
            else None
        )
        self.enterGroupRequested.emit(busbar)

    def selected_conductors(self) -> list:
        return [
            it.conductor
            for it in self._scene.selectedItems()
            if isinstance(it, ConductorItem)
        ]

    def _item_for(self, cond) -> "ConductorItem | None":
        for it in self._scene.items():
            if isinstance(it, ConductorItem) and it.conductor is cond:
                return it
        return None

    def select_conductor(self, cond) -> None:
        item = self._item_for(cond)
        if item is not None:
            self._scene.clearSelection()
            item.setSelected(True)

    def select_busbar(self, cond) -> None:
        """Select every shape of the conductor's busbar (whole-busbar mode)."""
        bb = self.scene_model.busbar_of(cond)
        self._expanding = True
        for it in self._scene.items():
            if isinstance(it, ConductorItem):
                it.setSelected(self.scene_model.busbar_of(it.conductor) == bb)
        self._expanding = False

    def refresh_item(self, cond) -> None:
        """Sync an item to its conductor after an edit from the property panel."""
        item = self._item_for(cond)
        if item is None:
            return
        item.prepareGeometryChange()
        item.setPos(cond.placement.x * MM, cond.placement.y * MM)
        item.setRotation(-cond.placement.rotation)
        item.update()

    def repaint_busbar(self, busbar) -> None:
        """Repaint all shapes of a busbar (e.g. after a phase/colour change)."""
        for it in self._scene.items():
            if isinstance(it, ConductorItem) and self.scene_model.busbar_of(it.conductor) == busbar:
                it.update()

    def drawBackground(self, painter: QtGui.QPainter, rect: QtCore.QRectF) -> None:
        super().drawBackground(painter, rect)
        g = GRID["mm"]
        if self.transform().m11() * g < 3:
            g *= 5
        minor = QtGui.QPen(QtGui.QColor("#e3e6ea"), 0)
        major = QtGui.QPen(QtGui.QColor("#cdd2d8"), 0)
        axis = QtGui.QPen(QtGui.QColor("#9aa0a6"), 0)
        x = math.floor(rect.left() / g) * g
        while x <= rect.right():
            n = round(x / g)
            painter.setPen(axis if abs(x) < 1e-6 else (major if n % 5 == 0 else minor))
            painter.drawLine(QtCore.QLineF(x, rect.top(), x, rect.bottom()))
            x += g
        y = math.floor(rect.top() / g) * g
        while y <= rect.bottom():
            n = round(y / g)
            painter.setPen(axis if abs(y) < 1e-6 else (major if n % 5 == 0 else minor))
            painter.drawLine(QtCore.QLineF(rect.left(), y, rect.right(), y))
            y += g

    def _fit(self) -> None:
        rect = self._scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        if rect.isValid():
            self.fitInView(rect, QtCore.Qt.KeepAspectRatio)

    def _on_selection(self) -> None:
        if self._expanding:
            return
        items = [it for it in self._scene.selectedItems() if isinstance(it, ConductorItem)]
        # outside isolation, selecting any shape selects its whole busbar
        if self._isolation is None and items:
            busbars = {self.scene_model.busbar_of(it.conductor) for it in items}
            self._expanding = True
            for it in self._scene.items():
                if isinstance(it, ConductorItem):
                    it.setSelected(self.scene_model.busbar_of(it.conductor) in busbars)
            self._expanding = False
            items = [it for it in self._scene.selectedItems() if isinstance(it, ConductorItem)]
        cond = items[0].conductor if items else None
        self.selectionChangedTo.emit(cond)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self._scene.update()
