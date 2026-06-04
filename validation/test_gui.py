r"""Milestone 7 smoke test: the PySide6 GUI runs end-to-end (headless).

Constructs the main window, runs a real solve through the worker, and switches
field views -- exercising the GUI's non-visual logic without a display. Skips
cleanly if PySide6 is not installed.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


def test_gui_build_solve_and_views():
    from PySide6 import QtWidgets

    from emsim.gui.mainwindow import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    win.show()
    assert len(win.scene_model.conductors) == 3  # balanced 3-phase A/B/C

    # exercise the real (synchronous) solve path used by the Solve button
    win._solve()
    app.processEvents()

    assert win.results.table.rowCount() == 3
    assert "R_AC" in win.results.summary.text()

    # three-phase currents are locked to 0 / -120 / +120 degrees
    import numpy as np

    phases = sorted(round(np.degrees(np.angle(cr.current))) for cr in
                    win.scene_model.analyse(win.scene_model.solve()).conductors)
    assert phases == [-120, 0, 120]

    for kind in ("|A_z|", "|B|", "|J_z|"):
        win.results.field_combo.setCurrentText(kind)
        app.processEvents()
