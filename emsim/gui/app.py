"""Application entry point: ``python -m emsim.gui``."""

from __future__ import annotations

import sys


def main() -> int:
    from PySide6 import QtWidgets

    from emsim.gui.mainwindow import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
