"""Shared app entry points for normal and dev launch modes."""

import sys

from PySide6.QtWidgets import QApplication

from onceworld.app.window import MainWindow
from onceworld.core.perf import set_debug


def run_app(save_unlabeled=False, debug=False):
    set_debug(debug)
    app = QApplication(sys.argv)
    win = MainWindow(save_unlabeled=save_unlabeled)
    if not win.detector:
        return
    win.show()
    win.raise_()
    win.activateWindow()
    app.exec()


def run(dev=False):
    dev_mode = bool(dev)
    run_app(save_unlabeled=dev_mode, debug=dev_mode)


if __name__ == "__main__":
    run(dev=False)

