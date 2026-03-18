"""Developer launcher.

Uses the same runtime config as detect.py, but also saves unlabeled icons.
"""

from onceworld.app.entry import run


if __name__ == "__main__":
    run(dev=True)
