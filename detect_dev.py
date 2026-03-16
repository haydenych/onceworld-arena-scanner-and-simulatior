"""Developer launcher.

Uses the same runtime config as detect.py, but also saves unlabeled icons.
"""

from detect_lib import run_app


if __name__ == "__main__":
    run_app(save_unlabeled=True)
