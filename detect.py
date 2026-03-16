"""End-user launcher.

Run this file for normal detection.
Edit settings in user_config.py only.
"""

from detect_lib import run_app


if __name__ == "__main__":
    run_app(save_unlabeled=False)
