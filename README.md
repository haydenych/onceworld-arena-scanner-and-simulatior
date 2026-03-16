# OnceWorld Arena Scanner & Simulator

Vibe-coding project, just for fun. No guaranteed maintenance.

## For users

Use this section if you only want to run detection.

### What to run
- Run `python detect.py`

### What to edit
- Edit `user_config.py` only.
- Main user knobs:
- `TESSERACT_CMD` (optional, only if Tesseract is not in PATH)
- `ASSET_DIR`, `ANCHOR_DIR`
- `ANCHOR_SCALES`
- `ANCHOR_THRESHOLD`
- `UNIT_THRESHOLD`
- `UNIT_MARGIN_THRESHOLD`

### Notes
- Checkpoints are auto-detected from `checkpoints/` (latest `unit_resnet18_*.pt`).
- `detect_dev.py` is not needed for normal use.
- For better detection accuracy, use a larger game/emulator window size so arena monster icons are roughly in the 50-90 px range.
- The icon classifier was mainly trained on monster icons around 50-90 px; much smaller icons can reduce detection quality.

## For developers

Use this section for dataset collection, labeling, and training workflows.

### Project layout
- Runtime app:
- `detect.py`, `detect_dev.py`, `detect_lib.py`, `detect_common.py`, `user_config.py`
- Dev tools:
- `dev_tools/screenshot.py`
- `dev_tools/ml_data.py`
- `dev_tools/label_tool.py`
- `dev_tools/ml.py`
- `dev_tools/generate_synthetic_labels.ipynb`

### Typical workflow
1. Collect icons with `python dev_tools/ml_data.py` or `python dev_tools/screenshot.py`.
2. Label unlabeled icons with `python dev_tools/label_tool.py`.
3. Train with `python dev_tools/ml.py`.
4. Run detector with `python detect.py` (or `python detect_dev.py` for dev mode).

### Shared internals
- `detect_common.py` keeps shared detector geometry/constants to reduce drift between runtime and dev collection code.
- `user_config.py` is intentionally minimal and end-user focused.

## Battle Simulation Credit

Battle simulation logic in this project follows and is adapted from:

https://github.com/nekocant/onceworld_arena_sim/tree/main

Full credit for the battle simulation logic and design goes to the original `onceworld_arena_sim` project and its author(s).
