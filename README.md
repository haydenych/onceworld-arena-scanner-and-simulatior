# OnceWorld Arena Scanner & Simulator

Vibe-coding project, just for fun. No guaranteed maintenance.

## Demo

<video src="assets/demo.mp4" controls muted playsinline width="960">
  Your browser does not support the video tag.
</video>

If inline playback does not render in your GitHub client, use [Watch demo](assets/demo.gif).

## For users

### Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Install Tesseract OCR (or set `TESSERACT_CMD` in `user_config.py`)

### Run
- Run `python detect.py`

### User config
Edit `user_config.py` only.

Main knobs:
- `TESSERACT_CMD` (optional if Tesseract is already in PATH)
- `ANCHOR_SCALES`
- `ANCHOR_THRESHOLD`

### Notes
- Use emulator full screen for more stable anchor and icon sizing.
- Best classifier accuracy is when enemy icons are roughly 50-90 px.
- Checkpoint is auto-selected from `checkpoints/` (latest `unit_resnet18_*.pt`).
- `detect_dev.py` is dev mode (debug/perf logs + unlabeled icon saving).

## For developers

### Project layout
- Runtime launchers: `detect.py`, `detect_dev.py`
- App/UI: `onceworld/app/entry.py`, `onceworld/app/window.py`
- Core runtime: `onceworld/core/detector.py`, `onceworld/core/vision.py`, `onceworld/core/ocr.py`, `onceworld/core/classifier.py`, `onceworld/core/sim.py`, `onceworld/core/perf.py`
- Core ML/data helpers: `onceworld/core/data_collection.py`, `onceworld/core/icon_preprocess.py`, `onceworld/core/modeling.py`
- Config: `onceworld/config/runtime.py`, `onceworld/config/geometry.py`, `user_config.py`
- Runtime data: `onceworld/data/monsters.csv`, `onceworld/data/attack_range.json`
- Dev tools: `dev_tools/screenshot.py`, `dev_tools/ml_data.py`, `dev_tools/label_tool.py`, `dev_tools/ml.py`, `dev_tools/quick_prelabelling.ipynb`

### ML preprocessing flow
- Real (train/val/inference): RGB -> pad to square (no crop) -> resize to 128 -> normalize.
- Synthetic train: template -> downsample to random `[50, 85]` -> low-res random shift (`+-max_shift_px`, padded) -> upsample to 128 -> normalize.
- Synthetic val: same as synthetic train but deterministic (`small_px` midpoint, zero shift).
- Real split uses `StratifiedShuffleSplit` for classes with at least two samples; singleton classes are assigned to train.
- Training mixes synthetic and real using `--real-mix-ratio-train` and `--real-mix-ratio-val`.

### Typical workflow
1. Collect icons: `python dev_tools/ml_data.py` for arena icons or `python dev_tools/screenshot.py` for monster dex synthetic icons
2. Run `dev_tools/quick_prelabelling.ipynb` for fast labels to expand dataset and check for correctness
3. Label real icons: `python dev_tools/label_tool.py`
4. Train model: `python dev_tools/ml.py`
5. Run detector: `python detect.py` (or `python detect_dev.py` for dev mode)

### Shared internals
- `onceworld/config/geometry.py` is the shared detector geometry/constants module.
- `user_config.py` is intentionally minimal and user-focused.
- `ANCHOR_SCALES` and `ANCHOR_THRESHOLD` from `user_config.py` are used by runtime detection and `dev_tools/ml_data.py`.

## Battle simulation credit

Battle simulation logic in this project follows and is adapted from:

https://github.com/nekocant/onceworld_arena_sim/tree/main

Full credit for the battle simulation logic and design goes to the original `onceworld_arena_sim` project and its author(s).
