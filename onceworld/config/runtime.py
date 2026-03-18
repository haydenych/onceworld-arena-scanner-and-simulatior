"""Runtime configuration for detection, OCR, model, and simulation."""

from pathlib import Path

import numpy as np

import user_config

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PACKAGE_ROOT / "data"

ASSET_DIR = PROJECT_ROOT / "assets"
ANCHOR_DIR = ASSET_DIR / "anchors"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
UNLABELED_ICON_DIR = PROJECT_ROOT / "dataset_enemy_icons"

ATTACK_RANGE_JSON = DATA_DIR / "attack_range.json"
MONSTERS_CSV = DATA_DIR / "monsters.csv"
TESSERACT_EXE_FALLBACK = PROJECT_ROOT / "Tesseract-OCR" / "tesseract.exe"

ANCHOR_SCALES = np.asarray(user_config.ANCHOR_SCALES, dtype=float)
ANCHOR_THRESHOLD = float(user_config.ANCHOR_THRESHOLD)

UNIT_THRESHOLD = 0.20
UNIT_MARGIN_THRESHOLD = 0.02
UNIT_WARN_THRESHOLD = 0.35

CHECKPOINT_GLOB = "unit_resnet18_*.pt"

UNLABELED_ICON_PREFIX = "img"
UNLABELED_ICON_DIGITS = 4

BATTLE_MONSTERS_CSV = MONSTERS_CSV
BATTLE_SIM_TRIALS = 100
BATTLE_SIM_DELTA_TIME = 0.02
BATTLE_SIM_DURATION = 40.0

TESS_CONFIG_LEVEL = r"--psm 7 -c tessedit_char_whitelist=Lv.0123456789"
TESS_CONFIG_COIN = r"--psm 7 -c tessedit_char_whitelist=0123456789xX, "
