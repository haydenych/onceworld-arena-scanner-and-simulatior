"""Canonical package and project paths."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PACKAGE_ROOT / "data"
ASSET_DIR = PROJECT_ROOT / "assets"
ANCHOR_DIR = ASSET_DIR / "anchors"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
UNLABELED_ICON_DIR = PROJECT_ROOT / "dataset_enemy_icons"

ATTACK_RANGE_JSON = DATA_DIR / "attack_range.json"
MONSTERS_CSV = DATA_DIR / "monsters.csv"
TESSERACT_EXE_FALLBACK = PROJECT_ROOT / "Tesseract-OCR" / "tesseract.exe"

