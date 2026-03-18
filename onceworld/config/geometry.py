"""Shared detector geometry constants.

This module is used by runtime detection and developer tools.
"""

# Row box ratios relative to anchor size.
ROW_X1 = -0.50
ROW_Y1 = -0.20
ROW_X2 = 9.00
ROW_Y2 = 3.60

# White portrait-frame detection inside dark-brown row.
ENEMY_MAX_SLOTS = 4
ENEMY_WHITE_HSV_LOW = (0, 0, 165)
ENEMY_WHITE_HSV_HIGH = (180, 70, 255)
ENEMY_BOX_MIN_AREA_RATIO = 0.015
ENEMY_BOX_MAX_AREA_RATIO = 0.22
ENEMY_BOX_ASPECT_MIN = 0.75
ENEMY_BOX_ASPECT_MAX = 1.30
ENEMY_BOX_MAX_X_RATIO = 0.78
ENEMY_BOX_MIN_Y_RATIO = 0.15
ENEMY_BOX_MAX_Y_RATIO = 0.92

# Coin OCR area (x1, y1, x2, y2) relative to row.
COIN_ZONE = (0.80, 0.12, 0.98, 0.35)

__all__ = [
    "ROW_X1",
    "ROW_Y1",
    "ROW_X2",
    "ROW_Y2",
    "ENEMY_MAX_SLOTS",
    "ENEMY_WHITE_HSV_LOW",
    "ENEMY_WHITE_HSV_HIGH",
    "ENEMY_BOX_MIN_AREA_RATIO",
    "ENEMY_BOX_MAX_AREA_RATIO",
    "ENEMY_BOX_ASPECT_MIN",
    "ENEMY_BOX_ASPECT_MAX",
    "ENEMY_BOX_MAX_X_RATIO",
    "ENEMY_BOX_MIN_Y_RATIO",
    "ENEMY_BOX_MAX_Y_RATIO",
    "COIN_ZONE",
]

