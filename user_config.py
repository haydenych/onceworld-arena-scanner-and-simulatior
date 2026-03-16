import numpy as np

# ============================================================
# End-user config for arena detection
# ============================================================
# Edit this file only.
# - Tesseract override is optional.
# - Checkpoints are auto-picked from ./checkpoints.

# Optional OCR override.
# If Tesseract is already in PATH, keep this as None.
# Example:
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = None

# Asset paths
# Root folder for runtime assets.
# Usually keep this as "assets" unless you reorganize the project.
ASSET_DIR = "assets"

# Folder containing team anchor templates (team_a/team_b/team_c images).
# Detector uses these to find each team row on screen.
ANCHOR_DIR = f"{ASSET_DIR}/anchors"

# Anchor multi-scale search
# Scale multipliers for anchor template matching.
# Wider range = more robust to emulator/window size changes, but slower.
# Narrower range = faster, but can miss anchors if UI scale changes.
ANCHOR_SCALES = np.linspace(0.75, 1.35, 13)

# Detection thresholds
# Minimum template-match score to accept an anchor hit.
# Raise to reduce false anchors; lower if anchors are missed.
ANCHOR_THRESHOLD = 0.72

# Minimum top-1 classifier probability to accept a unit prediction.
# If best probability is below this, detector labels slot as "unknown".
UNIT_THRESHOLD = 0.20

# Minimum gap between top-1 and top-2 class probabilities.
# Helps reject ambiguous predictions where multiple classes are close.
UNIT_MARGIN_THRESHOLD = 0.02
