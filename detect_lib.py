import os
import re
import csv
import random
import math
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pytesseract
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QFrame,
    QMessageBox,
    QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QPalette, QTextCursor, QIcon, QPixmap, QImage, QPainter, QPen, QFontMetrics
import sys
from battle_logic import Monster, Field
from detect_common import (
    COIN_ZONE,
    ENEMY_BOX_ASPECT_MAX,
    ENEMY_BOX_ASPECT_MIN,
    ENEMY_BOX_MAX_AREA_RATIO,
    ENEMY_BOX_MAX_X_RATIO,
    ENEMY_BOX_MAX_Y_RATIO,
    ENEMY_BOX_MIN_AREA_RATIO,
    ENEMY_BOX_MIN_Y_RATIO,
    ENEMY_MAX_SLOTS,
    ENEMY_WHITE_HSV_HIGH,
    ENEMY_WHITE_HSV_LOW,
    ROW_X1,
    ROW_X2,
    ROW_Y1,
    ROW_Y2,
)
import user_config

default_tess = os.path.join(os.path.dirname(__file__), "Tesseract-OCR", "tesseract.exe")
tess_cmd = user_config.TESSERACT_CMD
if not tess_cmd and os.path.isfile(default_tess):
    tess_cmd = default_tess
if tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = tess_cmd

# ============================================================
# Config
# ============================================================

ASSET_DIR = "assets"
ANCHOR_DIR = os.path.join(ASSET_DIR, "anchors")

ANCHOR_SCALES = np.asarray(user_config.ANCHOR_SCALES, dtype=float)

ANCHOR_THRESHOLD = float(user_config.ANCHOR_THRESHOLD)
UNIT_THRESHOLD = 0.20
UNIT_MARGIN_THRESHOLD = 0.02
UNIT_WARN_THRESHOLD = 0.35

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_GLOB = "unit_resnet18_*.pt"
DEBUG = True
UNLABELED_ICON_DIR = "dataset_enemy_icons"
UNLABELED_ICON_PREFIX = "img"
UNLABELED_ICON_DIGITS = 4
BATTLE_MONSTERS_CSV = "monsters.csv"
BATTLE_SIM_TRIALS = 100
BATTLE_SIM_DELTA_TIME = 0.02
BATTLE_SIM_DURATION = 40.0

# Row/enemy/coin geometry is shared from detect_common.py

# OCR tuning
TESS_CONFIG_LEVEL = r'--psm 7 -c tessedit_char_whitelist=Lv.0123456789'
TESS_CONFIG_COIN = r'--psm 7 -c tessedit_char_whitelist=0123456789xX, '

# ============================================================
# Utilities
# ============================================================

def _fmt_pct(x):
    try:
        return f"{float(x):.1%}"
    except Exception:
        return "n/a"


def _fmt_float(x, digits=3):
    try:
        return f"{float(x):.{int(digits)}f}"
    except Exception:
        return "n/a"


def _debug_perf(label, start_ts):
    if not DEBUG:
        return
    elapsed_ms = (time.time() - float(start_ts)) * 1000.0
    pretty_label = str(label).replace(":total", " total")
    print(f"[perf] {pretty_label}: {elapsed_ms:.2f} ms")


def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, float(t)))
    r = int(c1.red() + (c2.red() - c1.red()) * t)
    g = int(c1.green() + (c2.green() - c1.green()) * t)
    b = int(c1.blue() + (c2.blue() - c1.blue()) * t)
    return QColor(r, g, b)


def load_templates(folder):
    templates = {}
    if not os.path.isdir(folder):
        return templates

    for fn in os.listdir(folder):
        path = os.path.join(folder, fn)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        name, _ = os.path.splitext(fn)
        templates[name] = img
    return templates


def grab_screen_bgr():
    """Capture the full screen into memory only. Nothing is saved."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        shot = sct.grab(monitor)
        img = np.array(shot)[:, :, :3]  # BGRA -> BGR
        return img


def resize_keep_aspect(img, scale):
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def match_template_multiscale(search_img, template_bgr, scales, threshold, debug_label=None):
    """
    Returns the best match above threshold:
    {
      'score': float,
      'box': (x1, y1, x2, y2),
      'size': (w, h)
    }
    or None
    """
    tpl_orig = template_bgr
    best = None
    sh, sw = search_img.shape[:2]

    for scale in scales:
        tpl = resize_keep_aspect(tpl_orig, scale)
        th, tw = tpl.shape[:2]

        if th < 8 or tw < 8:
            continue
        if th >= sh or tw >= sw:
            continue

        res = cv2.matchTemplate(search_img, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if DEBUG and debug_label is not None:
            tag = "PASS" if max_val >= threshold else "FAIL"
            print(f"[anchor-scale] {debug_label} scale={scale:.3f} score={max_val:.3f} {tag}")

        if max_val >= threshold:
            x1, y1 = max_loc
            x2, y2 = x1 + tw, y1 + th
            cand = {
                "score": float(max_val),
                "box": (x1, y1, x2, y2),
                "size": (tw, th),
                "scale": float(scale),
            }
            if best is None or cand["score"] > best["score"]:
                best = cand

    return best


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return 0.0 if union <= 0 else inter / union


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    return (x1, y1, x2, y2)


def crop(img, box):
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


def next_unlabeled_index(out_dir, prefix=UNLABELED_ICON_PREFIX, digits=UNLABELED_ICON_DIGITS):
    pat = re.compile(rf"^{re.escape(prefix)}(\d{{{digits}}})\.png$", re.IGNORECASE)
    max_idx = 0
    if out_dir.exists():
        for p in out_dir.glob(f"{prefix}*.png"):
            m = pat.match(p.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def make_row_box_from_anchor(anchor_box, screen_bgr):
    ax1, ay1, ax2, ay2 = anchor_box
    aw = ax2 - ax1
    ah = ay2 - ay1

    x1 = int(round(ax1 + ROW_X1 * aw))
    y1 = int(round(ay1 + ROW_Y1 * ah))
    x2 = int(round(ax1 + ROW_X2 * aw))
    y2 = int(round(ay1 + ROW_Y2 * ah))

    h, w = screen_bgr.shape[:2]
    row_box = clamp_box((x1, y1, x2, y2), w, h)

    # Combine row estimation + dark-brown panel extraction in one step.
    row_img = crop(screen_bgr, row_box)
    if row_img.size == 0:
        return row_box

    hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (8, 90, 15), (24, 255, 130))

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n_labels <= 1:
        return row_box

    # Prefer the component touching the anchor center; otherwise largest area.
    anchor_cx = int(round((ax1 + ax2) / 2.0)) - row_box[0]
    anchor_cy = int(round((ay1 + ay2) / 2.0)) - row_box[1]
    anchor_cx = max(0, min(row_img.shape[1] - 1, anchor_cx))
    anchor_cy = max(0, min(row_img.shape[0] - 1, anchor_cy))
    anchor_label = int(labels[anchor_cy, anchor_cx])

    best_idx = None
    best_key = None
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 200:
            continue
        touches_anchor = 1 if i == anchor_label else 0
        key = (touches_anchor, area)
        if best_key is None or key > best_key:
            best_key = key
            best_idx = i

    if best_idx is None:
        return row_box

    bx = int(stats[best_idx, cv2.CC_STAT_LEFT])
    by = int(stats[best_idx, cv2.CC_STAT_TOP])
    bw = int(stats[best_idx, cv2.CC_STAT_WIDTH])
    bh = int(stats[best_idx, cv2.CC_STAT_HEIGHT])

    refined = (
        row_box[0] + bx,
        row_box[1] + by,
        row_box[0] + bx + bw,
        row_box[1] + by + bh,
    )
    row_box = clamp_box(refined, w, h)

    return row_box


def sub_box(parent_box, frac_box):
    px1, py1, px2, py2 = parent_box
    pw = px2 - px1
    ph = py2 - py1

    fx1, fy1, fx2, fy2 = frac_box
    x1 = int(round(px1 + fx1 * pw))
    y1 = int(round(py1 + fy1 * ph))
    x2 = int(round(px1 + fx2 * pw))
    y2 = int(round(py1 + fy2 * ph))

    return (x1, y1, x2, y2)


def detect_enemy_icon_boxes(row_img, row_box, max_slots=ENEMY_MAX_SLOTS):
    if row_img is None or row_img.size == 0:
        return []

    rh, rw = row_img.shape[:2]
    hsv = cv2.cvtColor(row_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ENEMY_WHITE_HSV_LOW, ENEMY_WHITE_HSV_HIGH)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    row_area = float(max(1, rh * rw))
    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area_ratio = (w * h) / row_area
        aspect = w / float(max(1, h))

        if area_ratio < ENEMY_BOX_MIN_AREA_RATIO or area_ratio > ENEMY_BOX_MAX_AREA_RATIO:
            continue
        if aspect < ENEMY_BOX_ASPECT_MIN or aspect > ENEMY_BOX_ASPECT_MAX:
            continue
        if x > int(ENEMY_BOX_MAX_X_RATIO * rw):
            continue
        if y < int(ENEMY_BOX_MIN_Y_RATIO * rh):
            continue
        if y + h > int(ENEMY_BOX_MAX_Y_RATIO * rh):
            continue

        # Normalize to a square icon box so level text below is excluded.
        side = max(1, min(w, h))
        sx = x + max(0, (w - side) // 2)
        sy = y
        if sx + side > rw:
            sx = max(0, rw - side)
        if sy + side > rh:
            sy = max(0, rh - side)

        abs_box = (
            row_box[0] + sx,
            row_box[1] + sy,
            row_box[0] + sx + side,
            row_box[1] + sy + side,
        )
        candidates.append(abs_box)

    candidates = sorted(candidates, key=lambda b: b[0])

    # Deduplicate overlapping candidates.
    dedup = []
    for box in candidates:
        replaced = False
        bx1, by1, bx2, by2 = box
        b_area = max(1, bx2 - bx1) * max(1, by2 - by1)

        for j, prev in enumerate(dedup):
            if iou(box, prev) <= 0.40:
                continue

            px1, py1, px2, py2 = prev
            p_area = max(1, px2 - px1) * max(1, py2 - py1)
            if b_area > p_area:
                dedup[j] = box
            replaced = True
            break

        if not replaced:
            dedup.append(box)

    dedup = sorted(dedup, key=lambda b: b[0])
    return dedup[:max_slots]


def preprocess_for_ocr(img_bgr):
    img = cv2.resize(img_bgr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def read_level_text(img_bgr):
    proc = preprocess_for_ocr(img_bgr)
    text = pytesseract.image_to_string(proc, config=TESS_CONFIG_LEVEL).strip()
    m = re.search(r'Lv\.?\s*(\d+)', text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), text
    return None, text


def read_coin_text(img_bgr):
    proc = preprocess_for_ocr(img_bgr)
    text = pytesseract.image_to_string(proc, config=TESS_CONFIG_COIN).strip()
    digits = re.sub(r'[^0-9]', '', text)
    if digits:
        return int(digits), text
    return None, text


# ============================================================
# Model-based unit classifier
# ============================================================

def _estimate_bg_fill_bgr(img_bgr):
    h, w = img_bgr.shape[:2]
    corners = [
        img_bgr[0, 0],
        img_bgr[0, max(0, w - 1)],
        img_bgr[max(0, h - 1), 0],
        img_bgr[max(0, h - 1), max(0, w - 1)],
    ]
    mean = np.mean(np.array(corners, dtype=np.float32), axis=0)
    return tuple(int(x) for x in mean.tolist())


def _pad_to_square_bgr(img_bgr, fill_bgr):
    h, w = img_bgr.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    canvas[:, :] = np.array(fill_bgr, dtype=np.uint8)
    x = (side - w) // 2
    y = (side - h) // 2
    canvas[y:y + h, x:x + w] = img_bgr
    return canvas


class TorchUnitClassifier:
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR):
        try:
            import torch
            import torch.nn as nn
            from torchvision import models
        except Exception as e:
            raise RuntimeError(
                "Unit classifier requires torch + torchvision. "
                "Install them in your environment."
            ) from e

        self.torch = torch
        self.nn = nn
        self.models = models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = self._find_latest_checkpoint(checkpoint_dir)
        payload = torch.load(ckpt_path, map_location=self.device)

        self.class_names = list(payload.get("class_names", []))
        if not self.class_names:
            raise RuntimeError(f"Checkpoint missing class_names: {ckpt_path}")

        self.image_size = int(payload.get("image_size", 128))
        self.mean = np.array(payload.get("normalize_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.std = np.array(payload.get("normalize_std", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.std = np.where(self.std == 0, 1.0, self.std)

        model = self.models.resnet18(weights=None)
        model.fc = self.nn.Linear(model.fc.in_features, len(self.class_names))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(self.device)
        model.eval()

        self.model = model
        self.checkpoint_path = str(ckpt_path)

        if DEBUG:
            print(f"[unit-model] loaded checkpoint: {self.checkpoint_path}")

    def _find_latest_checkpoint(self, checkpoint_dir):
        root = Path(checkpoint_dir)
        if not root.exists():
            raise RuntimeError(f"Checkpoint directory not found: {root}")

        candidates = list(root.glob(CHECKPOINT_GLOB))
        if not candidates:
            candidates = list(root.glob("*.pt"))
        if not candidates:
            raise RuntimeError(f"No checkpoint files found in: {root}")

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _preprocess(self, icon_bgr):
        fill = _estimate_bg_fill_bgr(icon_bgr)
        sq = _pad_to_square_bgr(icon_bgr, fill)
        resized = cv2.resize(sq, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        chw = np.transpose(rgb, (2, 0, 1))
        x = self.torch.from_numpy(chw).unsqueeze(0).float().to(self.device)
        return x

    def predict(self, icon_bgr, top_k=5):
        if icon_bgr is None or icon_bgr.size == 0:
            return None

        x = self._preprocess(icon_bgr)
        with self.torch.no_grad():
            logits = self.model(x)
            probs = self.torch.softmax(logits, dim=1)[0]

        k = min(max(1, int(top_k)), probs.shape[0])
        vals, idxs = self.torch.topk(probs, k=k)
        vals = vals.detach().cpu().numpy().tolist()
        idxs = idxs.detach().cpu().numpy().tolist()

        top = [(self.class_names[i], float(v)) for i, v in zip(idxs, vals)]
        best_name, best_score = top[0]
        second_score = top[1][1] if len(top) > 1 else 0.0
        return {
            "name": best_name,
            "score": float(best_score),
            "margin": float(best_score - second_score),
            "top": top,
        }


# ============================================================
# Detection
# ============================================================

class ScreenDetector:
    def __init__(self, save_unlabeled=False):
        self.anchor_templates = load_templates(ANCHOR_DIR)

        required_anchors = {"team_a", "team_b", "team_c"}
        missing = [x for x in required_anchors if x not in self.anchor_templates]
        if missing:
            raise RuntimeError(
                f"Missing anchor templates: {missing}\n"
                f"Expected files in {ANCHOR_DIR}"
            )

        self.unit_classifier = TorchUnitClassifier(CHECKPOINT_DIR)
        self.cached_anchor_scale = None
        self.save_unlabeled = bool(save_unlabeled)
        if self.save_unlabeled:
            self.unlabeled_icon_dir = Path(UNLABELED_ICON_DIR)
            self.unlabeled_icon_dir.mkdir(parents=True, exist_ok=True)
            self.next_unlabeled_idx = next_unlabeled_index(self.unlabeled_icon_dir)
        else:
            self.unlabeled_icon_dir = None
            self.next_unlabeled_idx = 0
        self.battle_monster_base = self._load_battle_monster_base(BATTLE_MONSTERS_CSV)

    def _load_battle_monster_base(self, csv_path):
        base = {}
        p = Path(csv_path)

        with p.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["PET_NAME"].strip()
                base[name] = {
                    "NO": int(row["NO"]),
                    "PET_NAME": name,
                    "ELEMENT": row["ELEMENT"].strip(),
                    "TYPE": row["TYPE"].strip(),
                    "RANGE": row["RANGE"].strip(),
                    "VIT": int(row["VIT"]),
                    "SPD": int(row["SPD"]),
                    "ATK": int(row["ATK"]),
                    "INT": int(row["INT"]),
                    "DEF": int(row["DEF"]),
                    "MDEF": int(row["MDEF"]),
                    "LUCK": int(row["LUCK"]),
                    "MOV": int(row["MOV"]),
                }
        return base

    def _simulate_battle_probs(self, results):
        t_sim_total = time.time()
        team_map = {"team_a": "A", "team_b": "B", "team_c": "C"}
        # Follow app_arena-style spawn setup.
        positions = {"A": (450.0, 450.0), "B": (550.0, 450.0), "C": (500.0, 500.0)}
        seed_entries = {"A": [], "B": [], "C": []}
        unknown_count = 0
        t_seed_start = time.time()

        for team_key, team_letter in team_map.items():
            for u in results.get(team_key, {}).get("units", []):
                unit_name = str(u.get("unit_name", "") or "")
                if not unit_name or unit_name == "unknown":
                    unknown_count += 1
                    continue
                base = self.battle_monster_base.get(unit_name)
                if base is None:
                    unknown_count += 1
                    continue
                level = int(u.get("level") or 1)
                seed_entries[team_letter].append((base, level))

        teams_with_known = sum(1 for t in ["A", "B", "C"] if len(seed_entries[t]) > 0)
        if teams_with_known < 2:
            _debug_perf("battle_sim:seed_setup", t_seed_start)
            _debug_perf("battle_sim:total", t_sim_total)
            return {"available": False, "reason": "not_enough_known_units", "unknown_count": unknown_count}

        wins = {"A": 0, "B": 0, "C": 0, "Draw": 0}
        total_steps = 0
        t_trials_start = time.time()

        for _ in range(BATTLE_SIM_TRIALS):
            teams = {"A": [], "B": [], "C": []}
            all_placed = []

            for t in ["A", "B", "C"]:
                base_x, base_y = positions[t]
                for base, level in seed_entries[t]:
                    m = Monster(t, dict(base), level=level)

                    placed = False
                    nx = base_x
                    ny = base_y
                    for _k in range(100):
                        nx = float(base_x + random.randint(-350, 350))
                        ny = float(base_y + random.randint(-350, 350))
                        nx = max(50.0, min(950.0, nx))
                        ny = max(50.0, min(950.0, ny))

                        overlap = False
                        for pm in all_placed:
                            if math.hypot(nx - pm.x, ny - pm.y) < 80.0:
                                overlap = True
                                break
                        if not overlap:
                            m.x = nx
                            m.y = ny
                            placed = True
                            break

                    if not placed:
                        m.x = nx
                        m.y = ny

                    teams[t].append(m)
                    all_placed.append(m)

            field = Field(teams)
            while not field.is_finished() and field.time_elapsed < BATTLE_SIM_DURATION:
                field.step(BATTLE_SIM_DELTA_TIME)
                total_steps += 1

            winner = field.get_winner()
            if winner in wins:
                wins[winner] += 1
            else:
                wins["Draw"] += 1
        t_trials_end = time.time()

        total = max(1, sum(wins.values()))
        team_probs = {
            "team_a": wins["A"] / total,
            "team_b": wins["B"] / total,
            "team_c": wins["C"] / total,
        }
        best_team = max(team_probs, key=team_probs.get)

        if DEBUG:
            seed_ms = (t_trials_start - t_seed_start) * 1000.0
            trials_ms = (t_trials_end - t_trials_start) * 1000.0
            avg_steps = total_steps / max(1, BATTLE_SIM_TRIALS)
            print(
                f"[perf] battle_sim: seed={seed_ms:.2f} ms "
                f"trials={trials_ms:.2f} ms "
                f"steps={total_steps} avg_steps/trial={avg_steps:.1f}"
            )
        _debug_perf("battle_sim:total", t_sim_total)

        return {
            "available": True,
            "team_probs": team_probs,
            "best_team": best_team,
            "best_prob": team_probs[best_team],
            "draw_prob": wins["Draw"] / total,
            "unknown_count": unknown_count,
        }

    def detect(self):
        t_detect_total = time.time()
        t_capture = time.time()
        screen_bgr = grab_screen_bgr()
        _debug_perf("capture_screen", t_capture)

        results = {}
        run_anchor_scale = self.cached_anchor_scale

        for team_name in ["team_a", "team_b", "team_c"]:
            t_team_total = time.time()
            t_anchor = time.time()
            if run_anchor_scale is None:
                anchor_match = match_template_multiscale(
                    screen_bgr,
                    self.anchor_templates[team_name],
                    ANCHOR_SCALES,
                    ANCHOR_THRESHOLD,
                    debug_label=team_name,
                )
            else:
                anchor_match = match_template_multiscale(
                    screen_bgr,
                    self.anchor_templates[team_name],
                    np.array([run_anchor_scale], dtype=np.float32),
                    ANCHOR_THRESHOLD,
                    debug_label=f"{team_name}:cached",
                )
                if anchor_match is None:
                    print(
                        f"[anchor] {team_name}: cached scale {run_anchor_scale:.3f} failed, "
                        "rerunning full search"
                    )
                    anchor_match = match_template_multiscale(
                        screen_bgr,
                        self.anchor_templates[team_name],
                        ANCHOR_SCALES,
                        ANCHOR_THRESHOLD,
                        debug_label=f"{team_name}:fallback",
                    )
            _debug_perf(f"{team_name}:anchor_search", t_anchor)

            if anchor_match is None:
                results[team_name] = {
                    "found": False,
                    "reason": "anchor not found"
                }
                _debug_perf(f"{team_name}:total", t_team_total)
                continue

            run_anchor_scale = float(anchor_match["scale"])
            row_box = make_row_box_from_anchor(anchor_match["box"], screen_bgr)
            print(
                f"[anchor] {team_name}: score={anchor_match['score']:.3f} "
                f"scale={anchor_match['scale']:.3f}"
            )
            t_row_detect = time.time()
            team_result = self._detect_team_row(screen_bgr, row_box, team_name)
            _debug_perf(f"{team_name}:row_detect", t_row_detect)
            team_result["found"] = True
            team_result["anchor_score"] = round(anchor_match["score"], 3)
            team_result["anchor_scale"] = round(anchor_match["scale"], 3)
            team_result["row_box"] = row_box
            results[team_name] = team_result
            _debug_perf(f"{team_name}:total", t_team_total)

        if run_anchor_scale is not None:
            self.cached_anchor_scale = run_anchor_scale

        results["_battle"] = self._simulate_battle_probs(results)
        _debug_perf("detect_total", t_detect_total)
        return results

    def _detect_team_row(self, screen_bgr, row_box, team_name):
        t_row_total = time.time()
        row_img = crop(screen_bgr, row_box)

        # Coin zone
        t_coin_ocr = time.time()
        coin_box_abs = sub_box(row_box, COIN_ZONE)
        coin_img = crop(screen_bgr, coin_box_abs)
        coin_value, coin_raw_text = read_coin_text(coin_img)
        coin_ocr_ms = (time.time() - t_coin_ocr) * 1000.0

        t_icon_detect = time.time()
        icon_boxes = detect_enemy_icon_boxes(row_img, row_box)
        icon_detect_ms = (time.time() - t_icon_detect) * 1000.0
        units = []
        unit_ml_ms = 0.0
        level_ocr_ms = 0.0
        save_icon_ms = 0.0

        for i, icon_box in enumerate(icon_boxes, start=1):
            icon_box = clamp_box(icon_box, screen_bgr.shape[1], screen_bgr.shape[0])
            icon_img = crop(screen_bgr, icon_box)

            if self.save_unlabeled:
                t_save_icon = time.time()
                unlabeled_name = f"{UNLABELED_ICON_PREFIX}{self.next_unlabeled_idx:0{UNLABELED_ICON_DIGITS}d}.png"
                unlabeled_path = self.unlabeled_icon_dir / unlabeled_name
                if cv2.imwrite(str(unlabeled_path), icon_img):
                    self.next_unlabeled_idx += 1
                save_icon_ms += (time.time() - t_save_icon) * 1000.0

            t_unit_ml = time.time()
            pred = self.unit_classifier.predict(icon_img, top_k=5)
            unit_ml_ms += (time.time() - t_unit_ml) * 1000.0
            if pred is None:
                best_name = None
                best_score = -1.0
                margin = None
                # print(f"[slot-best] {team_name} enemy_{i}: no-pred")
            else:
                best_name = pred["name"]
                best_score = pred["score"]
                margin = pred["margin"]
                # print(
                #     f"[slot-best] {team_name} enemy_{i}: "
                #     f"{best_name} score={best_score:.3f} margin={margin:.3f}"
                # )

            x1, y1, x2, y2 = icon_box
            icon_w = x2 - x1
            icon_h = y2 - y1

            # Level box is a small area below the icon
            level_box = (
                x1 - int(0.03 * icon_w),
                y2 + int(0.02 * icon_h),
                x2 + int(0.20 * icon_w),
                y2 + int(0.62 * icon_h),
            )
            level_box = clamp_box(level_box, screen_bgr.shape[1], screen_bgr.shape[0])
            level_img = crop(screen_bgr, level_box)
            t_level_ocr = time.time()
            level_value, level_raw_text = read_level_text(level_img)
            level_ocr_ms += (time.time() - t_level_ocr) * 1000.0

            if (
                best_name is None
                or best_score < UNIT_THRESHOLD
                or margin is None
                or margin < UNIT_MARGIN_THRESHOLD
            ):
                display_name = "unknown"
            else:
                display_name = best_name

            units.append({
                "unit_name": display_name,
                "score": round(best_score, 3) if best_score >= 0 else None,
                "icon_box": icon_box,
                "level": level_value,
                "level_raw_text": level_raw_text,
            })

        if DEBUG:
            total_ms = (time.time() - t_row_total) * 1000.0
            print(
                f"[perf] {team_name}: coin_ocr={coin_ocr_ms:.2f} ms "
                f"icon_detect={icon_detect_ms:.2f} ms "
                f"unit_ml={unit_ml_ms:.2f} ms "
                f"level_ocr={level_ocr_ms:.2f} ms "
                f"save_icons={save_icon_ms:.2f} ms "
                f"slots={len(icon_boxes)} "
                f"total={total_ms:.2f} ms"
            )

        return {
            "coins": coin_value,
            "coin_raw_text": coin_raw_text,
            "units": units,
        }


class MainWindow(QMainWindow):
    def __init__(self, save_unlabeled=False):
        super().__init__()
        self._base_window_width = 400
        self._base_window_height = 400
        self._font_targets = []
        self._enemy_text_boxes = []
        self.setWindowTitle("OnceWorld Arena")
        self.setMinimumSize(self._base_window_width, self._base_window_height)
        self.resize(self._base_window_width, self._base_window_height)
        self._set_app_icon()
        self.detector = None
        try:
            self.detector = ScreenDetector(save_unlabeled=save_unlabeled)
        except Exception as e:
            QMessageBox.critical(self, "Start", str(e))
            self.close()
            return
        self._check_ocr()
        self._init_palette()
        self._build_ui()
        self._apply_scaled_fonts()
        self._set_status("")

    def _set_app_icon(self):
        size = 64
        img = QImage(size, size, QImage.Format_ARGB32)
        img.fill(QColor("#0D1117"))

        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing, True)

        pen = QPen(QColor("#21262D"))
        pen.setWidth(2)
        p.setPen(pen)
        p.setBrush(QColor("#161B22"))
        p.drawRoundedRect(4, 4, size - 8, size - 8, 12, 12)

        pen2 = QPen(QColor("#58A6FF"))
        pen2.setWidth(5)
        pen2.setCapStyle(Qt.RoundCap)
        p.setPen(pen2)
        p.drawLine(18, 38, 30, 26)
        p.drawLine(30, 26, 46, 42)

        p.end()

        px = QPixmap.fromImage(img)
        self.setWindowIcon(QIcon(px))

    def _init_palette(self):
        self.c_bg = QColor("#0D1117")
        self.c_panel = QColor("#161B22")
        self.c_panel_soft = QColor("#0D1117")
        self.c_text = QColor("#C9D1D9")
        self.c_muted = QColor("#8B949E")
        self.c_accent = QColor("#58A6FF")
        self.c_good = QColor("#3FB950")
        self.c_warn = QColor("#D29922")
        self.c_bad = QColor("#F85149")
        self.c_team_a = QColor("#EF4444")  # A: red
        self.c_team_b = QColor("#3B82F6")  # B: blue
        self.c_team_c = QColor("#22C55E")  # C: green
        self.c_prob_lo = QColor("#22C55E")  # green
        self.c_prob_hi = QColor("#EF4444")  # red
        self.c_prob_gold = QColor("#FACC15")
        pal = self.palette()
        pal.setColor(QPalette.Window, self.c_bg)
        pal.setColor(QPalette.WindowText, self.c_text)
        pal.setColor(QPalette.Base, self.c_panel_soft)
        pal.setColor(QPalette.AlternateBase, self.c_panel)
        pal.setColor(QPalette.Text, self.c_text)
        pal.setColor(QPalette.Button, self.c_panel)
        pal.setColor(QPalette.ButtonText, self.c_text)
        self.setPalette(pal)

    def _card_frame(self):
        f = QFrame()
        f.setFrameShape(QFrame.StyledPanel)
        f.setFrameShadow(QFrame.Plain)
        f.setStyleSheet(
            "QFrame { background-color: #161B22; border: 1px solid #21262D; border-radius: 6px; }"
        )
        return f

    def _label(self, text, muted=False, bold=False, large=False):
        lbl = QLabel(text)
        base_size = 14 if large else 8
        font = QFont("Segoe UI", base_size)
        if bold:
            font.setWeight(QFont.DemiBold)
        lbl.setFont(font)
        if large:
            self._register_font_target(lbl, base_size, min_pt=10, max_pt=24)
        else:
            self._register_font_target(lbl, base_size, min_pt=7, max_pt=14)
        if muted:
            lbl.setStyleSheet("color: #8B949E;")
        else:
            lbl.setStyleSheet("color: #C9D1D9;")
        return lbl

    def _register_font_target(self, widget, base_pt, min_pt=7, max_pt=None):
        self._font_targets.append({
            "widget": widget,
            "base": float(base_pt),
            "min": int(min_pt) if min_pt is not None else None,
            "max": int(max_pt) if max_pt is not None else None,
        })

    def _font_scale_factor(self):
        w_scale = self.width() / float(self._base_window_width)
        h_scale = self.height() / float(self._base_window_height)
        scale = min(w_scale, h_scale)
        return max(0.85, min(1.8, scale))

    def _apply_scaled_fonts(self):
        if not self._font_targets:
            return
        scale = self._font_scale_factor()
        for spec in self._font_targets:
            widget = spec["widget"]
            if widget is None:
                continue
            font = widget.font()
            point_size = int(round(spec["base"] * scale))
            if spec["min"] is not None:
                point_size = max(spec["min"], point_size)
            if spec["max"] is not None:
                point_size = min(spec["max"], point_size)
            point_size = max(1, point_size)
            if font.pointSize() != point_size:
                font.setPointSize(point_size)
                widget.setFont(font)

        for text in self._enemy_text_boxes:
            fm = QFontMetrics(text.font())
            text.setMinimumHeight(int(fm.lineSpacing() * 4 + 8))

        if hasattr(self, "scan_btn"):
            btn_fm = QFontMetrics(self.scan_btn.font())
            self.scan_btn.setMinimumHeight(max(28, int(btn_fm.height() + 12)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_scaled_fonts()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout()
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(3)
        central.setLayout(root)
        self.status_label = None
        top_cards = QHBoxLayout()
        top_cards.setSpacing(4)
        card_win = self._card_frame()
        card_win_layout = QVBoxLayout(card_win)
        card_win_layout.setContentsMargins(6, 4, 6, 4)
        card_win_layout.setSpacing(2)
        card_win_layout.addWidget(self._label("Win", muted=True))
        win_row = QHBoxLayout()
        win_row.setSpacing(4)
        self.win_team_label = self._label("—", bold=True, large=True)
        win_row.addWidget(self.win_team_label, 0, Qt.AlignLeft)
        self.win_prob_label = QLabel("—")
        prob_font = QFont("Segoe UI", 10)
        prob_font.setWeight(QFont.DemiBold)
        self.win_prob_label.setFont(prob_font)
        self._register_font_target(self.win_prob_label, 10, min_pt=8, max_pt=18)
        self.win_prob_label.setStyleSheet("color: #FACC15;")
        win_row.addWidget(self.win_prob_label, 0, Qt.AlignLeft)
        self.win_coin_label = self._label("—", muted=True, bold=True)
        win_row.addWidget(self.win_coin_label, 0, Qt.AlignLeft)
        win_row.addStretch()
        card_win_layout.addLayout(win_row)
        self.scan_btn = QPushButton("Scan")
        scan_font = QFont("Segoe UI", 8)
        scan_font.setWeight(QFont.DemiBold)
        self.scan_btn.setFont(scan_font)
        self._register_font_target(self.scan_btn, 8, min_pt=7, max_pt=14)
        self.scan_btn.setCursor(Qt.PointingHandCursor)
        self._scan_btn_style_normal = (
            "QPushButton { background-color: #238636; color: #FFFFFF; border-radius: 4px; padding: 4px 10px; }"
            "QPushButton:hover { background-color: #2ea043; }"
            "QPushButton:disabled { background-color: #30363D; color: #8B949E; }"
        )
        self._scan_btn_style_scanning = (
            "QPushButton { background-color: #30363D; color: #8B949E; border-radius: 4px; padding: 4px 10px; }"
        )
        self.scan_btn.setStyleSheet(self._scan_btn_style_normal)
        self.scan_btn.clicked.connect(self.on_scan)
        self.scan_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.scan_btn.setMinimumHeight(28)
        card_win_layout.addWidget(self.scan_btn)
        top_cards.addWidget(card_win)
        root.addLayout(top_cards)
        sides_frame = self._card_frame()
        sides_layout = QVBoxLayout(sides_frame)
        sides_layout.setContentsMargins(6, 2, 6, 2)
        sides_layout.setSpacing(2)
        self.side_units = {}
        for key, label in [("team_a", "A"), ("team_b", "B"), ("team_c", "C")]:
            row = QVBoxLayout()
            row.setSpacing(2)
            head = QHBoxLayout()
            side_lbl = self._label(label, muted=True, bold=True)
            head.addWidget(side_lbl, 0, Qt.AlignLeft)
            prob_lbl = self._label("—", muted=True)
            head.addWidget(prob_lbl, 0, Qt.AlignLeft)
            head.addStretch()
            coin_lbl = self._label("Coin -", muted=True)
            head.addWidget(coin_lbl, 0, Qt.AlignRight)
            row.addLayout(head)
            text = QTextEdit()
            text.setReadOnly(True)
            text.setFrameShape(QFrame.NoFrame)
            text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            text.setStyleSheet(
                "QTextEdit { background-color: #0D1117; color: #C9D1D9; border-radius: 4px; }"
            )
            mono = QFont("Cascadia Mono", 8)
            text.setFont(mono)
            self._register_font_target(text, 8, min_pt=7, max_pt=14)
            self._enemy_text_boxes.append(text)
            fm = QFontMetrics(mono)
            text.setMinimumHeight(int(fm.lineSpacing() * 4 + 8))
            text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            row.addWidget(text, 1)
            sides_layout.addLayout(row, 1)
            self.side_units[key] = {"coin": coin_lbl, "prob": prob_lbl, "list": text}
        root.addWidget(sides_frame, 1)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setVisible(False)

    def _check_ocr(self):
        try:
            _ = pytesseract.get_tesseract_version()
        except Exception:
            QMessageBox.warning(
                self,
                "OCR",
                "Tesseract OCR not found.\nInstall Tesseract and set user_config.TESSERACT_CMD.",
            )

    def _set_status(self, text):
        return

    def _append_log(self, text, level=None):
        color = "#C9D1D9"
        if level == "bad":
            color = "#F85149"
        elif level == "warn":
            color = "#D29922"
        self.log_text.setTextColor(QColor(color))
        self.log_text.append(text)
        self.log_text.moveCursor(QTextCursor.End)
        if len(self.log_text.toPlainText()) > 4000:
            txt = self.log_text.toPlainText()
            self.log_text.clear()
            self.log_text.setText(txt[-3000:])

    def on_scan(self):
        if not self.detector:
            return
        self.scan_btn.setStyleSheet(self._scan_btn_style_scanning)
        self.scan_btn.setEnabled(False)
        self.scan_btn.repaint()
        QApplication.processEvents()
        self._set_status("")
        self.log_text.clear()
        for key in self.side_units:
            self.side_units[key]["coin"].setText("Coin -")
            self.side_units[key]["list"].clear()
        try:
            results = self.detector.detect()
            self._render_results(results)
            self._set_status("")
        except Exception as e:
            QMessageBox.critical(self, "Scan", str(e))
            self._set_status("")
        finally:
            self.scan_btn.setStyleSheet(self._scan_btn_style_normal)
            self.scan_btn.setEnabled(True)
            self.scan_btn.repaint()

    def _render_results(self, results):
        battle = results.get("_battle", {})
        probs = {}
        if battle.get("available"):
            best_team = battle.get("best_team")
            best_prob = battle.get("best_prob", 0.0)
            probs = battle.get("team_probs", {}) or {}
            label = {"team_a": "A", "team_b": "B", "team_c": "C"}.get(best_team, "?")
            self.win_team_label.setText(label)
            if best_team == "team_a":
                self.win_team_label.setStyleSheet(f"color: {self.c_team_a.name()};")
            elif best_team == "team_b":
                self.win_team_label.setStyleSheet(f"color: {self.c_team_b.name()};")
            elif best_team == "team_c":
                self.win_team_label.setStyleSheet(f"color: {self.c_team_c.name()};")
            else:
                self.win_team_label.setStyleSheet("color: #C9D1D9;")
            self.win_prob_label.setText(f"{_fmt_pct(best_prob)}")
            if best_prob >= 0.999:
                c = self.c_prob_gold
            else:
                t = 0.0
                if best_prob <= 0.5:
                    t = 0.0
                elif best_prob >= 0.99:
                    t = 1.0
                else:
                    t = (best_prob - 0.5) / (0.99 - 0.5)
                c = _lerp_color(self.c_prob_hi, self.c_prob_lo, t)
            self.win_prob_label.setStyleSheet(f"color: {c.name()};")
            coins = results.get(best_team, {}).get("coins", None)
            self.win_coin_label.setText(f"Coin {coins if coins is not None else '-'}")
        else:
            self.win_team_label.setText("?")
            self.win_team_label.setStyleSheet("color: #8B949E;")
            self.win_prob_label.setText("n/a")
            self.win_coin_label.setText("Coin -")
        for team in ["team_a", "team_b", "team_c"]:
            r = results.get(team, {})
            side = self.side_units.get(team)
            if not side:
                continue
            side_prob = probs.get(team, None)
            if side.get("prob") is not None:
                side["prob"].setText(_fmt_pct(side_prob) if side_prob is not None else "—")
            box = side["list"]
            if not r.get("found"):
                side["coin"].setText("Coin -")
                box.clear()
                box.append("none")
                continue
            coins = r.get("coins")
            side["coin"].setText(f"Coin {coins if coins is not None else '-'}")
            box.clear()
            units = r.get("units", [])
            if not units:
                box.append("none")
                continue
            for i, u in enumerate(units, 1):
                name = u.get("unit_name")
                level = u.get("level")
                score = u.get("score")
                score_txt = _fmt_float(score, 2) if score is not None else "n/a"
                label = {"team_a": "A", "team_b": "B", "team_c": "C"}.get(team, team)
                row_prefix = f"{label}{i} Lv{level}-{name} "
                if name == "unknown":
                    box.setTextColor(self.c_bad)
                    box.insertPlainText(f"{row_prefix}{score_txt} (low confidence)\n")
                elif score is not None and score < UNIT_WARN_THRESHOLD:
                    box.setTextColor(self.c_text)
                    box.insertPlainText(row_prefix)
                    box.setTextColor(self.c_warn)
                    box.insertPlainText(f"{score_txt} (low confidence)\n")
                else:
                    box.setTextColor(self.c_text)
                    box.insertPlainText(f"{row_prefix}{score_txt}\n")
                box.setTextColor(self.c_text)


def run_app(save_unlabeled=False):
    app = QApplication(sys.argv)
    win = MainWindow(save_unlabeled=save_unlabeled)
    if not win.detector:
        return
    win.show()
    win.raise_()
    win.activateWindow()
    app.exec()


if __name__ == "__main__":
    run_app(save_unlabeled=False)
