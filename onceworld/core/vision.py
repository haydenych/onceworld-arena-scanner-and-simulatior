"""Computer-vision helpers for screen capture and region detection."""

import os
import re

import cv2
import mss
import numpy as np

from onceworld.config.geometry import (
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
from onceworld.config.runtime import UNLABELED_ICON_DIGITS, UNLABELED_ICON_PREFIX
from onceworld.core.perf import is_debug


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
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        return np.array(shot)[:, :, :3]


def resize_keep_aspect(img, scale):
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def match_template_multiscale(search_img, template_bgr, scales, threshold, debug_label=None):
    best = None
    sh, sw = search_img.shape[:2]

    for scale in scales:
        tpl = resize_keep_aspect(template_bgr, scale)
        th, tw = tpl.shape[:2]

        if th < 8 or tw < 8:
            continue
        if th >= sh or tw >= sw:
            continue

        res = cv2.matchTemplate(search_img, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if is_debug() and debug_label is not None:
            tag = "PASS" if max_val >= threshold else "FAIL"
            print(f"[anchor-scale] {debug_label} scale={scale:.3f} score={max_val:.3f} {tag}")

        if max_val < threshold:
            continue

        x1, y1 = max_loc
        cand = {
            "score": float(max_val),
            "box": (x1, y1, x1 + tw, y1 + th),
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
        for path in out_dir.glob(f"{prefix}*.png"):
            match = pat.match(path.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
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
    return clamp_box(refined, w, h)


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

    candidates.sort(key=lambda b: b[0])

    dedup = []
    for box in candidates:
        replaced = False
        bx1, by1, bx2, by2 = box
        b_area = max(1, bx2 - bx1) * max(1, by2 - by1)

        for idx, prev in enumerate(dedup):
            if iou(box, prev) <= 0.40:
                continue

            px1, py1, px2, py2 = prev
            p_area = max(1, px2 - px1) * max(1, py2 - py1)
            if b_area > p_area:
                dedup[idx] = box
            replaced = True
            break

        if not replaced:
            dedup.append(box)

    dedup.sort(key=lambda b: b[0])
    return dedup[:max_slots]

