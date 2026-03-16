import os
import re
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import mss
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from detect_common import (
    DEFAULT_ANCHOR_SCALES,
    DEFAULT_ANCHOR_THRESHOLD,
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


# ============================================================
# Config (aligned with detect.py crop pipeline)
# ============================================================

ASSET_DIR = "assets"
ANCHOR_DIR = os.path.join(ASSET_DIR, "anchors")

ANCHOR_SCALES = DEFAULT_ANCHOR_SCALES.copy()
ANCHOR_THRESHOLD = DEFAULT_ANCHOR_THRESHOLD

# Row/enemy geometry is shared from detect_common.py


# ============================================================
# Utilities
# ============================================================

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
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)


def match_template_multiscale(search_img, template_bgr, scales, threshold):
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
        if max_val < threshold:
            continue

        x1, y1 = max_loc
        cand = {
            "score": float(max_val),
            "box": (x1, y1, x1 + tw, y1 + th),
            "scale": float(scale),
        }
        if best is None or cand["score"] > best["score"]:
            best = cand

    return best


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
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
    refined = (row_box[0] + bx, row_box[1] + by, row_box[0] + bx + bw, row_box[1] + by + bh)
    return clamp_box(refined, w, h)


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

        abs_box = (row_box[0] + sx, row_box[1] + sy, row_box[0] + sx + side, row_box[1] + sy + side)
        candidates.append(abs_box)

    candidates = sorted(candidates, key=lambda b: b[0])

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


def next_image_index(out_dir, prefix="img", digits=4):
    pat = re.compile(rf"^{re.escape(prefix)}(\d{{{digits}}})\.png$", re.IGNORECASE)
    max_idx = 0
    if out_dir.exists():
        for p in out_dir.glob(f"{prefix}*.png"):
            m = pat.match(p.name)
            if not m:
                continue
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


class DataCollector:
    def __init__(self, output_dir="dataset_enemy_icons", prefix="img", digits=4):
        self.required = ("team_a", "team_b", "team_c")
        self.anchors = load_templates(ANCHOR_DIR)
        missing = [x for x in self.required if x not in self.anchors]
        if missing:
            raise RuntimeError(f"Missing anchor templates: {missing} in {ANCHOR_DIR}")

        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.digits = int(digits)
        self.next_idx = next_image_index(self.out_dir, prefix=self.prefix, digits=self.digits)
        self.cached_anchor_scale = None

    def _match_team_anchor(self, screen, team_name, run_anchor_scale):
        if run_anchor_scale is None:
            return match_template_multiscale(
                screen,
                self.anchors[team_name],
                ANCHOR_SCALES,
                ANCHOR_THRESHOLD,
            )

        m = match_template_multiscale(
            screen,
            self.anchors[team_name],
            np.array([run_anchor_scale], dtype=np.float32),
            ANCHOR_THRESHOLD,
        )
        if m is not None:
            return m

        return match_template_multiscale(
            screen,
            self.anchors[team_name],
            ANCHOR_SCALES,
            ANCHOR_THRESHOLD,
        )

    def capture_and_save(self):
        screen = grab_screen_bgr()
        run_anchor_scale = self.cached_anchor_scale
        saved = []
        logs = []

        for team_name in self.required:
            m = self._match_team_anchor(screen, team_name, run_anchor_scale)
            if m is None:
                logs.append(f"[{team_name}] anchor not found")
                continue

            run_anchor_scale = float(m["scale"])
            logs.append(
                f"[{team_name}] anchor score={m['score']:.3f} scale={m['scale']:.3f}"
            )

            row_box = make_row_box_from_anchor(m["box"], screen)
            row_img = crop(screen, row_box)
            icon_boxes = detect_enemy_icon_boxes(row_img, row_box, max_slots=ENEMY_MAX_SLOTS)
            if not icon_boxes:
                logs.append(f"[{team_name}] no enemy boxes found")
                continue

            for slot, box in enumerate(icon_boxes, start=1):
                icon_img = crop(screen, clamp_box(box, screen.shape[1], screen.shape[0]))
                if icon_img.size == 0:
                    continue

                name = f"{self.prefix}{self.next_idx:0{self.digits}d}.png"
                path = self.out_dir / name
                cv2.imwrite(str(path), icon_img)
                saved.append(path)
                logs.append(f"[saved] {team_name} enemy_{slot} -> {path}")
                self.next_idx += 1

        if run_anchor_scale is not None:
            self.cached_anchor_scale = run_anchor_scale

        logs.append(f"done. saved={len(saved)} images to {self.out_dir.resolve()}")
        return saved, logs


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Data Collector")
        self.root.geometry("900x650")

        try:
            self.collector = DataCollector()
        except Exception as e:
            messagebox.showerror("Startup error", str(e))
            root.destroy()
            return

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")

        self.save_btn = ttk.Button(top, text="Save Data", command=self.on_save_data)
        self.save_btn.pack(side="left")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status_var).pack(side="left", padx=12)

        self.text = tk.Text(root, wrap="word", font=("Consolas", 11))
        self.text.pack(fill="both", expand=True, padx=10, pady=10)
        self.write("Ready.\n")

    def write(self, s):
        self.text.insert("end", s)
        self.text.see("end")

    def on_save_data(self):
        self.save_btn.config(state="disabled")
        self.status_var.set("Capturing...")
        self.root.update_idletasks()
        try:
            saved, logs = self.collector.capture_and_save()
            for line in logs:
                self.write(line + "\n")
            self.write("\n")
            self.status_var.set(f"Saved {len(saved)}")
        except Exception as e:
            messagebox.showerror("Capture error", str(e))
            self.status_var.set("Error")
        finally:
            self.save_btn.config(state="normal")


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
