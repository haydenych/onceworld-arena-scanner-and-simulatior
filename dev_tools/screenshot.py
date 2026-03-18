import re
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import cv2
import mss
import numpy as np
import pytesseract


# ============================================================
# Config
# ============================================================

OUT_DIR = Path("assets/units")
OCR_DEBUG_DIR = OUT_DIR / "_ocr_debug"
FILE_PREFIX = "img"
FILE_DIGITS = 4
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
REMINDER_TEXT = (
    "Reminder: Set emulator to full screen before taking screenshots.\n"
    "Screenshot 1: Captured pets, Screenshot 2: Non-captured."
)
SHOT_1_NAME = "Screenshot 1: Captured pets"
SHOT_2_NAME = "Screenshot 2: Non-captured"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# Hardcoded portrait rectangle ratios on the captured monitor:
# (x1, y1, x2, y2). Width is intentionally larger than height.
PORTRAIT_RECT_1 = (0.35, 0.20, 0.635, 0.347)
# Slightly lower variant for monsters whose portrait box sits lower.
PORTRAIT_RECT_2 = (0.35, 0.307, 0.635, 0.454)


# ============================================================
# Utilities
# ============================================================

def grab_screen_bgr():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        return np.array(shot)[:, :, :3]


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


def next_image_index(out_dir, prefix=FILE_PREFIX, digits=FILE_DIGITS):
    pat = re.compile(rf"^{re.escape(prefix)}(\d{{{digits}}})\.png$", re.IGNORECASE)
    max_idx = 0
    if out_dir.exists():
        for p in out_dir.glob(f"{prefix}*.png"):
            m = pat.match(p.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def sanitize_name_for_filename(text):
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text or None


def build_output_path(out_dir, base_name, fallback_idx):
    if base_name:
        return out_dir / f"{base_name}.png"
    return out_dir / f"{FILE_PREFIX}{fallback_idx:0{FILE_DIGITS}d}.png"


def fixed_portrait_box(screen_bgr, rect_ratio):
    h, w = screen_bgr.shape[:2]
    fx1, fy1, fx2, fy2 = rect_ratio
    box = (
        int(round(fx1 * w)),
        int(round(fy1 * h)),
        int(round(fx2 * w)),
        int(round(fy2 * h)),
    )
    return clamp_box(box, w, h)


def square_from_rect_by_height(rect_box, w, h):
    x1, y1, x2, y2 = rect_box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    side = bh

    # Keep vertical range, crop width around horizontal center.
    cx = 0.5 * (x1 + x2)
    sx1 = int(round(cx - side / 2.0))
    sx2 = sx1 + side
    sy1 = y1
    sy2 = y1 + side

    sq = clamp_box((sx1, sy1, sx2, sy2), w, h)
    sx1, sy1, sx2, sy2 = sq

    # Ensure exact square after clamping.
    side2 = min(sx2 - sx1, sy2 - sy1)
    return (sx1, sy1, sx1 + side2, sy1 + side2)


def extract_monster_name(screen_bgr, portrait_box, debug_tag=None):
    h, w = screen_bgr.shape[:2]
    x1, y1, x2, y2 = portrait_box
    pw = max(1, x2 - x1)
    ph = max(1, y2 - y1)

    # Name text line above portrait box, ending before "Lv."
    name_box = (
        x1 + int(round(0.4 * pw)),
        y1 - int(round(0.36 * ph)),
        x1 + int(round(1 * pw)),
        y1 - int(round(0.08 * ph)),
    )
    name_box = clamp_box(name_box, w, h)
    roi = crop(screen_bgr, name_box)
    if roi is None or roi.size == 0:
        return None

    if debug_tag:
        OCR_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OCR_DEBUG_DIR / f"ocr_box_{debug_tag}.png"), roi)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if debug_tag:
        cv2.imwrite(str(OCR_DEBUG_DIR / f"ocr_box_bw_{debug_tag}.png"), bw)

    text = pytesseract.image_to_string(bw, config="--oem 3 --psm 7")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None

    m = re.search(r"\bL[vV][\.\s]*\d*", text)
    if m:
        text = text[: m.start()].strip()

    text = text.strip(" -_|")
    return sanitize_name_for_filename(text)


# ============================================================
# Collector + GUI
# ============================================================

class ScreenshotCollector:
    def __init__(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.next_idx = next_image_index(OUT_DIR)

    def capture_and_save(self, rect_ratio, shot_name):
        screen = grab_screen_bgr()
        h, w = screen.shape[:2]
        rect = fixed_portrait_box(screen, rect_ratio)
        box = square_from_rect_by_height(rect, w, h)
        monster_name = extract_monster_name(
            screen,
            rect,
            debug_tag=f"{FILE_PREFIX}{self.next_idx:0{FILE_DIGITS}d}",
        )

        icon = crop(screen, box)
        if icon is None or icon.size == 0:
            return None, "Fixed box crop was empty."

        out_path = build_output_path(OUT_DIR, monster_name, self.next_idx)
        ok = cv2.imwrite(str(out_path), icon)
        if not ok:
            return None, f"Failed to save {out_path}"

        self.next_idx += 1
        if monster_name:
            return out_path, f"Saved ({shot_name}): {out_path} | name={monster_name} | box={box}"
        return out_path, f"Saved ({shot_name}): {out_path} | name=OCR_FAILED | box={box}"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Screenshot Gray Box Cropper")
        self.root.geometry("900x650")

        self.collector = ScreenshotCollector()

        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")

        self.capture1_btn = ttk.Button(top, text="Screenshot 1: Captured pets (F1)", command=self.on_capture_1)
        self.capture1_btn.pack(side="left")
        self.capture2_btn = ttk.Button(top, text="Screenshot 2: Non-captured (F2)", command=self.on_capture_2)
        self.capture2_btn.pack(side="left", padx=8)

        self.root.bind("<F1>", lambda _e: self.on_capture_1())
        self.root.bind("<F2>", lambda _e: self.on_capture_2())

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status_var).pack(side="left", padx=12)

        reminder = ttk.Label(root, text=REMINDER_TEXT, justify="left", padding=(10, 0, 10, 0))
        reminder.pack(fill="x")

        self.text = tk.Text(root, wrap="word", font=("Consolas", 11))
        self.text.pack(fill="both", expand=True, padx=10, pady=10)
        self._write(REMINDER_TEXT + "\n")
        self._write("Ready.\n\n")

    def _write(self, s):
        self.text.insert("end", s)
        self.text.see("end")

    def _capture_with_rect(self, rect_ratio, shot_name):
        self.capture1_btn.config(state="disabled")
        self.capture2_btn.config(state="disabled")
        self.status_var.set("Capturing...")
        self.root.update_idletasks()
        try:
            out_path, msg = self.collector.capture_and_save(rect_ratio, shot_name)
            self._write(msg + "\n")
            self._write("\n")
            if out_path is None:
                self.status_var.set("Error")
            else:
                self.status_var.set("Saved 1")
        except Exception as e:
            messagebox.showerror("Capture error", str(e))
            self.status_var.set("Error")
        finally:
            self.capture1_btn.config(state="normal")
            self.capture2_btn.config(state="normal")

    def on_capture_1(self):
        self._capture_with_rect(PORTRAIT_RECT_1, SHOT_1_NAME)

    def on_capture_2(self):
        self._capture_with_rect(PORTRAIT_RECT_2, SHOT_2_NAME)


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
