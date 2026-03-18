"""Detection orchestration: anchors, OCR, classifier, and simulation."""

import time
from pathlib import Path

import cv2
import numpy as np

from onceworld.config.runtime import (
    ANCHOR_DIR,
    ANCHOR_SCALES,
    ANCHOR_THRESHOLD,
    CHECKPOINT_DIR,
    UNIT_MARGIN_THRESHOLD,
    UNIT_THRESHOLD,
    UNLABELED_ICON_DIGITS,
    UNLABELED_ICON_DIR,
    UNLABELED_ICON_PREFIX,
)
from onceworld.core.classifier import UnitClassifier
from onceworld.core.ocr import read_coin_text, read_level_text
from onceworld.core.perf import debug_perf, is_debug
from onceworld.core.sim import BattleSimulator
from onceworld.core.vision import (
    clamp_box,
    crop,
    detect_enemy_icon_boxes,
    grab_screen_bgr,
    load_templates,
    make_row_box_from_anchor,
    match_template_multiscale,
    next_unlabeled_index,
    sub_box,
)
from onceworld.config.geometry import COIN_ZONE


class ScreenDetector:
    def __init__(self, save_unlabeled=False):
        self.anchor_templates = load_templates(ANCHOR_DIR)

        required_anchors = {"team_a", "team_b", "team_c"}
        missing = [name for name in required_anchors if name not in self.anchor_templates]
        if missing:
            raise RuntimeError(
                f"Missing anchor templates: {missing}\n"
                f"Expected files in {ANCHOR_DIR}"
            )

        self.unit_classifier = UnitClassifier(CHECKPOINT_DIR)
        self.simulator = BattleSimulator()
        self.cached_anchor_scale = None
        self.save_unlabeled = bool(save_unlabeled)
        if self.save_unlabeled:
            self.unlabeled_icon_dir = Path(UNLABELED_ICON_DIR)
            self.unlabeled_icon_dir.mkdir(parents=True, exist_ok=True)
            self.next_unlabeled_idx = next_unlabeled_index(self.unlabeled_icon_dir)
        else:
            self.unlabeled_icon_dir = None
            self.next_unlabeled_idx = 0

    def detect(self):
        t_detect_total = time.time()
        t_capture = time.time()
        screen_bgr = grab_screen_bgr()
        debug_perf("capture_screen", t_capture)

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
                    if is_debug():
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
            debug_perf(f"{team_name}:anchor_search", t_anchor)

            if anchor_match is None:
                results[team_name] = {"found": False, "reason": "anchor not found"}
                debug_perf(f"{team_name}:total", t_team_total)
                continue

            run_anchor_scale = float(anchor_match["scale"])
            row_box = make_row_box_from_anchor(anchor_match["box"], screen_bgr)
            if is_debug():
                print(
                    f"[anchor] {team_name}: score={anchor_match['score']:.3f} "
                    f"scale={anchor_match['scale']:.3f}"
                )
            t_row_detect = time.time()
            team_result = self._detect_team_row(screen_bgr, row_box, team_name)
            debug_perf(f"{team_name}:row_detect", t_row_detect)
            team_result["found"] = True
            team_result["anchor_score"] = round(anchor_match["score"], 3)
            team_result["anchor_scale"] = round(anchor_match["scale"], 3)
            team_result["row_box"] = row_box
            results[team_name] = team_result
            debug_perf(f"{team_name}:total", t_team_total)

        if run_anchor_scale is not None:
            self.cached_anchor_scale = run_anchor_scale

        results["_battle"] = self.simulator.simulate(results)
        debug_perf("detect_total", t_detect_total)
        return results

    def _detect_team_row(self, screen_bgr, row_box, team_name):
        t_row_total = time.time()
        row_img = crop(screen_bgr, row_box)

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

        for idx, icon_box in enumerate(icon_boxes, start=1):
            icon_box = clamp_box(icon_box, screen_bgr.shape[1], screen_bgr.shape[0])
            icon_img = crop(screen_bgr, icon_box)

            if self.save_unlabeled:
                t_save_icon = time.time()
                unlabeled_name = (
                    f"{UNLABELED_ICON_PREFIX}{self.next_unlabeled_idx:0{UNLABELED_ICON_DIGITS}d}.png"
                )
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
            else:
                best_name = pred["name"]
                best_score = pred["score"]
                margin = pred["margin"]

            x1, y1, x2, y2 = icon_box
            icon_w = x2 - x1
            icon_h = y2 - y1

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

            units.append(
                {
                    "unit_name": display_name,
                    "score": round(best_score, 3) if best_score >= 0 else None,
                    "icon_box": icon_box,
                    "level": level_value,
                    "level_raw_text": level_raw_text,
                }
            )

        if is_debug():
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
