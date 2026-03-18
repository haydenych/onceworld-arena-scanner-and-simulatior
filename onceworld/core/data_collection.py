"""Purpose-built icon data collection service for ML dataset generation."""

from pathlib import Path

import cv2
import numpy as np

from onceworld.config.geometry import ENEMY_MAX_SLOTS
from onceworld.config.runtime import (
    ANCHOR_DIR,
    ANCHOR_SCALES,
    ANCHOR_THRESHOLD,
    UNLABELED_ICON_DIGITS,
    UNLABELED_ICON_DIR,
    UNLABELED_ICON_PREFIX,
)
from onceworld.core.vision import (
    clamp_box,
    crop,
    detect_enemy_icon_boxes,
    grab_screen_bgr,
    load_templates,
    make_row_box_from_anchor,
    match_template_multiscale,
    next_unlabeled_index,
)


class IconDataCollector:
    def __init__(
        self,
        output_dir=UNLABELED_ICON_DIR,
        prefix=UNLABELED_ICON_PREFIX,
        digits=UNLABELED_ICON_DIGITS,
        anchor_scales=ANCHOR_SCALES,
        anchor_threshold=ANCHOR_THRESHOLD,
    ):
        self.required = ("team_a", "team_b", "team_c")
        self.anchors = load_templates(ANCHOR_DIR)
        missing = [x for x in self.required if x not in self.anchors]
        if missing:
            raise RuntimeError(f"Missing anchor templates: {missing} in {ANCHOR_DIR}")

        self.anchor_scales = np.asarray(anchor_scales, dtype=float)
        self.anchor_threshold = float(anchor_threshold)

        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = str(prefix)
        self.digits = int(digits)
        self.next_idx = next_unlabeled_index(
            self.out_dir, prefix=self.prefix, digits=self.digits
        )
        self.cached_anchor_scale = None

    def _match_team_anchor(self, screen, team_name, run_anchor_scale):
        if run_anchor_scale is None:
            return match_template_multiscale(
                screen,
                self.anchors[team_name],
                self.anchor_scales,
                self.anchor_threshold,
                debug_label=f"{team_name}:collect",
            )

        match = match_template_multiscale(
            screen,
            self.anchors[team_name],
            np.array([run_anchor_scale], dtype=np.float32),
            self.anchor_threshold,
            debug_label=f"{team_name}:collect:cached",
        )
        if match is not None:
            return match

        return match_template_multiscale(
            screen,
            self.anchors[team_name],
            self.anchor_scales,
            self.anchor_threshold,
            debug_label=f"{team_name}:collect:fallback",
        )

    def capture_and_save(self):
        screen = grab_screen_bgr()
        run_anchor_scale = self.cached_anchor_scale
        saved = []
        logs = []

        for team_name in self.required:
            match = self._match_team_anchor(screen, team_name, run_anchor_scale)
            if match is None:
                logs.append(f"[{team_name}] anchor not found")
                continue

            run_anchor_scale = float(match["scale"])
            logs.append(
                f"[{team_name}] anchor score={match['score']:.3f} scale={match['scale']:.3f}"
            )

            row_box = make_row_box_from_anchor(match["box"], screen)
            row_img = crop(screen, row_box)
            icon_boxes = detect_enemy_icon_boxes(
                row_img, row_box, max_slots=ENEMY_MAX_SLOTS
            )
            if not icon_boxes:
                logs.append(f"[{team_name}] no enemy boxes found")
                continue

            for slot, box in enumerate(icon_boxes, start=1):
                icon_img = crop(screen, clamp_box(box, screen.shape[1], screen.shape[0]))
                if icon_img.size == 0:
                    continue

                name = f"{self.prefix}{self.next_idx:0{self.digits}d}.png"
                path = self.out_dir / name
                if not cv2.imwrite(str(path), icon_img):
                    logs.append(f"[save-failed] {team_name} enemy_{slot} -> {path}")
                    continue

                saved.append(path)
                logs.append(f"[saved] {team_name} enemy_{slot} -> {path}")
                self.next_idx += 1

        if run_anchor_scale is not None:
            self.cached_anchor_scale = run_anchor_scale

        logs.append(f"done. saved={len(saved)} images to {self.out_dir.resolve()}")
        return saved, logs

