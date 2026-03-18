"""Torch-based unit icon classifier."""

from pathlib import Path

import cv2
import numpy as np

from onceworld.config.runtime import CHECKPOINT_DIR, CHECKPOINT_GLOB
from onceworld.core.modeling import build_resnet18_classifier
from onceworld.core.perf import is_debug


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
        except Exception as exc:
            raise RuntimeError(
                "Unit classifier requires torch + torchvision. Install them in your environment."
            ) from exc

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

        model = build_resnet18_classifier(
            num_classes=len(self.class_names),
            models_module=self.models,
            nn_module=self.nn,
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(self.device)
        model.eval()

        self.model = model
        self.checkpoint_path = str(ckpt_path)
        if is_debug():
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
        return self.torch.from_numpy(chw).unsqueeze(0).float().to(self.device)

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
