"""Unit icon classifier runtime wrapper."""

from pathlib import Path

from onceworld.config.runtime import CHECKPOINT_DIR, CHECKPOINT_GLOB
from onceworld.core.icon_preprocess import build_real_eval_preprocess, pil_from_bgr
from onceworld.core.modeling import build_resnet18_classifier
from onceworld.core.perf import is_debug


def _find_latest_checkpoint(checkpoint_dir):
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


class UnitClassifier:
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = _find_latest_checkpoint(checkpoint_dir)
        payload = torch.load(ckpt_path, map_location=self.device)

        class_names = payload.get("class_names")
        if not isinstance(class_names, list) or not class_names:
            raise RuntimeError(f"Checkpoint missing valid class_names: {ckpt_path}")
        self.class_names = class_names

        preprocess = payload.get("preprocess")
        if not isinstance(preprocess, dict):
            raise RuntimeError(f"Checkpoint missing preprocess config: {ckpt_path}")

        mode = str(preprocess.get("mode", ""))
        if mode != "pad_square_then_resize":
            raise RuntimeError(f"Unsupported preprocess mode in checkpoint: {mode}")

        image_size = int(preprocess["image_size"])
        mean = tuple(float(x) for x in preprocess["mean"])
        std = tuple(float(x) for x in preprocess["std"])
        if any(x == 0.0 for x in std):
            raise RuntimeError(f"Checkpoint preprocess std contains zero: {ckpt_path}")

        model = build_resnet18_classifier(
            num_classes=len(self.class_names),
            models_module=models,
            nn_module=nn,
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(self.device)
        model.eval()

        self.model = model
        self.preprocess = build_real_eval_preprocess(
            image_size=image_size,
            mean=mean,
            std=std,
        )
        self.checkpoint_path = str(ckpt_path)
        if is_debug():
            print(f"[unit-model] loaded checkpoint: {self.checkpoint_path}")

    def _preprocess(self, icon_bgr):
        pil_img = pil_from_bgr(icon_bgr)
        x = self.preprocess(pil_img)
        return x.unsqueeze(0).float().to(self.device)

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
