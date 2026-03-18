import argparse
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from onceworld.core.icon_preprocess import (
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    build_real_eval_preprocess,
    build_tensor_normalize,
    shift_lowres_with_padding,
)
from onceworld.core.modeling import build_resnet18_classifier


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class SyntheticIconDataset(Dataset):
    def __init__(
        self,
        items,
        image_size=128,
        samples_per_class=10,
        small_min_px=50,
        small_max_px=85,
        max_shift_px=2,
        train=True,
    ):
        self.items = items
        self.image_size = int(image_size)
        self.samples_per_class = int(samples_per_class)
        self.small_min_px = int(small_min_px)
        self.small_max_px = int(small_max_px)
        self.max_shift_px = int(max_shift_px)
        self.train = bool(train)

        if not self.items:
            raise RuntimeError("SyntheticIconDataset needs at least one template.")
        if self.image_size <= 0:
            raise RuntimeError("--image-size must be > 0")
        if not (8 <= self.small_min_px <= self.small_max_px):
            raise RuntimeError("--small-min-px/--small-max-px must satisfy 8 <= min <= max.")
        if self.max_shift_px < 0:
            raise RuntimeError("--max-shift-px must be >= 0")

        self.mid_small_px = (self.small_min_px + self.small_max_px) // 2
        self.to_tensor = build_tensor_normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        self.base_images = [Image.open(item["path"]).convert("RGB") for item in self.items]

    def __len__(self):
        return len(self.items) * self.samples_per_class

    def __getitem__(self, index):
        class_idx = int(index % len(self.base_images))
        img = self.base_images[class_idx]

        if self.train:
            small_px = random.randint(self.small_min_px, self.small_max_px)
            dx = random.randint(-self.max_shift_px, self.max_shift_px)
            dy = random.randint(-self.max_shift_px, self.max_shift_px)
        else:
            small_px = self.mid_small_px
            dx = 0
            dy = 0

        low = F.resize(img, [small_px, small_px], interpolation=InterpolationMode.NEAREST)
        low_shifted = shift_lowres_with_padding(
            low,
            dx=dx,
            dy=dy,
            pad_px=self.max_shift_px,
        )
        up = F.resize(
            low_shifted,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        x = self.to_tensor(up)
        y = class_idx
        return x, y


class RealIconDataset(Dataset):
    def __init__(self, samples, image_size=128):
        self.samples = samples
        self.preprocess = build_real_eval_preprocess(
            image_size=int(image_size),
            mean=NORMALIZE_MEAN,
            std=NORMALIZE_STD,
        )
        self.images = [Image.open(s["path"]).convert("RGB") for s in self.samples]
        self.labels = [int(s["class_idx"]) for s in self.samples]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.preprocess(self.images[index]), self.labels[index]


class MixedDataset(Dataset):
    def __init__(self, synthetic_ds=None, real_ds=None, real_ratio=0.5, train=True):
        self.synthetic_ds = synthetic_ds
        self.real_ds = real_ds
        self.real_ratio = float(real_ratio)
        self.train = bool(train)

        self.syn_len = len(self.synthetic_ds) if self.synthetic_ds is not None else 0
        self.real_len = len(self.real_ds) if self.real_ds is not None else 0
        if self.syn_len == 0 and self.real_len == 0:
            raise RuntimeError("MixedDataset needs at least one non-empty dataset.")
        self.length = max(self.syn_len, self.real_len)

    def __len__(self):
        return self.length

    def _use_real(self, index):
        if self.real_len == 0:
            return False
        if self.syn_len == 0:
            return True
        if self.train:
            return random.random() < self.real_ratio
        cutoff = int(round(1000 * self.real_ratio))
        return (index % 1000) < cutoff

    def __getitem__(self, index):
        if self._use_real(index):
            return self.real_ds[index % self.real_len]
        return self.synthetic_ds[index % self.syn_len]


def scan_templates(units_dir: Path):
    files = [
        p
        for p in sorted(units_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return [{"name": p.stem, "path": p} for p in files]


def parse_labeled_class_name(stem: str, class_to_idx):
    if stem in class_to_idx:
        return stem

    m = re.match(r"(.+)__(\d+)$", stem)
    if m and m.group(1) in class_to_idx:
        return m.group(1)

    m = re.match(r"(.+)_(\d+)$", stem)
    if m and m.group(1) in class_to_idx:
        return m.group(1)

    return None


def scan_real_labeled(real_dir: Path, class_to_idx):
    files = [
        p
        for p in sorted(real_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    samples = []
    skipped = 0
    for p in files:
        cls = parse_labeled_class_name(p.stem, class_to_idx)
        if cls is None:
            skipped += 1
            continue
        samples.append(
            {
                "path": p,
                "class_idx": int(class_to_idx[cls]),
                "class_name": cls,
            }
        )
    return samples, skipped


def split_real_samples(samples, train_ratio, seed):
    if not samples:
        return [], []

    train_ratio = float(train_ratio)
    if train_ratio <= 0.0:
        return [], list(samples)
    if train_ratio >= 1.0:
        return list(samples), []

    labels = [int(s["class_idx"]) for s in samples]
    counts = Counter(labels)

    split_indices = [i for i, y in enumerate(labels) if counts[y] >= 2]
    singleton_indices = [i for i, y in enumerate(labels) if counts[y] < 2]

    train_indices = []
    val_indices = []

    if split_indices:
        split_labels = [labels[i] for i in split_indices]
        n = len(split_indices)
        n_classes = len(set(split_labels))
        requested_train = int(round(n * train_ratio))
        min_train = n_classes
        max_train = n - n_classes
        train_size = max(min_train, min(max_train, requested_train))

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=train_size,
            random_state=int(seed),
        )
        idx_range = list(range(n))
        train_rel, val_rel = next(splitter.split(idx_range, split_labels))
        train_indices.extend(split_indices[i] for i in train_rel)
        val_indices.extend(split_indices[i] for i in val_rel)

    train_indices.extend(singleton_indices)
    train = [samples[i] for i in train_indices]
    val = [samples[i] for i in val_indices]
    return train, val


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch = y.size(0)
        total += batch
        total_loss += float(loss.item()) * batch
        total_correct += int((logits.argmax(dim=1) == y).sum().item())

    return total_loss / max(1, total), total_correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        batch = y.size(0)
        total += batch
        total_loss += float(loss.item()) * batch
        total_correct += int((logits.argmax(dim=1) == y).sum().item())

    return total_loss / max(1, total), total_correct / max(1, total)


def build_model(num_classes):
    return build_resnet18_classifier(num_classes, models, nn)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train static icon classifier with synthetic + real data."
    )
    parser.add_argument("--units-dir", type=str, default="assets/units")
    parser.add_argument("--real-dir", type=str, default="dataset_enemy_icons")
    parser.add_argument("--real-train-ratio", type=float, default=0.7 )
    parser.add_argument("--real-mix-ratio-train", type=float, default=0.8)
    parser.add_argument("--real-mix-ratio-val", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--small-min-px", type=int, default=50)
    parser.add_argument("--small-max-px", type=int, default=85)
    parser.add_argument("--max-shift-px", type=int, default=2)
    parser.add_argument("--train-samples-per-class", type=int, default=10)
    parser.add_argument("--val-samples-per-class", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if not (0.0 <= args.real_train_ratio <= 1.0):
        raise RuntimeError("--real-train-ratio must be in [0, 1]")
    if not (0.0 <= args.real_mix_ratio_train <= 1.0):
        raise RuntimeError("--real-mix-ratio-train must be in [0, 1]")
    if not (0.0 <= args.real_mix_ratio_val <= 1.0):
        raise RuntimeError("--real-mix-ratio-val must be in [0, 1]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    units_dir = Path(args.units_dir)
    if not units_dir.exists():
        raise RuntimeError(f"Units directory not found: {units_dir}")

    items = scan_templates(units_dir)
    if not items:
        raise RuntimeError(f"No unit template files found in: {units_dir}")

    class_names = [x["name"] for x in items]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    train_synth_ds = SyntheticIconDataset(
        items=items,
        image_size=args.image_size,
        samples_per_class=args.train_samples_per_class,
        small_min_px=args.small_min_px,
        small_max_px=args.small_max_px,
        max_shift_px=args.max_shift_px,
        train=True,
    )
    val_synth_ds = SyntheticIconDataset(
        items=items,
        image_size=args.image_size,
        samples_per_class=args.val_samples_per_class,
        small_min_px=args.small_min_px,
        small_max_px=args.small_max_px,
        max_shift_px=args.max_shift_px,
        train=False,
    )

    real_train_ds = None
    real_val_ds = None
    real_dir = Path(args.real_dir)
    if real_dir.exists():
        real_all, real_skipped = scan_real_labeled(real_dir, class_to_idx)
        real_train_samples, real_val_samples = split_real_samples(
            real_all, args.real_train_ratio, args.seed
        )

        if real_train_samples:
            real_train_ds = RealIconDataset(
                real_train_samples,
                image_size=args.image_size,
            )
        if real_val_samples:
            real_val_ds = RealIconDataset(
                real_val_samples,
                image_size=args.image_size,
            )

        print(
            f"real labeled: total={len(real_all)} "
            f"(skipped_unmatched={real_skipped}) | "
            f"train={len(real_train_samples)} val={len(real_val_samples)}"
        )
    else:
        print(f"real labeled dir not found, using synthetic-only: {real_dir}")

    train_ds = MixedDataset(
        synthetic_ds=train_synth_ds,
        real_ds=real_train_ds,
        real_ratio=args.real_mix_ratio_train,
        train=True,
    )
    val_ds = MixedDataset(
        synthetic_ds=val_synth_ds,
        real_ds=real_val_ds,
        real_ratio=args.real_mix_ratio_val,
        train=False,
    )

    print(
        f"mix ratios | train_real={args.real_mix_ratio_train:.2f} "
        f"val_real={args.real_mix_ratio_val:.2f}"
    )
    print(
        f"dataset lengths | train_syn={len(train_synth_ds)} "
        f"train_real={len(real_train_ds) if real_train_ds is not None else 0} "
        f"train_mix={len(train_ds)} | "
        f"val_syn={len(val_synth_ds)} "
        f"val_real={len(real_val_ds) if real_val_ds is not None else 0} "
        f"val_mix={len(val_ds)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_dir) / f"unit_resnet18_{ts}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            payload = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "preprocess": {
                    "mode": "pad_square_then_resize",
                    "image_size": int(args.image_size),
                    "mean": list(NORMALIZE_MEAN),
                    "std": list(NORMALIZE_STD),
                },
            }
            torch.save(payload, out_path)
            print(f"saved best -> {out_path} (val_loss={val_loss:.4f})")

    print(f"done. best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
