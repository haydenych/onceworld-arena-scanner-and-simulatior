import argparse
import random
import re
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import functional as TF

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from onceworld.core.modeling import build_resnet18_classifier


def center_crop_square(img: Image.Image):
    rgb = img.convert("RGB")
    w, h = rgb.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return rgb.crop((left, top, left + side, top + side))


class SyntheticIconDataset(Dataset):
    def __init__(
        self,
        items,
        image_size=128,
        samples_per_class=128,
        template_size_px=150,
        small_min_px=50,
        small_max_px=85,
        max_shift_px=4,
        seed=1337,
        train=True,
    ):
        self.items = items
        self.image_size = int(image_size)
        self.samples_per_class = int(samples_per_class)
        self.template_size_px = int(template_size_px)
        self.small_min_px = int(small_min_px)
        self.small_max_px = int(small_max_px)
        self.max_shift_px = int(max_shift_px)
        self.seed = int(seed)
        self.train = bool(train)
        self.epoch = 0

        if self.template_size_px < self.image_size:
            raise RuntimeError("--template-size-px must be >= --image-size")
        if not (8 <= self.small_min_px <= self.small_max_px <= self.template_size_px):
            raise RuntimeError(
                "--small-min-px/--small-max-px must satisfy: 8 <= min <= max <= template-size-px"
            )

        self.base_images = []
        for item in self.items:
            img = Image.open(item["path"]).convert("RGB")
            self.base_images.append(center_crop_square(img))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.items) * self.samples_per_class

    def _rng(self, index):
        salt = 1000003 if self.train else 2000003
        return random.Random(self.seed + self.epoch * salt + index * 9176)

    def _augment(self, img, rng):
        if self.train:
            small = rng.randint(self.small_min_px, self.small_max_px)
        else:
            small = (self.small_min_px + self.small_max_px) // 2

        down = img.resize((small, small), resample=Image.NEAREST)
        up = down.resize((self.template_size_px, self.template_size_px), resample=Image.NEAREST)

        max_off = self.template_size_px - self.image_size
        center_off = max_off // 2

        if self.train:
            x_jitter = rng.randint(-self.max_shift_px, self.max_shift_px)
            y_jitter = rng.randint(-self.max_shift_px, self.max_shift_px)
        else:
            x_jitter = 0
            y_jitter = 0

        x0 = max(0, min(max_off, center_off + x_jitter))
        y0 = max(0, min(max_off, center_off + y_jitter))
        return up.crop((x0, y0, x0 + self.image_size, y0 + self.image_size))

    def __getitem__(self, index):
        class_idx = index % len(self.items)
        rng = self._rng(index)

        img = self.base_images[class_idx]
        aug = self._augment(img, rng)

        x = TF.to_tensor(aug)
        x = (x - 0.5) / 0.5
        y = class_idx
        return x, y


class RealIconDataset(Dataset):
    def __init__(
        self,
        samples,
        image_size=128,
        template_size_px=150,
        max_shift_px=4,
        seed=1337,
        train=True,
    ):
        self.samples = samples
        self.image_size = int(image_size)
        self.template_size_px = int(template_size_px)
        self.max_shift_px = int(max_shift_px)
        self.seed = int(seed)
        self.train = bool(train)
        self.epoch = 0

        if self.template_size_px < self.image_size:
            raise RuntimeError("--template-size-px must be >= --image-size")

        self.base_images = []
        self.labels = []
        for s in self.samples:
            img = Image.open(s["path"]).convert("RGB")
            self.base_images.append(center_crop_square(img))
            self.labels.append(int(s["class_idx"]))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.labels)

    def _rng(self, index):
        salt = 3000007 if self.train else 4000007
        return random.Random(self.seed + self.epoch * salt + index * 7919)

    def _augment(self, img, rng):
        up = img.resize((self.template_size_px, self.template_size_px), resample=Image.NEAREST)

        max_off = self.template_size_px - self.image_size
        center_off = max_off // 2

        if self.train:
            x_jitter = rng.randint(-self.max_shift_px, self.max_shift_px)
            y_jitter = rng.randint(-self.max_shift_px, self.max_shift_px)
        else:
            x_jitter = 0
            y_jitter = 0

        x0 = max(0, min(max_off, center_off + x_jitter))
        y0 = max(0, min(max_off, center_off + y_jitter))
        return up.crop((x0, y0, x0 + self.image_size, y0 + self.image_size))

    def __getitem__(self, index):
        rng = self._rng(index)
        img = self.base_images[index]
        aug = self._augment(img, rng)

        x = TF.to_tensor(aug)
        x = (x - 0.5) / 0.5
        y = self.labels[index]
        return x, y


class MixedDataset(Dataset):
    def __init__(self, synthetic_ds=None, real_ds=None, real_ratio=0.5, seed=1337, train=True):
        self.synthetic_ds = synthetic_ds
        self.real_ds = real_ds
        self.real_ratio = float(real_ratio)
        self.seed = int(seed)
        self.train = bool(train)
        self.epoch = 0

        self.syn_len = len(synthetic_ds) if synthetic_ds is not None else 0
        self.real_len = len(real_ds) if real_ds is not None else 0

        if self.syn_len == 0 and self.real_len == 0:
            raise RuntimeError("MixedDataset needs at least one non-empty dataset.")

        if self.syn_len > 0 and self.real_len > 0:
            self.length = max(self.syn_len, self.real_len)
        else:
            self.length = self.syn_len if self.syn_len > 0 else self.real_len

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.synthetic_ds is not None and hasattr(self.synthetic_ds, "set_epoch"):
            self.synthetic_ds.set_epoch(epoch)
        if self.real_ds is not None and hasattr(self.real_ds, "set_epoch"):
            self.real_ds.set_epoch(epoch)

    def __len__(self):
        return self.length

    def _rng(self, index):
        salt = 5000011 if self.train else 6000011
        return random.Random(self.seed + self.epoch * salt + index * 6151)

    def __getitem__(self, index):
        if self.syn_len == 0:
            return self.real_ds[index % self.real_len]
        if self.real_len == 0:
            return self.synthetic_ds[index % self.syn_len]

        use_real = self._rng(index).random() < self.real_ratio
        if use_real:
            return self.real_ds[index % self.real_len]
        return self.synthetic_ds[index % self.syn_len]


def scan_templates(units_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = [p for p in sorted(units_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    items = []
    for p in files:
        items.append({"name": p.stem, "path": p})
    return items


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
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = [p for p in sorted(real_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]

    samples = []
    skipped = 0
    for p in files:
        cls = parse_labeled_class_name(p.stem, class_to_idx)
        if cls is None:
            skipped += 1
            continue
        samples.append({
            "path": p,
            "class_idx": int(class_to_idx[cls]),
            "class_name": cls,
        })

    return samples, skipped


def split_real_samples_random(samples, train_ratio, seed):
    """
    Stratified split by class_idx (kept function name for compatibility).
    """
    if not samples:
        return [], []

    train_ratio = float(train_ratio)
    if train_ratio <= 0.0:
        return [], list(samples)
    if train_ratio >= 1.0:
        return list(samples), []

    by_class = {}
    for s in samples:
        by_class.setdefault(int(s["class_idx"]), []).append(s)

    train = []
    val = []

    for class_idx in sorted(by_class.keys()):
        group = by_class[class_idx]
        n = len(group)

        rng = random.Random(seed + class_idx * 104729)
        order = list(range(n))
        rng.shuffle(order)

        if n == 1:
            # With one sample we cannot split; bias toward training so class is learnable.
            train.append(group[order[0]])
            continue

        n_train = int(round(n * train_ratio))
        # Best-effort stratification: keep both splits populated for n>=2.
        n_train = max(1, min(n - 1, n_train))

        train.extend(group[i] for i in order[:n_train])
        val.extend(group[i] for i in order[n_train:])

    rng_all = random.Random(seed + 99991)
    rng_all.shuffle(train)
    rng_all.shuffle(val)
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
    parser = argparse.ArgumentParser(description="Train static-icon classifier with in-memory augmentation.")
    parser.add_argument("--units-dir", type=str, default="assets/units")
    parser.add_argument("--real-dir", type=str, default="dataset_enemy_icons")
    parser.add_argument("--real-train-ratio", type=float, default=0.9)
    parser.add_argument("--real-mix-ratio-train", type=float, default=0.5)
    parser.add_argument("--real-mix-ratio-val", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--train-samples-per-class", type=int, default=10)
    parser.add_argument("--val-samples-per-class", type=int, default=10)
    parser.add_argument("--template-size-px", type=int, default=150)
    parser.add_argument("--small-min-px", type=int, default=50)
    parser.add_argument("--small-max-px", type=int, default=85)
    parser.add_argument("--max-shift-px", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    units_dir = Path(args.units_dir)
    if not units_dir.exists():
        raise RuntimeError(f"Units directory not found: {units_dir}")

    items = scan_templates(units_dir)
    if not items:
        raise RuntimeError(f"No unit template files found in: {units_dir}")

    class_names = [x["name"] for x in items]
    num_classes = len(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    if not (0.0 <= args.real_train_ratio <= 1.0):
        raise RuntimeError("--real-train-ratio must be in [0, 1]")
    if not (0.0 <= args.real_mix_ratio_train <= 1.0):
        raise RuntimeError("--real-mix-ratio-train must be in [0, 1]")
    if not (0.0 <= args.real_mix_ratio_val <= 1.0):
        raise RuntimeError("--real-mix-ratio-val must be in [0, 1]")

    train_synth_ds = SyntheticIconDataset(
        items=items,
        image_size=args.image_size,
        samples_per_class=args.train_samples_per_class,
        template_size_px=args.template_size_px,
        small_min_px=args.small_min_px,
        small_max_px=args.small_max_px,
        max_shift_px=args.max_shift_px,
        seed=args.seed,
        train=True,
    )
    val_synth_ds = SyntheticIconDataset(
        items=items,
        image_size=args.image_size,
        samples_per_class=args.val_samples_per_class,
        template_size_px=args.template_size_px,
        small_min_px=args.small_min_px,
        small_max_px=args.small_max_px,
        max_shift_px=0,
        seed=args.seed + 9999,
        train=False,
    )

    real_train_ds = None
    real_val_ds = None
    real_dir = Path(args.real_dir)
    if real_dir.exists():
        real_samples_all, real_skipped = scan_real_labeled(real_dir, class_to_idx)
        real_train_samples, real_val_samples = split_real_samples_random(
            real_samples_all, args.real_train_ratio, args.seed + 12345
        )

        if real_train_samples:
            real_train_ds = RealIconDataset(
                real_train_samples,
                image_size=args.image_size,
                template_size_px=args.template_size_px,
                max_shift_px=args.max_shift_px,
                seed=args.seed + 23456,
                train=True,
            )
        if real_val_samples:
            real_val_ds = RealIconDataset(
                real_val_samples,
                image_size=args.image_size,
                template_size_px=args.template_size_px,
                max_shift_px=0,
                seed=args.seed + 34567,
                train=False,
            )

        print(
            f"real labeled: total={len(real_samples_all)} "
            f"(skipped_unmatched={real_skipped}) | "
            f"train={len(real_train_samples)} val={len(real_val_samples)}"
        )
    else:
        print(f"real labeled dir not found, using synthetic-only: {real_dir}")

    train_ds = MixedDataset(
        synthetic_ds=train_synth_ds,
        real_ds=real_train_ds,
        real_ratio=args.real_mix_ratio_train,
        seed=args.seed + 45678,
        train=True,
    )
    val_ds = MixedDataset(
        synthetic_ds=val_synth_ds,
        real_ds=real_val_ds,
        real_ratio=args.real_mix_ratio_val,
        seed=args.seed + 56789,
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
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out_dir) / f"unit_resnet18_{ts}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)

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
                "image_size": args.image_size,
                "normalize_mean": [0.5, 0.5, 0.5],
                "normalize_std": [0.5, 0.5, 0.5],
            }
            torch.save(payload, out_path)
            print(f"saved best -> {out_path} (val_loss={val_loss:.4f})")

    print(f"done. best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
