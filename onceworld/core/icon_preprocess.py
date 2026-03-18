"""Shared icon preprocessing used by ML training and runtime inference."""

from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def estimate_corner_mean_rgb(img: Image.Image):
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    h, w = arr.shape[:2]
    corners = np.array(
        [
            arr[0, 0],
            arr[0, max(0, w - 1)],
            arr[max(0, h - 1), 0],
            arr[max(0, h - 1), max(0, w - 1)],
        ],
        dtype=np.float32,
    )
    mean = corners.mean(axis=0)
    return tuple(int(round(v)) for v in mean.tolist())


def pad_to_square(img: Image.Image, fill_rgb=None):
    rgb = img.convert("RGB")
    w, h = rgb.size
    if w == h:
        return rgb

    side = max(w, h)
    if fill_rgb is None:
        fill_rgb = estimate_corner_mean_rgb(rgb)

    canvas = Image.new("RGB", (side, side), color=tuple(int(v) for v in fill_rgb))
    x = (side - w) // 2
    y = (side - h) // 2
    canvas.paste(rgb, (x, y))
    return canvas


def shift_lowres_with_padding(low_img: Image.Image, dx, dy, pad_px):
    low = low_img.convert("RGB")
    s_w, s_h = low.size
    if s_w != s_h:
        raise RuntimeError("shift_lowres_with_padding expects a square low-res image.")

    pad_px = int(max(0, pad_px))
    dx = int(dx)
    dy = int(dy)
    if pad_px == 0:
        return low
    if dx < -pad_px or dx > pad_px or dy < -pad_px or dy > pad_px:
        raise RuntimeError("dx/dy must be in [-pad_px, pad_px].")

    side = s_w
    fill_rgb = estimate_corner_mean_rgb(low)
    canvas_side = side + 2 * pad_px
    canvas = Image.new("RGB", (canvas_side, canvas_side), color=fill_rgb)
    canvas.paste(low, (pad_px, pad_px))

    x0 = pad_px + dx
    y0 = pad_px + dy
    return F.crop(canvas, y0, x0, side, side)


def build_tensor_normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_real_eval_preprocess(image_size=128, mean=NORMALIZE_MEAN, std=NORMALIZE_STD):
    image_size = int(image_size)
    if image_size <= 0:
        raise RuntimeError("image_size must be > 0")

    return transforms.Compose(
        [
            transforms.Lambda(lambda img: pad_to_square(img, fill_rgb=None)),
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def pil_from_bgr(icon_bgr):
    rgb = np.ascontiguousarray(icon_bgr[:, :, ::-1])
    return Image.fromarray(rgb, mode="RGB")
