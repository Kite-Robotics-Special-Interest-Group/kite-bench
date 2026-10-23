"""Input perturbations for VLA observation frames.

Every function takes a 224x224 RGB :class:`PIL.Image` and returns a new image of
the same size. Random perturbations take an explicit ``seed`` so a sweep is
reproducible on any machine.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image, ImageFilter

IMAGE_SIZE = 224


def _as_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def brightness(img: Image.Image, factor: float = 1.2) -> Image.Image:
    """Scale pixel values by ``factor`` (1.0 is identity)."""
    arr = np.asarray(_as_rgb(img), dtype=np.float32) * factor
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def contrast(img: Image.Image, factor: float = 1.2) -> Image.Image:
    arr = np.asarray(_as_rgb(img), dtype=np.float32)
    mean = arr.mean(axis=(0, 1), keepdims=True)
    return Image.fromarray(np.clip((arr - mean) * factor + mean, 0, 255).astype(np.uint8))


def gamma(img: Image.Image, value: float = 1.5) -> Image.Image:
    arr = np.asarray(_as_rgb(img), dtype=np.float32) / 255.0
    out = np.power(arr, value) * 255.0
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def noise(img: Image.Image, sigma: float = 10.0, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(_as_rgb(img), dtype=np.float32)
    arr = arr + rng.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def blur(img: Image.Image, radius: float = 2.0) -> Image.Image:
    return _as_rgb(img).filter(ImageFilter.GaussianBlur(radius=radius))


def jpeg(img: Image.Image, quality: int = 40) -> Image.Image:
    import io

    buf = io.BytesIO()
    _as_rgb(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def translate(img: Image.Image, dx: float = 0.05, dy: float = 0.0) -> Image.Image:
    """Shift the frame by a fraction of its width/height, padding with edge pixels."""
    src = _as_rgb(img)
    w, h = src.size
    px, py = int(round(dx * w)), int(round(dy * h))
    out = Image.new("RGB", (w, h))
    out.paste(src, (px, py))
    if px > 0:
        out.paste(src.crop((0, 0, px, h)).resize((px, h)), (0, 0))
    elif px < 0:
        out.paste(src.crop((w + px, 0, w, h)).resize((-px, h)), (w + px, 0))
    if py > 0:
        band = out.crop((0, 0, w, py))
        out.paste(band, (0, 0))
    return out


def scale(img: Image.Image, factor: float = 1.1) -> Image.Image:
    src = _as_rgb(img)
    w, h = src.size
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    resized = src.resize((nw, nh))
    if factor >= 1.0:
        left, top = (nw - w) // 2, (nh - h) // 2
        return resized.crop((left, top, left + w, top + h))
    out = Image.new("RGB", (w, h))
    out.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    return out


def rotate(img: Image.Image, degrees: float = 5.0) -> Image.Image:
    return _as_rgb(img).rotate(degrees, resample=Image.BILINEAR, expand=False)


def downsample(img: Image.Image, size: int = 112) -> Image.Image:
    src = _as_rgb(img)
    return src.resize((size, size), Image.BILINEAR).resize(src.size, Image.BILINEAR)


def occlude_rect(img: Image.Image, area: float = 0.12, seed: int = 0) -> Image.Image:
    """Paste a gray rectangle covering roughly ``area`` of the frame."""
    src = _as_rgb(img).copy()
    rng = np.random.default_rng(seed)
    w, h = src.size
    side = int(round((area * w * h) ** 0.5))
    x = int(rng.integers(0, max(1, w - side)))
    y = int(rng.integers(0, max(1, h - side)))
    patch = Image.new("RGB", (side, side), (96, 96, 100))
    src.paste(patch, (x, y))
    return src


def occlude_arm(img: Image.Image) -> Image.Image:
    """Simulate the robot arm entering the third-person view from the bottom."""
    src = _as_rgb(img).copy()
    w, h = src.size
    band = Image.new("RGB", (w, int(h * 0.42)), (34, 36, 40))
    src.paste(band, (0, h - int(h * 0.42)))
    return src


def registry() -> dict[str, Callable[..., Image.Image]]:
    """All perturbations by name, for generic sweeps."""
    return {
        "brightness": brightness,
        "contrast": contrast,
        "gamma": gamma,
        "noise": noise,
        "blur": blur,
        "jpeg": jpeg,
        "translate": translate,
        "scale": scale,
        "rotate": rotate,
        "downsample": downsample,
        "occlude_rect": occlude_rect,
        "occlude_arm": occlude_arm,
    }
