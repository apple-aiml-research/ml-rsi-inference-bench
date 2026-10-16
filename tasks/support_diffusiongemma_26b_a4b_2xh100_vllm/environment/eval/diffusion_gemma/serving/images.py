#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Deterministic synthetic RGB images for the served checks (public)."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw

COLOURS = {"red": (220, 40, 40), "green": (40, 170, 70), "blue": (40, 90, 220), "yellow": (240, 210, 40),
           "purple": (140, 60, 190), "orange": (245, 140, 30), "black": (20, 20, 20), "white": (250, 250, 250)}
SHAPES = ("circle", "square", "triangle")


def spec(seed: int, n_shapes: int = 3) -> list[dict]:
    """The shapes an image carries: kind, colour and box, drawn from the seed."""
    rng = np.random.default_rng(seed)
    names = list(COLOURS)
    out = []
    for i in range(n_shapes):
        out.append({"kind": SHAPES[int(rng.integers(len(SHAPES)))], "colour": names[int(rng.integers(len(names) - 2))],
                    "cx": float(0.2 + 0.6 * (i + 0.5) / n_shapes), "cy": float(0.3 + 0.4 * rng.random()),
                    "r": float(0.08 + 0.06 * rng.random())})
    return out


def image(seed: int, size: tuple[int, int] = (896, 672), n_shapes: int = 3) -> Image.Image:
    w, h = size
    rng = np.random.default_rng(10_000 + seed)
    top = rng.integers(170, 255, size=3)
    bottom = rng.integers(120, 220, size=3)
    t = np.linspace(0.0, 1.0, h)[:, None, None]
    grad = (top[None, None, :] * (1 - t) + bottom[None, None, :] * t).astype(np.uint8)
    img = Image.fromarray(np.broadcast_to(grad, (h, w, 3)).copy(), "RGB")
    d = ImageDraw.Draw(img)
    for s in spec(seed, n_shapes):
        cx, cy, r = s["cx"] * w, s["cy"] * h, s["r"] * min(w, h)
        col = COLOURS[s["colour"]]
        if s["kind"] == "circle":
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        elif s["kind"] == "square":
            d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=col)
        else:
            d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=col)
    return img


def data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
