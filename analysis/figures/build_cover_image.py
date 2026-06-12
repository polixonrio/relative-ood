"""Build the decorative thesis cover image (cover.png) for draft/aauReportTemplate/AAUgraphics.

Run: python analysis/figures/build_cover_image.py
"""
from pathlib import Path
import math
import random

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[2] / 'draft' / 'aauReportTemplate'
OUT = ROOT / 'AAUgraphics' / 'cover.png'

WIDTH = 1800
HEIGHT = 980

BG_TOP = (244, 244, 241)
BG_BOTTOM = (233, 236, 240)
NAVY = (23, 33, 71)
BLUE = (47, 107, 255)
GREEN = (19, 134, 107)
AMBER = (214, 137, 27)
RUST = (165, 77, 42)


def vertical_gradient(size, top, bottom):
    width, height = size
    image = Image.new('RGB', size, top)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        t = y / max(height - 1, 1)
        color = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        draw.line((0, y, width, y), fill=color)
    return image


def add_soft_blob(base, center, radius, color, alpha):
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color + (alpha,))
    layer = layer.filter(ImageFilter.GaussianBlur(radius // 3))
    base.alpha_composite(layer)


def add_ribbon(base, color, width_scale, amp, phase, y_center, thickness, alpha):
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    points_top = []
    points_bottom = []
    for x in range(-40, WIDTH + 41, 18):
        y = y_center + math.sin((x / WIDTH) * math.pi * width_scale + phase) * amp
        points_top.append((x, y - thickness / 2))
        points_bottom.append((x, y + thickness / 2))
    polygon = points_top + list(reversed(points_bottom))
    draw.polygon(polygon, fill=color + (alpha,))
    layer = layer.filter(ImageFilter.GaussianBlur(10))
    base.alpha_composite(layer)


def add_contour_lines(base):
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    line_color = NAVY + (48,)
    for offset in range(-5, 6):
        points = []
        for x in range(0, WIDTH + 1, 22):
            y = 0.62 * HEIGHT
            y += 70 * math.sin((x / WIDTH) * math.pi * 1.7 + offset * 0.17)
            y += 22 * math.sin((x / WIDTH) * math.pi * 5.2 + offset * 0.4)
            y += offset * 18
            points.append((x, y))
        draw.line(points, fill=line_color, width=2)
    base.alpha_composite(layer)


def add_node_field(base):
    rng = random.Random(7)
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    nodes = []
    for _ in range(34):
        x = rng.randint(150, WIDTH - 150)
        y = rng.randint(180, HEIGHT - 160)
        nodes.append((x, y))

    for i, (x1, y1) in enumerate(nodes):
        nearest = sorted(
            ((math.dist((x1, y1), (x2, y2)), j, x2, y2) for j, (x2, y2) in enumerate(nodes) if j != i),
            key=lambda item: item[0],
        )[:2]
        for dist, _, x2, y2 in nearest:
            if dist < 300:
                draw.line((x1, y1, x2, y2), fill=NAVY + (32,), width=1)

    palette = [BLUE, GREEN, AMBER, RUST]
    for idx, (x, y) in enumerate(nodes):
        fill = palette[idx % len(palette)]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=fill + (170,), outline=(255, 255, 255, 180), width=2)

    layer = layer.filter(ImageFilter.GaussianBlur(0))
    base.alpha_composite(layer)


def add_focus_arcs(base):
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    boxes = [
        (120, 120, 760, 760),
        (420, 120, 1180, 820),
        (980, 80, 1710, 760),
    ]
    colors = [BLUE, GREEN, AMBER]
    for box, color in zip(boxes, colors):
        draw.arc(box, start=200, end=340, fill=color + (95,), width=12)
        draw.arc(box, start=20, end=120, fill=color + (60,), width=6)
    layer = layer.filter(ImageFilter.GaussianBlur(2))
    base.alpha_composite(layer)


def main():
    base = vertical_gradient((WIDTH, HEIGHT), BG_TOP, BG_BOTTOM).convert('RGBA')

    add_soft_blob(base, (320, 220), 190, BLUE, 70)
    add_soft_blob(base, (760, 340), 220, GREEN, 60)
    add_soft_blob(base, (1180, 220), 210, AMBER, 58)
    add_soft_blob(base, (1510, 300), 170, RUST, 54)
    add_soft_blob(base, (1430, 740), 210, NAVY, 30)

    add_ribbon(base, BLUE, width_scale=1.5, amp=46, phase=0.7, y_center=300, thickness=56, alpha=68)
    add_ribbon(base, GREEN, width_scale=1.9, amp=58, phase=2.0, y_center=430, thickness=64, alpha=56)
    add_ribbon(base, AMBER, width_scale=1.6, amp=52, phase=3.4, y_center=560, thickness=60, alpha=52)
    add_ribbon(base, RUST, width_scale=1.8, amp=44, phase=4.4, y_center=690, thickness=54, alpha=48)

    add_contour_lines(base)
    add_node_field(base)
    add_focus_arcs(base)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    base.convert('RGB').save(OUT, quality=95)


if __name__ == '__main__':
    main()
