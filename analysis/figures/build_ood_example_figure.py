"""Build the illustrative full-spectrum grouping figure (ood_examples.png).

Downloads and caches the Creative Commons Wikimedia source photos into
AAUgraphics/example_sources on first run. Superseded by build_ood_examples_v2.py,
which reads the same local images without any download.
Run: python analysis/figures/build_ood_example_figure.py
"""
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2] / 'draft' / 'aauReportTemplate'
ASSET_DIR = ROOT / 'AAUgraphics' / 'example_sources'
OUT_PATH = ROOT / 'AAUgraphics' / 'ood_examples.png'

SOURCES = {
    'clean_id': (
        'https://upload.wikimedia.org/wikipedia/commons/8/88/Mi_Golden_Retriever.jpg',
        ASSET_DIR / 'clean_id_dog.jpg',
        'Clean ID',
        'Golden retriever',
    ),
    'near_ood': (
        'https://upload.wikimedia.org/wikipedia/commons/a/a0/Red_fox.jpg',
        ASSET_DIR / 'near_ood_fox.jpg',
        'Near-OOD',
        'Red fox',
    ),
    'far_ood': (
        'https://upload.wikimedia.org/wikipedia/commons/e/eb/Airbus_A320_%281%29.jpg',
        ASSET_DIR / 'far_ood_airplane.jpg',
        'Far-OOD',
        'Airplane',
    ),
}


def ensure_download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(request) as response, path.open('wb') as handle:
        handle.write(response.read())


def load_panel(path, size):
    image = Image.open(path).convert('RGB')
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def add_label(draw, xy, title, subtitle, font_title, font_body):
    x, y = xy
    draw.rounded_rectangle((x, y, x + 230, y + 62), radius=10, fill=(255, 255, 255))
    draw.text((x + 12, y + 8), title, fill=(0, 0, 0), font=font_title)
    draw.text((x + 12, y + 34), subtitle, fill=(50, 50, 50), font=font_body)


def main():
    for url, path, _, _ in SOURCES.values():
        ensure_download(url, path)

    panel_size = (620, 420)
    clean = load_panel(SOURCES['clean_id'][1], panel_size)
    csid = clean.filter(ImageFilter.GaussianBlur(radius=4))
    near = load_panel(SOURCES['near_ood'][1], panel_size)
    far = load_panel(SOURCES['far_ood'][1], panel_size)

    canvas = Image.new('RGB', (1340, 980), color=(247, 247, 247))
    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()

    positions = {
        'clean_id': (40, 60),
        'cs_id': (680, 60),
        'near_ood': (40, 520),
        'far_ood': (680, 520),
    }
    panels = {
        'clean_id': clean,
        'cs_id': csid,
        'near_ood': near,
        'far_ood': far,
    }
    labels = {
        'clean_id': ('Clean ID', 'Known class'),
        'cs_id': ('cs-ID', 'Same class under blur'),
        'near_ood': ('Near-OOD', 'Semantically nearby'),
        'far_ood': ('Far-OOD', 'Clearly unrelated'),
    }

    draw.text((40, 20), 'Illustrative full-spectrum grouping', fill=(0, 0, 0), font=font_title)

    for key, image in panels.items():
        x, y = positions[key]
        canvas.paste(image, (x, y))
        add_label(draw, (x + 16, y + 16), labels[key][0], labels[key][1], font_title, font_body)
        draw.rectangle((x, y, x + panel_size[0], y + panel_size[1]), outline=(180, 180, 180), width=2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH)


if __name__ == '__main__':
    main()
