"""Build the dataset-example galleries (dataset_examples_shift.png and dataset_examples_ood.png).

Samples the first available image from each benchmark imglist under data/ and tiles them
into the clean/cs-ID shift gallery and the near/far-OOD gallery.
Run: python analysis/figures/build_dataset_gallery.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2] / 'draft' / 'aauReportTemplate'
DATA_ROOT = ROOT.parents[1] / 'data'
LARGE_ROOT = DATA_ROOT / 'images_largescale'
CLASSIC_ROOT = DATA_ROOT / 'images_classic'

SHIFT_OUT = ROOT / 'AAUgraphics' / 'dataset_examples_shift.png'
OOD_OUT = ROOT / 'AAUgraphics' / 'dataset_examples_ood.png'

PANEL_W = 360
PANEL_H = 250
GAP = 28
MARGIN = 34
HEADER_H = 88
FOOTER_H = 76


SHIFT_DATASETS = [
    {
        'name': 'ImageNet-200',
        'role': 'Clean ID',
        'imglist': DATA_ROOT / 'benchmark_imglist' / 'imagenet200' / 'val_imagenet200.txt',
        'root': LARGE_ROOT,
    },
    {
        'name': 'ImageNet-1K',
        'role': 'Clean ID',
        'imglist': DATA_ROOT / 'benchmark_imglist' / 'imagenet' / 'val_imagenet.txt',
        'root': LARGE_ROOT,
    },
    {
        'name': 'ImageNet-C',
        'role': 'cs-ID',
        'imglist': DATA_ROOT / 'benchmark_imglist' / 'imagenet200' / 'test_imagenet200_c.txt',
        'root': LARGE_ROOT,
    },
    {
        'name': 'ImageNet-R',
        'role': 'cs-ID',
        'imglist': DATA_ROOT / 'benchmark_imglist' / 'imagenet200' / 'test_imagenet200_r.txt',
        'root': LARGE_ROOT,
    },
    {
        'name': 'ImageNet-V2',
        'role': 'cs-ID',
        'imglist': DATA_ROOT / 'benchmark_imglist' / 'imagenet200' / 'test_imagenet200_v2.txt',
        'root': LARGE_ROOT,
    },
    {
        'name': 'ImageNet-ES',
        'role': 'cs-ID',
        'imglist': DATA_ROOT / 'benchmark_imglist' / 'imagenet' / 'test_imagenet_es.txt',
        'root': LARGE_ROOT,
        'allow_missing': True,
        'missing_note': 'Configured in ImageNet-1K branch; local sample not mirrored',
    },
]

OOD_DATASETS = [
    {
        'name': 'SSB-hard',
        'role': 'Near-OOD',
        'imglist': DATA_ROOT / 'benchmark_imglist' / 'imagenet' / 'test_ssb_hard.txt',
        'root': LARGE_ROOT,
    },
    {
        'name': 'NINCO',
        'role': 'Near-OOD',
        'imglist': DATA_ROOT / 'benchmark_imglist' / 'imagenet' / 'test_ninco.txt',
        'root': LARGE_ROOT,
    },
    {
        'name': 'iNaturalist',
        'role': 'Far-OOD',
        'imglist': DATA_ROOT / 'benchmark_imglist' / 'imagenet' / 'test_inaturalist.txt',
        'root': LARGE_ROOT,
    },
    {
        'name': 'Textures',
        'role': 'Far-OOD',
        'imglist': DATA_ROOT / 'benchmark_imglist' / 'imagenet' / 'test_textures.txt',
        'root': CLASSIC_ROOT,
    },
    {
        'name': 'OpenImage-O',
        'role': 'Far-OOD',
        'imglist': DATA_ROOT / 'benchmark_imglist' / 'imagenet' / 'test_openimage_o.txt',
        'root': LARGE_ROOT,
    },
]


def first_existing_sample(imglist_path, root):
    with imglist_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rel_path = line.split()[0]
            candidate = root / rel_path
            if candidate.is_file():
                return candidate
    return None


def load_panel(sample_path):
    image = Image.open(sample_path).convert('RGB')
    return ImageOps.fit(image, (PANEL_W, PANEL_H), method=Image.Resampling.LANCZOS)


def wrap_text(draw, text, font, max_width):
    words = text.split()
    lines = []
    current = ''
    for word in words:
        trial = word if not current else f'{current} {word}'
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_missing_panel(draw, x0, y0, note, font_note):
    draw.rounded_rectangle((x0, y0, x0 + PANEL_W, y0 + PANEL_H), radius=18, fill=(231, 233, 236), outline=(180, 184, 189), width=3)
    lines = wrap_text(draw, note, font_note, PANEL_W - 40)
    total_h = len(lines) * 18
    y = y0 + (PANEL_H - total_h) / 2
    for line in lines:
        w = draw.textlength(line, font=font_note)
        draw.text((x0 + (PANEL_W - w) / 2, y), line, fill=(70, 74, 80), font=font_note)
        y += 18


def draw_gallery(items, title, subtitle, out_path):
    cols = 3
    rows = (len(items) + cols - 1) // cols
    width = MARGIN * 2 + cols * PANEL_W + (cols - 1) * GAP
    height = HEADER_H + rows * (PANEL_H + FOOTER_H) + (rows - 1) * GAP + MARGIN
    canvas = Image.new('RGB', (width, height), (247, 247, 245))
    draw = ImageDraw.Draw(canvas)
    font_title = ImageFont.load_default()
    font_body = ImageFont.load_default()
    font_note = ImageFont.load_default()

    draw.text((MARGIN, 20), title, fill=(18, 18, 18), font=font_title)
    draw.text((MARGIN, 46), subtitle, fill=(70, 70, 70), font=font_body)

    for idx, item in enumerate(items):
        row = idx // cols
        col = idx % cols
        x = MARGIN + col * (PANEL_W + GAP)
        y = HEADER_H + row * (PANEL_H + FOOTER_H + GAP)

        sample = first_existing_sample(item['imglist'], item['root'])
        if sample is not None:
            canvas.paste(load_panel(sample), (x, y))
            draw.rectangle((x, y, x + PANEL_W, y + PANEL_H), outline=(180, 180, 180), width=2)
        else:
            note = item.get('missing_note', 'Sample unavailable in local artifact bundle')
            draw_missing_panel(draw, x, y, note, font_note)

        draw.text((x, y + PANEL_H + 10), item['name'], fill=(0, 0, 0), font=font_title)
        draw.text((x, y + PANEL_H + 34), item['role'], fill=(70, 70, 70), font=font_body)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    draw_gallery(
        SHIFT_DATASETS,
        'Examples from the clean-ID and covariate-shifted datasets',
        'Direct benchmark samples from the local artifact bundle used by the thesis runtime.',
        SHIFT_OUT,
    )
    draw_gallery(
        OOD_DATASETS,
        'Examples from the semantic near-OOD and far-OOD datasets',
        'Direct benchmark samples shared across the ImageNet-200 and ImageNet-1K study.',
        OOD_OUT,
    )


if __name__ == '__main__':
    main()
