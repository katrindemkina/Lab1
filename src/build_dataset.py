from __future__ import annotations
import hashlib
from pathlib import Path
import subprocess
import string
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

PROJECT = Path(__file__).resolve().parent.parent
FONTS = {
    'sans-serif': [
        'Inter', 'Inter Display', 'Lato', 'Roboto', 'Roboto Condensed',
        'Open Sans', 'Open Sans Condensed', 'Liberation Sans',
        'Liberation Sans Narrow', 'DejaVu Sans', 'Noto Sans',
        'PT Sans', 'PT Sans Caption', 'PT Sans Narrow', 'Cantarell', 'Nimbus Sans',
    ],
    'serif': [
        'DejaVu Serif', 'Liberation Serif', 'Noto Serif', 'Noto Serif Display',
        'PT Serif', 'PT Serif Caption', 'Roboto Slab', 'Nimbus Roman',
    ],
    'monospace': [
        'DejaVu Sans Mono', 'Liberation Mono', 'Noto Sans Mono',
        'PT Mono', 'Nimbus Mono PS',
    ],
    'display': ['Comic Neue', 'Lobster Two'],
}
FONT_PX = 128
CHARACTERS = string.ascii_letters


def locate_font(family: str) -> Path:
    """Resolve installed Regular face, with an explicit family check."""
    r = subprocess.check_output(
        ['fc-match', '-f', '%{family[0]}|%{style[0]}|%{file}', f'{family}:style=Regular'],
        text=True,
    ).strip()
    resolved_family, style, path = r.split('|', 2)
    if resolved_family.casefold() != family.casefold():
        raise ValueError(f'fontconfig substituted {family!r} -> {resolved_family!r}')
    return Path(path)


def face_info(path: Path) -> tuple[dict, dict, int]:
    with TTFont(path, lazy=True) as f:
        cmap = f.getBestCmap() or {}
        if 'hmtx' not in f or 'head' not in f:
            raise ValueError(f'Missing font metrics in {path}')
        hmtx = f['hmtx'].metrics
        upm = f['head'].unitsPerEm
        return cmap, hmtx, upm


def glyph_metrics(font_path: Path, family: str, category: str, char: str,
                  cmap: dict, hmtx: dict, upm: int, draw_font: ImageFont.FreeTypeFont):
    cp = ord(char)
    if cp not in cmap:
        raise ValueError(f'Glyph {char!r} is missing from {font_path}')
    adv_units = hmtx[cmap[cp]][0]
    bbox = draw_font.getbbox(char, anchor='ls')
    height_px = max(0, bbox[3] - bbox[1])
    img = Image.new('L', (256, 256), 0)
    ImageDraw.Draw(img).text((40, 160), char, font=draw_font,
                             fill=255, anchor='ls')
    ink_em2 = float(np.asarray(img, dtype=np.float64).sum()) / (255.0 * FONT_PX**2)
    return {
        'family': family, 'category': category, 'character': char,
        'is_uppercase': int(char.isupper()),
        'advance_em': adv_units / upm,
        'ink_area_em2': ink_em2,
        'bbox_height_em': height_px / FONT_PX,
    }


def build():
    rows, provenance = [], []
    for category, families in FONTS.items():
        for family in families:
            path = locate_font(family)
            cmap, hmtx, upm = face_info(path)
            drawable = ImageFont.truetype(str(path), FONT_PX)
            source_package = subprocess.run(['dpkg-query', '-S', str(path)],
                                            capture_output=True, text=True)
            package = (source_package.stdout.split(':', 1)[0]
                       if source_package.returncode == 0 else 'not reported')
            provenance.append({
                'family': family, 'category': category,
                'source_file': str(path),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'package': package,
                'units_per_em': upm,
                'file_size_bytes': path.stat().st_size,
            })
            for char in CHARACTERS:
                rows.append(glyph_metrics(path, family, category, char,
                                          cmap, hmtx, upm, drawable))
    primary = pd.DataFrame(rows)
    manifest = pd.DataFrame(provenance)
    assert primary.shape[0] == len(manifest) * len(CHARACTERS)
    assert not primary[['advance_em','ink_area_em2','bbox_height_em']].isna().any().any()
    assert primary[['advance_em','ink_area_em2','bbox_height_em']].gt(0).all().all()
    data_dir = PROJECT / 'data'
    primary.to_csv(data_dir / 'glyph_metrics.csv', index=False, float_format='%.12g')
    manifest.to_csv(data_dir / 'primary_font_manifest.csv', index=False)

    extra_rows = []
    for path in sorted(Path('/usr/share/fonts/truetype/noto').glob('*-Regular.ttf')):
        if path.name in {'NotoSans-Regular.ttf', 'NotoSerif-Regular.ttf',
                         'NotoSansDisplay-Regular.ttf', 'NotoSerifDisplay-Regular.ttf',
                         'NotoSansMono-Regular.ttf'}:
            continue
        with TTFont(path, lazy=True) as tt:
            cmap = tt.getBestCmap() or {}
            glyph_count = len(cmap)
        extra_rows.append({
            'family': path.stem.removesuffix('-Regular'),
            'glyph_count': glyph_count,
            'file_size_kib': path.stat().st_size / 1024,
            'source_file': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    secondary = pd.DataFrame(extra_rows)
    secondary.to_csv(data_dir / 'noto_multiscript.csv', index=False, float_format='%.12g')
    print('Primary:', primary.shape, primary.category.value_counts().to_dict())
    print('Secondary:', secondary.shape)
    print('Statistics:')
    print(primary[['advance_em', 'ink_area_em2', 'bbox_height_em']].describe())
    print(secondary[['glyph_count','file_size_kib']].describe())
    return primary, secondary


if __name__ == '__main__':
    build()
