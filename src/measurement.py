from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

DPI = 144
FONT_SIZE_PT = 12
TEXT_WIDTH_MM = 170
FONT_NAMES = ("DejaVu Serif", "DejaVu Sans", "DejaVu Sans Mono")


def load_font(name):
    file_path = font_manager.findfont(font_manager.FontProperties(family=name))
    return ImageFont.truetype(file_path, round(FONT_SIZE_PT * DPI / 72))


def split_lines(text, font):
    max_width_px = round(TEXT_WIDTH_MM * DPI / 25.4)
    result, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and font.getlength(candidate) > max_width_px:
            result.append(current)
            current = word
        else:
            current = candidate
    if current:
        result.append(current)
    return result


def measure(text, font):
    lines = split_lines(text, font)
    width_px = round(TEXT_WIDTH_MM * DPI / 25.4)
    line_step = round(FONT_SIZE_PT * DPI / 72 * 1.4)
    canvas = Image.new("L", (width_px, (len(lines) + 1) * line_step), 0)
    drawer = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        drawer.text((0, i * line_step), line, font=font, fill=255)
    dark_pixels = np.asarray(canvas, dtype=np.float32).sum() / 255
    ink_cm2 = dark_pixels * (2.54 / DPI) ** 2
    return len(lines), round(float(ink_cm2), 4)


def preview(text, font_name, path):
    font = load_font(font_name)
    lines = split_lines(text, font)
    line_step = round(FONT_SIZE_PT * DPI / 72 * 1.4)
    im = Image.new("RGB", (round(210 * DPI / 25.4), round(297 * DPI / 25.4)), "white")
    d = ImageDraw.Draw(im)
    x, y = round(20 * DPI / 25.4), round(20 * DPI / 25.4)
    for i, line in enumerate(lines):
        d.text((x, y + i * line_step), line, font=font, fill="#222222")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
