from pathlib import Path
import re

import numpy as np
import pandas as pd

from .measurement import FONT_NAMES, load_font, measure, preview

ROOT = Path(__file__).resolve().parents[1]
BOOKS = {
    "alice": ("Алиса в Стране чудес", "https://www.gutenberg.org/ebooks/11"),
    "pride": ("Гордость и предубеждение", "https://www.gutenberg.org/ebooks/1342"),
    "holmes": ("Приключения Шерлока Холмса", "https://www.gutenberg.org/ebooks/1661"),
}


def fragments(text, rng):
    pieces = []
    for paragraph in re.split(r"\n\s*\n", text):
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z“])', paragraph.strip())
        chunk, target = [], int(rng.integers(30, 47))
        for sentence in sentences:
            chunk.append(sentence)
            if len(" ".join(chunk).split()) >= target:
                pieces.append(" ".join(chunk).strip())
                chunk, target = [], int(rng.integers(30, 47))
        if chunk and len(" ".join(chunk).split()) >= 12:
            pieces.append(" ".join(chunk).strip())
    return [s for s in pieces if len(s.split()) >= 20]


def main():
    rng = np.random.default_rng(2026)
    paragraphs = []
    for key, (book, url) in BOOKS.items():
        text = (ROOT / "data" / "texts" / f"{key}.txt").read_text(encoding="utf-8")
        variants = fragments(text, rng)
        if len(variants) < 18:
            raise ValueError(f"Слишком мало фрагментов в {key}: {len(variants)}")
        selected = rng.choice(len(variants), 18, replace=False)
        for i, index in enumerate(selected, start=1):
            paragraphs.append({"id": f"{key}-{i:02d}", "book": book, "source_url": url, "text": variants[index]})

    p = pd.DataFrame(paragraphs)
    p.to_csv(ROOT / "data" / "fragments.csv", index=False)
    fonts = {name: load_font(name) for name in FONT_NAMES}
    rows = []
    for item in p.itertuples(index=False):
        for font_name, font in fonts.items():
            lines, ink_cm2 = measure(item.text, font)
            rows.append({"id": item.id, "book": item.book, "font": font_name,
                         "chars": len(item.text), "words": len(item.text.split()),
                         "lines": lines, "ink_cm2": ink_cm2})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "layout.csv", index=False)
    preview(p.iloc[0].text, "DejaVu Serif", ROOT / "figures" / "example_a4.png")
    print(f"Фрагментов: {len(p)}; измерений: {len(df)}")
    print(df.groupby("font")["lines"].mean().round(2).to_string())


if __name__ == "__main__":
    main()
