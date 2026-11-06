from src.measurement import load_font, measure, split_lines


def test_split_and_measure():
    font = load_font("DejaVu Serif")
    text = "This is a short test sentence to measure how much space it occupies."
    lines = split_lines(text, font)
    count, area = measure(text, font)
    assert count == len(lines)
    assert area > 0


def test_repeatable():
    font = load_font("DejaVu Sans")
    assert measure("An example line.", font) == measure("An example line.", font)
